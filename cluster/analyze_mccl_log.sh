#!/bin/bash

# 分析MCCL日志文件脚本
# 用法: ./analyze_mccl_logs.sh <目录路径> <文件前缀>
# 示例: ./analyze_mccl_logs.sh /path/to/logs mccl_

if [ $# -lt 2 ]; then
    echo "用法: $0 <目录路径> <文件前缀>"
    echo "示例: $0 /path/to/logs mccl_"
    exit 1
fi

LOG_DIR="$1"
FILE_PREFIX="$2"

if [ ! -d "$LOG_DIR" ]; then
    echo "错误: 目录 '$LOG_DIR' 不存在"
    exit 1
fi

# 临时文件存储分析结果
TEMP_BW_FILE=$(mktemp)
TEMP_WRONG_FILE=$(mktemp)

# 清理临时文件
cleanup() {
    rm -f "$TEMP_BW_FILE" "$TEMP_WRONG_FILE"
}
trap cleanup EXIT

# 从文件中提取hostname列表（去重）
extract_hostnames() {
    local file="$1"
    grep -E "Rank\s+[0-9]+\s+Group\s+[0-9]+\s+Pid\s+[0-9]+\s+on\s+" "$file" | \
        sed -E 's/.*on\s+([a-zA-Z0-9_-]+)\s+device.*/\1/' | \
        sort -u | tr '\n' ',' | sed 's/,$//'
}

# 从文件中提取Avg bus bandwidth值
extract_avg_bandwidth() {
    local file="$1"
    grep "Avg bus bandwidth" "$file" 2>/dev/null | \
        sed -E 's/.*Avg bus bandwidth\s*:\s*([0-9.]+).*/\1/'
}

# 检查文件中是否有#wrong列非零的行
check_wrong_values() {
    local file="$1"
    # 跳过注释行，查找数据行中#wrong列非零的情况
    # 数据行格式: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
    # #wrong 在第9列和第13列
    awk '
    /^[[:space:]]*[0-9]/ {
        # 这是数据行
        if (NF >= 13) {
            wrong1 = $9
            wrong2 = $13
            if (wrong1 != "0" || wrong2 != "0") {
                print $0
            }
        }
    }
    ' "$file"
}

echo "=============================================="
echo "MCCL日志分析报告"
echo "目录: $LOG_DIR"
echo "文件前缀: $FILE_PREFIX"
echo "=============================================="
echo ""

# 遍历所有匹配的文件
file_count=0
for file in "$LOG_DIR"/"$FILE_PREFIX"*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        hostnames=$(extract_hostnames "$file")
        avg_bw=$(extract_avg_bandwidth "$file")
        
        if [ -n "$avg_bw" ]; then
            echo "$avg_bw|$filename|$hostnames" >> "$TEMP_BW_FILE"
            file_count=$((file_count + 1))
        fi
        
        # 检查wrong值
        wrong_lines=$(check_wrong_values "$file")
        if [ -n "$wrong_lines" ]; then
            echo "FILE:$filename|HOSTS:$hostnames" >> "$TEMP_WRONG_FILE"
            echo "$wrong_lines" >> "$TEMP_WRONG_FILE"
            echo "---" >> "$TEMP_WRONG_FILE"
        fi
    fi
done

if [ $file_count -eq 0 ]; then
    echo "警告: 没有找到匹配 '$FILE_PREFIX*' 的文件"
    exit 0
fi

echo "找到 $file_count 个文件"
echo ""

# ==================== 1. 按Avg bus bandwidth排序 ====================
echo "=============================================="
echo "1. Avg Bus Bandwidth 排序分析"
echo "=============================================="

if [ -s "$TEMP_BW_FILE" ]; then
    # 获取最大值
    max_bw=$(sort -t'|' -k1 -rn "$TEMP_BW_FILE" | head -1 | cut -d'|' -f1)
    
    echo ""
    printf "%-5s %-40s %-20s %-15s %s\n" "序号" "文件名" "Hostnames" "Avg BW (GB/s)" "与最大值差距"
    printf "%-5s %-40s %-20s %-15s %s\n" "-----" "----------------------------------------" "--------------------" "---------------" "---------------"
    
    # 按带宽降序排序并输出
    index=1
    sort -t'|' -k1 -rn "$TEMP_BW_FILE" | while IFS='|' read -r bw filename hosts; do
        if [ -n "$bw" ] && [ -n "$max_bw" ]; then
            # 计算与最大值的差距百分比
            diff_percent=$(awk -v bw="$bw" -v max="$max_bw" 'BEGIN {
                if (max > 0) {
                    diff = (max - bw) / max * 100
                    printf "%.2f%%", diff
                } else {
                    print "N/A"
                }
            }')
            printf "%-5s %-40s %-20s %-15s %s\n" "$index" "$filename" "$hosts" "$bw" "$diff_percent"
            ((index++))
        fi
    done
    
    echo ""
    echo "最大 Avg Bus Bandwidth: $max_bw GB/s"
else
    echo "没有找到包含 Avg bus bandwidth 的文件"
fi

echo ""

# ==================== 2. #wrong非零的文件 ====================
echo "=============================================="
echo "2. #wrong 列非零的文件分析"
echo "=============================================="
echo ""

if [ -s "$TEMP_WRONG_FILE" ]; then
    echo "发现以下文件包含 #wrong 非零的记录:"
    echo ""
    
    current_file=""
    current_hosts=""
    file_index=1
    
    while IFS= read -r line; do
        if [[ "$line" == FILE:* ]]; then
            # 解析文件名和hosts
            current_file=$(echo "$line" | sed 's/FILE:\([^|]*\)|.*/\1/')
            current_hosts=$(echo "$line" | sed 's/.*HOSTS://')
            echo "[$file_index] 文件: $current_file"
            echo "    Hostnames: $current_hosts"
            echo "    异常数据行:"
            printf "      %-15s %-15s %-8s %-8s %-8s %-10s %-10s %-10s %-8s %-10s %-10s %-10s %-8s\n" \
                   "Size(B)" "Count" "Type" "Redop" "Root" "Time(us)" "AlgBW" "BusBW" "#Wrong" "Time(us)" "AlgBW" "BusBW" "#Wrong"
            ((file_index++))
        elif [[ "$line" == "---" ]]; then
            echo ""
        elif [[ -n "$line" ]]; then
            echo "      $line"
        fi
    done < "$TEMP_WRONG_FILE"
else
    echo "所有文件的 #wrong 列均为零，没有发现异常"
fi

echo ""
echo "=============================================="
echo "分析完成"
echo "=============================================="

