#!/bin/bash
#
# 检查 hostfile 中所有节点的 ulimit -a 和 ulimit -aH 输出，
# 并统计每个节点之间的差异。
#
# 用法:
#   bash check_ulimit.sh <hostfile> [选项]
#
# 示例:
#   bash check_ulimit.sh ../hostfile.test
#   bash check_ulimit.sh ../hostfile.test -p 16
#   bash check_ulimit.sh ../hostfile.test -t 15
#   bash check_ulimit.sh ../hostfile.test -o ulimit_report
#

set -euo pipefail

# ===================== 颜色定义 =====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ===================== 默认参数 =====================
MAX_PARALLEL=8
SSH_TIMEOUT=10
OUTPUT_DIR=""

# ===================== 使用说明 =====================
usage() {
    echo "用法: $0 <hostfile> [选项]"
    echo ""
    echo "功能: 收集所有节点的 ulimit -a (soft) 和 ulimit -aH (hard) 输出，并对比节点间差异"
    echo ""
    echo "选项:"
    echo "  -p <并发数>       最大并发SSH连接数 (默认: 8)"
    echo "  -t <超时秒数>     SSH连接超时时间 (默认: 10)"
    echo "  -o <输出目录>     结果输出目录 (默认: ulimit_report_<时间戳>)"
    echo "  -h                显示帮助信息"
    exit 1
}

# ===================== 参数解析 =====================
if [ $# -lt 1 ]; then
    usage
fi

HOSTFILE="$1"
shift

while getopts "p:t:o:h" opt; do
    case $opt in
        p) MAX_PARALLEL="$OPTARG" ;;
        t) SSH_TIMEOUT="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ ! -f "$HOSTFILE" ]; then
    echo -e "${RED}错误: hostfile 不存在: $HOSTFILE${NC}"
    exit 1
fi

# 生成输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="ulimit_report_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUTPUT_DIR"

# ===================== 解析 hostfile =====================
# 支持 hostfile 格式: IP [slots] 或纯 IP，每行一个
parse_hostfile() {
    local file="$1"
    local ips=()

    while IFS= read -r line || [ -n "$line" ]; do
        # 跳过空行和注释
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        # 提取第一个字段作为 IP
        local node_ip=$(echo "$line" | awk '{print $1}')
        [[ -n "$node_ip" ]] && ips+=("$node_ip")
    done < "$file"

    printf '%s\n' "${ips[@]}"
}

mapfile -t ALL_IPS < <(parse_hostfile "$HOSTFILE")
TOTAL=${#ALL_IPS[@]}

if [ $TOTAL -eq 0 ]; then
    echo -e "${RED}错误: hostfile 中没有有效 IP${NC}"
    exit 1
fi

echo -e "${BOLD}======================================"
echo "  ulimit 差异检查工具"
echo -e "======================================${NC}"
echo "hostfile: $HOSTFILE"
echo "节点数:   $TOTAL"
echo "并发数:   $MAX_PARALLEL"
echo "超时:     ${SSH_TIMEOUT}s"
echo "输出目录: $OUTPUT_DIR"
echo "--------------------------------------"

# ===================== 收集 ulimit 信息 =====================
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=${SSH_TIMEOUT} -o LogLevel=ERROR"

collect_ulimit() {
    local ip="$1"
    local soft_file="${OUTPUT_DIR}/${ip}.soft"
    local hard_file="${OUTPUT_DIR}/${ip}.hard"
    local status_file="${OUTPUT_DIR}/${ip}.status"

    # 执行远程 ulimit 命令
    local output
    if output=$(ssh -n $SSH_OPTS "$ip" "echo '===SOFT==='; ulimit -a; echo '===HARD==='; ulimit -aH" 2>&1); then
        # 分离 soft 和 hard 输出
        local in_soft=0
        local in_hard=0
        > "$soft_file"
        > "$hard_file"

        while IFS= read -r line; do
            if [[ "$line" == "===SOFT===" ]]; then
                in_soft=1; in_hard=0; continue
            elif [[ "$line" == "===HARD===" ]]; then
                in_soft=0; in_hard=1; continue
            fi
            if [ $in_soft -eq 1 ]; then
                echo "$line" >> "$soft_file"
            elif [ $in_hard -eq 1 ]; then
                echo "$line" >> "$hard_file"
            fi
        done <<< "$output"

        echo "OK" > "$status_file"
        echo -e "  ${GREEN}✓${NC} ${ip}"
    else
        echo "FAIL" > "$status_file"
        echo "$output" > "${OUTPUT_DIR}/${ip}.error"
        echo -e "  ${RED}✗${NC} ${ip} (连接失败)"
    fi
}

echo -e "${CYAN}正在收集各节点 ulimit 信息...${NC}"

# 并发收集
PIDS=()
RUNNING=0

for ip in "${ALL_IPS[@]}"; do
    # 控制并发数
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" 2>/dev/null
                unset 'PIDS[$i]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        sleep 0.3
    done

    collect_ulimit "$ip" &
    PIDS+=($!)
    RUNNING=$((RUNNING + 1))
done

# 等待所有任务完成
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null
done

echo ""

# ===================== 统计成功/失败 =====================
SUCCESS_IPS=()
FAIL_IPS=()

for ip in "${ALL_IPS[@]}"; do
    if [ -f "${OUTPUT_DIR}/${ip}.status" ] && grep -q "OK" "${OUTPUT_DIR}/${ip}.status"; then
        SUCCESS_IPS+=("$ip")
    else
        FAIL_IPS+=("$ip")
    fi
done

echo -e "收集完成: ${GREEN}${#SUCCESS_IPS[@]} 成功${NC}, ${RED}${#FAIL_IPS[@]} 失败${NC}"

if [ ${#SUCCESS_IPS[@]} -eq 0 ]; then
    echo -e "${RED}没有成功收集到任何节点的 ulimit 信息，退出${NC}"
    exit 1
fi

# ===================== 解析 ulimit 输出为 key=value 格式 =====================
# ulimit -a 的输出格式示例:
#   core file size          (blocks, -c) 0
#   data seg size           (kbytes, -d) unlimited
#   ...
# 我们提取为: key=value 方便比较
parse_ulimit_file() {
    local file="$1"
    local parsed_file="${file%.soft}.soft.parsed"
    if [[ "$file" == *.hard ]]; then
        parsed_file="${file%.hard}.hard.parsed"
    fi

    > "$parsed_file"

    while IFS= read -r line; do
        # 跳过空行
        [[ -z "$line" ]] && continue
        # 解析格式: 描述 (单位, 选项) 值
        # 提取选项字母和值
        if [[ "$line" =~ \((([^)]+)\)[[:space:]]+([^[:space:]]+) ]]; then
            local opt="${BASH_REMATCH[1]}"
            local val="${BASH_REMATCH[2]}"
            # 清理选项，取最后一个字母 (如 "-c" -> "c")
            opt=$(echo "$opt" | sed 's/.*-//')
            echo "${opt}=${val}" >> "$parsed_file"
        fi
    done < "$file"
}

echo -e "${CYAN}正在解析 ulimit 输出...${NC}"

for ip in "${SUCCESS_IPS[@]}"; do
    parse_ulimit_file "${OUTPUT_DIR}/${ip}.soft"
    parse_ulimit_file "${OUTPUT_DIR}/${ip}.hard"
done

# ===================== 差异分析 =====================
echo ""
echo -e "${BOLD}======================================"
echo "  差异分析"
echo -e "======================================${NC}"

# --- 1. Soft limit 差异 ---
echo -e "${CYAN}【1】Soft limit (ulimit -a) 节点间差异${NC}"
echo "--------------------------------------"

soft_diff_file="${OUTPUT_DIR}/soft_diff.txt"
> "$soft_diff_file"

# 收集所有 soft limit 的 key
ALL_SOFT_KEYS=()
declare -A SOFT_KEY_SEEN

for ip in "${SUCCESS_IPS[@]}"; do
    while IFS='=' read -r key val; do
        [[ -z "$key" ]] && continue
        if [[ -z "${SOFT_KEY_SEEN[$key]:-}" ]]; then
            SOFT_KEY_SEEN[$key]=1
            ALL_SOFT_KEYS+=("$key")
        fi
    done < "${OUTPUT_DIR}/${ip}.soft.parsed"
done

# 对每个 key，检查各节点值是否一致
SOFT_DIFF_COUNT=0

for key in "${ALL_SOFT_KEYS[@]}"; do
    declare -A SOFT_VAL_MAP
    FIRST_VAL=""
    HAS_DIFF=0

    for ip in "${SUCCESS_IPS[@]}"; do
        val=$(grep "^${key}=" "${OUTPUT_DIR}/${ip}.soft.parsed" 2>/dev/null | cut -d'=' -f2-)
        [[ -z "$val" ]] && val="NOT_SET"

        if [[ -z "$FIRST_VAL" ]]; then
            FIRST_VAL="$val"
        elif [[ "$val" != "$FIRST_VAL" ]]; then
            HAS_DIFF=1
        fi

        # 按 val 分组记录 IP
        if [[ -z "${SOFT_VAL_MAP[$val]:-}" ]]; then
            SOFT_VAL_MAP[$val]="$ip"
        else
            SOFT_VAL_MAP[$val]="${SOFT_VAL_MAP[$val]}, $ip"
        fi
    done

    if [ $HAS_DIFF -eq 1 ]; then
        SOFT_DIFF_COUNT=$((SOFT_DIFF_COUNT + 1))

        # 获取 key 的描述名
        key_desc=$(head -1 "${OUTPUT_DIR}/${SUCCESS_IPS[0]}.soft" | cat)  # fallback
        # 从原始文件提取描述
        key_desc=""
        for ip in "${SUCCESS_IPS[@]}"; do
            while IFS= read -r line; do
                if [[ "$line" =~ \(-${key}\) ]]; then
                    # 提取括号前的描述部分
                    key_desc=$(echo "$line" | sed "s/(.*)//" | xargs)
                    break 2
                fi
            done < "${OUTPUT_DIR}/${ip}.soft"
        done

        echo "" | tee -a "$soft_diff_file"
        echo -e "  ${YELLOW}► 选项 -${key} (${key_desc:-unknown}) 存在差异:${NC}" | tee -a "$soft_diff_file"
        for val in "${!SOFT_VAL_MAP[@]}"; do
            echo "      值: ${val}  ->  节点: ${SOFT_VAL_MAP[$val]}" | tee -a "$soft_diff_file"
        done
    fi

    unset SOFT_VAL_MAP
done

if [ $SOFT_DIFF_COUNT -eq 0 ]; then
    echo -e "  ${GREEN}✓ 所有节点 soft limit 完全一致${NC}" | tee -a "$soft_diff_file"
else
    echo "" | tee -a "$soft_diff_file"
    echo "  Soft limit 差异项数: ${SOFT_DIFF_COUNT}" | tee -a "$soft_diff_file"
fi

# --- 2. Hard limit 差异 ---
echo ""
echo -e "${CYAN}【2】Hard limit (ulimit -aH) 节点间差异${NC}"
echo "--------------------------------------"

hard_diff_file="${OUTPUT_DIR}/hard_diff.txt"
> "$hard_diff_file"

ALL_HARD_KEYS=()
declare -A HARD_KEY_SEEN

for ip in "${SUCCESS_IPS[@]}"; do
    while IFS='=' read -r key val; do
        [[ -z "$key" ]] && continue
        if [[ -z "${HARD_KEY_SEEN[$key]:-}" ]]; then
            HARD_KEY_SEEN[$key]=1
            ALL_HARD_KEYS+=("$key")
        fi
    done < "${OUTPUT_DIR}/${ip}.hard.parsed"
done

HARD_DIFF_COUNT=0

for key in "${ALL_HARD_KEYS[@]}"; do
    declare -A HARD_VAL_MAP
    FIRST_VAL=""
    HAS_DIFF=0

    for ip in "${SUCCESS_IPS[@]}"; do
        val=$(grep "^${key}=" "${OUTPUT_DIR}/${ip}.hard.parsed" 2>/dev/null | cut -d'=' -f2-)
        [[ -z "$val" ]] && val="NOT_SET"

        if [[ -z "$FIRST_VAL" ]]; then
            FIRST_VAL="$val"
        elif [[ "$val" != "$FIRST_VAL" ]]; then
            HAS_DIFF=1
        fi

        if [[ -z "${HARD_VAL_MAP[$val]:-}" ]]; then
            HARD_VAL_MAP[$val]="$ip"
        else
            HARD_VAL_MAP[$val]="${HARD_VAL_MAP[$val]}, $ip"
        fi
    done

    if [ $HAS_DIFF -eq 1 ]; then
        HARD_DIFF_COUNT=$((HARD_DIFF_COUNT + 1))

        key_desc=""
        for ip in "${SUCCESS_IPS[@]}"; do
            while IFS= read -r line; do
                if [[ "$line" =~ \(-${key}\) ]]; then
                    key_desc=$(echo "$line" | sed "s/(.*)//" | xargs)
                    break 2
                fi
            done < "${OUTPUT_DIR}/${ip}.hard"
        done

        echo "" | tee -a "$hard_diff_file"
        echo -e "  ${YELLOW}► 选项 -${key} (${key_desc:-unknown}) 存在差异:${NC}" | tee -a "$hard_diff_file"
        for val in "${!HARD_VAL_MAP[@]}"; do
            echo "      值: ${val}  ->  节点: ${HARD_VAL_MAP[$val]}" | tee -a "$hard_diff_file"
        done
    fi

    unset HARD_VAL_MAP
done

if [ $HARD_DIFF_COUNT -eq 0 ]; then
    echo -e "  ${GREEN}✓ 所有节点 hard limit 完全一致${NC}" | tee -a "$hard_diff_file"
else
    echo "" | tee -a "$hard_diff_file"
    echo "  Hard limit 差异项数: ${HARD_DIFF_COUNT}" | tee -a "$hard_diff_file"
fi

# --- 3. Soft vs Hard 差异 (同一节点内) ---
echo ""
echo -e "${CYAN}【3】Soft vs Hard 差异 (同一节点内 soft != hard 的项)${NC}"
echo "--------------------------------------"

svh_diff_file="${OUTPUT_DIR}/soft_vs_hard_diff.txt"
> "$svh_diff_file"

# 使用第一个成功节点获取所有 key 作为参考
REF_IP="${SUCCESS_IPS[0]}"
ALL_KEYS=()

declare -A REF_KEY_SEEN
while IFS='=' read -r key val; do
    [[ -z "$key" ]] && continue
    if [[ -z "${REF_KEY_SEEN[$key]:-}" ]]; then
        REF_KEY_SEEN[$key]=1
        ALL_KEYS+=("$key")
    fi
done < "${OUTPUT_DIR}/${REF_IP}.soft.parsed"

SVH_DIFF_COUNT=0

for ip in "${SUCCESS_IPS[@]}"; do
    NODE_DIFF=0
    NODE_DIFF_ITEMS=""

    for key in "${ALL_KEYS[@]}"; do
        soft_val=$(grep "^${key}=" "${OUTPUT_DIR}/${ip}.soft.parsed" 2>/dev/null | cut -d'=' -f2-)
        hard_val=$(grep "^${key}=" "${OUTPUT_DIR}/${ip}.hard.parsed" 2>/dev/null | cut -d'=' -f2-)
        [[ -z "$soft_val" ]] && soft_val="NOT_SET"
        [[ -z "$hard_val" ]] && hard_val="NOT_SET"

        if [[ "$soft_val" != "$hard_val" ]]; then
            NODE_DIFF=$((NODE_DIFF + 1))
            NODE_DIFF_ITEMS="${NODE_DIFF_ITEMS}    -${key}: soft=${soft_val}, hard=${hard_val}\n"
        fi
    done

    if [ $NODE_DIFF -gt 0 ]; then
        SVH_DIFF_COUNT=$((SVH_DIFF_COUNT + 1))
        echo -e "  ${YELLOW}► ${ip} 有 ${NODE_DIFF} 项 soft != hard:${NC}" | tee -a "$svh_diff_file"
        echo -e "$NODE_DIFF_ITEMS" | tee -a "$svh_diff_file"
    else
        echo -e "  ${GREEN}✓ ${ip} soft == hard (完全一致)${NC}" | tee -a "$svh_diff_file"
    fi
done

# --- 4. 汇总表 ---
echo ""
echo -e "${CYAN}【4】各节点 ulimit 值汇总表 (soft)${NC}"
echo "--------------------------------------"

summary_file="${OUTPUT_DIR}/summary_table.txt"
> "$summary_file"

# 打印表头
HEADER=$(printf "%-16s" "选项")
for ip in "${SUCCESS_IPS[@]}"; do
    HEADER="${HEADER}$(printf "%-16s" "$ip")"
done
echo "$HEADER" | tee -a "$summary_file"
SEP_LENGTH=$((16 * (${#SUCCESS_IPS[@]} + 1)))
SEP=$(head -c "$SEP_LENGTH" /dev/zero | tr '\0' '-')
echo "$SEP" | tee -a "$summary_file"

for key in "${ALL_SOFT_KEYS[@]}"; do
    # 获取描述
    key_desc=""
    for ip in "${SUCCESS_IPS[@]}"; do
        while IFS= read -r line; do
            if [[ "$line" =~ \(-${key}\) ]]; then
                key_desc=$(echo "$line" | sed "s/(.*)//" | xargs)
                break 2
            fi
        done < "${OUTPUT_DIR}/${ip}.soft"
    done

    ROW=$(printf "%-16s" "-${key}(${key_desc:0:6})")
    for ip in "${SUCCESS_IPS[@]}"; do
        val=$(grep "^${key}=" "${OUTPUT_DIR}/${ip}.soft.parsed" 2>/dev/null | cut -d'=' -f2-)
        [[ -z "$val" ]] && val="-"
        # 截断过长的值
        val_short="${val:0:14}"
        ROW="${ROW}$(printf "%-16s" "$val_short")"
    done
    echo "$ROW" | tee -a "$summary_file"
done

# --- 5. 失败节点列表 ---
if [ ${#FAIL_IPS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}【5】连接失败节点${NC}"
    echo "--------------------------------------"
    for ip in "${FAIL_IPS[@]}"; do
        echo -e "  ${RED}✗ ${ip}${NC}"
        if [ -f "${OUTPUT_DIR}/${ip}.error" ]; then
            echo "    错误: $(cat "${OUTPUT_DIR}/${ip}.error" | head -1)"
        fi
    done
fi

# ===================== 最终汇总 =====================
echo ""
echo -e "${BOLD}======================================"
echo "  检查结果汇总"
echo -e "======================================${NC}"
echo "总节点数:         $TOTAL"
echo "成功:             ${#SUCCESS_IPS[@]}"
echo "失败:             ${#FAIL_IPS[@]}"
echo "Soft limit 差异:  ${SOFT_DIFF_COUNT} 项"
echo "Hard limit 差异:  ${HARD_DIFF_COUNT} 项"
echo "Soft≠Hard 节点:   ${SVH_DIFF_COUNT} 个"
echo ""
echo "输出目录: ${OUTPUT_DIR}/"
echo "  ├── *.soft          各节点 soft limit 原始输出"
echo "  ├── *.hard          各节点 hard limit 原始输出"
echo "  ├── *.soft.parsed   解析后的 key=value"
echo "  ├── *.hard.parsed   解析后的 key=value"
echo "  ├── soft_diff.txt   soft limit 差异详情"
echo "  ├── hard_diff.txt   hard limit 差异详情"
echo "  ├── soft_vs_hard_diff.txt  同节点 soft vs hard 差异"
echo "  └── summary_table.txt      汇总表"
echo "======================================"

