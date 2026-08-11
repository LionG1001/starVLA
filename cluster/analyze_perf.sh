#!/bin/bash
# Usage: analyze_perf.sh <path> <start_step> <end_step> [output_file]
#   path: logfile or directory
#   output_file: optional, output to CSV file
# Example: ./analyze_perf.sh train.log 5291 5305
# Example: ./analyze_perf.sh ./logs/ 5291 5305 result.csv

if [ $# -lt 3 ]; then
    echo "Usage: $0 <logfile|directory> <start_step> <end_step> [output_file]"
    exit 1
fi

PATH_INPUT="$1"
START_STEP="$2"
END_STEP="$3"
OUTPUT_FILE="$4"

extract_ip() {
    local filename=$(basename "$1")
    # Extract IP from format: log.X.IP.IP.IP.IP
    if [[ "$filename" =~ log\.[0-9]+\.([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "$filename"
    fi
}

extract_hostname() {
    local ip="$1"
    # Convert IP to hostname: 10.121.32.1 -> worker32001
    if [[ "$ip" =~ ([0-9]+)\.([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        local octet3="${BASH_REMATCH[3]}"
        local octet4="${BASH_REMATCH[4]}"
        printf "worker%s%03d" "$octet3" "$octet4"
    else
        echo "$ip"
    fi
}

analyze_file() {
    local file="$1"
    local start="$2"
    local end="$3"
    
    awk -v start="$start" -v end="$end" '
        {
            # Extract step number and training time
            # Format: Steps: 1%| | 5296/849600 [11:14:50<1624:01:28, 6.92s/it, lr=2e-5, step_loss=0.0216]
            step = ""
            time = ""

            # Step from explicit field: "global step: 5296" or "step=5296"
            if (match($0, /(global[[:space:]]+)?step[[:space:]]*[:=][[:space:]]*[0-9]+/)) {
                step = substr($0, RSTART, RLENGTH)
                sub(/.*[:=][[:space:]]*/, "", step)
            } else if (match($0, /\|[[:space:]]*[0-9]+\/[0-9]+/)) {
                # Step from progress bar format: "| 5296/849600"
                step = substr($0, RSTART + 1, RLENGTH - 1)
                sub(/^\s*/, "", step)
                sub(/\/.*/, "", step)
            }

            # Extract time like "6.92s/it" or "56s/it"
            if (match($0, /[0-9]+(\.[0-9]+)?s\/it/)) {
                time = substr($0, RSTART, RLENGTH)
                sub(/s\/it/, "", time)
            }

            if (step != "" && step + 0 >= start && step + 0 <= end && time != "") {
                time = time + 0  # 强制转换为数值
                if (n == 0) {
                    min = time
                    max = time
                    sum = time
                } else {
                    if (time < min) min = time
                    if (time > max) max = time
                    sum += time
                }
                n++
            }
        }
        END {
            if (n == 0) {
                print "N/A|N/A|N/A|0"
            } else {
                avg = sum / n
                printf "%.2f|%.2f|%.2f|%d\n", min, max, avg, n
            }
        }
    ' "$file"
}

if [ -f "$PATH_INPUT" ]; then
    # Single file
    ip=$(extract_ip "$PATH_INPUT")
    hostname=$(extract_hostname "$ip")
    if [ -n "$OUTPUT_FILE" ]; then
        echo "IP地址,Hostname,最小(s),最大(s),均值(s),样本数" > "$OUTPUT_FILE"
    fi
    echo "IP地址          Hostname        最小(s)  最大(s)  均值(s)  样本数"
    echo "-------------------------------------------------------------"
    result=$(analyze_file "$PATH_INPUT" "$START_STEP" "$END_STEP")
    IFS='|' read -r min max avg samples <<< "$result"
    printf "%-15s %-15s %-8s %-8s %-8s %s\n" "$ip" "$hostname" "$min" "$max" "$avg" "$samples"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "$ip,$hostname,$min,$max,$avg,$samples" >> "$OUTPUT_FILE"
        echo "结果已保存到: $OUTPUT_FILE"
    fi
elif [ -d "$PATH_INPUT" ]; then
    # Directory - process all files and sort by average
    if [ -n "$OUTPUT_FILE" ]; then
        echo "IP地址,Hostname,最小(s),最大(s),均值(s),样本数" > "$OUTPUT_FILE"
    fi
    echo "IP地址          Hostname        最小(s)  最大(s)  均值(s)  样本数"
    echo "-------------------------------------------------------------"
    temp_file=$(mktemp)
    for file in "$PATH_INPUT"/*; do
        if [ -f "$file" ]; then
            # 跳过非日志文件（只处理 log.* 格式的文件）
            filename=$(basename "$file")
            if [[ ! "$filename" =~ ^log\. ]]; then
                continue
            fi
            ip=$(extract_ip "$file")
            hostname=$(extract_hostname "$ip")
            result=$(analyze_file "$file" "$START_STEP" "$END_STEP")
            IFS='|' read -r min max avg samples <<< "$result"
            echo "$avg|$ip|$hostname|$min|$max|$samples" >> "$temp_file"
        fi
    done
    # Sort by average (first field) numerically
    sort -t'|' -k1 -n "$temp_file" | while IFS='|' read -r avg ip hostname min max samples; do
        printf "%-15s %-15s %-8s %-8s %-8s %s\n" "$ip" "$hostname" "$min" "$max" "$avg" "$samples"
        if [ -n "$OUTPUT_FILE" ]; then
            echo "$ip,$hostname,$min,$max,$avg,$samples" >> "$OUTPUT_FILE"
        fi
    done
    rm -f "$temp_file"
    if [ -n "$OUTPUT_FILE" ]; then
        echo "结果已保存到: $OUTPUT_FILE"
    fi
else
    echo "Error: $PATH_INPUT is not a valid file or directory"
    exit 1
fi

