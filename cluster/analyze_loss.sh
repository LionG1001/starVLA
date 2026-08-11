#!/bin/bash
# Usage: analyze_loss.sh <logfile|directory> <start_step> <end_step> [output_file]
#   path: logfile or directory
#   output_file: optional, output to CSV file
# Example: ./analyze_loss.sh train.log 1 1200
# Example: ./analyze_loss.sh ./logs/ 1 1200 result.csv

if [ $# -lt 3 ]; then
    echo "Usage: $0 <logfile|directory> <start_step> <end_step> [output_file]"
    exit 1
fi

PATH_INPUT="$1"
START_STEP="$2"
END_STEP="$3"
OUTPUT_FILE="$4"

extract_ip() {
    local filename
    filename=$(basename "$1")
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

analyze_loss_file() {
    local file="$1"
    local start="$2"
    local end="$3"

    awk -v start="$start" -v end="$end" '
        /global step loss/ {
            step = ""
            loss = ""

            # Parse numbers after "global step loss"; first number is step, last number is loss
            pos = index($0, "global step loss")
            tail = substr($0, pos)
            while (match(tail, /[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?/)) {
                num = substr(tail, RSTART, RLENGTH)
                if (step == "") {
                    step = num
                }
                loss = num
                tail = substr(tail, RSTART + RLENGTH)
            }

            # Fallback: step from progress bar format: "| 1123/9000"
            if (step == "" && match($0, /\|[[:space:]]*[0-9]+\/[0-9]+/)) {
                step = substr($0, RSTART + 1, RLENGTH - 1)
                sub(/^\s*/, "", step)
                sub(/\/.*/, "", step)
            }

            # Fallback: loss from step_loss=0.132
            if (loss == "" && match($0, /step_loss=([-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?)/)) {
                loss = substr($0, RSTART, RLENGTH)
                sub(/step_loss=/, "", loss)
            }

            if (step != "" && step + 0 >= start && step + 0 <= end && loss != "") {
                loss = loss + 0
                print step "|" loss
            }
        }
    ' "$file"
}

emit_file_csv() {
    local file="$1"
    local start="$2"
    local end="$3"
    local ip="$4"
    local hostname="$5"

    analyze_loss_file "$file" "$start" "$end" | while IFS='|' read -r step loss; do
        printf "%s,%s,%s,%s\n" "$ip" "$hostname" "$step" "$loss"
    done
}

if [ -f "$PATH_INPUT" ]; then
    ip=$(extract_ip "$PATH_INPUT")
    hostname=$(extract_hostname "$ip")
    if [ -n "$OUTPUT_FILE" ]; then
        echo "IP地址,Hostname,Step,Loss" > "$OUTPUT_FILE"
    fi
    echo "IP地址          Hostname        Step      Loss"
    echo "-------------------------------------------------------------"
    analyze_loss_file "$PATH_INPUT" "$START_STEP" "$END_STEP" | while IFS='|' read -r step loss; do
        printf "%-15s %-15s %-9s %s\n" "$ip" "$hostname" "$step" "$loss"
        if [ -n "$OUTPUT_FILE" ]; then
            printf "%s,%s,%s,%s\n" "$ip" "$hostname" "$step" "$loss" >> "$OUTPUT_FILE"
        fi
    done
    if [ -n "$OUTPUT_FILE" ]; then
        echo "结果已保存到: $OUTPUT_FILE"
    fi
elif [ -d "$PATH_INPUT" ]; then
    if [ -n "$OUTPUT_FILE" ]; then
        echo "IP地址,Hostname,Step,Loss" > "$OUTPUT_FILE"
    fi
    echo "IP地址          Hostname        Step      Loss"
    echo "-------------------------------------------------------------"
    for file in "$PATH_INPUT"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            if [[ ! "$filename" =~ ^log\. ]]; then
                continue
            fi
            ip=$(extract_ip "$file")
            hostname=$(extract_hostname "$ip")
            analyze_loss_file "$file" "$START_STEP" "$END_STEP" | while IFS='|' read -r step loss; do
                printf "%-15s %-15s %-9s %s\n" "$ip" "$hostname" "$step" "$loss"
                if [ -n "$OUTPUT_FILE" ]; then
                    printf "%s,%s,%s,%s\n" "$ip" "$hostname" "$step" "$loss" >> "$OUTPUT_FILE"
                fi
            done
        fi
    done
    if [ -n "$OUTPUT_FILE" ]; then
        echo "结果已保存到: $OUTPUT_FILE"
    fi
else
    echo "Error: $PATH_INPUT is not a valid file or directory"
    exit 1
fi

