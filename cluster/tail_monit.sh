#!/bin/bash

################################################################################
# GPU Network Tail Latency Monitor for AI Training
# 
# 监控AI训练网络中的尾部延迟问题，通过 ethtool 获取网络指标：
# - 网络丢包/错包
# - 网络延迟和拥塞
# - 重传次数
# - 流控暂停帧
################################################################################

CONFIG_FILE="${1:-tail_latency.conf}"

# Default values
NODELIST="nodelist.txt"
NET_IFACE="eth0"
NET_PKT_LOSS_THRESHOLD=1
NET_ERR_THRESHOLD=1
NET_RETRY_THRESHOLD=10
OUTPUT_DIR="/tmp/tail_latency_logs"
INTERVAL=60
CONTINUOUS=false

if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "Configuration file '$CONFIG_FILE' not found. Using default settings."
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--continuous)
            CONTINUOUS=true
            shift
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [CONFIG_FILE] [-c|--continuous] [-i|--interval SECONDS]"
            echo "  -c, --continuous: Run continuously (default: false)"
            echo "  -i, --interval:  Monitor interval in seconds (default: 60)"
            exit 0
            ;;
        *)
            CONFIG_FILE="$1"
            shift
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/tail_latency_${TIMESTAMP}.log"

echo "==========================================="
echo " GPU Network Tail Latency Monitor"
echo " Date: $(date)"
echo " Mode: $([ "$CONTINUOUS" = true ] && echo "Continuous (interval=${INTERVAL}s)" || echo "One-shot")"
echo "==========================================="

################################################################################
# Function: Run command on host (local or remote via SSH)
################################################################################
run_on_host() {
    local host=$1
    shift
    local cmd="$@"
    
    if [ "$host" = "localhost" ] || [ "$host" = "$(hostname)" ]; then
        eval "$cmd" 2>/dev/null
    else
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes "$host" "$cmd" 2>/dev/null
    fi
}

################################################################################
# Function: Get unique hosts from nodelist
# nodelist format: 10.18.34.1mccxadmin/mt@33113!
# cut -d'm' -f1 extracts IP part before 'm'
################################################################################
get_hosts() {
    cat "$NODELIST" 2>/dev/null | cut -d'm' -f1 | sort -u | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'
}

################################################################################
# Function: Get ethtool statistics
################################################################################
get_ethtool_stats() {
    local host=$1
    local iface=$2
    run_on_host "$host" "ethtool -S $iface"
}

################################################################################
# Function: Get ethtool link info
################################################################################
get_ethtool_link() {
    local host=$1
    local iface=$2
    run_on_host "$host" "ethtool $iface"
}

################################################################################
# Function: Parse ethtool counter value
################################################################################
parse_counter() {
    local stats="$1"
    local pattern="$2"
    echo "$stats" | grep -iE "$pattern" | awk -F': ' '{print $2}' | head -1
}

################################################################################
# Function: Check if interface exists
################################################################################
iface_exists() {
    local host=$1
    local iface=$2
    run_on_host "$host" "ip link show $iface >/dev/null 2>&1 && echo yes || echo no"
}

################################################################################
# Function: Get IB devices via ibdev2netdev
################################################################################
get_ibdev2netdev() {
    local host=$1
    run_on_host "$host" "ibdev2netdev 2>/dev/null"
}

################################################################################
# Function: Parse ibdev2netdev output to get net device names
################################################################################
parse_ibdev2netdev() {
    local ibdev2netdev_output="$1"
    echo "$ibdev2netdev_output" | awk -F'==> ' '{print $2}' | awk '{print $1}' | grep -v '^$'
}

################################################################################
# 1. Detect Network Interfaces (Prioritize ibdev2netdev for InfiniBand/RoCE)
################################################################################
echo ""
echo "[1] Detecting Network Interfaces..."

declare -A host_ifaces

for host in $(get_hosts); do
    echo "    Checking $host..."
    
    # First, try to get high-speed network devices via ibdev2netdev (IB/RoCE)
    ibdev2netdev_output=$(get_ibdev2netdev "$host")
    active_ifaces=""
    
    if [ -n "$ibdev2netdev_output" ]; then
        echo "      Found ibdev2netdev devices:"
        # Parse ibdev2netdev output to get net device names (e.g., ens12np0, mlx5_bond_0)
        ib_ifaces=$(parse_ibdev2netdev "$ibdev2netdev_output")
        
        for iface in $ib_ifaces; do
            # Check if interface exists and is up
            state=$(run_on_host "$host" "cat /sys/class/net/$iface/operstate 2>/dev/null")
            if [ "$state" = "up" ]; then
                active_ifaces="$active_ifaces $iface"
                # Show corresponding IB device
                ib_dev=$(echo "$ibdev2netdev_output" | grep "$iface" | head -1 | awk '{print $1}')
                echo "        $ib_dev -> $iface (Up)"
            else
                echo "        $iface (Down/Unknown)"
            fi
        done
        
        # Also show full ibdev2netdev output for reference
        echo "      Full ibdev2netdev:"
        echo "$ibdev2netdev_output" | sed 's/^/        /'
    else
        echo "      [INFO] ibdev2netdev not available, falling back to ip link"
        
        # Fallback: Try to find active interfaces via ip link
        ifaces=$(run_on_host "$host" "ip -o link show | awk -F': ' '{print \$2}' | grep -v '^lo$'")
        
        for iface in $ifaces; do
            state=$(run_on_host "$host" "cat /sys/class/net/$iface/operstate 2>/dev/null")
            if [ "$state" = "up" ]; then
                active_ifaces="$active_ifaces $iface"
            fi
        done
    fi
    
    if [ -n "$active_ifaces" ]; then
        host_ifaces[$host]="$active_ifaces"
        echo "      Active interfaces:$active_ifaces"
    else
        echo "      [WARN] No active interfaces found on $host"
    fi
done

################################################################################
# 2. Analyze Network Statistics via ethtool
################################################################################
echo ""
echo "[2] Analyzing Network Statistics (ethtool)..."

for host in $(get_hosts); do
    echo "" >> "$LOG_FILE"
    echo "## $host" >> "$LOG_FILE"
    echo "    === $host ==="
    
    ifaces=${host_ifaces[$host]:-$NET_IFACE}
    
    for iface in $ifaces; do
        exists=$(iface_exists "$host" "$iface")
        if [ "$exists" != "yes" ]; then
            continue
        fi
        
        echo "      Interface: $iface"
        
        # Get ethtool stats
        stats=$(get_ethtool_stats "$host" "$iface")
        link=$(get_ethtool_link "$host" "$iface")
        
        # Save raw data to log file immediately
        echo "" >> "$LOG_FILE"
        echo "### $iface" >> "$LOG_FILE"
        echo "--- ethtool -S ---" >> "$LOG_FILE"
        if [ -n "$stats" ]; then
            echo "$stats" >> "$LOG_FILE"
        else
            echo "[empty or failed]" >> "$LOG_FILE"
        fi
        echo "--- ethtool link ---" >> "$LOG_FILE"
        if [ -n "$link" ]; then
            echo "$link" >> "$LOG_FILE"
        else
            echo "[empty or failed]" >> "$LOG_FILE"
        fi
        
        if [ -z "$stats" ] && [ -z "$link" ]; then
            echo "        [WARN] ethtool failed for $iface (need root?)"
            continue
        fi
        
        # Link speed and status
        if [ -n "$link" ]; then
            speed=$(echo "$link" | grep "Speed:" | awk '{print $2}')
            duplex=$(echo "$link" | grep "Duplex:" | awk '{print $2}')
            link_detected=$(echo "$link" | grep "Link detected:" | awk '{print $3}')
            echo "        Speed: $speed, Duplex: $duplex, Link: $link_detected"
            
            if [ "$link_detected" = "no" ]; then
                echo "        [FAIL] Link down on $iface!"
            fi
        fi
        
        # === Mellanox/ConnectX Specific Counters ===
        
        # RX packet buffer drops (critical for tail latency)
        rx_out_of_buffer=$(parse_counter "$stats" "rx_out_of_buffer")
        rx_buff_alloc_err=$(parse_counter "$stats" "rx_buff_alloc_err")
        rx_steer_missed=$(parse_counter "$stats" "rx_steer_missed_packets")
        
        if [ -n "$rx_out_of_buffer" ] && [ "$rx_out_of_buffer" -gt "$NET_PKT_LOSS_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_out_of_buffer: $rx_out_of_buffer (RX buffer exhaustion - check CPU/IRQ)"
        fi
        if [ -n "$rx_buff_alloc_err" ] && [ "$rx_buff_alloc_err" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_buff_alloc_err: $rx_buff_alloc_err (buffer allocation failure)"
        fi
        if [ -n "$rx_steer_missed" ] && [ "$rx_steer_missed" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_steer_missed_packets: $rx_steer_missed (packet steering miss)"
        fi
        
        # TX queue status
        tx_queue_stopped=$(parse_counter "$stats" "tx_queue_stopped")
        tx_queue_dropped=$(parse_counter "$stats" "tx_queue_dropped")
        
        if [ -n "$tx_queue_stopped" ] && [ "$tx_queue_stopped" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] tx_queue_stopped: $tx_queue_stopped (queue full - check congestion)"
        fi
        if [ -n "$tx_queue_dropped" ] && [ "$tx_queue_dropped" -gt "$NET_PKT_LOSS_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] tx_queue_dropped: $tx_queue_dropped (TX drops due to queue full)"
        fi
        
        # WQE/CQE errors (indicates HW processing issues)
        rx_wqe_err=$(parse_counter "$stats" "rx_wqe_err")
        tx_cqe_err=$(parse_counter "$stats" "tx_cqe_err")
        
        if [ -n "$rx_wqe_err" ] && [ "$rx_wqe_err" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_wqe_err: $rx_wqe_err (RX WQE errors)"
        fi
        if [ -n "$tx_cqe_err" ] && [ "$tx_cqe_err" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] tx_cqe_err: $tx_cqe_err (TX CQE errors)"
        fi
        
        # RX cache utilization
        rx_cache_busy=$(parse_counter "$stats" "rx_cache_busy")
        rx_cache_full=$(parse_counter "$stats" "rx_cache_full")
        rx_cache_empty=$(parse_counter "$stats" "rx_cache_empty")
        
        if [ -n "$rx_cache_busy" ] && [ "$rx_cache_busy" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_cache_busy: $rx_cache_busy (RX cache contention)"
        fi
        if [ -n "$rx_cache_full" ] && [ "$rx_cache_full" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_cache_full: $rx_cache_full (RX cache full - consider increase)"
        fi
        
        # Packet errors (CRC, length, etc)
        rx_crc_errors=$(parse_counter "$stats" "rx_crc_errors_phy")
        rx_length_errors=$(parse_counter "$stats" "rx_in_range_len_errors_phy|rx_out_of_range_len_phy")
        rx_oversize=$(parse_counter "$stats" "rx_oversize_pkts_phy|rx_oversize_pkts_sw_drop")
        rx_undersize=$(parse_counter "$stats" "rx_undersize_pkts_phy")
        rx_fragments=$(parse_counter "$stats" "rx_fragments_phy")
        rx_jabbers=$(parse_counter "$stats" "rx_jabbers_phy")
        
        if [ -n "$rx_crc_errors" ] && [ "$rx_crc_errors" -gt "$NET_ERR_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_crc_errors_phy: $rx_crc_errors (physical layer CRC errors)"
        fi
        if [ -n "$rx_length_errors" ] && [ "$rx_length_errors" -gt "$NET_ERR_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_length_errors: $rx_length_errors (packet length errors)"
        fi
        if [ -n "$rx_oversize" ] && [ "$rx_oversize" -gt "$NET_ERR_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_oversize_pkts: $rx_oversize (oversized packets dropped)"
        fi
        if [ -n "$rx_undersize" ] && [ "$rx_undersize" -gt "$NET_ERR_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_undersize_pkts: $rx_undersize (undersized packets)"
        fi
        if [ -n "$rx_fragments" ] && [ "$rx_fragments" -gt "$NET_ERR_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_fragments: $rx_fragments (fragmented packets)"
        fi
        if [ -n "$rx_jabbers" ] && [ "$rx_jabbers" -gt "$NET_ERR_THRESHOLD" ] 2>/dev/null; then
            echo "        [WARN] rx_jabbers: $rx_jabbers (jabber frames - check cable/firmware)"
        fi
        
        # PHY symbol/corrected bits errors (Mellanox specific)
        rx_corrected_bits=$(parse_counter "$stats" "rx_corrected_bits_phy")
        rx_symbol_err=$(parse_counter "$stats" "rx_pcs_symbol_err_phy")
        rx_err_lane_0=$(parse_counter "$stats" "rx_err_lane_0_phy")
        rx_err_lane_1=$(parse_counter "$stats" "rx_err_lane_1_phy")
        rx_err_lane_2=$(parse_counter "$stats" "rx_err_lane_2_phy")
        rx_err_lane_3=$(parse_counter "$stats" "rx_err_lane_3_phy")
        
        if [ -n "$rx_corrected_bits" ] && [ "$rx_corrected_bits" -gt 0 ] 2>/dev/null; then
            echo "        [INFO] rx_corrected_bits_phy: $rx_corrected_bits (FEC corrections - monitor trend)"
        fi
        if [ -n "$rx_symbol_err" ] && [ "$rx_symbol_err" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_pcs_symbol_err_phy: $rx_symbol_err (PCS symbol errors)"
        fi
        # Lane errors typically non-zero indicates HW issue
        if [ -n "$rx_err_lane_0" ] && [ "$rx_err_lane_0" -gt 1000 ] 2>/dev/null; then
            echo "        [WARN] rx_err_lane_0_phy: $rx_err_lane_0 (lane 0 errors - check cable)"
        fi
        if [ -n "$rx_err_lane_1" ] && [ "$rx_err_lane_1" -gt 1000 ] 2>/dev/null; then
            echo "        [WARN] rx_err_lane_1_phy: $rx_err_lane_1 (lane 1 errors - check cable)"
        fi
        
        # Flow control pause frames (indicates congestion)
        rx_pause_ctrl=$(parse_counter "$stats" "rx_pause_ctrl_phy")
        tx_pause_ctrl=$(parse_counter "$stats" "tx_pause_ctrl_phy")
        rx_prio3_pause=$(parse_counter "$stats" "rx_prio3_pause")
        tx_prio3_pause=$(parse_counter "$stats" "tx_prio3_pause")
        pause_storm_warn=$(parse_counter "$stats" "tx_pause_storm_warning_events")
        pause_storm_err=$(parse_counter "$stats" "tx_pause_storm_error_events")
        
        if [ -n "$rx_pause_ctrl" ] && [ "$rx_pause_ctrl" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_pause_ctrl_phy: $rx_pause_ctrl (RX flow control frames)"
        fi
        if [ -n "$tx_pause_ctrl" ] && [ "$tx_pause_ctrl" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] tx_pause_ctrl_phy: $tx_pause_ctrl (TX flow control frames)"
        fi
        if [ -n "$pause_storm_warn" ] && [ "$pause_storm_warn" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] tx_pause_storm_warning: $pause_storm_warn (pause storm warning)"
        fi
        if [ -n "$pause_storm_err" ] && [ "$pause_storm_err" -gt 0 ] 2>/dev/null; then
            echo "        [CRIT] tx_pause_storm_error: $pause_storm_err (pause storm error - severe congestion!)"
        fi
        
        # RDMA / vPort statistics (key for GPU interconnects)
        rx_vport_rdma_unicast=$(parse_counter "$stats" "rx_vport_rdma_unicast_packets")
        tx_vport_rdma_unicast=$(parse_counter "$stats" "tx_vport_rdma_unicast_packets")
        rx_rdma_ooo=$(parse_counter "$stats" "tx_tls_ooo")
        
        if [ -n "$rx_rdma_ooo" ] && [ "$rx_rdma_ooo" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] tx_tls_ooo (out-of-order): $rx_rdma_ooo (TLS out-of-order)"
        fi
        
        # XDP related drops (if XDP is in use)
        rx_xdp_drop=$(parse_counter "$stats" "rx_xdp_drop")
        rx_xdp_redirect=$(parse_counter "$stats" "rx_xdp_redirect")
        
        if [ -n "$rx_xdp_drop" ] && [ "$rx_xdp_drop" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_xdp_drop: $rx_xdp_drop (XDP drops - check XDP program)"
        fi
        if [ -n "$rx_xdp_redirect" ] && [ "$rx_xdp_redirect" -gt 0 ] 2>/dev/null; then
            echo "        [INFO] rx_xdp_redirect: $rx_xdp_redirect (XDP redirects)"
        fi
        
        # Congestion / UMR
        rx_congst_umr=$(parse_counter "$stats" "rx_congst_umr")
        if [ -n "$rx_congst_umr" ] && [ "$rx_congst_umr" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_congst_umr: $rx_congst_umr (UMR congestion)"
        fi
        
        # Module health
        module_unplug=$(parse_counter "$stats" "module_unplug")
        module_high_temp=$(parse_counter "$stats" "module_high_temp")
        module_bad_shorted=$(parse_counter "$stats" "module_bad_shorted")
        
        if [ -n "$module_unplug" ] && [ "$module_unplug" -gt 0 ] 2>/dev/null; then
            echo "        [CRIT] module_unplug: $module_unplug (module unplugged!)"
        fi
        if [ -n "$module_high_temp" ] && [ "$module_high_temp" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] module_high_temp: $module_high_temp (module over temperature)"
        fi
        if [ -n "$module_bad_shorted" ] && [ "$module_bad_shorted" -gt 0 ] 2>/dev/null; then
            echo "        [CRIT] module_bad_shorted: $module_bad_shorted (module bad/shorted!)"
        fi
        
        # PCI signal integrity
        rx_pci_signal=$(parse_counter "$stats" "rx_pci_signal_integrity")
        tx_pci_signal=$(parse_counter "$stats" "tx_pci_signal_integrity")
        pci_stalled_rd=$(parse_counter "$stats" "outbound_pci_stalled_rd")
        pci_stalled_wr=$(parse_counter "$stats" "outbound_pci_stalled_wr")
        
        if [ -n "$rx_pci_signal" ] && [ "$rx_pci_signal" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] rx_pci_signal_integrity: $rx_pci_signal (PCI RX signal issues)"
        fi
        if [ -n "$tx_pci_signal" ] && [ "$tx_pci_signal" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] tx_pci_signal_integrity: $tx_pci_signal (PCI TX signal issues)"
        fi
        if [ -n "$pci_stalled_rd" ] && [ "$pci_stalled_rd" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] outbound_pci_stalled_rd: $pci_stalled_rd (PCI read stalls)"
        fi
        if [ -n "$pci_stalled_wr" ] && [ "$pci_stalled_wr" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] outbound_pci_stalled_wr: $pci_stalled_wr (PCI write stalls)"
        fi
        
        # Link down events
        link_down_events=$(parse_counter "$stats" "link_down_events_phy")
        if [ -n "$link_down_events" ] && [ "$link_down_events" -gt 0 ] 2>/dev/null; then
            echo "        [WARN] link_down_events_phy: $link_down_events (link down events)"
        fi
        
    done
done

################################################################################
# 3. Check TCP Retransmissions (sysfs / netstat)
################################################################################
echo ""
echo "[3] Checking TCP Retransmissions..."

for host in $(get_hosts); do
    echo "    === $host ==="
    
    # Get TCP retransmits from netstat -s
    tcp_retrans=$(run_on_host "$host" "netstat -s 2>/dev/null | grep -i 'retransm' | head -5")
    if [ -n "$tcp_retrans" ]; then
        echo "$tcp_retrans" | sed 's/^/      /'
    fi
    
    # Get from ss -ti if available
    tcp_rto=$(run_on_host "$host" "ss -ti 2>/dev/null | grep -c 'retrans:' || echo 0")
    if [ "$tcp_rto" -gt 0 ] 2>/dev/null; then
        echo "      Active connections with retransmissions: $tcp_rto"
    fi
done

################################################################################
# 4. Summary
################################################################################
echo ""
echo "============================================"
echo " Network Tail Latency Analysis Summary"
echo "============================================"

echo ""
echo "Key Network Indicators for Tail Latency (Mellanox ConnectX):"
echo "  1. rx_out_of_buffer             -> RX buffer exhaustion (check CPU/IRQ)"
echo "  2. rx_buff_alloc_err            -> Buffer allocation failure"
echo "  3. rx_steer_missed_packets      -> Packet steering miss"
echo "  4. tx_queue_stopped / dropped   -> TX queue full (congestion)"
echo "  5. rx_wqe_err / tx_cqe_err      -> HW processing errors"
echo "  6. rx_cache_busy / full         -> RX cache contention"
echo "  7. rx_crc_errors_phy            -> Physical layer CRC errors"
echo "  8. rx_corrected_bits_phy       -> FEC corrections (monitor trend)"
echo "  9. rx_err_lane_N_phy           -> Per-lane errors (cable issues)"
echo "  10. rx_pause_ctrl_phy / tx_pause_ctrl_phy -> Flow control"
echo "  11. tx_pause_storm_error       -> Severe congestion (critical)"
echo "  12. rx_xdp_drop                -> XDP drops (check XDP program)"
echo "  13. module_unplug/high_temp     -> Module health issues"
echo "  14. pci_stalled_rd/wr          -> PCIe bandwidth issues"
echo ""
echo "Recommendations:"
echo "  - High rx_out_of_buffer: Increase ring buffer (ethtool -G) or check IRQ"
echo "  - High rx_corrected_bits: Monitor trend, may indicate cable degradation"
echo "  - High lane errors: Check fiber cables and connectors"
echo "  - Pause storms: Enable PFC/ECN on switch, check congestion"
echo "  - PCI stalls: Check PCIe slot, update firmware"
echo ""

echo "Log saved to: $LOG_FILE"
echo "============================================"

echo ""
echo "Analysis complete."

