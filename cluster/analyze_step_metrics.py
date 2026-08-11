#!/usr/bin/env python3
"""
分析训练日志中 "Step X, Loss: {dict}" 格式的指标，计算均值、最大值、最小值。

用法:
    python analyze_step_metrics.py <logfile> [logfile2 ...]
    python analyze_step_metrics.py <logfile> --start 100 --end 500
    python analyze_step_metrics.py "通配符路径"
    python analyze_step_metrics.py <logfile> --csv output.csv
    python analyze_step_metrics.py <logfile> --sort-by model_time

示例:
    python analyze_step_metrics.py train.log
    python analyze_step_metrics.py train.log --start 50 --end 200
    python analyze_step_metrics.py "logs/**/stdout.log"
    python analyze_step_metrics.py train.log --csv metrics.csv
    python analyze_step_metrics.py "logs/**/stdout.log" --sort-by model_time

说明:
    - 解析日志行格式: Step 100, Loss: {'action_dit_loss': 0.26, 'mse_score': 0.04, ...}
    - 自动提取 Python 字典中的所有 key-value 对作为指标
    - 统计每个指标的均值、最小值、最大值、样本数
    - 支持 --start / --end 过滤 step 范围
    - 支持 --csv 导出为 CSV 文件
    - 支持 --sort-by 按指定指标均值对文件排序，从文件名 log.编号.ip 中提取 IP
"""

import sys
import re
import ast
import glob
import os
import csv
import argparse
from collections import defaultdict


# 匹配 "Step <number>, Loss: <dict>" 的行
LINE_PATTERN = re.compile(
    r'Step\s+(\d+)\s*,\s*Loss\s*:\s*(\{.*\})',
    re.IGNORECASE
)

# 匹配文件名中的 IP: log.编号.IP.IP.IP.IP
FILENAME_IP_PATTERN = re.compile(
    r'^log\.\d+\.((?:\d+\.){3}\d+)$'
)


def extract_ip_from_filename(filepath: str) -> str:
    """
    从文件名中提取 IP 地址。
    文件名格式: log.编号.IP.IP.IP.IP
    例如: log.0.10.121.32.1 -> 10.121.32.1
    若无法提取则返回文件名本身。
    """
    basename = os.path.basename(filepath)
    match = FILENAME_IP_PATTERN.match(basename)
    if match:
        return match.group(1)
    return basename


def parse_dict_safely(dict_str: str) -> dict:
    """
    安全地将字符串形式的 Python 字典转为 dict。
    先尝试 ast.literal_eval，失败则尝试正则提取 key-value 对。
    """
    try:
        result = ast.literal_eval(dict_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # 回退: 用正则逐个提取 'key': value 对
    metrics = {}
    kv_pattern = re.compile(r"'([^']+)'\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")
    for match in kv_pattern.finditer(dict_str):
        key = match.group(1)
        try:
            value = float(match.group(2))
            metrics[key] = value
        except ValueError:
            continue
    return metrics


def parse_log_line(line: str):
    """
    解析单行日志，返回 (step, metrics_dict) 或 None。
    """
    match = LINE_PATTERN.search(line)
    if not match:
        return None
    step = int(match.group(1))
    dict_str = match.group(2)
    metrics = parse_dict_safely(dict_str)
    if not metrics:
        return None
    return step, metrics


def find_log_files(pattern: str) -> list:
    """使用 glob 模式查找匹配的日志文件。"""
    matched = glob.glob(pattern, recursive=True)
    if not matched:
        expanded = os.path.expanduser(pattern)
        matched = glob.glob(expanded, recursive=True)
    matched = sorted(set(f for f in matched if os.path.isfile(f)))
    return matched


def read_metrics_from_file(filepath: str, start_step=None, end_step=None) -> list:
    """
    从文件中读取所有匹配行的指标。
    返回 [(step, {metric: value, ...}), ...]
    """
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 快速预过滤
                if 'Step' not in line or 'Loss' not in line:
                    continue
                parsed = parse_log_line(line)
                if parsed is None:
                    continue
                step, metrics = parsed
                if start_step is not None and step < start_step:
                    continue
                if end_step is not None and step > end_step:
                    continue
                results.append((step, metrics))
    except (OSError, IOError) as e:
        print(f"  [警告] 无法读取文件 {filepath}: {e}", file=sys.stderr)
    return results


def compute_stats(values: list) -> dict:
    """计算列表的统计信息。"""
    if not values:
        return None
    return {
        'min': min(values),
        'max': max(values),
        'avg': sum(values) / len(values),
        'count': len(values),
    }


def format_value(val) -> str:
    """格式化数值输出。"""
    if val is None:
        return "nan"
    abs_val = abs(val) if val != 0 else 1
    if abs_val < 0.001 or abs_val >= 1e6:
        return f"{val:.4e}"
    elif abs_val < 1:
        return f"{val:.6f}"
    else:
        return f"{val:.4f}"


def build_stats_from_records(records: list) -> dict:
    """
    从记录列表构建各指标统计。
    records: [(step, {metric: value, ...}), ...]
    返回: {metric_name: {'min', 'max', 'avg', 'count'}}
    """
    values = defaultdict(list)
    for step, metrics in records:
        for metric_name, val in metrics.items():
            if isinstance(val, (int, float)):
                values[metric_name].append(val)

    stats = {}
    for metric_name, vals in values.items():
        stats[metric_name] = compute_stats(vals)
    return stats


def print_stats_table(stats: dict, title: str = None, metric_order: list = None):
    """打印指标统计表格。"""
    if title:
        print(f"  {title}")

    col1 = 30
    col2 = 16
    header = f"  {'指标':<{col1}} {'均值':>{col2}} {'最小值':>{col2}} {'最大值':>{col2}} {'样本数':>6}"
    separator = "  " + "-" * (len(header) - 2)
    print(separator)
    print(header)
    print(separator)

    # 排序: 优先按 metric_order，其余按字母序
    if metric_order:
        ordered = [m for m in metric_order if m in stats]
        remaining = sorted(m for m in stats if m not in metric_order)
        ordered.extend(remaining)
    else:
        ordered = sorted(stats.keys())

    for metric_name in ordered:
        s = stats[metric_name]
        if s is None:
            continue
        avg_str = format_value(s['avg'])
        min_str = format_value(s['min'])
        max_str = format_value(s['max'])
        cnt_str = str(s['count'])
        print(f"  {metric_name:<{col1}} {avg_str:>{col2}} {min_str:>{col2}} {max_str:>{col2}} {cnt_str:>6}")

    print(separator)


def export_csv(stats: dict, filepath: str, metric_order: list = None):
    """将统计结果导出为 CSV 文件。"""
    if metric_order:
        ordered = [m for m in metric_order if m in stats]
        remaining = sorted(m for m in stats if m not in metric_order)
        ordered.extend(remaining)
    else:
        ordered = sorted(stats.keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', '均值', '最小值', '最大值', '样本数'])
        for metric_name in ordered:
            s = stats[metric_name]
            if s is None:
                continue
            writer.writerow([
                metric_name,
                format_value(s['avg']),
                format_value(s['min']),
                format_value(s['max']),
                s['count'],
            ])
    print(f"结果已导出到: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='分析训练日志中 "Step X, Loss: {dict}" 格式的指标，计算均值、最大值、最小值'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='日志文件路径（支持通配符，如 "logs/**/stdout.log"）'
    )
    parser.add_argument(
        '--start', '-s',
        type=int,
        default=None,
        help='起始 step（含），不指定则从最开始'
    )
    parser.add_argument(
        '--end', '-e',
        type=int,
        default=None,
        help='结束 step（含），不指定则到最后'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='将统计结果导出为 CSV 文件'
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        default=None,
        help='按指定指标均值对文件排序，输出 IP 和该指标均值（如 --sort-by model_time）'
    )
    args = parser.parse_args()

    # 收集所有文件
    all_files = []
    for path in args.paths:
        if any(c in path for c in '*?[]'):
            files = find_log_files(path)
        else:
            if os.path.isfile(path):
                all_files.append(path)
            elif os.path.isdir(path):
                for root, _, filenames in os.walk(path):
                    for fn in sorted(filenames):
                        all_files.append(os.path.join(root, fn))
            else:
                # 尝试当作 glob 模式
                files = find_log_files(path)
                if not files:
                    print(f"[警告] 未找到匹配的文件: {path}", file=sys.stderr)
                all_files.extend(files)

    all_files = sorted(set(all_files))

    if not all_files:
        print("未找到任何日志文件", file=sys.stderr)
        sys.exit(1)

    # 读取所有文件数据
    all_records = []
    file_records = {}

    for filepath in all_files:
        records = read_metrics_from_file(filepath, args.start, args.end)
        file_records[filepath] = records
        all_records.extend(records)

    if not all_records:
        print("未找到匹配的日志行 (格式: Step X, Loss: {...})", file=sys.stderr)
        sys.exit(1)

    # 确定 step 范围
    steps = [r[0] for r in all_records]
    print("=" * 90)
    print("训练日志指标统计分析 (Step, Loss: {dict} 格式)")
    print("=" * 90)
    print(f"  Step 范围: {min(steps)} ~ {max(steps)}")

    if args.start is not None or args.end is not None:
        range_info = f"  过滤范围: step {args.start or '∞'} ~ {args.end or '∞'}"
        print(range_info)

    # 每个文件的统计
    for filepath, records in file_records.items():
        if records:
            stats = build_stats_from_records(records)
            print(f"\n  文件: {filepath}")
            file_steps = [r[0] for r in records]
            print(f"  匹配行数: {len(records)}, Step 范围: {min(file_steps)} ~ {max(file_steps)}")
            print_stats_table(stats)
        else:
            print(f"\n  文件: {filepath} — 无匹配行")

    # 全局汇总
    print("\n" + "=" * 90)
    overall_stats = build_stats_from_records(all_records)
    print_stats_table(overall_stats, title="所有文件汇总统计")
    print(f"共 {len(all_files)} 个文件，匹配日志行数: {len(all_records)}")
    print("=" * 90)

    # 按 --sort-by 指标均值排序输出
    if args.sort_by:
        sort_metric = args.sort_by
        # 收集每个文件的 IP 和该指标均值
        file_sort_data = []
        for filepath, records in file_records.items():
            if not records:
                continue
            ip = extract_ip_from_filename(filepath)
            # 计算该指标均值
            metric_values = []
            for step, metrics in records:
                if sort_metric in metrics and isinstance(metrics[sort_metric], (int, float)):
                    metric_values.append(metrics[sort_metric])
            if metric_values:
                avg_val = sum(metric_values) / len(metric_values)
                file_sort_data.append((filepath, ip, avg_val))

        if not file_sort_data:
            print(f"\n[警告] 未找到任何文件包含指标 '{sort_metric}'", file=sys.stderr)
        else:
            # 按均值降序排序
            file_sort_data.sort(key=lambda x: x[2], reverse=True)

            print("\n" + "=" * 90)
            print(f"按 {sort_metric} 均值排序（降序）")
            print("=" * 90)

            col_ip = 20
            col_val = 16
            col_file = 50
            header = f"  {'IP':<{col_ip}} {sort_metric + ' 均值':>{col_val}}  {'文件':<{col_file}}"
            separator = "  " + "-" * (len(header) - 2)
            print(separator)
            print(header)
            print(separator)

            for filepath, ip, avg_val in file_sort_data:
                val_str = format_value(avg_val)
                print(f"  {ip:<{col_ip}} {val_str:>{col_val}}  {filepath:<{col_file}}")

            print(separator)
            print(f"共 {len(file_sort_data)} 个文件")

    # 导出 CSV
    if args.csv:
        export_csv(overall_stats, args.csv)


if __name__ == '__main__':
    main()

