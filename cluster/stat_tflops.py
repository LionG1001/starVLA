#!/usr/bin/env python3
"""
分析指定路径下的日志文件，统计 throughput per GPU (TFLOP/s/GPU)。

用法:
    python analyze_throughput.py "通配符路径"

示例:
    python analyze_throughput.py "2026-04-13_18-38-09/qwen3_vl_20260413_183*/logs/distributed/**/**/**/7/stdout.log"

说明:
    - 路径支持通配符（glob 模式），使用 pathlib.glob 或 glob 模块匹配
    - 匹配日志行格式: [...] throughput per GPU (TFLOP/s/GPU): 230.8 | ...
    - 统计每个匹配文件中所有匹配行的 throughput 最小值、最大值、均值和样本数
    - 若文件中无匹配行，输出 nan
    - 结果以表格形式输出: 路径、均值、最小值、最大值、样本数
    - 最后输出所有文件的汇总统计
"""

import sys
import glob
import re
import os
import argparse
from pathlib import Path


# 匹配 "throughput per GPU (TFLOP/s/GPU): <number>" 的正则
THROUGHPUT_PATTERN = re.compile(
    r'throughput per GPU \(TFLOP/s/GPU\):\s+([\d.eE+\-]+)'
)


def find_log_files(pattern: str) -> list:
    """
    使用 glob 模式查找匹配的日志文件。
    支持 ** 递归通配符。
    """
    # 使用 glob.glob 启用 recursive 以支持 ** 通配符
    # 先尝试从当前工作目录解析
    matched = glob.glob(pattern, recursive=True)

    # 如果没有匹配，尝试将 pattern 视为绝对路径或相对于当前目录
    if not matched:
        # 尝试展开用户目录 ~
        expanded = os.path.expanduser(pattern)
        matched = glob.glob(expanded, recursive=True)

    # 去重并排序
    matched = sorted(set(matched))

    # 只保留文件（排除目录）
    files = [f for f in matched if os.path.isfile(f)]
    return files


def analyze_throughput(filepath: str) -> dict:
    """
    分析单个日志文件，提取所有 throughput per GPU 值并计算统计信息。
    返回字典包含: min, max, avg, count。
    若无匹配行，返回 None。
    """
    throughputs = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = THROUGHPUT_PATTERN.search(line)
                if match:
                    try:
                        val = float(match.group(1))
                        throughputs.append(val)
                    except ValueError:
                        continue
    except (OSError, IOError) as e:
        print(f"  [警告] 无法读取文件 {filepath}: {e}", file=sys.stderr)
        return None

    if not throughputs:
        return None

    return {
        'min': min(throughputs),
        'max': max(throughputs),
        'avg': sum(throughputs) / len(throughputs),
        'count': len(throughputs),
    }


def main():
    parser = argparse.ArgumentParser(
        description='分析日志文件中的 throughput per GPU'
    )
    parser.add_argument(
        'pattern',
        help='日志文件路径（支持通配符，如 "2026-04-13_18-38-09/qwen3_vl_20260413_183*/logs/distributed/**/**/**/7/stdout.log"）'
    )
    args = parser.parse_args()

    pattern = args.pattern
    files = find_log_files(pattern)

    if not files:
        print(f"未找到匹配的文件: {pattern}", file=sys.stderr)
        sys.exit(1)

    # 收集结果
    results = []
    for filepath in files:
        stats = analyze_throughput(filepath)
        results.append((filepath, stats))

    # 输出表格
    # 计算第一列最大宽度
    max_path_len = max(len(r[0]) for r in results)
    col1_width = max(max_path_len + 2, len("路径") + 2)

    # 表头
    header = f"{'路径':<{col1_width}} {'均值':>8} {'最小值':>8} {'最大值':>8} {'样本数':>6}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for filepath, stats in results:
        if stats is None:
            avg_str, min_str, max_str, cnt_str = "nan", "nan", "nan", "0"
        else:
            avg_str = f"{stats['avg']:.2f}"
            min_str = f"{stats['min']:.2f}"
            max_str = f"{stats['max']:.2f}"
            cnt_str = str(stats['count'])
        print(f"{filepath:<{col1_width}} {avg_str:>8} {min_str:>8} {max_str:>8} {cnt_str:>6}")

    print(separator)
    print(f"共 {len(results)} 个文件")

    # 输出所有文件的 throughput 统计信息（基于单文件均值汇总）
    valid_stats = [s for _, s in results if s is not None]
    if valid_stats:
        all_avgs = [s['avg'] for s in valid_stats]
        all_mins = [s['min'] for s in valid_stats]
        all_maxs = [s['max'] for s in valid_stats]
        total_count = sum(s['count'] for s in valid_stats)
        overall_avg = sum(all_avgs) / len(all_avgs)
        overall_min = min(all_mins)
        overall_max = max(all_maxs)
        print(f"throughput 均值: {overall_avg:.2f} | 最小值: {overall_min:.2f} | 最大值: {overall_max:.2f} | 样本数: {total_count}")
    else:
        print("throughput 均值: nan | 最小值: nan | 最大值: nan | 样本数: 0")


if __name__ == '__main__':
    main()

