#!/usr/bin/env python3
"""
给定一个全集 hostfile 和多个子集 hostfile，输出不在任何子集中的 IP。

默认按每行第 1 列提取 IP，忽略空行与注释行。
输出顺序与全集 hostfile 中首次出现的顺序一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入全集 hostfile 和多个子集 hostfile，输出不在任何子集中的 IP"
    )
    parser.add_argument(
        "--all-hostfile",
        required=True,
        help="全集 hostfile 路径",
    )
    parser.add_argument(
        "--subset-hostfiles",
        required=True,
        nargs="+",
        help="一个或多个子集 hostfile 路径",
    )
    parser.add_argument(
        "--column",
        type=int,
        default=1,
        help="IP 所在列，从 1 开始计数，默认第 1 列",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="额外输出统计信息到 stderr",
    )
    return parser.parse_args()


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def extract_ips(hostfile: Path, column: int) -> list[str]:
    if column < 1:
        eprint("错误: --column 必须大于等于 1")
        sys.exit(1)

    if not hostfile.exists():
        eprint(f"错误: 文件不存在: {hostfile}")
        sys.exit(1)

    if not hostfile.is_file():
        eprint(f"错误: 路径不是文件: {hostfile}")
        sys.exit(1)

    ips: list[str] = []
    seen: set[str] = set()

    with hostfile.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < column:
                eprint(
                    f"警告: {hostfile} 第 {line_no} 行列数不足，已跳过: {raw_line.rstrip()}"
                )
                continue

            ip = fields[column - 1]
            if ip not in seen:
                ips.append(ip)
                seen.add(ip)

    return ips


def main() -> None:
    args = parse_args()

    all_hostfile = Path(args.all_hostfile)
    subset_hostfiles = [Path(path) for path in args.subset_hostfiles]

    all_ips = extract_ips(all_hostfile, args.column)
    subset_ip_set: set[str] = set()

    for subset_hostfile in subset_hostfiles:
        subset_ips = extract_ips(subset_hostfile, args.column)
        subset_ip_set.update(subset_ips)

    remaining_ips = [ip for ip in all_ips if ip not in subset_ip_set]

    for ip in remaining_ips:
        print(ip)

    if args.show_summary:
        eprint(f"全集 IP 数量: {len(all_ips)}")
        eprint(f"子集去重后 IP 数量: {len(subset_ip_set)}")
        eprint(f"未命中任何子集的 IP 数量: {len(remaining_ips)}")


if __name__ == "__main__":
    main()

