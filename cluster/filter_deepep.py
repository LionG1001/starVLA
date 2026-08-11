#!/usr/bin/env python3
"""过滤 deepep.txt 中满足以下阈值的行，并按 best_combine_gbps 降序排序：

- best_combine_gbps >= 132
- best_dispatch_bf16_gbps >= 93
- all2all_gbps >= 141
- h2d_gbps >= 53.5
"""

import sys

INPUT_FILE = "deepep_perf.txt"

# 阈值定义
THRESHOLDS = {
    "best_combine_gbps": 132,
    "best_dispatch_bf16_gbps": 91.5,
    "all2all_gbps": 141,
    "h2d_gbps": 53.5,
}

# 列索引（0-based）
COL_FILE = 0
COL_COMBINE = 1
COL_DISPATCH = 2
COL_ALL2ALL = 3
COL_D2H = 4
COL_H2D = 5


def main():
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    header = lines[0].strip()
    print(header)
    print("-" * len(header))

    filtered = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue

        try:
            combine = float(parts[COL_COMBINE])
            dispatch = float(parts[COL_DISPATCH])
            all2all = float(parts[COL_ALL2ALL])
            h2d = float(parts[COL_H2D])
        except (ValueError, IndexError):
            continue

        if (
            combine >= THRESHOLDS["best_combine_gbps"]
            and dispatch >= THRESHOLDS["best_dispatch_bf16_gbps"]
            and all2all >= THRESHOLDS["all2all_gbps"]
            and h2d >= THRESHOLDS["h2d_gbps"]
        ):
            filtered.append((combine, line.rstrip()))

    # 按 best_combine_gbps 降序排序
    filtered.sort(key=lambda x: x[0], reverse=True)

    for _, row in filtered:
        print(row)

    print()
    print(f"共 {len(filtered)} 行满足条件（总阈值：combine>={THRESHOLDS['best_combine_gbps']}, "
          f"dispatch>={THRESHOLDS['best_dispatch_bf16_gbps']}, "
          f"all2all>={THRESHOLDS['all2all_gbps']}, "
          f"h2d>={THRESHOLDS['h2d_gbps']}）")


if __name__ == "__main__":
    main()

