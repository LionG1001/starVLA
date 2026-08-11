#!/usr/bin/env python3
import sys
import re

def extract_ip(path):
    """从路径中提取 IP，如 'rank_0_10.121.32.12' -> '10.121.32.12'"""
    match = re.search(r'rank_0_([0-9.]+)', path)
    return match.group(1) if match else None

def parse_value(val):
    """将字符串转为浮点数，若为 'nan' 或无法转换则返回 None（表示无穷大）"""
    if val.lower() == 'nan':
        return None
    try:
        return float(val)
    except ValueError:
        return None

def main():
    records = []  # 存储 (数值或None, IP, 原始值字符串)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        path = parts[0]
        value_str = parts[-1]  # 第二列（最后一个字段）
        ip = extract_ip(path)
        if ip is None:
            continue
        num = parse_value(value_str)
        records.append((num, ip, value_str))

    # 排序：None（nan）视为无穷大，排最后；其余按数值升序
    records.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else float('inf')))

    # 输出两列表格，IP与数值（原字符串）用制表符分隔
    for num, ip, val_str in records:
        print(f"{ip}\t{val_str}")

if __name__ == "__main__":
    main()
