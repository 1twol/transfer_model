#!/usr/bin/env python3
"""
interactive.py — ⁷Li + ²³²Th 三体转移模型 交互式入口

逐项提问代替命令行参数, 复用 li7_th232_main.main() 的全部计算与输出逻辑。
等价于:
  python li7_th232_main.py --model <m> --energy <a> <b> <c> ...

用法:
  python interactive.py
"""

import argparse
import sys

from li7_th232_main import main

# 推荐模型 (第一级)
MODELS = [
    ("icf",   "ICF 占比校准 (默认, 推荐)"),
    ("fermi", "费米积分: t-Th 俘获模型"),
]

# 示意模型 (仅通过"其他"选入; 绝对标度未标定, 只适合看形状)
SCHEMATIC_MODELS = [
    ("tunneling", "简单指数隧穿"),
    ("qwindow",   "Q 窗口隧穿"),
    ("dwba",      "半经典 DWBA"),
]

ALL_MODELS = MODELS + SCHEMATIC_MODELS

ENERGY_MODES = [
    ("默认范围",   "E_lab 20-40 MeV, 步长 2"),
    ("自定义范围", "输入 E_lab 下限/上限/步长"),
    ("单个能量点", "只算一个 E_lab"),
]

PACE4_MODES = [
    ("中位能量",  "只生成一份 .pace (默认)"),
    ("全部能量",  "每个能量点各生成一份"),
    ("不生成",    "只算截面/谱, 不写 .pace"),
]


def _choice(prompt, options, default=0):
    """从序号列表选择, 回车取默认"""
    print(prompt)
    for i, (label, desc) in enumerate(options):
        marker = "  [默认]" if i == default else ""
        print(f"  {i + 1}) {label}{marker}  —  {desc}")
    while True:
        s = input(f"输入序号 1-{len(options)} [默认 {default + 1}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if 1 <= v <= len(options):
                return v - 1
        except ValueError:
            pass
        print("  无效输入, 请重试。")


def _num(prompt, default, cast=float):
    """输入数值, 回车取默认"""
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            return cast(s)
        except ValueError:
            print("  无效数值, 请重试。")


def _yesno(prompt, default=True):
    hint = "Y/n" if default else "y/N"
    s = input(f"{prompt} [{hint}]: ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes")


def _q(prompt):
    return input(f"{prompt}: ").strip()


def run():
    print("=" * 62)
    print("  ⁷Li + ²³²Th 三体转移模型 — 交互式运行")
    print("  (与 python li7_th232_main.py 相同计算链路)")
    print("=" * 62)
    print()

    # 1. 模型 (两级: 推荐模型 + "其他"示意模型)
    mi = _choice("选择转移模型:", MODELS + [("其他", "示意模型 (tunneling/qwindow/dwba)")], default=0)
    if mi < len(MODELS):
        model = MODELS[mi][0]
    else:
        print()
        print("  ⚠ 以下为示意模型, 绝对标度未标定 (P₀/D₀ 经验值), 仅适合看形状。")
        si = _choice("选择示意模型:", SCHEMATIC_MODELS, default=0)
        model = SCHEMATIC_MODELS[si][0]

    # 2. 快速测试?
    quick = _yesno("快速测试模式? (减少抽样/级联数)", default=False)

    # 3. 能量模式
    em = _choice("能量模式:", ENERGY_MODES, default=0)
    energy = None
    e_lab = None
    if em == 1:  # 自定义范围
        lo = _num("  E_lab 下限 (MeV)", 20.0)
        hi = _num("  E_lab 上限 (MeV)", 40.0)
        st = _num("  步长 (MeV)", 2.0)
        energy = (lo, hi, st)
    elif em == 2:  # 单个能量点
        e_lab = _num("  E_lab (MeV)", 32.0)
        energy = (e_lab, e_lab, 2.0)

    # 4. PACE4 模式
    p4 = _choice("PACE4 输入生成:", PACE4_MODES, default=0)
    no_pace4 = (p4 == 2)
    all_ = (p4 == 1)

    # 5. 费米动量抽样数
    n_fermi = int(_num("费米动量抽样数", 1000 if quick else 5000))

    # 6. 画图
    no_plot = not _yesno("生成图表?", default=True)

    # 7. 输出目录
    out = _q("输出目录 (回车自动命名)")

    args = argparse.Namespace(
        model=model,
        energy=energy,
        e_lab=e_lab,
        all=all_,
        quick=quick,
        n_fermi=n_fermi,
        no_plot=no_plot,
        no_pace4=no_pace4,
        cascades=1000 if quick else 10000,
        facla=10.0,
        output_dir=(out or None),
    )

    print()
    print("参数确认:")
    print(f"  model    = {model}")
    print(f"  energy   = {energy or '默认范围'}")
    print(f"  e_lab    = {e_lab or '-'}")
    print(f"  PACE4    = {'全部能量' if all_ else ('不生成' if no_pace4 else '中位能量')}")
    print(f"  n_fermi  = {n_fermi}")
    print(f"  output   = {out or '(自动)'}")
    print()

    main(args)


if __name__ == "__main__":
    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        sys.exit(1)
