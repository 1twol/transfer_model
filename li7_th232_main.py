#!/usr/bin/env python3
"""
li7_th232_main.py — ⁷Li + ²³²Th 三体转移模型主程序

反应:
  ⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa*

三体模型:
  - 初态: α-t 团簇 (束缚于 ⁷Li) + ²³²Th
  - 转移: t 从 ⁷Li 转移到 ²³²Th
  - 末态: α + ²³⁵Pa* (两体)

包含物理:
  ✓ ⁷Li 内部费米动量分布 (高斯近似)
  ✓ 卢瑟福擦边轨道
  ✓ 指数隧穿转移概率 + Q 值窗口
  ✓ 库仑后加速
  ✓ 角度依赖
  ✓ 激发能谱 → PACE4 输入

用法:
  python li7_th232_main.py                    # 默认 ICF 模型, 完整计算
  python li7_th232_main.py --quick            # 快速测试模式
  python li7_th232_main.py --model icf        # ICF 占比校准 (推荐)
  python li7_th232_main.py --model fermi      # 费米动量积分 + 隧穿
  python li7_th232_main.py --model tunneling  # 简单隧穿模型
  python li7_th232_main.py --energy 20 45 5   # 自定义能量范围
"""

import argparse
import os
import sys
import time
import numpy as np

# 将父目录加入 path (支持直接运行)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import config
from model.structure import FermiMomentumSampler
from model.kinematics import (grazing_angle, grazing_angular_momentum,
                               rutherford_trajectory)
from model.potentials import (woods_saxon, coulomb_uniform_sphere,
                                total_potential, find_barrier)
from model.transfer import (TunnelingModel, QWindowTunnelingModel,
                               FermiIntegratedModel, create_model)
from model.cross_section import (compute_excitation_function,
                                    compute_angular_distribution,
                                    compute_excitation_energy_spectrum,
                                    compute_full)
from post_process import (generate_pace4_from_spectrum, generate_pace4_single,
                          generate_eexcn_table, plot_all)

_sys = config.system
_mod = config.model


# ============================================================
# 辅助: 打印体系信息
# ============================================================

def print_system_info():
    """打印反应体系信息"""
    print("=" * 60)
    print("  ⁷Li + ²³²Th 三体转移模型")
    print("  Three-Body Transfer Model for ⁷Li + ²³²Th → α + ²³⁵Pa*")
    print("=" * 60)
    print()
    print("  [核素]")
    print(f"    ⁷Li  : Z={_sys.proj.Z}, A={_sys.proj.A}, m={_sys.proj.mass_MeV:.1f} MeV")
    print(f"    α    : Z={_sys.spectator.Z}, A={_sys.spectator.A}, m={_sys.spectator.mass_MeV:.1f} MeV")
    print(f"    t    : Z={_sys.cluster.Z}, A={_sys.cluster.A}, m={_sys.cluster.mass_MeV:.1f} MeV")
    print(f"    ²³²Th: Z={_sys.targ.Z}, A={_sys.targ.A}, m={_sys.targ.mass_MeV:.1f} MeV")
    print(f"    ²³⁵Pa: Z={_sys.product.Z}, A={_sys.product.A}, m={_sys.product.mass_MeV:.1f} MeV")
    print()
    print("  [Q 值]")
    print(f"    ⁷Li → α + t   : Q = {_sys.q_breakup:.3f} MeV")
    print(f"    t + ²³²Th → ²³⁵Pa : Q = {_sys.q_capture:.3f} MeV")
    print(f"    净 Q               : Q = {_sys.q_total:.3f} MeV")
    print()
    print("  [约化质量]")
    print(f"    ⁷Li-²³²Th : {_sys.mu_proj_targ:.1f} MeV/c²")
    print(f"    α-t       : {_sys.mu_alpha_t:.1f} MeV/c²")
    print(f"    α-²³⁵Pa   : {_sys.mu_alpha_pa:.1f} MeV/c²")
    print()
    print("  [模型参数]")
    print(f"    半径参数 r₀={_mod.r0} fm, 弥散 a₀={_mod.a0} fm")
    print(f"    零程常数 D₀={_mod.d0_manual} MeV·fm³/²")
    print(f"    费米动量 σ_k≈{_mod.sigma_k_manual:.2f} fm⁻¹ (手动)")
    print(f"    b 网格: {_mod.n_b} 点, E* bin: 50")
    print()


def print_kinematics_table(e_lab_range: np.ndarray):
    """打印运动学参考表"""
    print("  [运动学参考]")
    print(f"    {'E_lab':>6s}  {'E_cm':>8s}  {'η':>8s}  {'k':>8s}  "
          f"{'θ_g(cm)':>8s}  {'L_g':>6s}  {'b_g':>8s}  {'V_cb':>8s}")
    print("    " + "-" * 72)

    for e_lab in e_lab_range:
        e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
        eta = config.sommerfeld(_sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm)
        k = config.wavenumber(_sys.mu_proj_targ, e_cm)
        r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A)
        theta_g, _, l_g = grazing_angle(e_cm, r_int)
        b_g = l_g / k if k > 0 else 0
        v_cb = config.coulomb_barrier(_sys.proj.Z, _sys.targ.Z, r_int)

        print(f"    {e_lab:6.1f}  {e_cm:8.2f}  {eta:8.2f}  {k:8.4f}  "
              f"{np.degrees(theta_g):7.1f}°  {l_g:6.1f}  {b_g:8.2f}  {v_cb:8.2f}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="⁷Li+²³²Th 三体转移模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 完整计算 (Fermi-Integrated 模型)
  %(prog)s --quick                  # 快速测试 (Tunneling 模型, 少抽样)
  %(prog)s --model qwindow          # Q 值窗口隧穿模型
  %(prog)s --energy 25 40 5         # 自定义能量 25-40 MeV, 步长 5
  %(prog)s --no-plot                # 不画图
  %(prog)s --output-dir ./results   # 指定输出目录
        """
    )

    parser.add_argument('--model', type=str, default='icf',
                        choices=['tunneling', 'qwindow', 'dwba', 'fermi', 'icf'],
                        help='转移概率模型 (default: fermi)')
    parser.add_argument('--energy', type=float, nargs=3,
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='E_lab 范围 (MeV), 例: --energy 20 40 2')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (简化参数)')
    parser.add_argument('--n-fermi', type=int, default=5000,
                        help='费米动量抽样数 (default: 5000)')
    parser.add_argument('--no-plot', action='store_true',
                        help='不调用 matplotlib 画图')
    parser.add_argument('--no-pace4', action='store_true',
                        help='不生成 PACE4 输入文件')
    parser.add_argument('--cascades', type=int, default=10000,
                        help='PACE4 cascades (default: 10000)')
    parser.add_argument('--facla', type=float, default=10.0,
                        help='能级密度参数 FACLA (default: 10.0)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: ./output_<model>_<timestamp>)')

    args = parser.parse_args()

    # ---- 参数设置 ----
    if args.quick:
        _mod.n_b = 30
        _mod.n_theta = 20
        args.n_fermi = 1000
        args.cascades = 1000
        print(">>> 快速测试模式 <<<")

    if args.energy:
        e_lab_range = np.arange(args.energy[0], args.energy[1] + args.energy[2]/2,
                                  args.energy[2])
    else:
        e_lab_range = np.arange(_mod.e_lab_min,
                                 _mod.e_lab_max + _mod.e_lab_step / 2,
                                 _mod.e_lab_step)

    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output_{args.model}_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 打印体系信息 ----
    print_system_info()
    print_kinematics_table(e_lab_range)
    print()

    # ---- 创建模型 ----
    print(f"  [模型] 使用 {args.model} 模型")
    model = create_model(args.model)
    print()

    # ---- 计算 ----
    t_start = time.time()

    if args.quick:
        # 快速模式: 只算激发函数
        print(">>> 快速模式: 仅计算激发函数 <<<")
        exc = compute_excitation_function(model, e_lab_range, args.n_fermi, verbose=True)
        result = {'excitation': exc}
    else:
        result = compute_full(model, e_lab_range, args.n_fermi, verbose=True)

    t_elapsed = time.time() - t_start
    print(f"\n  总耗时: {t_elapsed:.1f} s")

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    print("  结果汇总")
    print("=" * 60)

    exc = result.get('excitation', {})
    if exc:
        print("\n  [激发函数]")
        print(f"    {'E_lab':>6s}  {'σ_tr(mb)':>12s}  {'L_g':>6s}")
        print("    " + "-" * 30)
        for i in range(len(exc['e_lab'])):
            print(f"    {exc['e_lab'][i]:6.1f}  {exc['sigma'][i]:12.4e}  {exc['l_g'][i]:6.1f}")

    ang = result.get('angular', {})
    if ang:
        idx_peak = np.argmax(ang['dsigma_domega'])
        print(f"\n  [角分布]")
        print(f"    峰值角度: θ_cm = {ang['theta_cm_deg'][idx_peak]:.1f}°")
        print(f"    峰值截面: dσ/dΩ = {ang['dsigma_domega'][idx_peak]:.4e} mb/sr")

    spec = result.get('e_star_spectrum', {})
    if spec:
        print(f"\n  [激发能谱]")
        print(f"    平均 E*  = {spec.get('e_star_mean', 0):.2f} MeV")
        print(f"    标准差   = {spec.get('e_star_std', 0):.2f} MeV")
        print(f"    Q_capture = {spec.get('q_capture', 0):.2f} MeV")

    # ---- 画图 ----
    if not args.no_plot:
        print("\n  [画图]")
        try:
            plot_all(result, args.output_dir)
        except Exception as e:
            print(f"    画图失败: {e}")

    # ---- PACE4 输入 ----
    if not args.no_pace4 and spec:
        print("\n  [PACE4 输入]")
        pace_dir = os.path.join(args.output_dir, f"Li7+Th232_{args.model}")
        # 用中位能量计算 spin distribution
        e_mid = e_lab_range[len(e_lab_range)//2]
        e_cm_mid = config.e_lab_to_e_cm(e_mid, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
        meta = generate_pace4_from_spectrum(spec, e_cm=e_cm_mid,
                                              output_dir=pace_dir,
                                              label=f"Li7+Th232 {args.model}",
                                              cascades=args.cascades,
                                              facla=args.facla)
        print(f"    输出目录: {pace_dir}")
        print(f"    生成 {len(meta['files'])} 个 .pace 文件")
        print(f"    总截面: {meta['total_sigma_mb']:.4e} mb")
        print(f"    [WARNING] spin dist: sharp-cutoff L_g (not CCFULL partial waves)")
        print(f"    汇总文件: {meta['summary_path']}")

        # 同时生成 EEXCN 表 (供 generate_pace.py 使用)
        eexcn_path = os.path.join(args.output_dir, "eexcn_table.txt")
        generate_eexcn_table(result['excitation'], output_path=eexcn_path)
        print(f"    EEXCN 表: {eexcn_path}")
        print(f"    用法: python generate_pace.py ... --e-star <表中EEXCN值>")

    # ---- 保存原始结果 ----
    result_path = os.path.join(args.output_dir, "result_summary.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"# Three-Body Transfer Model Results\n")
        f.write(f"# Reaction: ⁷Li + ²³²Th → α + ²³⁵Pa*\n")
        f.write(f"# Model: {args.model}\n")
        f.write(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#\n")
        if exc:
            f.write("# Excitation Function\n")
            f.write("# E_lab(MeV)  σ_tr(mb)  L_g(ħ)\n")
            for i in range(len(exc['e_lab'])):
                f.write(f"  {exc['e_lab'][i]:.1f}  {exc['sigma'][i]:.6e}  {exc['l_g'][i]:.1f}\n")
    print(f"\n  结果文件: {result_path}")

    print(f"\n  所有输出保存在: {args.output_dir}")
    print("=" * 60)

    return result


if __name__ == '__main__':
    main()