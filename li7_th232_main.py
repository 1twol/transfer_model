#!/usr/bin/env python3
"""
li7_th232_main.py — ⁷Li + ²³²Th 三体转移模型主程序

反应:
  ⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa*
                └── PACE4 蒸发一个中子 → ²³⁴Pa

三体模型:
  - 初态: α-t 团簇 (束缚于 ⁷Li) + ²³²Th
  - 转移: t 从 ⁷Li 转移到 ²³²Th
  - 末态: α + ²³⁵Pa* (两体)

包含物理:
  ✓ ⁷Li 内部费米动量分布 (高斯近似)
  ✓ 卢瑟福擦边轨道
  ✓ Hill-Wheeler 势垒穿透 + 经典角动量截断
  ✓ 出口道两体能量守恒 (渐近动能含库仑后加速, T_rel = E_cm+Q₀−E*)
  ✓ 角度依赖
  ✓ 激发能谱 → PACE4 输入

用法:
  python li7_th232_main.py                        # 默认 ICF 模型, 中位能量 PACE4
  python li7_th232_main.py --quick                # 快速测试 (只算激发函数)
  python li7_th232_main.py --e-lab 32            # 指定 E_lab=32 MeV 生成 PACE4
  python li7_th232_main.py --all                  # 所有能量点各生成 PACE4
  python li7_th232_main.py --energy 20 45 5       # 自定义能量范围
  python li7_th232_main.py --model fermi          # 费米动量积分模型
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
                                    compute_alpha_double_differential,
                                    compute_full)
from post_process import (generate_pace4_from_spectrum, generate_pace4_single,
                          generate_eexcn_table, plot_all,
                          plot_e_star_spec_to_file, plot_e_star_spectra_map,
                          plot_alpha_double_diff,
                          plot_angular_distribution_to_file)

_sys = config.system
_mod = config.model


# ============================================================
# 辅助: 打印体系信息
# ============================================================

def print_system_info():
    """打印反应体系信息"""
    print("=" * 60)
    print("  ⁷Li + ²³²Th 三体转移模型 (目标产物 ²³⁴Pa)")
    print("  Three-Body Transfer Model: ⁷Li + ²³²Th → α + ²³⁵Pa* → ²³⁴Pa + n")
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
# PACE4 批量生成
# ============================================================

def generate_pace4_for_energies(model, e_lab_list, exc_result,
                                 output_dir, label_prefix, cascades, facla,
                                 n_fermi=5000, n_b=None, verbose=True):
    """对多个能量点各生成 PACE4 文件 (含各自的 E* 谱)

    Returns
    -------
    e_star_specs : {float(e_lab): spec} 每个能量的 E* 谱 (供 EEXCN 汇总表)
    """
    if n_b is None:
        n_b = min(_mod.n_b, 40)
    n_fermi_es = max(n_fermi, 2000)

    e_star_specs = {}
    for e_lab in e_lab_list:
        e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)

        if verbose:
            print(f"\n  E_lab={e_lab:.0f} MeV (E_cm={e_cm:.2f} MeV)")

        # 算 E* 谱
        spec = compute_excitation_energy_spectrum(
            model, e_lab=e_lab, n_b=n_b, n_fermi=n_fermi_es, verbose=False
        )
        e_star_specs[float(e_lab)] = spec

        # 输出目录
        pace_dir = os.path.join(output_dir, f"E={e_lab:.0f}MeV")

        meta = generate_pace4_from_spectrum(
            spec, e_cm=e_cm, output_dir=pace_dir,
            label=f"{label_prefix} E={e_lab:.0f}MeV",
            cascades=cascades, facla=facla, model=model
        )

        # 每个能量点的 E* 谱图放进对应目录
        plot_e_star_spec_to_file(spec,
                                 os.path.join(pace_dir, "e_star_spectrum.png"),
                                 e_lab=e_lab)

        # α 旁观者双微分热图 (THM 坐标系)
        d2 = compute_alpha_double_differential(
            model, e_lab=e_lab, n_b=n_b, n_fermi=n_fermi_es, verbose=False)
        plot_alpha_double_diff(d2,
                               os.path.join(pace_dir, "alpha_double_diff.png"))

        # 每个能量点的角分布图 (与 E* 谱、双微分并列)
        ang_ev = compute_angular_distribution(
            model, e_lab=e_lab, n_theta=_mod.n_theta,
            n_fermi=n_fermi_es, verbose=False)
        plot_angular_distribution_to_file(
            ang_ev, os.path.join(pace_dir, "angular_distribution.png"))

        if verbose:
            print(f"    <E*>={spec['e_star_mean']:.1f} MeV, "
                  f"σ={meta['total_sigma_mb']:.4e} mb, "
                  f"{len(meta['files'])} files → {pace_dir}")

    # 全部能量算完后, 画二维频谱图 (含 E*_opt 曲线)
    plot_e_star_spectra_map(e_star_specs,
                            os.path.join(output_dir, "e_star_spectra_map.png"))

    return e_star_specs


# ============================================================
# 主入口
# ============================================================

def main(args=None):
    parser = argparse.ArgumentParser(
        description="⁷Li+²³²Th 三体转移模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 默认: 激发函数 + 中位能量 PACE4
  %(prog)s --quick                  # 快速测试 (只算激发函数)
  %(prog)s --e-lab 32               # 指定 E_lab=32 MeV 生成 PACE4
  %(prog)s --all                    # 所有能量点各生成 PACE4
  %(prog)s --all --no-plot          # 全能量 PACE4, 不画图
  %(prog)s --energy 25 40 5         # 自定义能量范围
  %(prog)s --model fermi            # 费米动量积分模型
        """
    )

    parser.add_argument('--model', type=str, default='icf',
                        choices=['tunneling', 'qwindow', 'dwba', 'fermi', 'icf'],
                        help='转移概率模型 (default: icf)')
    parser.add_argument('--energy', type=float, nargs=3,
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='E_lab 范围 (MeV), 例: --energy 20 40 2')
    parser.add_argument('--e-lab', type=float, default=None,
                        help='指定单个 E_lab (MeV) 生成 PACE4')
    parser.add_argument('--all', action='store_true',
                        help='对所有能量点各生成 PACE4')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式 (只算激发函数)')
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
    parser.add_argument('--all-spectra', action='store_true',
                        help='每个能量点各算一张 E* 谱并画叠加图 (与 .pace 粒度一致)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: ./output_<model>_<timestamp>)')

    if args is None:
        args = parser.parse_args()

    # ---- 互斥检查 ----
    if args.e_lab is not None and args.all:
        print("[ERROR] --e-lab 和 --all 不能同时使用")
        sys.exit(1)

    # ---- 参数设置 ----
    if args.quick:
        _mod.n_theta = 20
        args.n_fermi = 500
        args.cascades = 1000
        args.no_pace4 = True   # quick 只算激发函数, 无 E* 谱可喂 PACE4
        print(">>> 快速测试模式 (减少抽样数, 精度与标准模式一致) <<<")

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
        print(">>> 快速模式: 仅计算激发函数 <<<")
        exc = compute_excitation_function(model, e_lab_range, args.n_fermi, verbose=True)
        result = {'excitation': exc}
    else:
        result = compute_full(model, e_lab_range, args.n_fermi,
                              all_spectra=args.all_spectra, verbose=True)

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
        print(f"\n  [激发能谱 (中位能量)]")
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
    if not args.no_pace4:
        print("\n  [PACE4 输入]")

        # 各能量点的 E* 谱 (供 EEXCN 汇总表; 口径与 .pace 文件一致)
        e_star_specs = {}

        if args.all:
            # 所有能量点各算 E* 谱 + PACE4
            print(f"  模式: --all ({len(e_lab_range)} 个能量点)")
            specs = generate_pace4_for_energies(
                model, e_lab_range, exc,
                output_dir=os.path.join(args.output_dir, "Li7+Th232_icf"),
                label_prefix=f"Li7+Th232 {args.model}",
                cascades=args.cascades, facla=args.facla,
                n_fermi=args.n_fermi, verbose=True
            )
            e_star_specs.update(specs)

        elif args.e_lab is not None:
            # 单能量
            e_lab = args.e_lab
            e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
            print(f"  模式: --e-lab {e_lab:.0f} MeV")

            spec_e = compute_excitation_energy_spectrum(
                model, e_lab=e_lab, n_b=min(_mod.n_b, 40), n_fermi=max(args.n_fermi, 2000),
                verbose=False
            )
            e_star_specs[float(e_lab)] = spec_e
            pace_dir = os.path.join(args.output_dir, f"Li7+Th232_E={e_lab:.0f}MeV")
            meta = generate_pace4_from_spectrum(
                spec_e, e_cm=e_cm, output_dir=pace_dir,
                label=f"Li7+Th232 {args.model} E={e_lab:.0f}MeV",
                cascades=args.cascades, facla=args.facla, model=model
            )
            print(f"    E* = {spec_e['e_star_mean']:.1f} MeV, "
                  f"σ = {meta['total_sigma_mb']:.4e} mb, "
                  f"{len(meta['files'])} files → {pace_dir}")

        else:
            # 默认: 中位能量
            e_mid = e_lab_range[len(e_lab_range)//2]
            e_cm_mid = config.e_lab_to_e_cm(e_mid, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
            print(f"  模式: 默认 (中位能量 E_lab={e_mid:.0f} MeV)")

            if spec and spec.get('e_star') is not None:
                e_star_specs[float(e_mid)] = spec
            pace_dir = os.path.join(args.output_dir, f"Li7+Th232_E={e_mid:.0f}MeV")
            meta = generate_pace4_from_spectrum(
                spec, e_cm=e_cm_mid, output_dir=pace_dir,
                label=f"Li7+Th232 {args.model} E={e_mid:.0f}MeV",
                cascades=args.cascades, facla=args.facla, model=model
            )
            print(f"    {len(meta['files'])} files → {pace_dir}")
            print(f"    [spin dist: sharp-cutoff L_g, not CCFULL partial waves]")

        # EEXCN 汇总表 (全能量; 未在上面的 PACE4 分支里算过 E* 谱的能量点, 补算)
        for e_lab in exc['e_lab']:
            el = float(e_lab)
            if el not in e_star_specs:
                e_star_specs[el] = compute_excitation_energy_spectrum(
                    model, e_lab=el, n_b=min(_mod.n_b, 20),
                    n_fermi=min(args.n_fermi, 3000), verbose=False
                )
        eexcn_path = os.path.join(args.output_dir, "eexcn_table.txt")
        generate_eexcn_table(exc, e_star_specs, output_path=eexcn_path)
        print(f"    EEXCN 表: {eexcn_path}")

    # ---- 保存原始结果 ----
    result_path = os.path.join(args.output_dir, "result_summary.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"# Three-Body Transfer Model Results\n")
        f.write(f"# Reaction: ⁷Li + ²³²Th → α + ²³⁵Pa* → ²³⁴Pa + n\n")
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
