"""
post_process.py — 结果后处理、画图、PACE4 接口

输出:
  1. 激发函数图 (σ vs E, 对数/线性)
  2. 角分布图 (dσ/dΩ vs θ)
  3. E* 激发能谱图
  4. PACE4 .pace 输入文件 (正确格式, 与 generate_pace.py 兼容)

PACE4 .pace 格式 (与 generate_pace.py 一致):
  行1: cascades  IOUT  MODE  IDSC  ITRACK  IREVERSE  IDIFF  FACLA  DELANG  IROT2  IFB
  行2: Z  A  EEXCN  0.0000  AJNUC  0
  行3: n_L  sigma_0 sigma_1 ... sigma_{Lmax}
  行4-8: 空行

注意:
  transfer_model 不产生分波截面 — 使用 sharp-cutoff L_g 近似
  如需精确分波, 将本模型的 EEXCN 传给 generate_pace.py 配合 CCFULL partial.dat
"""

import numpy as np
import os
from typing import Dict, Optional, Tuple

import model.config as config
from model.cross_section import (compute_excitation_function,
                                     compute_angular_distribution,
                                     compute_excitation_energy_spectrum,
                                     compute_full)
from model.kinematics import grazing_angular_momentum

_sys = config.system
_mod = config.model


# ============================================================
# 辅助: 生成 PACE4 标题行
# ============================================================

def _make_pace_header(cascades: int, facla: int = 10) -> str:
    """构建 PACE4 第一行 (Fixed-header line)

    格式:
        cascades  IOUT  MODE  IDSC  ITRACK  IREVERSE  IDIFF  FACLA  DELANG  IROT2  IFB
    其中 FACLA 为能级密度参数 a = A / FACLA。
    """
    return f"{int(cascades):5d}    1    3    0    0    1    0{int(facla):5d}    0    0    0"


def _build_spin_distribution(e_cm: float, total_sigma_mb: float,
                                l_max: int = 0) -> Tuple[np.ndarray, int]:
    """构建分波截面数组 (sharp-cutoff 近似)

    对于给定质心能量，使用擦边角动量 L_g 作为截断:
      σ_L = σ_total / (L_g + 1)   for L ≤ L_g
      σ_L = 0                      for L > L_g

    返回 (sigma_L 数组, 实际 l_max)
    """
    r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A, _mod.r0)
    l_g = int(grazing_angular_momentum(e_cm, r_int,
                                        _sys.proj.Z, _sys.targ.Z))

    if l_max <= 0:
        l_max = l_g

    n_l = l_max + 1
    sigmas = np.zeros(n_l)
    n_active = l_g + 1 if l_g < n_l else n_l
    sigmas[:n_active] = total_sigma_mb / n_active

    return sigmas, l_max


def _get_ajnuc(z_cn: int, a_cn: int, l_max: int) -> float:
    """PACE4 的 AJNUC 参数: L_max (+ 0.5 for odd-A)"""
    is_odd = (a_cn % 2 == 1)
    return l_max + 0.5 if is_odd else float(l_max)


# ============================================================
# 1. PACE4 输入生成 (正确格式)
# ============================================================

def write_single_pace(outfile: str,
                       e_cm: float,
                       eexcn: float,
                       total_sigma_mb: float,
                       z_cn: int = None,
                       a_cn: int = None,
                       cascades: int = 10000,
                       facla: int = 10,
                       l_max: int = 0,
                       label: str = "") -> None:
    """写单个 .pace 文件 (PACE4 正确格式)

    Parameters
    ----------
    outfile : 输出路径
    e_cm : 质心系能量 (用于计算 L_g)
    eexcn : 激发能 E* (MeV)
    total_sigma_mb : 总截面 (mb)
    z_cn, a_cn : 复合核 Z, A
    cascades : Monte Carlo 级联数
    facla : 能级密度参数
    l_max : 最大角动量 (0=自动从 L_g 取)
    label : 反应标签 (写入注释)
    """
    if z_cn is None:
        z_cn = _sys.product.Z
    if a_cn is None:
        a_cn = _sys.product.A

    sigma_l, l_max_actual = _build_spin_distribution(e_cm, total_sigma_mb, l_max)
    ajnuc = _get_ajnuc(z_cn, a_cn, l_max_actual)
    n_l = len(sigma_l)
    sigma_str = " ".join(f"{s:.6e}" for s in sigma_l)

    lines = []
    # .pace 文件不加注释行 — PACE4 固定列格式解析器可能不兼容
    # 所有元信息写入同目录下的 pace4_summary.txt

    lines.append(_make_pace_header(cascades, facla))
    lines.append(f"   {z_cn}  {a_cn}  {eexcn:.4f}    0.0000   {ajnuc:.4f}    0")
    lines.append(f"   {n_l}        {sigma_str}")

    # 5 行空行
    for _ in range(5):
        lines.append("")

    content = "\r\n".join(lines) + "\r\n"

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        f.write(content)


def generate_pace4_from_spectrum(e_star_spec: Dict,
                                   e_cm: float,
                                   output_dir: str = ".",
                                   z_cn: int = None, a_cn: int = None,
                                   label: str = "Li7+Th232",
                                   cascades: int = 10000,
                                   facla: int = 10,
                                   n_e_star_bins: int = 10) -> Dict:
    """从激发能谱生成多份 .pace 文件 (每个 E* bin 一份)

    与 generate_pace.py 的区别:
      - 本函数用 transfer_model 计算的 E* 谱 (含费米运动展宽)
      - 分波截面用 sharp-cutoff L_g 近似 (非 CCFULL partial.dat)
      - 建议: 用本函数的 EEXCN 值, 但分波数据从 generate_pace.py 获取

    Parameters
    ----------
    e_star_spec : compute_excitation_energy_spectrum 的返回
    e_cm : 质心系能量 (MeV) — 用于 spin distribution
    output_dir : 输出目录
    z_cn, a_cn : 复合核 Z, A
    label : 反应标签
    cascades : 级联数
    facla : 能级密度参数
    n_e_star_bins : 保留的 E* bin 数量

    Returns
    -------
    meta : {'files', 'total_sigma_mb', 'e_star_mean', 'e_star_std', 'summary_path'}
    """
    if z_cn is None:
        z_cn = _sys.product.Z
    if a_cn is None:
        a_cn = _sys.product.A

    os.makedirs(output_dir, exist_ok=True)

    e_star = e_star_spec['e_star']
    dsigma_de = e_star_spec['dsigma_de']

    # 总截面
    de = e_star[1] - e_star[0]
    total_sigma = np.sum(dsigma_de) * de

    # 选显著的 E* bin
    mask = dsigma_de > 0.005 * dsigma_de.max()
    e_selected = e_star[mask]
    w_selected = dsigma_de[mask]
    w_selected = w_selected / w_selected.sum()

    if len(e_selected) > n_e_star_bins:
        idx = np.argsort(w_selected)[-n_e_star_bins:]
        e_selected = e_selected[idx]
        w_selected = w_selected[idx]
        w_selected = w_selected / w_selected.sum()

    if len(e_selected) == 0:
        e_selected = np.array([float(e_star_spec.get('e_star_mean', 15.0))])
        w_selected = np.array([1.0])

    files = []
    for i, (e, w) in enumerate(zip(e_selected, w_selected)):
        sigma_mb = total_sigma * w
        fname = f"EEXCN={e:.2f}.pace"
        fpath = os.path.join(output_dir, fname)
        write_single_pace(fpath, e_cm, float(e), sigma_mb,
                           z_cn, a_cn, cascades, facla, l_max=0,
                           label=f"{label} E* bin {i+1}/{len(e_selected)}")
        files.append({
            'path': fpath,
            'eexcn': float(e),
            'sigma_mb': sigma_mb,
            'weight': w,
        })

    # 汇总文件
    summary_path = os.path.join(output_dir, "pace4_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"# PACE4 Summary: {label}\n")
        f.write(f"# CN: Z={z_cn}, A={a_cn}\n")
        f.write(f"# Total σ = {total_sigma:.4e} mb\n")
        f.write(f"# Mean E* = {e_star_spec.get('e_star_mean', 0):.2f} MeV\n")
        f.write(f"# Std E*  = {e_star_spec.get('e_star_std', 0):.2f} MeV\n")
        f.write(f"# Spin distribution: sharp-cutoff at L_g (not CCFULL partial waves)\n")
        f.write(f"#\n")
        f.write(f"# {'Idx':>4s} {'EEXCN':>8s} {'σ(mb)':>12s} {'%':>6s}  File\n")
        for i, fi in enumerate(files):
            f.write(f"  {i+1:4d} {fi['eexcn']:8.2f} {fi['sigma_mb']:12.4e} "
                    f"{fi['weight']*100:5.1f}  {os.path.basename(fi['path'])}\n")

    return {
        'files': files,
        'total_sigma_mb': total_sigma,
        'e_star_mean': e_star_spec.get('e_star_mean', 0),
        'e_star_std': e_star_spec.get('e_star_std', 0),
        'summary_path': summary_path,
    }


def generate_pace4_single(eexcn: float,
                            e_cm: float,
                            total_sigma_mb: float,
                            output_dir: str = ".",
                            z_cn: int = None, a_cn: int = None,
                            label: str = "Li7+Th232",
                            cascades: int = 10000,
                            facla: int = 10) -> Dict:
    """从单值 EEXCN 生成一份 .pace (配合 generate_pace.py 使用)

    典型用法:
      >> eexcn = transfer_model 算出的 ⟨E*⟩
      >> generate_pace4_single(eexcn, e_cm, sigma_mb, ...)
      >> 将生成的 .pace 喂给 PACE4

    Parameters
    ----------
    eexcn : 激发能 (MeV)
    e_cm : 质心系能量 (MeV)
    total_sigma_mb : 总截面 (mb)
    output_dir, z_cn, a_cn, label, cascades, facla : 同上

    Returns
    -------
    meta : {'path', 'eexcn', 'sigma_mb'}
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = f"EEXCN={eexcn:.2f}.pace"
    fpath = os.path.join(output_dir, fname)
    write_single_pace(fpath, e_cm, eexcn, total_sigma_mb,
                       z_cn, a_cn, cascades, facla,
                       label=label)
    return {
        'path': fpath,
        'eexcn': eexcn,
        'sigma_mb': total_sigma_mb,
    }


# ============================================================
# 2. 生成与 generate_pace.py 对接的 EEXCN 汇总表
# ============================================================

def generate_eexcn_table(exc_result: Dict,
                           e_star_specs: Dict = None,
                           output_path: str = None) -> str:
    """生成 EEXCN 汇总表, 供 generate_pace.py 的 --e-star 参数使用

    输出格式 (可直接阅读或供脚本解析):
      E_lab  E_cm   EEXCN_mean  EEXCN_std  σ_total(mb)  L_g

    Parameters
    ----------
    exc_result : compute_excitation_function 返回
    e_star_specs : {E_lab: e_star_spec_dict} 每个能量的 E* 谱
    output_path : 保存路径 (None=只返回字符串)

    Returns
    -------
    table : 格式化的字符串表
    """
    lines = []
    lines.append("# EEXCN table from transfer_model")
    lines.append("# For use with: python generate_pace.py --e-star <EEXCN>")
    lines.append("#")
    lines.append(f"# {'E_lab':>6s}  {'E_cm':>8s}  {'EEXCN_mean':>10s}  "
                 f"{'EEXCN_std':>10s}  {'σ(mb)':>12s}  {'L_g':>5s}")
    lines.append("# " + "-" * 62)

    for i in range(len(exc_result['e_lab'])):
        e_lab = exc_result['e_lab'][i]
        e_cm = exc_result['e_cm'][i]
        sigma = exc_result['sigma'][i]
        l_g = exc_result['l_g'][i]

        if e_star_specs and e_lab in e_star_specs:
            eexcn_mean = e_star_specs[e_lab].get('e_star_mean', 0)
            eexcn_std = e_star_specs[e_lab].get('e_star_std', 0)
        else:
            # 从 Q 值公式估算
            ratio = (_sys.spectator.Z * _sys.product.Z) / (_sys.proj.Z * _sys.targ.Z)
            q_opt = (ratio - 1.0) * e_cm
            eexcn_mean = _sys.q_total - q_opt
            eexcn_std = 0.0

        lines.append(f"  {e_lab:6.1f}  {e_cm:8.2f}  {eexcn_mean:10.2f}  "
                     f"{eexcn_std:10.2f}  {sigma:12.4e}  {l_g:5.0f}")

    table = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(table + "\n")
        print(f"  [eexcn] EEXCN 表 → {output_path}")

    return table


# ============================================================
# 3. 画图工具 (需要 matplotlib)
# ============================================================

def plot_excitation_function(result: Dict, output_path: str = None,
                               log_scale: bool = True):
    """画激发函数图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    exc = result['excitation']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(exc['e_lab'], exc['sigma'], 'o-', color='C0', lw=2, ms=6,
             label=f"σ_tr ({_sys.proj.symbol}+{_sys.targ.symbol}→α+{_sys.product.symbol})")
    ax1.plot(exc['e_lab'], exc['sigma_rutherford'], '--', color='gray', lw=1,
             label="Geometric (Rutherford)")
    ax1.set_xlabel("E_lab (MeV)")
    ax1.set_ylabel("σ (mb)")
    if log_scale:
        ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.set_title("Transfer Cross Section")
    ax1.grid(True, alpha=0.3)

    ax2.plot(exc['e_lab'], exc['l_g'], 's-', color='C1', lw=2, ms=6)
    ax2.set_xlabel("E_lab (MeV)")
    ax2.set_ylabel("L_g (ħ)")
    ax2.set_title("Grazing Angular Momentum")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 激发函数图 → {output_path}")
    else:
        plt.show()


def plot_angular_distribution(result: Dict, output_path: str = None):
    """画角分布图"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    ang = result['angular']
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(ang['theta_cm_deg'], ang['dsigma_domega'], 'o-', color='C0',
            lw=2, ms=5, label="dσ/dΩ (transfer)")
    ax.plot(ang['theta_cm_deg'], ang['dsigma_domega_ruth'], '--',
            color='gray', lw=1, alpha=0.7, label="dσ/dΩ (Rutherford)")

    idx_peak = np.argmax(ang['dsigma_domega'])
    theta_peak = ang['theta_cm_deg'][idx_peak]
    ax.axvline(theta_peak, color='red', ls=':', lw=1.5,
               label=f"Peak ≈ {theta_peak:.1f}°")

    ax.set_xlabel("θ_cm (deg)")
    ax.set_ylabel("dσ/dΩ (mb/sr)")
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.set_title("Angular Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 角分布图 → {output_path}")
    else:
        plt.show()


def plot_e_star_spectrum(result: Dict, output_path: str = None):
    """画激发能谱"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    spec = result['e_star_spectrum']
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.fill_between(spec['e_star'], 0, spec['dsigma_de'],
                     color='C2', alpha=0.4)
    ax.plot(spec['e_star'], spec['dsigma_de'], '-', color='C2', lw=2)

    ax.axvline(spec['q_capture'], color='gray', ls='--', lw=1,
               label=f"Q_capture={spec['q_capture']:.1f} MeV")
    ax.axvline(spec['e_star_mean'], color='red', ls='-', lw=1.5,
               label=f"⟨E*⟩={spec['e_star_mean']:.1f} MeV")

    ax.set_xlabel("E* (MeV)")
    ax.set_ylabel("dσ/dE* (mb/MeV)")
    ax.legend(fontsize=9)
    ax.set_title(f"Excitation Energy Spectrum of {_sys.product.symbol}*")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 激发能谱图 → {output_path}")
    else:
        plt.show()


def plot_all(result: Dict, output_dir: str = "."):
    """一次性画所有图"""
    os.makedirs(output_dir, exist_ok=True)
    plot_excitation_function(result,
                              os.path.join(output_dir, "excitation_function.png"))
    plot_angular_distribution(result,
                               os.path.join(output_dir, "angular_distribution.png"))
    plot_e_star_spectrum(result,
                           os.path.join(output_dir, "e_star_spectrum.png"))
