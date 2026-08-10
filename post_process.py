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
                                l_max: int = 0,
                                model=None) -> Tuple[np.ndarray, int]:
    """构建分波截面数组 σ_L

    标准分波展开公式（与 CCFULL partial.dat / PACE4 约定一致）:

      σ_L(mb) = (πħ²/2μE) × (2L+1) × P(L/k) × 10

    其中 P(L/k) 是半经典穿透系数, b ↔ L 映射为 b = L/k。
    (2L+1) 是量子角动量简并度 —— 标准分波分解的固有因子。
    πħ²/2μE = π/k² 是约化波长平方的面积因子。

    形状: L=0 非零, (2L+1) 线性上升, P(b) 在擦边区截断下降,
    峰值在 L ≈ L_g/2 ~ L_g（取决于截断宽度）。

    返回 (sigma_L 数组, 实际 l_max)
    """
    r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A, _mod.r0)
    l_g = int(grazing_angular_momentum(e_cm, r_int,
                                        _sys.proj.Z, _sys.targ.Z))

    if l_max <= 0:
        l_max = max(l_g, 1)

    n_l = l_max + 1
    k = config.wavenumber(_sys.mu_proj_targ, e_cm)

    # 约化波长面积因子: π/k² (fm²), 乘以 10 → mb
    lambda2_pi = np.pi / k**2 * 10 if k > 0 else 1e-10

    # σ_L = λ²π × (2L+1) × P(L/k), 未归一化
    raw = np.zeros(n_l)
    for L in range(n_l):
        b = L / k if k > 0 else 0
        if model is not None and k > 0:
            try:
                p = model.probability(e_cm, b, n_fermi_samples=0)
            except TypeError:
                p = model.probability(e_cm, b)
        else:
            p = 1.0 / (1.0 + np.exp((b - r_int) / 0.5))
        raw[L] = max(lambda2_pi * (2 * L + 1) * abs(p), 0.0)

    # 归一化使总截面对等: Σ σ_L ≈ total_sigma_mb
    total_sum = raw.sum()
    if total_sum > 1e-30:
        sigmas = raw * (total_sigma_mb / total_sum)
    else:
        sigmas = raw

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
                       model=None,
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

    sigma_l, l_max_actual = _build_spin_distribution(e_cm, total_sigma_mb, l_max, model=model)
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
                                   n_e_star_bins: int = 10,
                                   model=None) -> Dict:
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
                           model=model,
                           label=f"{label} E* bin {i+1}/{len(e_selected)}")
        files.append({
            'path': fpath,
            'eexcn': float(e),
            'sigma_mb': sigma_mb,
            'weight': w,
        })

    # 汇总文件
    summary_path = os.path.join(output_dir, "pace4_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
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
      >> eexcn = transfer_model 算出的 <E*>
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
    lines.append("# NOTE: EEXCN_mean/std 来自模型 E* 谱 (E*=Q_capture+E_rel(t-Th), 含费米展宽),")
    lines.append("#       与 .pace 文件口径一致; 不再是旧的 Q_opt 单值法。")
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
            # 从 Q 值公式估算 (仅当未提供 E* 谱时的回退)
            ratio = (_sys.spectator.Z * _sys.product.Z) / (_sys.proj.Z * _sys.targ.Z)
            q_opt = (ratio - 1.0) * e_cm
            eexcn_mean = _sys.q_total - q_opt
            eexcn_std = 0.0

        lines.append(f"  {e_lab:6.1f}  {e_cm:8.2f}  {eexcn_mean:10.2f}  "
                     f"{eexcn_std:10.2f}  {sigma:12.4e}  {l_g:5.0f}")

    table = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
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


def _plot_angular(ang: Dict, ax, title: str = "Angular Distribution"):
    """在 ax 上画一条角分布 (transfer + Rutherford 参照)"""
    ax.plot(ang['theta_cm_deg'], ang['dsigma_domega'], 'o-', color='C0',
            lw=2, ms=5, label="dσ/dΩ (transfer)")
    if 'dsigma_domega_ruth' in ang and ang['dsigma_domega_ruth'] is not None:
        ax.plot(ang['theta_cm_deg'], ang['dsigma_domega_ruth'], '--',
                color='gray', lw=1, alpha=0.7, label="dσ/dΩ (Rutherford)")

    ds = ang['dsigma_domega']
    idx_peak = np.argmax(ds)
    theta_peak = ang['theta_cm_deg'][idx_peak]
    ax.axvline(theta_peak, color='red', ls=':', lw=1.5,
               label=f"Peak ≈ {theta_peak:.1f}°")

    ax.set_xlabel("θ_cm (deg)")
    ax.set_ylabel("dσ/dΩ (mb/sr)")
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_angular_distribution(result: Dict, output_path: str = None):
    """画角分布图 (中位能量, 由 result['angular'] 提供)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    ang = result.get('angular')
    if not ang:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _plot_angular(ang, ax)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 角分布图 → {output_path}")
    else:
        plt.show()


def plot_angular_distribution_to_file(ang: Dict, output_path: str,
                                      e_lab: float = None):
    """把单个能量点的角分布存为图片 (用于每个能量点目录下的独立角分布图)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    if not ang:
        return
    e_lab_ = ang.get('e_lab', e_lab)
    title = f"α angular distribution, E_lab={e_lab_:.0f} MeV"
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    _plot_angular(ang, ax, title=title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] 角分布图 → {output_path}")


def _optimal_e_star(e_cm: float) -> float:
    """库仑匹配最优剩余激发能 E*_opt = Q₀ − Q_opt

    Q_opt = (Z₃Z₄/Z₁Z₂ − 1)·E_cm, 其中 (1,2)=入射道 (7Li,Th), (3,4)=出口道 (α,Pa)。
    这是经典单值预测; 与模型算出的 E* 分布均值对比, 体现费米运动带来的展宽。
    """
    ratio = (_sys.spectator.Z * _sys.product.Z) / (_sys.proj.Z * _sys.targ.Z)
    q_opt = (ratio - 1.0) * e_cm
    return _sys.q_total - q_opt


def _plot_e_star_spec(spec: Dict, ax, title: str = ""):
    """在一张 ax 上画单条 E* 谱 (谱线 + Q_capture/<E*> 竖线 + E*_opt 竖线)"""
    ax.fill_between(spec['e_star'], 0, spec['dsigma_de'],
                    color='C2', alpha=0.4)
    ax.plot(spec['e_star'], spec['dsigma_de'], '-', color='C2', lw=2)

    ax.axvline(spec['q_capture'], color='gray', ls='--', lw=1,
               label=f"Q_capture={spec['q_capture']:.1f} MeV")
    ax.axvline(spec['e_star_mean'], color='red', ls='-', lw=1.5,
               label=f"<E*>={spec['e_star_mean']:.1f} MeV")

    # 库仑匹配最优激发能 (Q_opt 单值)
    if 'e_cm' in spec:
        e_opt = _optimal_e_star(spec['e_cm'])
        ax.axvline(e_opt, color='purple', ls=':', lw=2,
                   label=f"E*_opt(Q_opt)={e_opt:.1f} MeV")

    ax.set_xlabel("E* (MeV)")
    ax.set_ylabel("dσ/dE* (mb/MeV)")
    ax.legend(fontsize=9)
    ax.set_title(title or f"Excitation Energy Spectrum of {_sys.product.symbol}*")
    ax.grid(True, alpha=0.3)


def plot_e_star_spectrum(result: Dict, output_path: str = None):
    """画激发能谱 (中位能量, 由 result['e_star_spectrum'] 提供)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    spec = result.get('e_star_spectrum')
    if spec is None:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _plot_e_star_spec(spec, ax)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 激发能谱图 → {output_path}")
    else:
        plt.show()


def plot_alpha_double_diff(result: Dict, output_path: str = None,
                           label: str = ""):
    """画 α 旁观者双微分截面热图 d²σ/dE_α dΩ_α (θ_α, E_α)

    坐标系与 THM 实验图 (Cook et al. 2019) 一致: 横轴 θ_lab, 纵轴 E_α。
    result 为 compute_alpha_double_differential 的返回。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    th = result['theta_alpha_deg']
    ea = result['e_alpha']
    d2s = result['d2sigma']

    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # 对数色标: 刻度自动为幂值 (10⁻³, 10⁻², 10⁻¹, 1, 10, PRL Fig.1 风格),
    # 标签不写 log; 范围按数据自适应 (不写死)
    d2s_pos = d2s[d2s > 0]
    vmin = max(float(np.min(d2s_pos)) if d2s_pos.size else 1e-3, 1e-6)
    mesh = ax.pcolormesh(th, ea, d2s, shading='auto', cmap='viridis',
                         norm=LogNorm(vmin=vmin, vmax=float(np.max(d2s))))
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("d²σ/dE_α dΩ_α (mb/sr/MeV)")

    ax.set_xlabel("θ_α (deg)")
    ax.set_ylabel("E_α (MeV)")
    e_lab = result.get('e_lab')
    ax.set_title(f"α spectator double-diff. E_lab={e_lab:.0f} MeV {label}".strip())
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] α 双微分热图 → {output_path}")
    else:
        plt.show()


def plot_e_star_spec_to_file(spec: Dict, output_path: str,
                             e_lab: float = None):
    """把单个 E* 谱存为图片 (用于每个能量点目录下的独立谱图)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    if spec is None:
        return
    e_lab_ = spec.get('e_lab', e_lab)
    title = f"E* spectrum, E_lab={e_lab_:.0f} MeV"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    _plot_e_star_spec(spec, ax, title=title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] E* 谱图 → {output_path}")


def _alpha_opt_energy(spec: Dict) -> float:
    """单值参照: E* = E*_opt (Q_opt 库仑匹配) 对应的单一 α 动能

    由能量守恒 E* = E_cm + Q_total − E_α − E_Pa, E_Pa = (m_α/m_Pa)·E_α
    (Pa 静止近似) 解出: E_α = (E_cm + Q_total − E*) / (1 + m_α/m_Pa)。
    与模型算出的分布对比, 展示"α 动能是分布而非固定值"。
    """
    e_cm = spec['e_cm']
    e_opt = _optimal_e_star(e_cm)
    m_alpha = _sys.spectator.mass_MeV
    m_pa = _sys.product.mass_MeV
    return (e_cm + _sys.q_total - e_opt) / (1.0 + m_alpha / m_pa)


def plot_alpha_energy_distribution(spec: Dict, output_path: str,
                                   e_lab: float = None):
    """α 旁观者动能分布图 (核心输出)

    分布 + <E_α> 红竖线 + 单值参照虚线 (E* = E*_opt 对应的 E_α)。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    if spec is None:
        return
    e_lab_ = spec.get('e_lab', e_lab)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.fill_between(spec['e_alpha'], 0, spec['dsigma_de'],
                    color='C0', alpha=0.4)
    ax.plot(spec['e_alpha'], spec['dsigma_de'], '-', color='C0', lw=2)

    ax.axvline(spec['e_alpha_mean'], color='red', ls='-', lw=1.5,
               label=f"<E_α>={spec['e_alpha_mean']:.1f} MeV")

    e_opt = _alpha_opt_energy(spec)
    ax.axvline(e_opt, color='purple', ls=':', lw=2,
               label=f"E_α(Q_opt)={e_opt:.1f} MeV (single-value)")

    ax.set_xlabel("E_α (MeV)")
    ax.set_ylabel("dσ/dE_α (mb/MeV)")
    ax.legend(fontsize=9)
    ax.set_title(f"α spectator kinetic energy, E_lab={e_lab_:.0f} MeV")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] α 动能分布图 → {output_path}")


def write_alpha_energy_distribution(spec: Dict, output_path: str):
    """α 动能分布数据文件 (供与实验对比)"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Alpha spectator kinetic energy distribution\n")
        f.write(f"# E_lab = {spec['e_lab']:.1f} MeV   E_cm = {spec['e_cm']:.2f} MeV\n")
        f.write(f"# <E_alpha> = {spec['e_alpha_mean']:.3f} MeV   "
                f"std = {spec['e_alpha_std']:.3f} MeV\n")
        f.write(f"# Single-value reference (E* = E*_opt): "
                f"E_alpha = {_alpha_opt_energy(spec):.3f} MeV\n")
        f.write(f"#\n")
        f.write(f"# {'E_alpha(MeV)':>14s}  {'dsigma/dE_alpha(mb/MeV)':>24s}\n")
        for e, d in zip(spec['e_alpha'], spec['dsigma_de']):
            f.write(f"  {e:12.4f}  {d:22.6e}\n")
    print(f"  [data] α 动能分布 → {output_path}")


def plot_e_star_spectra_map(specs: Dict, output_path: str = None,
                            label: str = ""):
    """二维"频谱图": 横轴 E*, 纵轴 E_lab, 颜色 = log10(dσ/dE*)

    每个能量点的 dσ/dE* 插值到统一 E* 网格; 叠加库仑匹配最优激发能
    E*_opt(Q_opt) 曲线 (随 E_lab 变化的一条线)。

    Parameters
    ----------
    specs : {float(E_lab): spec} 由 compute_full all_spectra=True 生成
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib 未安装, 跳过画图")
        return

    if not specs:
        print("[WARNING] 没有多能量谱 (e_star_spectra), 跳过频谱图")
        return

    energies = sorted(specs.keys())
    # 统一 E* 网格 (覆盖所有能量)
    e_lo = min(spec['e_star'].min() for spec in specs.values())
    e_hi = max(spec['e_star'].max() for spec in specs.values())
    e_star_grid = np.linspace(e_lo, e_hi, 300)

    # 构建 2D 数组 [E_lab, E*]
    Z = np.zeros((len(energies), len(e_star_grid)))
    for i, e_lab in enumerate(energies):
        spec = specs[e_lab]
        dsde = np.interp(e_star_grid, spec['e_star'], spec['dsigma_de'],
                         left=0.0, right=0.0)
        Z[i] = dsde

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    logZ = np.log10(np.maximum(Z, 1e-30))
    # 横轴 E_lab, 纵轴 E* (交换)
    mesh = ax.pcolormesh(np.array(energies), e_star_grid, logZ.T,
                         shading='auto', cmap='viridis')
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10(dσ/dE*) [mb/MeV]")

    # E*_opt(Q_opt) 曲线: 随 E_lab 上升的一条线 (x=E_lab, y=E*_opt)
    e_opt_curve = []
    for e_lab in energies:
        e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
        e_opt_curve.append(_optimal_e_star(e_cm))
    ax.plot(np.array(energies), e_opt_curve, '-', color='red', lw=2.5,
            label="E*_opt(Q_opt)")

    ax.axhline(_sys.q_capture, color='white', ls='--', lw=1, alpha=0.6)

    ax.set_xlabel("E_lab (MeV)")
    ax.set_ylabel("E* (MeV)")
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title(f"Excitation Energy Spectra map {label}".strip())
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  [plot] 频谱图 → {output_path}")
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
    # α 旁观者动能分布 (核心输出): 图 + 数据文件
    alpha = result.get('alpha_energy')
    if alpha:
        plot_alpha_energy_distribution(
            alpha, os.path.join(output_dir, "alpha_energy_distribution.png"))
        write_alpha_energy_distribution(
            alpha, os.path.join(output_dir, "alpha_energy_distribution.txt"))
    # 若有多能量谱, 额外画二维频谱图 (含 E*_opt 曲线)
    if result.get('e_star_spectra'):
        plot_e_star_spectra_map(result['e_star_spectra'],
                                os.path.join(output_dir, "e_star_spectra_map.png"))