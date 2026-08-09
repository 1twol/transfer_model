"""
cross_section.py — 相空间积分求截面

积分流程:
  1. 对碰撞参数 b 积分 → 总转移截面 σ(E)
  2. 角度变换 → dσ/dΩ
  3. 双重积分 ∫db∫d³k → 激发能分布 dσ/dE*
  4. 出口道后加速修正

截面公式:
  σ_tr(E) = 2π ∫₀^∞ b db P_tr(b, E)
  dσ_tr/dΩ = (b/sinθ) |db/dθ| P_tr(θ)
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.integrate import simpson

from . import config
from .kinematics import grazing_angular_momentum, coulomb_recoil
from .transfer import TransferModel, FermiIntegratedModel, ICFFractionModel

_sys = config.system
_mod = config.model


# ============================================================
# 1. 碰撞参数网格
# ============================================================

def make_b_grid(e_cm: float, n_b: int = None, b_max: float = None) -> np.ndarray:
    """生成碰撞参数网格

    策略: b 从 0 到 b_max, 在擦边角动量 L_g 附近加密
    """
    if n_b is None:
        n_b = _mod.n_b

    r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A, _mod.r0)
    l_g = grazing_angular_momentum(e_cm, r_int,
                                    _sys.proj.Z, _sys.targ.Z)
    k = config.wavenumber(_sys.mu_proj_targ, e_cm)
    b_g = l_g / k if k > 0 else r_int

    if b_max is None or b_max <= 0:
        b_max = max(2.0 * b_g, r_int * _mod.b_max_factor)

    # 在擦边附近加密
    n_inner = int(n_b * 0.4)
    n_outer = n_b - n_inner

    b_inner = np.linspace(0, b_g * 1.2, n_inner)
    b_outer = np.linspace(b_g * 1.2, b_max, n_outer)

    # 合并, 跳过重复点
    b_grid = np.unique(np.concatenate([b_inner, b_outer]))
    return b_grid


def _near_point_geometry(e_cm: float, b: float) -> Tuple[float, float]:
    """入射道卢瑟福轨道近点: 距离 D(b) 与近点方向角 φ_p (rad, 相对束流)

    近点方向角 = (π − θ_in)/2 = arctan(b/a), 其中 θ_in = 2 arctan(a/b)
    是入射道散射角, a = η/k 是卢瑟福半长轴。
    """
    eta = config.sommerfeld(_sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm)
    k = config.wavenumber(_sys.mu_proj_targ, e_cm)
    a = eta / k
    d = config.distance_of_closest_approach(_sys.proj.Z, _sys.targ.Z,
                                            _sys.mu_proj_targ, e_cm, b)
    # φ_p = arctan(b/a) = (π − θ_in)/2; b→0 正碰时近点在束流前方 (φ_p→0)
    phi_p = np.arctan(b / max(a, 1e-9))
    return d, phi_p


def _alpha_b_min(e_cm: float) -> float:
    """近正碰下界: 近点进入核区 (D(b) < R_int) 的 7Li 被完全融合吸收,
    不产生可测的旁观者 α。解 D(b_min) = R_int:
      a + √(a² + b²) = R_int  →  b_min = √(R_int² − 2aR_int),  a = η/k
    """
    r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A, _mod.r0)
    eta = config.sommerfeld(_sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm)
    k = config.wavenumber(_sys.mu_proj_targ, e_cm)
    a = eta / k
    return np.sqrt(max(r_int * r_int - 2.0 * a * r_int, 0.0))


def _alpha_velocity(e_cm: float, b: float, k_mag, k_theta) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """α 旁观者的初始速度 (近点切向推进 + 费米) 与库仑传播结果

    α 在近点 (D, φ_p) 继承 ⁷Li 的近点切向速度 v_near = b·v_∞/D (角动量
    守恒, 非渐近速度 v_∞——近点处 7Li 大部分动能已转为库仑势能), 叠加上
    7Li 静止系内与 t 反向的费米速度, 再经 coulomb_recoil 传播到无穷远。

    Returns
    -------
    (theta_out, e_out, e_breakup) :
      theta_out : 渐近出射角 (rad, [0, π])
      e_out : 无穷远 α 动能 (MeV)
      e_breakup : 破裂点 α 动能 (MeV, 库仑增益前)
    """
    d, phi_p = _near_point_geometry(e_cm, b)
    v_inf = np.sqrt(2.0 * e_cm / _sys.mu_proj_targ)
    v_near = b * v_inf / max(d, 1e-9)  # 近点切向速度 (角动量守恒 L=μ b v_∞=μ D v_tan)
    m_t = _sys.cluster.mass_MeV
    m_alpha = _sys.spectator.mass_MeV

    v_t = config.HBARC * np.asarray(k_mag, float) / m_t
    # α 费米速度 (与 t 反向, 动量守恒 |p_α|=|p_t|=ħk)
    v_ax = -v_near * np.sin(phi_p) - (m_t / m_alpha) * v_t * np.cos(np.asarray(k_theta, float))
    v_aperp = v_near * np.cos(phi_p) - (m_t / m_alpha) * v_t * np.sin(np.asarray(k_theta, float))

    e_breakup = 0.5 * m_alpha * (v_ax**2 + v_aperp**2)
    theta_out, e_out = coulomb_recoil(d, phi_p, v_ax, v_aperp,
                                      _sys.spectator.Z, _sys.product.Z, m_alpha)
    return theta_out, e_out, e_breakup


# ============================================================
# 2. 截面积分主函数
# ============================================================

def compute_excitation_function(model: TransferModel,
                                 e_lab_range: np.ndarray = None,
                                 n_fermi: int = 5000,
                                 verbose: bool = True) -> Dict:
    """计算激发函数 σ_tr(E)

    σ_tr(E_lab) = 2π ∫ b db P_tr(b, E_cm(E_lab))

    Parameters
    ----------
    model : 转移模型
    e_lab_range : 实验室系能量数组 (MeV), None 则用 config 默认
    n_fermi : 费米动量抽样数 (仅对 FermiIntegratedModel)
    verbose : 是否打印进度

    Returns
    -------
    result : {'e_lab', 'e_cm', 'sigma', 'sigma_rutherford', 'lg'}
    """
    if e_lab_range is None:
        e_lab_range = np.arange(_mod.e_lab_min,
                                 _mod.e_lab_max + _mod.e_lab_step / 2,
                                 _mod.e_lab_step)

    n_energies = len(e_lab_range)
    sigma = np.zeros(n_energies)
    sigma_rutherford = np.zeros(n_energies)  # 纯卢瑟福参考截面
    l_g_values = np.zeros(n_energies)

    for i, e_lab in enumerate(e_lab_range):
        e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
        b_grid = make_b_grid(e_cm)
        p_grid = np.zeros_like(b_grid)

        l_g = grazing_angular_momentum(e_cm,
                                        config.interaction_radius(
                                            _sys.proj.A, _sys.targ.A),
                                        _sys.proj.Z, _sys.targ.Z)
        l_g_values[i] = l_g

        for j, b in enumerate(b_grid):
            if isinstance(model, (FermiIntegratedModel, ICFFractionModel)):
                p_grid[j] = model.probability(e_cm, b, n_fermi_samples=n_fermi)
            else:
                p_grid[j] = model.probability(e_cm, b)

        # σ = 2π ∫ b P(b) db
        # 1 fm² = 10 mb
        integrand = 2.0 * np.pi * b_grid * p_grid  # fm²
        sigma[i] = simpson(integrand, b_grid) * 10  # → mb

        # 卢瑟福截面 (全融合上限: P=1 for b < b_g, P=0 for b > b_g)
        b_g = l_g / config.wavenumber(_sys.mu_proj_targ, e_cm)
        idx_g = np.searchsorted(b_grid, b_g)
        sigma_rutherford[i] = np.pi * b_g**2 * 10  # geometric, mb

        if verbose:
            print(f"  E_lab={e_lab:.0f} MeV, E_cm={e_cm:.2f} MeV, "
                  f"σ={sigma[i]:.4e} mb, L_g={l_g:.0f}")

    return {
        'e_lab': e_lab_range,
        'e_cm': np.array([config.e_lab_to_e_cm(e, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
                           for e in e_lab_range]),
        'sigma': sigma,
        'sigma_rutherford': sigma_rutherford,
        'l_g': l_g_values,
    }


def compute_angular_distribution(model: TransferModel,
                                    e_lab: float,
                                    n_theta: int = None,
                                    n_fermi: int = 5000,
                                    verbose: bool = True) -> Dict:
    """计算 α 旁观者的角分布 dσ/dΩ(θ_α)

    α 旁观者运动学 (含出口道库仑后加速):
      每个费米事件, α 在 ⁷Li 破裂点 (近点) 继承 ⁷Li 的近点切向速度 +
      内部费米速度 (与 t 反向), 然后在 ²³⁵Pa 库仑排斥场中传播到无穷远,
      得到渐近出射角 θ_α。按转移概率 P_tr(b,k) 加权累加进角 bin。

    dσ/dΩ_α 归一化: Σ_θ dσ/dΩ·ΔΩ = σ_tr (总截面守恒)。

    Parameters
    ----------
    model : 转移模型
    e_lab : 固定实验室系能量 (MeV)
    n_theta : 角度网格点数
    n_fermi : 费米动量抽样数

    Returns
    -------
    result : {'theta_cm', 'theta_lab', 'dsigma_domega', 'dsigma_domega_ruth'}
    """
    e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
    if n_theta is None:
        n_theta = _mod.n_theta

    # α 出射角范围 [0, π] (朝后的 α 事件也计入)
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    b_grid = make_b_grid(e_cm, min(_mod.n_b, 40))
    # 剔除近正碰 (近点进入核区, 融合吸收, 无旁观 α)
    b_min = _alpha_b_min(e_cm)
    b_grid = b_grid[b_grid >= b_min]
    if len(b_grid) < 3:
        b_grid = make_b_grid(e_cm, min(_mod.n_b, 40))[1:]
    # b 积分的求积权重 (梯形): 非均匀 b_grid 必须用权重
    b_w = np.zeros_like(b_grid)
    b_w[0] = 0.5 * (b_grid[1] - b_grid[0])
    b_w[-1] = 0.5 * (b_grid[-1] - b_grid[-2])
    b_w[1:-1] = 0.5 * (b_grid[2:] - b_grid[:-2])

    all_theta = []
    all_w = []

    for j, b in enumerate(b_grid):
        k_mag, k_theta, p, _ = model.event_distribution(e_cm, b, n_fermi)
        p = np.asarray(p, dtype=float)
        k_mag = np.asarray(k_mag, dtype=float)
        k_theta = np.asarray(k_theta, dtype=float)

        # α 旁观者运动学: 近点切向初速 + 库仑后传播 → 渐近出射角
        th_alpha, _, _ = _alpha_velocity(e_cm, b, k_mag, k_theta)

        # 每个样本的截面权重: b_w·2π·b·(p_i/N)  (fm², 最后 ×10 → mb)
        w = b_w[j] * 2.0 * np.pi * b * p / len(p)

        all_theta.append(th_alpha)
        all_w.append(w)

        if verbose and (j % max(1, len(b_grid) // 4) == 0):
            print(f"  b={b:.1f} fm, <P>={np.mean(p):.4e}")

    theta_all = np.concatenate(all_theta)
    w_all = np.concatenate(all_w)

    # dσ/dΩ = (Σ bin 内权重 ×10) / ΔΩ,  ΔΩ = 2π sinθ Δθ
    dsdo, _ = np.histogram(theta_all, bins=theta_edges, weights=w_all)
    dtheta = theta_edges[1] - theta_edges[0]
    domega = 2.0 * np.pi * np.sin(theta_centers) * dtheta
    dsdo = dsdo * 10.0 / np.maximum(domega, 1e-10)  # → mb/sr

    # 入射道卢瑟福截面 (参照曲线, 前向峰)
    eta = config.sommerfeld(_sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm)
    k = config.wavenumber(_sys.mu_proj_targ, e_cm)
    a = eta / k
    sin_half = np.sin(theta_centers / 2.0)
    dsdo_ruth = (a / (2.0 * k * sin_half**2))**2 * 10.0  # mb/sr

    # 重靶反冲 → α 实验室角 ≈ 质心角
    return {
        'theta_cm': theta_centers,
        'theta_cm_deg': np.degrees(theta_centers),
        'theta_lab': theta_centers,
        'theta_lab_deg': np.degrees(theta_centers),
        'dsigma_domega': dsdo,
        'dsigma_domega_ruth': dsdo_ruth,
    }


def compute_alpha_double_differential(model: TransferModel,
                                        e_lab: float,
                                        n_b: int = None,
                                        n_fermi: int = 5000,
                                        n_theta: int = 80,
                                        n_e_alpha: int = 80,
                                        verbose: bool = False) -> Dict:
    """α 旁观者双微分截面 d²σ/dE_α dΩ_α (θ_α, E_α)

    每个费米事件: α 在近点继承 ⁷Li 近点切向速度 + 内部费米速度 (与 t 反向),
    经 coulomb_recoil 在 ²³⁵Pa 库仑排斥场中传播到无穷远, 得到渐近出射角
    θ_α 与动能 E_α (含库仑后加速增益)。按转移概率 P_tr 加权 bin 到二维网格。

    与 THM 实验图 (如 Cook et al. 2019) 坐标系一致: 横轴 θ_lab, 纵轴 E_α。

    Parameters
    ----------
    model : 转移模型
    e_lab : 实验室系能量 (MeV)
    n_b, n_fermi : b 网格数、费米抽样数
    n_theta, n_e_alpha : 二维网格点数

    Returns
    -------
    result : {'theta_alpha', 'theta_alpha_deg', 'e_alpha', 'd2sigma', 'e_lab'}
      theta_alpha : 角度网格中心 (rad)
      e_alpha : α 动能网格中心 (MeV)
      d2sigma : (n_e_alpha, n_theta) 数组, d²σ/dE dΩ (mb/sr/MeV)
    """
    if n_b is None:
        n_b = min(_mod.n_b, 40)

    e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
    b_grid = make_b_grid(e_cm, n_b)
    # 剔除近正碰 (近点进入核区, 融合吸收, 无旁观 α)
    b_min = _alpha_b_min(e_cm)
    b_grid = b_grid[b_grid >= b_min]
    if len(b_grid) < 3:
        b_grid = make_b_grid(e_cm, n_b)[1:]

    # b 梯形求积权重
    b_w = np.zeros_like(b_grid)
    b_w[0] = 0.5 * (b_grid[1] - b_grid[0])
    b_w[-1] = 0.5 * (b_grid[-1] - b_grid[-2])
    b_w[1:-1] = 0.5 * (b_grid[2:] - b_grid[:-2])

    all_th = []
    all_e = []
    all_w = []

    for j, b in enumerate(b_grid):
        k_mag, k_theta, p, _ = model.event_distribution(e_cm, b, n_fermi)
        p = np.asarray(p, dtype=float)
        k_mag = np.asarray(k_mag, dtype=float)
        k_theta = np.asarray(k_theta, dtype=float)

        # α 旁观者运动学: 近点切向初速 + 库仑后传播 → 渐近出射角/能量
        th_alpha, e_alpha, _ = _alpha_velocity(e_cm, b, k_mag, k_theta)

        w = b_w[j] * 2.0 * np.pi * b * p / len(p)  # fm²
        all_th.append(th_alpha)
        all_e.append(e_alpha)
        all_w.append(w)

    th_all = np.concatenate(all_th)
    e_all = np.concatenate(all_e)
    w_all = np.concatenate(all_w)

    # 二维网格
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    e_lo = max(float(np.percentile(e_all, 0.5)), 0.0)
    e_hi = float(np.percentile(e_all, 99.5))
    if e_hi - e_lo < 1.0:
        e_hi = e_lo + 30.0
    e_edges = np.linspace(e_lo, e_hi, n_e_alpha + 1)
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])

    d2sigma, _, _ = np.histogram2d(e_all, th_all, bins=[e_edges, theta_edges],
                                   weights=w_all)
    d2sigma *= 10.0  # fm² → mb

    # 除以 ΔE·ΔΩ: ΔΩ = 2π sinθ Δθ
    dtheta = theta_edges[1] - theta_edges[0]
    domega = 2.0 * np.pi * np.sin(theta_centers) * dtheta
    de = e_edges[1] - e_edges[0]
    d2sigma = d2sigma / (de * domega[np.newaxis, :])

    return {
        'theta_alpha': theta_centers,
        'theta_alpha_deg': np.degrees(theta_centers),
        'e_alpha': e_centers,
        'd2sigma': d2sigma,
        'e_lab': e_lab,
        'e_cm': e_cm,
    }


def compute_excitation_energy_spectrum(model: TransferModel,
                                         e_lab: float,
                                         n_b: int = None,
                                         n_fermi: int = 10000,
                                         e_star_bins: int = 50,
                                         return_weighted: bool = True,
                                         verbose: bool = True) -> Dict:
    """计算 ²³⁵Pa* 的激发能谱 dσ/dE*

    双重积分:
      dσ/dE* = ∫ b db · 2π · dP/dE*(b)

    其中 dP/dE* 来自费米动量分布投影到激发能。

    Parameters
    ----------
    model : FermiIntegratedModel (必须含费米动量积分)
    e_lab : 实验室系能量 (MeV)
    n_b : b 网格点数
    n_fermi : 费米动量抽样数
    e_star_bins : E* 分 bin 数
    return_weighted : 返回截面加权的谱 (mb/MeV)
    verbose : 是否打印进度

    Returns
    -------
    result : {'e_star', 'dsigma_de', 'e_star_mean', 'e_star_std'}
    """
    if n_b is None:
        n_b = min(_mod.n_b, 50)  # 平衡精度与速度

    e_cm = config.e_lab_to_e_cm(e_lab, _sys.proj.mass_MeV, _sys.targ.mass_MeV)
    b_grid = make_b_grid(e_cm, n_b)

    # 激发能范围下限 (E* = Q_capture + E_rel ≥ Q_capture, 故下限安全)
    q_capture = _sys.q_capture
    e_star_min = max(0.0, q_capture - 5.0)

    # b 积分的求积权重 (梯形): 非均匀 b_grid 必须用权重, 不能用普通求和
    b_w = np.zeros_like(b_grid)
    b_w[0] = 0.5 * (b_grid[1] - b_grid[0])
    b_w[-1] = 0.5 * (b_grid[-1] - b_grid[-2])
    b_w[1:-1] = 0.5 * (b_grid[2:] - b_grid[:-2])

    # 累积所有样本的 (E*, 截面权重); 权重按每个样本自身的 p_i 计
    all_e_star = []
    all_w = []

    for j, b in enumerate(b_grid):
        _, _, p_values, e_star_values = model.event_distribution(e_cm, b, n_fermi)
        p_values = np.asarray(p_values, dtype=float)
        e_star_values = np.asarray(e_star_values, dtype=float)

        # 每个样本的截面权重: dσ/dE* ∝ b_w·2π·b·(p_i/N)  (fm², 最后 ×10 → mb)
        all_e_star.append(e_star_values)
        all_w.append(b_w[j] * 2.0 * np.pi * b * p_values / len(p_values))

        if verbose and (j % max(1, n_b // 4) == 0):
            print(f"  b={b:.1f} fm, <P>={np.mean(p_values):.4e}")

    e_star_all = np.concatenate(all_e_star)
    w_all = np.concatenate(all_w)

    # 自适应 bin 上界: 覆盖全部采样到的 E* (不丢高能尾)
    e_star_max = float(e_star_all.max())
    if e_star_max - e_star_min < 1.0:
        e_star_max = e_star_min + 25.0
    e_star_edges = np.linspace(e_star_min, e_star_max, e_star_bins + 1)
    e_star_centers = 0.5 * (e_star_edges[:-1] + e_star_edges[1:])

    # 单位转换: fm² → mb (×10), bin 宽度归一化 → mb/MeV
    dsigma_de, _ = np.histogram(e_star_all, bins=e_star_edges, weights=w_all)
    dsigma_de *= 10.0
    de = e_star_edges[1] - e_star_edges[0]
    dsigma_de /= de

    e_star_mean = np.average(e_star_centers, weights=dsigma_de + 1e-30)
    e_star_std = np.sqrt(np.average((e_star_centers - e_star_mean)**2,
                                      weights=dsigma_de + 1e-30))

    return {
        'e_star': e_star_centers,
        'dsigma_de': dsigma_de,
        'e_star_mean': e_star_mean,
        'e_star_std': e_star_std,
        'q_capture': q_capture,
        'e_lab': e_lab,
        'e_cm': e_cm,
    }


# ============================================================
# 3. 完整计算: 含后加速修正
# ============================================================

def compute_full(model: TransferModel,
                  e_lab_range: np.ndarray = None,
                  n_fermi: int = 5000,
                  include_post_accel: bool = True,
                  all_spectra: bool = False,
                  verbose: bool = True) -> Dict:
    """完整计算: 激发函数 + 角分布(中位能量) + E* 谱

    Parameters
    ----------
    all_spectra : 为 True 时, 对能量范围内每个 E_lab 各算一张 E* 谱,
                  存入 result['e_star_spectra'] = {float(E_lab): spec}。
                  用较少的 n_fermi 控制耗时。

    Returns
    -------
    result : 包含所有计算结果的大字典
    """
    if e_lab_range is None:
        e_lab_range = np.arange(_mod.e_lab_min,
                                 _mod.e_lab_max + _mod.e_lab_step / 2,
                                 _mod.e_lab_step)

    result = {}

    # 1. 激发函数
    if verbose:
        print("=" * 50)
        print("1. 计算激发函数 σ(E)...")
        print("=" * 50)
    exc_func = compute_excitation_function(model, e_lab_range, n_fermi, verbose)
    result['excitation'] = exc_func

    # 2. 中位能量角分布
    e_mid = e_lab_range[len(e_lab_range) // 2]
    if verbose:
        print(f"\n{'='*50}")
        print(f"2. 计算角分布 dσ/dΩ (E_lab={e_mid:.0f} MeV)...")
        print("=" * 50)
    angular = compute_angular_distribution(model, e_mid,
                                            n_theta=_mod.n_theta,
                                            n_fermi=n_fermi,
                                            verbose=verbose)
    result['angular'] = angular

    # 3. 激发能谱
    if all_spectra:
        # 每个 E_lab 各算一张谱 (与 .pace 文件粒度一致)
        e_star_specs = {}
        n_b_es = min(_mod.n_b, 40)
        n_fermi_es = max(min(n_fermi, 3000), 1000)  # 控制耗时
        for e_lab in e_lab_range:
            if verbose:
                print(f"  谱 E_lab={e_lab:.1f} MeV ...")
            e_star_specs[float(e_lab)] = compute_excitation_energy_spectrum(
                model, e_lab=e_lab, n_b=n_b_es, n_fermi=n_fermi_es, verbose=False)
        result['e_star_spectra'] = e_star_specs
        # 兼容: 中位能量谱也保留单张
        result['e_star_spectrum'] = e_star_specs.get(float(e_mid))
    else:
        if verbose:
            print(f"\n{'='*50}")
            print(f"3. 计算激发能谱 dσ/dE* (E_lab={e_mid:.0f} MeV)...")
            print("=" * 50)
        result['e_star_spectrum'] = compute_excitation_energy_spectrum(
            model, e_mid,
            n_b=min(_mod.n_b, 40),
            n_fermi=n_fermi * 2,
            verbose=verbose)

    return result
