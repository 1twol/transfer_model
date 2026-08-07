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
from .kinematics import (rutherford_trajectory, impact_parameter_from_angle,
                          rutherford_trajectory, grazing_angular_momentum,
                          post_acceleration)
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


def make_theta_grid(e_cm: float, n_theta: int = None) -> np.ndarray:
    """生成质心系角度网格"""
    if n_theta is None:
        n_theta = _mod.n_theta
    theta_min = np.radians(_mod.theta_min_deg)
    theta_max = np.radians(_mod.theta_max_deg)
    return np.linspace(theta_min, theta_max, n_theta)


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
    """计算角分布 dσ/dΩ(θ)

    变换: dσ/dΩ = (dσ/dΩ)_Ruth × P_tr(θ)
          = (b / sinθ) |db/dθ| × P_tr(θ)

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
    theta_grid = make_theta_grid(e_cm, n_theta)
    n_theta_actual = len(theta_grid)

    dsdo = np.zeros(n_theta_actual)
    dsdo_ruth = np.zeros(n_theta_actual)

    for i, theta in enumerate(theta_grid):
        # 碰撞参数
        b = impact_parameter_from_angle(theta, e_cm,
                                         _sys.proj.Z, _sys.targ.Z,
                                         _sys.mu_proj_targ)

        # 转移概率
        if isinstance(model, (FermiIntegratedModel, ICFFractionModel)):
            p_tr = model.probability(e_cm, b, n_fermi_samples=n_fermi)
        else:
            p_tr = model.probability(e_cm, b)

        # 卢瑟福截面 (作为参照)
        eta = config.sommerfeld(_sys.proj.Z, _sys.targ.Z,
                                 _sys.mu_proj_targ, e_cm)
        k = config.wavenumber(_sys.mu_proj_targ, e_cm)
        a = eta / k
        sin_half = np.sin(theta / 2.0)
        dsdo_ruth[i] = (a / (2.0 * k * sin_half**2))**2  # fm²/sr
        dsdo_ruth[i] *= 10  # → mb/sr

        # 转移截面
        dsdo[i] = dsdo_ruth[i] * p_tr

        if verbose and (i % max(1, n_theta_actual // 5) == 0):
            print(f"  θ={np.degrees(theta):.1f}°, b={b:.1f} fm, "
                  f"P={p_tr:.4e}, dσ/dΩ={dsdo[i]:.4e} mb/sr")

    # 实验室系角度
    theta_lab = np.zeros(n_theta_actual)
    for i, theta in enumerate(theta_grid):
        from .kinematics import cm_to_lab
        theta_lab[i], _ = cm_to_lab(theta, e_cm,
                                      _sys.spectator.mass_MeV,
                                      _sys.product.mass_MeV,
                                      _sys.q_total)

    return {
        'theta_cm': theta_grid,
        'theta_cm_deg': np.degrees(theta_grid),
        'theta_lab': theta_lab,
        'theta_lab_deg': np.degrees(theta_lab),
        'dsigma_domega': dsdo,
        'dsigma_domega_ruth': dsdo_ruth,
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

    # 激发能范围
    q_capture = _sys.q_capture
    e_star_min = max(0, q_capture - 5.0)
    e_star_max = q_capture + 25.0
    e_star_edges = np.linspace(e_star_min, e_star_max, e_star_bins + 1)
    e_star_centers = 0.5 * (e_star_edges[:-1] + e_star_edges[1:])

    # 累积谱
    dsigma_de = np.zeros(e_star_bins)

    for j, b in enumerate(b_grid):
        if isinstance(model, (FermiIntegratedModel, ICFFractionModel)):
            _, details = model.probability(
                e_cm, b, n_fermi_samples=n_fermi, return_details=True
            )
            p_values = details['probabilities']
            e_star_values = details['e_star']
        else:
            # 非费米模型: 用简单的Q值
            p_values = np.array([model.probability(e_cm, b)])
            e_star_values = np.array([q_capture])

        # 加权: dσ/dE* ∝ b × P(b) × δ(E*)
        weight = 2.0 * np.pi * b * np.mean(p_values)  # fm²

        # 将概率分配到激发能 bins
        hist, _ = np.histogram(e_star_values, bins=e_star_edges,
                                weights=np.ones_like(e_star_values) * weight / len(e_star_values))
        dsigma_de += hist

        if verbose and (j % max(1, n_b // 4) == 0):
            print(f"  b={b:.1f} fm, <P>={np.mean(p_values):.4e}")

    # 单位转换: fm²/MeV → mb/MeV
    dsigma_de *= 10
    # bin宽度归一化
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
    }


# ============================================================
# 3. 完整计算: 含后加速修正
# ============================================================

def compute_full(model: TransferModel,
                  e_lab_range: np.ndarray = None,
                  n_fermi: int = 5000,
                  include_post_accel: bool = True,
                  verbose: bool = True) -> Dict:
    """完整计算: 激发函数 + 角分布(中位能量) + E* 谱(中位能量)

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

    # 3. 中位能量激发能谱
    if verbose:
        print(f"\n{'='*50}")
        print(f"3. 计算激发能谱 dσ/dE* (E_lab={e_mid:.0f} MeV)...")
        print("=" * 50)
    e_star_spec = compute_excitation_energy_spectrum(model, e_mid,
                                                      n_b=min(_mod.n_b, 40),
                                                      n_fermi=n_fermi * 2,
                                                      verbose=verbose)
    result['e_star_spectrum'] = e_star_spec

    return result
