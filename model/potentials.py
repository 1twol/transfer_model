"""
potentials.py — 库仑势、核势、形状因子

包含:
  1. 点电荷 / 均匀球库仑势
  2. Woods-Saxon 核势 (Akyüz-Winther 系统学)
  3. 总有效势 (入射道 / 出口道)
  4. 转移形状因子
  5. São Paulo 双折叠势接口 (预留)

参考:
  - Akyüz & Winther, in "Nuclear Structure and Heavy-Ion Reactions" (1981)
  - Broglia & Winther, "Heavy Ion Reactions" (2004)
  - São Paulo potential: Chamon et al., PRL 79, 5218 (1997)
"""

import numpy as np
from typing import Tuple, Optional, Callable
from . import config


# ============================================================
# 1. 库仑势
# ============================================================

def coulomb_point(r: np.ndarray, z1: int, z2: int) -> np.ndarray:
    """点电荷库仑势 (MeV)"""
    r_safe = np.maximum(r, 1e-12)
    return z1 * z2 * config.E2 / r_safe


def coulomb_uniform_sphere(r: np.ndarray, z1: int, z2: int,
                            a1: int, a2: int, r0c: float = 1.25) -> np.ndarray:
    """均匀带电球库仑势 (MeV)

    r >= R_c: V_c = Z₁Z₂ e² / r
    r <  R_c: V_c = Z₁Z₂ e²/(2R_c) · (3 - (r/R_c)²)
    """
    r_safe = np.maximum(r, 1e-12)
    rc = r0c * (a1**(1.0/3.0) + a2**(1.0/3.0))
    v = np.where(r_safe >= rc,
                 z1 * z2 * config.E2 / r_safe,
                 z1 * z2 * config.E2 / (2.0 * rc) * (3.0 - (r_safe / rc)**2))
    return v


# ============================================================
# 2. Woods-Saxon 核势
# ============================================================

def woods_saxon(r: np.ndarray, v0: float, r0: float, a: float,
                 ap: int, at: int) -> np.ndarray:
    """Woods-Saxon 势 (MeV)

    V(r) = -V₀ / [1 + exp((r - R)/a)]
    R = r₀ (A_p^{1/3} + A_t^{1/3})
    """
    radius = r0 * (ap**(1.0/3.0) + at**(1.0/3.0))
    return -v0 / (1.0 + np.exp((r - radius) / a))


def woods_saxon_derivative(r: np.ndarray, v0: float, r0: float, a: float,
                            ap: int, at: int) -> np.ndarray:
    """WS 势导数 dV/dr (用于形状因子)"""
    radius = r0 * (ap**(1.0/3.0) + at**(1.0/3.0))
    exp_arg = np.exp((r - radius) / a)
    common = 1.0 + exp_arg
    return v0 * exp_arg / (a * common**2)


def akyuz_winther_potential(ap: int, zt: int, at: int, zp: int) -> Tuple[float, float, float]:
    """Akyüz-Winther 系统学: 自动估算 WS 势参数

    Returns
    -------
    v0 : 势阱深度 (MeV)
    r0 : 约化半径 (fm)
    a  : 弥散 (fm)
    """
    # 基于 Broglia & Winther 参数化
    r0 = 1.17 - 0.34 * (ap**(-1.0/3.0) + at**(-1.0/3.0))
    # 简化: 对于中重/重核系统
    if at > 200:
        # 锕系靶
        v0 = 75.0
        r0 = 1.18
        a = 0.65
    elif at > 100:
        v0 = 70.0
        r0 = 1.17
        a = 0.63
    else:
        v0 = 60.0
        r0 = 1.16
        a = 0.60

    return v0, r0, a


# ============================================================
# 3. 复合势 (库仑 + 核)
# ============================================================

def total_potential(r: np.ndarray, e_cm: float,
                     z1: int, a1: int, z2: int, a2: int,
                     v0_nuc: float, r0_nuc: float, a_nuc: float,
                     include_centrifugal: bool = False,
                     l_val: float = 0.0, mu: float = 0.0) -> np.ndarray:
    """总有效势 V_tot = V_Coul + V_Nuc + V_cent

    Parameters
    ----------
    r : 径向坐标 (fm)
    e_cm : 质心系能量 (MeV)
    z1, a1 : 核1的Z, A
    z2, a2 : 核2的Z, A
    v0_nuc, r0_nuc, a_nuc : WS 核势参数
    include_centrifugal : 是否包含离心势
    l_val : 轨道角动量 (ħ 单位)
    mu : 约化质量 (MeV/c²)
    """
    v = coulomb_uniform_sphere(r, z1, z2, a1, a2)
    v += woods_saxon(r, v0_nuc, r0_nuc, a_nuc, a1, a2)

    if include_centrifugal and mu > 0:
        v += config.HBARC**2 * l_val * (l_val + 1.0) / (2.0 * mu * r**2)

    return v


def find_barrier(r_grid: np.ndarray, potential: np.ndarray) -> Tuple[float, float, float]:
    """找库仑势垒的位置、高度和曲率

    Returns
    -------
    rb : 势垒位置 (fm)
    vb : 势垒高度 (MeV)
    curv : 曲率 ħω = ħ√(-V''/μ) (MeV)
    """
    # 找最大值位置
    idx_max = np.argmax(potential)
    rb = r_grid[idx_max]
    vb = potential[idx_max]

    # 二阶导数求曲率
    dr = r_grid[1] - r_grid[0]
    if 1 <= idx_max <= len(r_grid) - 2:
        d2v = (potential[idx_max + 1] - 2 * potential[idx_max] + potential[idx_max - 1]) / dr**2
    else:
        d2v = -1.0  # fallback

    # 曲率 ħω = ħ √(-V''/μ)
    # 这里返回势的二阶导数值本身
    curv = np.sqrt(max(-d2v, 1e-6))
    return rb, vb, curv


# ============================================================
# 4. 转移形状因子
# ============================================================

def transfer_form_factor_zero_range(r: np.ndarray, d0: float,
                                     u_alphat: callable) -> np.ndarray:
    """零程转移形状因子

    F(R) = D₀ · φ_αt(R)

    其中 R 是 α-²³²Th (或等价地, t 从 ⁷Li 看出去)的距离

    Parameters
    ----------
    r : 径向点 (fm)
    d0 : 零程常数 (MeV·fm³/²)
    u_alphat : u(r) = r·φ_αt(r) 的插值函数
    """
    return d0 * u_alphat(np.maximum(r, 1e-12)) / np.maximum(r, 1e-12)


def transfer_form_factor_finite_range(r_alpha_t: np.ndarray,
                                       r_t_th: np.ndarray,
                                       u_alphat: callable,
                                       u_tth: callable) -> np.ndarray:
    """有限程转移形状因子 (占位, 后续扩展)

    完整形式需要对两个波函数的积分:
    F ∝ ∫ d³x φ*_tTh(x) V(x) φ_αt(x - R)

    此处留作接口。
    """
    raise NotImplementedError("有限程形状因子待实现")


# ============================================================
# 5. São Paulo 双折叠势接口 (预留)
# ============================================================

class SaoPauloPotential:
    """São Paulo 双折叠势 (接口预留)

    参考: Chamon et al., PRL 79, 5218 (1997)
          Chamon et al., PRC 66, 014610 (2002)

    V_SP(R) = V_F(R) · exp(-4 v²/c²)
    其中 V_F(R) 是冻结密度双折叠积分
    """

    def __init__(self):
        self._initialized = False

    def initialize(self, ap: int, zp: int, at: int, zt: int):
        """初始化核密度分布"""
        # TODO: 实现双折叠积分
        self._initialized = True
        raise NotImplementedError("São Paulo 势待实现")

    def __call__(self, r: np.ndarray, e_cm: float) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("São Paulo 势未初始化")
        raise NotImplementedError("São Paulo 势待实现")


# ============================================================
# 6. 光学模型势 (DWBA 用, 预留扩展)
# ============================================================

def optical_potential(r: np.ndarray, e_cm: float,
                       ap: int, zp: int, at: int, zt: int,
                       v0: float = 0, r0: float = 0, a: float = 0,
                       w0: float = 0, rw: float = 0, aw: float = 0) -> np.ndarray:
    """光学模型势 V_opt = V(r) + iW(r) + V_Coul(r)

    如果 V₀=0, 自动用 Akyüz-Winther 估算。
    """
    if v0 == 0:
        v0, r0, a = akyuz_winther_potential(ap, zt, at, zp)

    # 实部
    v_real = woods_saxon(r, v0, r0, a, ap, at)

    # 虚部 (如果提供)
    v_imag = np.zeros_like(r)
    if w0 > 0:
        if rw == 0:
            rw = r0 * 1.05
        if aw == 0:
            aw = a * 0.95
        v_imag = -1j * w0 / (1.0 + np.exp((r - rw * (ap**(1/3) + at**(1/3))) / aw))

    # 库仑
    v_coul = coulomb_uniform_sphere(r, zp, zt, ap, at)

    return v_real + v_imag + v_coul
