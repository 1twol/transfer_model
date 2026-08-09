"""
kinematics.py — 经典轨道与坐标变换

核心:
  1. 卢瑟福轨道: b ↔ θ, 最近接近距离 D(b)
  2. 擦边角与擦边角动量
  3. 三体 → 两体坐标变换 (⁷Li 质心 → α + ²³⁵Pa)
  4. 库仑后加速: 出口道双曲轨道

参考:
  - Broglia & Winther, "Heavy Ion Reactions", Ch.2-4
  - Bass, "Nuclear Reactions with Heavy Ions" (1980)
"""

import numpy as np
from typing import Tuple, NamedTuple
from . import config

_sys = config.system
_mod = config.model


# ============================================================
# 1. 卢瑟福散射轨道
# ============================================================

class RutherfordTrajectory(NamedTuple):
    """卢瑟福双曲轨道参数"""
    eta: float           # Sommerfeld 参数
    k: float             # 波数 (fm⁻¹)
    b: float             # 碰撞参数 (fm)
    d: float             # 最近接近距离 (fm)
    theta_cm: float      # 质心系散射角 (弧度)
    a_half: float        # 半长轴 = η/k
    eccentricity: float  # 离心率 e = √(1 + (b/a)²)


def rutherford_trajectory(e_cm: float, b: float,
                           z1: int, z2: int, mu: float) -> RutherfordTrajectory:
    """计算卢瑟福轨道参数

    卢瑟福散射角:
      θ = 2 arctan(η/(kb))

    最近接近距离:
      D = η/k + √((η/k)² + b²)

    Parameters
    ----------
    e_cm : 质心系能量 (MeV)
    b : 碰撞参数 (fm)
    z1, z2 : 核电荷数
    mu : 约化质量 (MeV/c²)

    Returns
    -------
    RutherfordTrajectory
    """
    eta = config.sommerfeld(z1, z2, mu, e_cm)
    k = config.wavenumber(mu, e_cm)
    a_half = eta / k                         # 卢瑟福轨道半长轴
    eccentricity = np.sqrt(1.0 + (b / a_half)**2)
    d = a_half * (1.0 + eccentricity)         # 最近接近距离

    # 散射角: θ = π - 2 arctan(b/a) = 2 arctan(a/b)
    theta_cm = 2.0 * np.arctan(a_half / max(b, 1e-6))

    return RutherfordTrajectory(
        eta=eta, k=k, b=b, d=d,
        theta_cm=theta_cm,
        a_half=a_half, eccentricity=eccentricity
    )


def impact_parameter_from_angle(theta_cm: float, e_cm: float,
                                  z1: int, z2: int, mu: float) -> float:
    """从质心系散射角反推碰撞参数

    b = η/k · cot(θ/2)
    """
    eta = config.sommerfeld(z1, z2, mu, e_cm)
    k = config.wavenumber(mu, e_cm)
    return eta / k / np.tan(theta_cm / 2.0)


# ============================================================
# 2. 擦边角与擦边角动量
# ============================================================

def grazing_angle(e_cm: float, r_int: float = 0,
                   z1: int = None, z2: int = None,
                   a1: int = None, a2: int = None,
                   mu: float = None,
                   r0: float = 1.25) -> Tuple[float, float, float]:
    """计算擦边角 (quarter-point angle)

    使用经典公式:
      θ_g = 2 arcsin[ Z₁Z₂e² / (2 E_cm R_int - Z₁Z₂e²) ]

    或等价的:
      θ_g = 2 arctan[ η / (k R_int) ]

    Parameters
    ----------
    e_cm : 质心系能量 (MeV)
    r_int : 强吸收半径 (fm), 若为0则自动计算
    z1, z2 : 核电荷数 (若None则用体系默认)
    a1, a2 : 质量数
    mu : 约化质量
    r0 : 半径参数

    Returns
    -------
    theta_g_cm : 擦边角 (弧度, 质心系)
    theta_g_lab : 擦边角 (弧度, 实验室系, 轻粒子)
    l_g : 擦边角动量 (ħ 单位)
    """
    if z1 is None:
        z1 = _sys.proj.Z
    if z2 is None:
        z2 = _sys.targ.Z
    if a1 is None:
        a1 = _sys.proj.A
    if a2 is None:
        a2 = _sys.targ.A
    if mu is None:
        mu = _sys.mu_proj_targ

    if r_int <= 0:
        r_int = config.interaction_radius(a1, a2, r0)

    eta = config.sommerfeld(z1, z2, mu, e_cm)
    k = config.wavenumber(mu, e_cm)

    # quarter-point 公式: θ_g = 2 arctan(η/(k R_int))
    theta_g_cm = 2.0 * np.arctan(eta / (k * r_int))

    # 擦边角动量
    b_g = impact_parameter_from_angle(theta_g_cm, e_cm, z1, z2, mu)
    l_g = k * b_g  # L_g = k b_g (半经典)

    # 实验室系擦边角 (轻粒子)
    # θ_lab ≈ θ_cm / 2 (当 m_proj << m_targ 时)
    gamma = np.arcsin(np.sin(theta_g_cm) *
                       _sys.proj.mass_MeV / (_sys.proj.mass_MeV + _sys.targ.mass_MeV))
    theta_g_lab = theta_g_cm - gamma

    return theta_g_cm, theta_g_lab, l_g


def grazing_angular_momentum(e_cm: float, r_int: float = 0,
                              z1: int = None, z2: int = None) -> float:
    """仅计算擦边角动量 L_g (ħ)

    基于 quarter-point 条件: θ_g = 2 arctan(η/(k R_int))
    由此 L_g = k b_g = k R_int (半经典)

    经典公式 (E_cm > V_CB): L_g = k R_int √(1 - V_CB/E_cm)
    当 E_cm ≤ V_CB 时, 经典 L_g = 0, 但 quarter-point 外推给出 k R_int > 0
    此处使用 quarter-point 外推以统一处理。
    """
    if z1 is None:
        z1 = _sys.proj.Z
    if z2 is None:
        z2 = _sys.targ.Z
    if r_int <= 0:
        r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A)

    k = config.wavenumber(_sys.mu_proj_targ, e_cm)
    # quarter-point 近似: b_g ≈ R_int, L_g = k * R_int
    l_g = k * r_int

    # 经典修正 (当 E_cm 远大于库仑势垒时):
    v_cb = z1 * z2 * config.E2 / r_int
    if e_cm > v_cb:
        l_g_classical = k * r_int * np.sqrt(1.0 - v_cb / e_cm)
        # 平滑过渡: 当 E > V_CB 时逐步从 quarter-point 过渡到经典公式
        # 此处直接用 quarter-point 值, 经典值作为参考
        # l_g = l_g_classical  # 如需严格经典公式, 取消此行注释

    return l_g


# ============================================================
# 3. 三体 → 两体坐标变换
# ============================================================

def breakup_kinematics(e_cm: float, b: float,
                        k_vec_cluster: np.ndarray,
                        cluster_label: str = "t") -> dict:
    """在转移瞬间的三体运动学

    初始态: ⁷Li (速度 v_cm) 接近 ²³²Th (静止在 CM)
    内部: α + t 以费米动量 k 相对运动

    Parameters
    ----------
    e_cm : 质心系能量 (MeV)
    b : 碰撞参数 (fm)
    k_vec_cluster : 费米动量矢量 (k_mag, theta, phi) — t 相对 α
    cluster_label : "t" 或 "α", 转移的团簇

    Returns
    -------
    kin : 包含所有运动学量的字典
    """
    # 入射道运动学
    traj = rutherford_trajectory(e_cm, b,
                                  _sys.proj.Z, _sys.targ.Z,
                                  _sys.mu_proj_targ)

    # 入射道运动学
    traj = rutherford_trajectory(e_cm, b,
                                  _sys.proj.Z, _sys.targ.Z,
                                  _sys.mu_proj_targ)

    # v_rel: ⁷Li-²³²Th 相对速度 (c 单位), 靶核静止时即 ⁷Li 的实验室系速度
    v_rel = config.HBARC * traj.k / _sys.mu_proj_targ
    v_li = v_rel

    # ⁷Li 内部: α-t 相对动量
    k_mag, k_theta, k_phi = k_vec_cluster

    # t 在 ⁷Li 内部的速度 (c 单位)
    v_t_in_li = config.HBARC * k_mag / _sys.cluster.mass_MeV

    # t 在 ⁷Li 内的速度分量 (各向同性方向)
    v_t_li_x = v_t_in_li * np.cos(k_theta)
    v_t_li_y = v_t_in_li * np.sin(k_theta) * np.cos(k_phi)
    v_t_li_z = v_t_in_li * np.sin(k_theta) * np.sin(k_phi)

    # t 在实验室系中的速度 (靶核静止, 用于计算 t+Th 相对动能)
    v_t_lab_x = v_li + v_t_li_x
    v_t_lab_y = v_t_li_y
    v_t_lab_z = v_t_li_z
    v_t_lab = np.sqrt(v_t_lab_x**2 + v_t_lab_y**2 + v_t_lab_z**2)

    # t 在总 CM 系中的速度
    v_t_cm_x = v_li + v_t_li_x
    v_t_cm_y = v_t_li_y
    v_t_cm_z = v_t_li_z
    v_t_cm = np.sqrt(v_t_cm_x**2 + v_t_cm_y**2 + v_t_cm_z**2)

    # t + ²³²Th 的相对动能 (靶核在实验室系静止)
    e_rel_t_th = 0.5 * _sys.mu_t_th * v_t_lab**2

    # 转移后: t 与 ²³²Th 结合 → ²³⁵Pa*
    # ²³⁵Pa 的激发能 = Q_capture + E_rel(t-Th)
    e_star_pa = _sys.q_capture + e_rel_t_th

    # α 在实验室系/总 CM 中的速度 (动量守恒: m_α v_α_in_li = -m_t v_t_in_li)
    v_alpha_lab_x = v_li - (_sys.cluster.mass_MeV / _sys.spectator.mass_MeV) * v_t_li_x
    v_alpha_lab_y = -(_sys.cluster.mass_MeV / _sys.spectator.mass_MeV) * v_t_li_y
    v_alpha_lab_z = -(_sys.cluster.mass_MeV / _sys.spectator.mass_MeV) * v_t_li_z
    v_alpha_lab = np.sqrt(v_alpha_lab_x**2 + v_alpha_lab_y**2 + v_alpha_lab_z**2)

    kin = {
        'trajectory': traj,
        'v_rel': v_rel,
        'v_li': v_li,
        'e_cm': e_cm,
        'b': b,
        # t 的运动学
        'v_t_lab': v_t_lab,
        'v_t_lab_vec': np.array([v_t_lab_x, v_t_lab_y, v_t_lab_z]),
        'v_t_cm': v_t_cm,
        'v_t_cm_vec': np.array([v_t_cm_x, v_t_cm_y, v_t_cm_z]),
        'e_rel_t_th': e_rel_t_th,
        'e_star_pa': e_star_pa,
        # α 的运动学
        'v_alpha_lab': v_alpha_lab,
        'v_alpha_lab_vec': np.array([v_alpha_lab_x, v_alpha_lab_y, v_alpha_lab_z]),
        # 出口道相对运动
        'v_rel_alpha_pa': v_alpha_lab,  # 近似 (²³⁵Pa ~ 静止在实验室系)
        'e_rel_alpha_pa': 0.5 * _sys.mu_alpha_pa * v_alpha_lab**2,
    }

    return kin


def t_th_relative_energy(e_cm: float, k_mag: float, k_theta: float) -> Tuple[float, float]:
    """t-²³²Th 准自由事件的相对动能与 ²³⁵Pa 激发能

    E_rel = ½ μ_tTh |v_rel + v_t|², 其中
      v_rel = √(2E_cm/μ_proj_targ) 是 ⁷Li-²³²Th 相对速度 (= 束流速度, 靶核静止)
      v_t   = ħ·k/m_t 是 t 在 ⁷Li 内的费米速度

    总质心系中 t 与 Th 的相对速度恰为 v_rel + v_t (Th 在总 CM 中反向运动,
    伽利略变换下相对速度不变), 故实验室系与质心系结果一致。

    Parameters
    ----------
    e_cm : 入射道质心系能量 (MeV)
    k_mag : 费米动量大小 (fm⁻¹)
    k_theta : 费米动量相对束流方向的角度 (弧度)

    Returns
    -------
    e_rel : t-Th 相对动能 (MeV)
    e_star : ²³⁵Pa 激发能 E* = Q_capture + E_rel (MeV)
    """
    v_rel = np.sqrt(2.0 * e_cm / _sys.mu_proj_targ)
    v_t = config.HBARC * k_mag / _sys.cluster.mass_MeV
    v_rel_sq = (v_rel + v_t * np.cos(k_theta))**2 + (v_t * np.sin(k_theta))**2
    e_rel = 0.5 * _sys.mu_t_th * v_rel_sq
    return e_rel, _sys.q_capture + e_rel


# ============================================================
# 4. 库仑后加速 (出口道)
# ============================================================

def coulomb_recoil(r0, phi_p, vx, vy, z1: int, z2: int, m: float):
    """α 旁观者的库仑排斥传播 (Pa 静止近似)

    破裂点 (近点): α 从距离 r0、方向角 phi_p (相对束流) 处以实验室速度
    (vx, vy) 出发, 在 ²³⁵Pa 库仑排斥场中运动到无穷远。解析求解排斥双曲
    轨道, 返回渐近出射角与无穷远动能 (库仑增益 Z_αZ_Pa·e²/r0 已计入)。

    近点处 ⁷Li 的速度方向是切向 (φ_p + π/2), 所以调用方应把初始速度
    建为: v_beam·t̂(φ_p) + 费米项, 而不是 v_beam·x̂。

    Parameters
    ----------
    r0, phi_p : 破裂距离 (fm) 与近点方向角 (rad, 相对束流)
    vx, vy : α 初始速度 (c 单位, 实验室系, x=束流方向)
    z1, z2 : α 与产物核电荷数
    m : α 质量 (MeV/c²)

    Returns
    -------
    theta_out : 渐近出射角 (rad, [0, π], 相对束流)
    e_out : 无穷远动能 (MeV)
    """
    r0 = np.asarray(r0, float)
    phi_p = np.asarray(phi_p, float)
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)

    C = z1 * z2 * config.E2
    # 速度分解到径向 (phi_p 方向) / 切向
    v_r = vx * np.cos(phi_p) + vy * np.sin(phi_p)
    v_t = -vx * np.sin(phi_p) + vy * np.cos(phi_p)
    E = 0.5 * m * (vx * vx + vy * vy) + C / np.maximum(r0, 1e-9)
    L = m * r0 * v_t

    theta_out = phi_p.copy()
    e_out = E

    L_abs = np.abs(L)
    ok = L_abs > 1e-9
    if not np.any(ok):
        return theta_out, e_out

    s = np.where(L > 0, 1.0, -1.0)
    L2 = L * L
    p = L2 / (m * C)
    eps = np.sqrt(1.0 + 2.0 * E * L2 / (m * C * C))
    cosA = np.clip((p / r0 + 1.0) / eps, -1.0, 1.0)
    A = np.arccos(cosA)
    # 近点角 φ_a 的符号由径向速度定: v_r>0 (向外) → φ_a<0
    phi_a = np.where(v_r > 0, -A, A)
    phi_a = np.where(v_r == 0, -A, phi_a)
    dphi = np.arccos(np.clip(1.0 / eps, -1.0, 1.0))

    th = phi_p + s * (phi_a + dphi)
    th = np.mod(th, 2.0 * np.pi)
    th = np.where(th > np.pi, 2.0 * np.pi - th, th)
    theta_out = np.where(ok, th, theta_out)
    return theta_out, e_out


def post_acceleration(r_transfer: float, e_initial: float,
                       z1: int = None, z2: int = None,
                       mu: float = None,
                       r_inf: float = 500.0) -> Tuple[float, float, float]:
    """库仑后加速计算 (出口道分解工具)

    在转移点 r = r_transfer 处, α 和 ²³⁵Pa 的初始相对动能
    为 E_initial。在无穷远处, 相对动能增加了库仑排斥能:

      E_final = E_initial + Z₁Z₂e² / r_transfer

    出射角由卢瑟福轨道决定:
      θ_out = 2 arctan(η_out / (k_out b_out))

    注意: 主计算路径中 α-²³⁵Pa 的渐近动能由两体能量守恒给出
    (T_rel(∞) = E_cm + Q_total − E*, 已含库仑后加速), 不需要调用本函数。
    本函数用于单独分解转移点处的局域动能与库仑增益。

    Parameters
    ----------
    r_transfer : 转移点距离 (fm)
    e_initial : 出口道初始相对动能 (MeV)
    z1, z2 : 核电荷数 (α 和 ²³⁵Pa)
    mu : 出口道约化质量
    r_inf : 无穷远近似距离 (fm)

    Returns
    -------
    e_final : 无穷远相对动能 (MeV)
    theta_out : 质心系出射角 (弧度)
    v_final : 末态相对速度 (c 单位)
    """
    if z1 is None:
        z1 = _sys.spectator.Z  # α: Z=2
    if z2 is None:
        z2 = _sys.product.Z    # ²³⁵Pa: Z=91
    if mu is None:
        mu = _sys.mu_alpha_pa

    # 能量守恒
    coulomb_energy = z1 * z2 * config.E2 / max(r_transfer, 1e-6)
    e_final = e_initial + coulomb_energy

    if e_final <= 0:
        return 0.0, 0.0, 0.0

    # 出口道卢瑟福轨道
    eta_out = config.sommerfeld(z1, z2, mu, e_final)
    k_out = config.wavenumber(mu, e_final)

    # 出射角由最近接近距离反推
    a_half_out = eta_out / k_out
    # 对于排斥库仑势, 无穷远的渐近偏转角
    # 出口道速度方向与转移时 α 的速度方向有关
    # 简化: 用 r_transfer 作为最近接近距离, 反推出射角
    if r_transfer > a_half_out:
        ecc = r_transfer / a_half_out - 1.0
        if ecc > 1e-6:
            theta_out = 2.0 * np.arctan(1.0 / np.sqrt(ecc**2 - 1.0))
        else:
            theta_out = np.pi
    else:
        theta_out = np.pi / 2.0

    v_final = config.HBARC * k_out / mu

    return e_final, theta_out, v_final


# ============================================================
# 5. 角度依赖 (实验室系 ↔ 质心系)
# ============================================================

def cm_to_lab(theta_cm: float, e_cm: float,
               m_ejectile: float, m_recoil: float,
               q_value: float = 0.0,
               m_proj: float = None, m_targ: float = None) -> Tuple[float, float]:
    """质心系角度 → 实验室系 (标准两体运动学)

    参数
    ----
    theta_cm : 质心系角度 (弧度)
    e_cm : 入射道质心系能量 (MeV)
    m_ejectile : 出射粒子质量 (MeV/c²)
    m_recoil : 反冲核质量 (MeV/c²)
    q_value : Q 值 (MeV)
    m_proj, m_targ : 入射弹核/靶核质量; 未提供时使用 _sys 默认值

    返回
    ----
    theta_lab : 实验室系角度 (弧度)
    e_lab : 实验室系动能 (MeV)
    """
    if m_proj is None:
        m_proj = _sys.proj.mass_MeV
    if m_targ is None:
        m_targ = _sys.targ.mass_MeV

    e_total = e_cm + q_value
    if e_total <= 0:
        return np.pi, 0.0

    # 质心系在实验室系中的速度 (靶核静止)
    M_in = m_proj + m_targ
    V = np.sqrt(2.0 * e_cm / M_in) * np.sqrt(m_proj / m_targ)

    # 出射粒子在质心系中的速度
    M_out = m_ejectile + m_recoil
    T_ej_cm = e_total * m_recoil / M_out
    u = np.sqrt(2.0 * T_ej_cm / m_ejectile)

    # 实验室系速度分量
    denom = np.cos(theta_cm) + V / u
    theta_lab = np.arctan2(np.sin(theta_cm), denom)
    if theta_lab < 0:
        theta_lab += np.pi

    e_lab = 0.5 * m_ejectile * (V**2 + u**2 +
                                2.0 * V * u * np.cos(theta_cm))

    return theta_lab, e_lab


def lab_to_cm(theta_lab: float, e_lab: float,
               m_ejectile: float, m_recoil: float,
               q_value: float = 0.0,
               m_proj: float = None, m_targ: float = None) -> Tuple[float, float]:
    """实验室系角度 → 质心系 (近似反变换)

    注: 严格反变换需要解非线性方程, 此处采用小质量比近似
    (m_ej << m_rec), 对 α + 重核产物足够精确。
    """
    if m_proj is None:
        m_proj = _sys.proj.mass_MeV
    if m_targ is None:
        m_targ = _sys.targ.mass_MeV

    M_in = m_proj + m_targ
    M_out = m_ejectile + m_recoil

    # 先粗略估计 e_cm (忽略 V/u 项)
    e_cm = e_lab * M_out / m_recoil - q_value
    e_cm = max(e_cm, 0.01)

    # 迭代一次使 lab 能量匹配
    for _ in range(10):
        V = np.sqrt(2.0 * e_cm / M_in) * np.sqrt(m_targ / m_proj)
        T_ej_cm = (e_cm + q_value) * m_recoil / M_out
        u = np.sqrt(2.0 * T_ej_cm / m_ejectile)
        e_lab_calc = 0.5 * m_ejectile * (
            V**2 + u**2 + 2.0 * V * u * np.cos(theta_lab))
        # 用差分校正 e_cm
        e_cm_new = e_cm * e_lab / max(e_lab_calc, 1e-6)
        if abs(e_cm_new - e_cm) < 1e-4:
            break
        e_cm = e_cm_new

    theta_cm = theta_lab
    return theta_cm, e_cm
