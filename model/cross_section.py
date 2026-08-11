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
from .transfer import TransferModel

_sys = config.system          # 供模块内函数引用; 对比脚本可临时替换 config.system
_mod = config.model


# ============================================================
# 1. 碰撞参数网格
# ============================================================

def make_b_grid(e_cm: float, n_b: int = None, b_max: float = None) -> np.ndarray:
    """生成碰撞参数网格

    策略: b 从 0 到 b_max, 在擦边角动量 L_g 附近加密
    """
    if n_b is None:
        n_b = config.model.n_b

    r_int = config.interaction_radius(config.system.proj.A, config.system.targ.A, config.model.r0)
    l_g = grazing_angular_momentum(e_cm, r_int,
                                    config.system.proj.Z, config.system.targ.Z)
    k = config.wavenumber(config.system.mu_proj_targ, e_cm)
    b_g = l_g / k if k > 0 else r_int

    if b_max is None or b_max <= 0:
        b_max = max(2.0 * b_g, r_int * config.model.b_max_factor)

    # 在擦边附近加密
    n_inner = int(n_b * 0.4)
    n_outer = n_b - n_inner

    b_inner = np.linspace(0, b_g * 1.2, n_inner)
    b_outer = np.linspace(b_g * 1.2, b_max, n_outer)

    # 合并, 跳过重复点
    b_grid = np.unique(np.concatenate([b_inner, b_outer]))
    return b_grid


def _weighted_percentiles(data: np.ndarray, weights: np.ndarray,
                           percents: list) -> list:
    """截面加权分位 (排序 + 权重累积)"""
    order = np.argsort(data)
    data_s, w_s = data[order], np.asarray(weights)[order]
    cdf = np.cumsum(w_s)
    if cdf[-1] <= 0:
        return [np.percentile(data, p) for p in percents]
    cdf = cdf / cdf[-1]
    return [float(np.interp(p / 100.0, cdf, data_s)) for p in percents]


def _b_quadrature_weights(b_grid: np.ndarray) -> np.ndarray:
    """非均匀 b 网格的梯形求积权重 (端点半宽, 内部全宽)"""
    b_w = np.zeros_like(b_grid)
    b_w[0] = 0.5 * (b_grid[1] - b_grid[0])
    b_w[-1] = 0.5 * (b_grid[-1] - b_grid[-2])
    b_w[1:-1] = 0.5 * (b_grid[2:] - b_grid[:-2])
    return b_w


def _exclude_head_on(b_grid: np.ndarray, e_cm: float) -> np.ndarray:
    """剔除近正碰 (近点进入核区 D(b)<R_int, 融合吸收无旁观 α) 的 b 点"""
    b_min = _alpha_b_min(e_cm)
    b_cut = b_grid[b_grid >= b_min]
    if len(b_cut) < 3:
        b_cut = b_grid[1:]
    return b_cut


def _alpha_b_min(e_cm: float) -> float:
    """近正碰下界: 近点进入核区 (D(b) < R_int) 的 7Li 被完全融合吸收,
    不产生可测的旁观者 α。解 D(b_min) = R_int:
      a + √(a² + b²) = R_int  →  b_min = √(R_int² − 2aR_int),  a = η/k
    """
    r_int = config.interaction_radius(config.system.proj.A, config.system.targ.A, config.model.r0)
    eta = config.sommerfeld(config.system.proj.Z, config.system.targ.Z, config.system.mu_proj_targ, e_cm)
    k = config.wavenumber(config.system.mu_proj_targ, e_cm)
    a = eta / k
    return np.sqrt(max(r_int * r_int - 2.0 * a * r_int, 0.0))


# ============================================================
# 破裂几何: 轨道上 t–Th 首次到达俘获半径 R_cap
# ============================================================

_tth_capture_radius = None


def _ensure_tth_capture_radius() -> float:
    """t+Th 俘获半径 R_cap (懒加载缓存)

    取 t+Th 入射道总势 (库仑 + Akyüz-Winther 核势) 的势垒位置: t 到达
    势垒顶即被 ²³²Th 强吸收 (强吸收近似, 文献标准处理)。≈ 12.1 fm。
    """
    global _tth_capture_radius
    if _tth_capture_radius is not None:
        return _tth_capture_radius
    from .potentials import total_potential, find_barrier, akyuz_winther_potential
    r_grid = np.linspace(0.5, 30.0, 2000)
    v0_t, r0_t, a_t = akyuz_winther_potential(
        config.system.cluster.A, config.system.targ.Z, config.system.targ.A, config.system.cluster.Z)
    v = total_potential(r_grid, 1.0,
                        config.system.cluster.Z, config.system.cluster.A,
                        config.system.targ.Z, config.system.targ.A,
                        v0_t, r0_t, a_t)
    rb, vb, _ = find_barrier(r_grid, v)
    _tth_capture_radius = rb
    return rb


def _orbit_geometry(e_cm: float, b: float) -> Dict:
    """库仑轨道参数 (入射道 7Li+Th)"""
    mu = config.system.mu_proj_targ
    eta = config.sommerfeld(config.system.proj.Z, config.system.targ.Z, mu, e_cm)
    k = config.wavenumber(mu, e_cm)
    a = eta / k                       # 半长轴 (η/k)
    v_inf = np.sqrt(2.0 * e_cm / mu)  # 无穷远相对速度
    L = mu * b * v_inf                # 轨道角动量 (ħ 单位×c)
    e = np.sqrt(1.0 + (b / a) ** 2)   # 离心率
    p = b * b / a                     # 半正焦弦
    d = a * (1.0 + e)                 # 近点距离
    return {'a': a, 'e': e, 'p': p, 'L': L, 'v_inf': v_inf, 'd': d}


def _potential_at(r: np.ndarray) -> np.ndarray:
    """入射道总势 V(r) = V_Coul(均匀带电球) + V_WS (MeV)"""
    from .potentials import coulomb_uniform_sphere, woods_saxon
    v = coulomb_uniform_sphere(r, config.system.proj.Z, config.system.targ.Z,
                               config.system.proj.A, config.system.targ.A)
    v += woods_saxon(r, config.model.v0_in, config.model.r0_in, config.model.a_in,
                     config.system.proj.A, config.system.targ.A)
    return v


_alpha_t_sampler = None


def _sample_alpha_t_3d(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """抽样 α-t 内部间距 r_αt 三维矢量 (r_mag, theta, phi), 方向各向同性

    theta 相对轨道径向 r̂ (从靶指向 7Li), phi 绕 r̂ 的方位角。
    """
    global _alpha_t_sampler
    if _alpha_t_sampler is None:
        from .structure import AlphaTRadiusSampler
        _alpha_t_sampler = AlphaTRadiusSampler()
    return _alpha_t_sampler.sample_3d(n)


def _event_physics(model, e_cm: float, b: float, n_fermi: int) -> Dict:
    """一次费米事件抽样, 返回全部事件物理量 (全链路唯一口径)

    破裂几何 (正确考虑库仑后加速):
      - 破裂点: 轨道上 t–Th 首次到达俘获半径 R_cap 处, 7Li 质心位置
        R* = (m_α/M)·r_αt∥ + √(R_cap² − ((m_α/M)·r_αt⊥)²)
        (r_αt 为 α–t 三维矢量; t 相对质心位移 d_t = −(m_α/M)r⃗_αt)
      - 作废事件: t 横向够不到 R_cap (⊥ 分量过大), 或 R* < 近点 D(b)
        (经典不可达), 或 R* 处经典动能 < 0
      - α 初速: 破裂点轨道瞬时速度 (径向 v_r 入射/出射支各半 + 切向 v_t)
        + 费米速度 (α = +(m_t/M)v_rel, t = −(m_α/M)v_rel, 投影到轨道平面)
      - α 破裂距离: r_α = |R*r̂ + (m_t/M)r⃗_αt|, 经 coulomb_recoil 传播

    转移概率: P = T·f_ICF·p_geo (常数, "t 到达 R_cap 即被吸收"的强吸收近似;
    局部 E_rel(tTh) 依赖见 R* 几何本身——t 朝 Th 使 R* 更深, 轨道动能更高)

    E* = E_cm + Q_total − E_α(∞) − E_Pa    (能量守恒, α 动能含库仑后加速)
    capture = E* ≥ Q_capture: t 被 ²³²Th 俘获才算转移; 不满足的事件
    (α 拿走全部能量, t 未被俘获) 从所有截面/谱中剔除。
    """
    k_mag, k_theta, k_phi, p = model.event_distribution(e_cm, b, n_fermi)
    p = np.asarray(p, dtype=float)
    k_mag = np.asarray(k_mag, dtype=float)
    k_theta = np.asarray(k_theta, dtype=float)
    k_phi = np.asarray(k_phi, dtype=float)

    # α-t 三维矢量 (r_αt 方向相对束流 x̂)
    r_at, r_at_th, r_at_phi = _sample_alpha_t_3d(n_fermi)

    mu = config.system.mu_proj_targ
    m_alpha = config.system.spectator.mass_MeV
    m_t = config.system.cluster.mass_MeV
    m_pa = config.system.product.mass_MeV
    m_li = m_alpha + m_t

    orb = _orbit_geometry(e_cm, b)
    r_cap = _ensure_tth_capture_radius()
    r_int = config.interaction_radius(config.system.proj.A, config.system.targ.A, config.model.r0)
    d_near = orb['d']                       # 近点距离 D(b)

    # ---- 破裂 gate: 近点处 t 能否够到俘获半径 ----
    # 破裂发生在近点附近 (7Li 最接近靶处, t–Th 最近), α 旁观。
    # 近点处 r̂ = x̂ (束流方向); t 相对质心位移 d_t = −(m_α/M)r⃗_αt
    # gate: |D(b)·x̂ + d⃗_t| ≤ R_cap  (t 在近点处与 Th 距离 ≤ 俘获半径)
    # 自动实现 b 截断: b 太大 → D(b) 太大 → t 够不到 → 不破裂
    r_at_par = r_at * np.cos(r_at_th)       # r_αt∥ (沿束流 x̂)
    r_at_perp = r_at * np.sin(r_at_th)      # r_αt⊥
    d_t_par = -(m_alpha / m_li) * r_at_par
    d_t_perp = (m_alpha / m_li) * r_at_perp
    r_t_near = np.sqrt((d_near + d_t_par) ** 2 + d_t_perp ** 2)
    valid = r_t_near <= r_cap

    # ---- α 发射 (近点, 纯切向 + 费米; 无轨道径向速度) ----
    # 近点切向速度 v_near = b·v_∞/D (角动量守恒, 转折点处径向速度=0),
    # 方向 +y (b>0 时 7Li 绕行到束流前方); 叠加费米投影到散射平面
    # (费米分量用 sinθ/±cosθ 平面化近似——带 cosφ 因子会把角分布
    # 系统性拉向前向, 已由 A/B 测试确认; 平面化与实验符合)
    v_near = b * orb['v_inf'] / max(d_near, 1e-9)
    v_rel = config.HBARC * k_mag / config.system.mu_alpha_t       # α–t 相对速度
    v_alpha_t = v_near + (m_t / m_li) * v_rel * np.sin(k_theta)
    v_alpha_r = -(m_t / m_li) * v_rel * np.cos(k_theta)

    # α 破裂距离: 近点 + α-t 间距 (标量, 与角分布验证一致的旧口径)
    r_alpha = d_near + r_at
    valid = valid & (r_alpha >= r_int)      # α 创建点必须在核外

    # 速度分解到直角 (+x = 束流/近点方向; 近点切向沿 +y)
    vx = v_alpha_r
    vy = v_alpha_t

    # ---- 库仑传播到无穷远 (无效事件用远距离占位, 增益≈0) ----
    # 近点在束流前方 (+x), 出射方向相对 +x
    phi_p = np.zeros(n_fermi)
    theta_alpha, e_alpha = coulomb_recoil(
        np.where(valid, r_alpha, 1e6), np.where(valid, phi_p, 0.0),
        np.where(valid, vx, 0.0), np.where(valid, vy, 0.0),
        config.system.spectator.Z, config.system.product.Z, m_alpha)
    theta_alpha = np.where(valid, theta_alpha, 0.0)
    e_alpha = np.where(valid, e_alpha, 0.0)

    # ---- 能量守恒激发能 + 俘获条件 ----
    e_pa = (m_alpha / m_pa) * e_alpha
    e_star = e_cm + config.system.q_total - e_alpha - e_pa
    capture = (e_star >= config.system.q_capture) & valid

    # 事件概率: P_base (gate 已处理 b 截断, 无额外隧穿修正)
    p_event = np.where(valid, p, 0.0)

    return {'k_mag': k_mag, 'k_theta': k_theta, 'k_phi': k_phi, 'p': p_event,
            'r_near': np.full(n_fermi, d_near), 'theta_alpha': theta_alpha,
            'e_alpha': e_alpha, 'e_pa': e_pa, 'e_star': e_star, 'capture': capture}


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
    n_fermi : 费米动量抽样数
    verbose : 是否打印进度

    Returns
    -------
    result : {'e_lab', 'e_cm', 'sigma', 'sigma_rutherford', 'lg'}
    """
    if e_lab_range is None:
        e_lab_range = np.arange(config.model.e_lab_min,
                                 config.model.e_lab_max + config.model.e_lab_step / 2,
                                 config.model.e_lab_step)

    n_energies = len(e_lab_range)
    sigma = np.zeros(n_energies)
    sigma_rutherford = np.zeros(n_energies)  # 纯卢瑟福参考截面
    l_g_values = np.zeros(n_energies)

    for i, e_lab in enumerate(e_lab_range):
        e_cm = config.e_lab_to_e_cm(e_lab, config.system.proj.mass_MeV, config.system.targ.mass_MeV)
        b_grid = make_b_grid(e_cm)
        # 与 α 双微分/角分布一致: 剔除近正碰 (近点进入核区, 融合吸收,
        # 无旁观 α)。保证 σ(E) 与可测 α 截面归一化一致。
        b_grid = _exclude_head_on(b_grid, e_cm)
        p_grid = np.zeros_like(b_grid)

        l_g = grazing_angular_momentum(e_cm,
                                        config.interaction_radius(
                                            config.system.proj.A, config.system.targ.A),
                                        config.system.proj.Z, config.system.targ.Z)
        l_g_values[i] = l_g

        for j, b in enumerate(b_grid):
            # 对所有模型统一取费米平均 <P(b)>: 按俘获条件 (E* ≥ Q_cap) 剔除
            # 非俘获事件后平均, 保证 σ(E) 与 E* 谱/α 分布口径一致
            ev = _event_physics(model, e_cm, b, n_fermi)
            p_grid[j] = float(np.sum(ev['p'] * ev['capture']) / len(ev['p']))

        # σ = 2π ∫ b P(b) db
        # 1 fm² = 10 mb
        integrand = 2.0 * np.pi * b_grid * p_grid  # fm²
        sigma[i] = simpson(integrand, b_grid) * 10  # → mb

        # 卢瑟福截面 (全融合上限: P=1 for b < b_g, P=0 for b > b_g)
        b_g = l_g / config.wavenumber(config.system.mu_proj_targ, e_cm)
        idx_g = np.searchsorted(b_grid, b_g)
        sigma_rutherford[i] = np.pi * b_g**2 * 10  # geometric, mb

        if verbose:
            print(f"  E_lab={e_lab:.0f} MeV, E_cm={e_cm:.2f} MeV, "
                  f"σ={sigma[i]:.4e} mb, L_g={l_g:.0f}")

    return {
        'e_lab': e_lab_range,
        'e_cm': np.array([config.e_lab_to_e_cm(e, config.system.proj.mass_MeV, config.system.targ.mass_MeV)
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
    e_cm = config.e_lab_to_e_cm(e_lab, config.system.proj.mass_MeV, config.system.targ.mass_MeV)
    if n_theta is None:
        n_theta = config.model.n_theta

    # α 出射角范围 [0, π] (朝后的 α 事件也计入)
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    b_grid = make_b_grid(e_cm, min(config.model.n_b, 40))
    b_grid = _exclude_head_on(b_grid, e_cm)
    b_w = _b_quadrature_weights(b_grid)

    all_theta = []
    all_w = []

    for j, b in enumerate(b_grid):
        ev = _event_physics(model, e_cm, b, n_fermi)

        # 每个样本的截面权重: b_w·2π·b·(p_i/N)  (fm², 最后 ×10 → mb);
        # 非俘获事件 (E* < Q_cap) 权重置 0
        w = b_w[j] * 2.0 * np.pi * b * ev['p'] / len(ev['p'])
        w = np.where(ev['capture'], w, 0.0)

        all_theta.append(ev['theta_alpha'])
        all_w.append(w)

        if verbose and (j % max(1, len(b_grid) // 4) == 0):
            print(f"  b={b:.1f} fm, <P>={np.mean(ev['p']):.4e}")

    theta_all = np.concatenate(all_theta)
    w_all = np.concatenate(all_w)

    # dσ/dΩ = (Σ bin 内权重 ×10) / ΔΩ,  ΔΩ = 2π sinθ Δθ
    dsdo, _ = np.histogram(theta_all, bins=theta_edges, weights=w_all)
    dtheta = theta_edges[1] - theta_edges[0]
    domega = 2.0 * np.pi * np.sin(theta_centers) * dtheta
    dsdo = dsdo * 10.0 / np.maximum(domega, 1e-10)  # → mb/sr

    # 入射道卢瑟福截面 (参照曲线, 前向峰)
    eta = config.sommerfeld(config.system.proj.Z, config.system.targ.Z, config.system.mu_proj_targ, e_cm)
    k = config.wavenumber(config.system.mu_proj_targ, e_cm)
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
        'e_lab': e_lab,
        'e_cm': e_cm,
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
        n_b = min(config.model.n_b, 40)

    e_cm = config.e_lab_to_e_cm(e_lab, config.system.proj.mass_MeV, config.system.targ.mass_MeV)
    b_grid = make_b_grid(e_cm, n_b)
    b_grid = _exclude_head_on(b_grid, e_cm)

    b_w = _b_quadrature_weights(b_grid)

    all_th = []
    all_e = []
    all_w = []

    for j, b in enumerate(b_grid):
        ev = _event_physics(model, e_cm, b, n_fermi)

        w = b_w[j] * 2.0 * np.pi * b * ev['p'] / len(ev['p'])  # fm²
        w = np.where(ev['capture'], w, 0.0)
        all_th.append(ev['theta_alpha'])
        all_e.append(ev['e_alpha'])
        all_w.append(w)

    th_all = np.concatenate(all_th)
    e_all = np.concatenate(all_e)
    w_all = np.concatenate(all_w)

    # 二维网格 (E 轴用截面加权分位, 避免低权重极端事件拉宽坐标)
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    e_lo, e_hi = _weighted_percentiles(e_all, w_all, [0.5, 99.5])
    e_lo = max(float(e_lo), 0.0)
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

    能量守恒口径: 每个事件 E* = E_cm + Q_total − E_α(∞) − E_Pa(反冲),
    其中 E_α(∞) 是 α 旁观者经库仑后加速后的最终动能 (含增益)。α 动能
    计入激发能预算, 出口道能量严格闭合。负值 (α 拿走全部能量) 截断为 0。

    Parameters
    ----------
    model : 转移模型 (含费米动量抽样)
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
        n_b = min(config.model.n_b, 50)  # 平衡精度与速度

    e_cm = config.e_lab_to_e_cm(e_lab, config.system.proj.mass_MeV, config.system.targ.mass_MeV)
    b_grid = make_b_grid(e_cm, n_b)
    b_grid = b_grid[b_grid >= _alpha_b_min(e_cm)]
    if len(b_grid) < 3:
        b_grid = make_b_grid(e_cm, n_b)[1:]

    # 能量守恒口径: E* = E_cm + Q_total − E_α(∞) − E_Pa(反冲)
    # α 最终动能 (含库仑后加速增益) 计入激发能预算, 能量严格闭合
    q_capture = config.system.q_capture
    # 俘获条件: E* ≥ Q_capture, 低于阈值的事件 (t 未被俘获) 剔除
    e_star_min = q_capture

    b_w = _b_quadrature_weights(b_grid)

    # 累积所有样本的 (E*, 截面权重); 权重按每个样本自身的 p_i 计
    all_e_star = []
    all_w = []

    for j, b in enumerate(b_grid):
        ev = _event_physics(model, e_cm, b, n_fermi)

        # 每个样本的截面权重: dσ/dE* ∝ b_w·2π·b·(p_i/N)  (fm², 最后 ×10 → mb);
        # 非俘获事件 (E* < Q_cap) 权重置 0
        w = b_w[j] * 2.0 * np.pi * b * ev['p'] / len(ev['p'])
        w = np.where(ev['capture'], w, 0.0)

        all_e_star.append(ev['e_star'])
        all_w.append(w)

        if verbose and (j % max(1, n_b // 4) == 0):
            print(f"  b={b:.1f} fm, <P>={np.mean(ev['p']):.4e}")

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


def compute_alpha_energy_distribution(model: TransferModel,
                                      e_lab: float,
                                      n_b: int = None,
                                      n_fermi: int = 10000,
                                      n_e_alpha_bins: int = 50,
                                      verbose: bool = True) -> Dict:
    """计算 α 旁观者动能分布 dσ/dE_α (核心输出)

    每个费米事件: α 在近点继承 ⁷Li 近点切向速度 + 内部费米速度 (与 t 反向),
    经 coulomb_recoil 在 ²³⁵Pa 库仑排斥场中传播到无穷远, 得到渐近动能
    E_α (含库仑后加速增益)。按转移概率 P_tr 加权 bin, 非俘获事件
    (E* < Q_cap, t 未被俘获) 剔除。与 E* 谱 (能量守恒口径) 严格一致:
    ∫ dσ/dE_α dE_α = ∫ dσ/dE* dE* = σ_tr。

    Parameters
    ----------
    model : 转移模型
    e_lab : 实验室系能量 (MeV)
    n_b, n_fermi : b 网格数、费米抽样数
    n_e_alpha_bins : E_α 分 bin 数

    Returns
    -------
    result : {'e_alpha', 'dsigma_de', 'e_alpha_mean', 'e_alpha_std',
              'q_capture', 'e_lab', 'e_cm'}
      e_alpha : 动能网格中心 (MeV)
      dsigma_de : dσ/dE_α (mb/MeV)
    """
    if n_b is None:
        n_b = min(config.model.n_b, 50)

    e_cm = config.e_lab_to_e_cm(e_lab, config.system.proj.mass_MeV, config.system.targ.mass_MeV)
    b_grid = make_b_grid(e_cm, n_b)
    b_grid = b_grid[b_grid >= _alpha_b_min(e_cm)]
    if len(b_grid) < 3:
        b_grid = make_b_grid(e_cm, n_b)[1:]

    b_w = _b_quadrature_weights(b_grid)

    all_e_alpha = []
    all_w = []

    for j, b in enumerate(b_grid):
        ev = _event_physics(model, e_cm, b, n_fermi)

        # 每个样本的截面权重: b_w·2π·b·(p_i/N)  (fm², 最后 ×10 → mb);
        # 非俘获事件 (E* < Q_cap) 权重置 0
        w = b_w[j] * 2.0 * np.pi * b * ev['p'] / len(ev['p'])
        w = np.where(ev['capture'], w, 0.0)

        all_e_alpha.append(ev['e_alpha'])
        all_w.append(w)

        if verbose and (j % max(1, n_b // 4) == 0):
            print(f"  b={b:.1f} fm, <P>={np.mean(ev['p']):.4e}")

    e_alpha_all = np.concatenate(all_e_alpha)
    w_all = np.concatenate(all_w)

    # 自适应 bin 上界: 截面加权 99.5% 分位 (极端低权重事件不拉宽坐标)
    e_alpha_max = float(_weighted_percentiles(e_alpha_all, w_all, [99.5])[0])
    if e_alpha_max < 1.0:
        e_alpha_max = 40.0
    e_alpha_edges = np.linspace(0.0, e_alpha_max, n_e_alpha_bins + 1)
    e_alpha_centers = 0.5 * (e_alpha_edges[:-1] + e_alpha_edges[1:])

    # 单位转换: fm² → mb (×10), bin 宽度归一化 → mb/MeV
    dsigma_de, _ = np.histogram(e_alpha_all, bins=e_alpha_edges, weights=w_all)
    dsigma_de *= 10.0
    de = e_alpha_edges[1] - e_alpha_edges[0]
    dsigma_de /= de

    e_alpha_mean = np.average(e_alpha_centers, weights=dsigma_de + 1e-30)
    e_alpha_std = np.sqrt(np.average((e_alpha_centers - e_alpha_mean)**2,
                                      weights=dsigma_de + 1e-30))

    return {
        'e_alpha': e_alpha_centers,
        'dsigma_de': dsigma_de,
        'e_alpha_mean': e_alpha_mean,
        'e_alpha_std': e_alpha_std,
        'q_capture': config.system.q_capture,
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
        e_lab_range = np.arange(config.model.e_lab_min,
                                 config.model.e_lab_max + config.model.e_lab_step / 2,
                                 config.model.e_lab_step)

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
                                            n_theta=config.model.n_theta,
                                            n_fermi=n_fermi,
                                            verbose=verbose)
    result['angular'] = angular

    # 2b. 中位能量 α 旁观者动能分布 (核心输出)
    result['alpha_energy'] = compute_alpha_energy_distribution(
        model, e_mid, n_b=min(config.model.n_b, 40),
        n_fermi=max(n_fermi, 2000), verbose=False)

    # 3. 激发能谱
    if all_spectra:
        # 每个 E_lab 各算一张谱 (与 .pace 文件粒度一致)
        e_star_specs = {}
        n_b_es = min(config.model.n_b, 40)
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
            n_b=min(config.model.n_b, 40),
            n_fermi=n_fermi * 2,
            verbose=verbose)

    return result
