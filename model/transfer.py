"""
transfer.py — 转移振幅与转移概率

核心方法:
  1. 零程 DWBA (半经典近似): 扭曲波 → 经典轨道 + 费米动量积分
  2. 指数隧穿模型: P_tr ∼ exp(-2κD)
  3. Q 值窗口匹配
  4. 角动量匹配

转移截面:
  dσ/dΩ = (dσ/dΩ)_Ruth × P_tr(θ)
  σ = 2π ∫ b db P_tr(b)

参考:
  - Brink, Phys. Lett. B 40, 37 (1972)
  - Broglia & Winther, "Heavy Ion Reactions", Ch. 7
  - Satchler, "Direct Nuclear Reactions", Ch. 11
"""

import numpy as np
from typing import Callable, Optional, Tuple
from . import config
from .kinematics import grazing_angular_momentum
from .kinematics import RutherfordTrajectory, rutherford_trajectory
from .kinematics import t_th_relative_energy
from .structure import FermiMomentumSampler

_sys = config.system
_mod = config.model


# ============================================================
# 1. 转移概率模型基类
# ============================================================

class TransferModel:
    """转移概率模型的抽象基类"""

    def __init__(self, name: str = "base"):
        self.name = name
        self._fermi_sampler: Optional[FermiMomentumSampler] = None

    @property
    def fermi_sampler(self) -> FermiMomentumSampler:
        if self._fermi_sampler is None:
            self._fermi_sampler = FermiMomentumSampler(use_numerov=False)
        return self._fermi_sampler

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """转移概率 P(b) 在给定碰撞参数下"""
        raise NotImplementedError

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        """转移概率作为散射角的函数"""
        raise NotImplementedError

    def average_over_fermi(self, e_cm: float, b: float, n_samples: int = 1000) -> float:
        """对费米动量取平均的转移概率

        ⟨P⟩ = ∫ d³k P(k) × P_tr(b, k)
        """
        k_mag, k_theta, k_phi = self.fermi_sampler.sample(n_samples)
        p_total = 0.0

        for i in range(n_samples):
            k_vec = (k_mag[i], k_theta[i], k_phi[i])
            p_total += self.probability(e_cm, b, k_vec=k_vec)

        return p_total / n_samples

    def event_distribution(self, e_cm: float, b: float, n_samples: int):
        """抽样费米事件, 返回每个事件的 (k_mag, k_theta, p_event, e_star) 数组

        p_event 是单个费米事件的转移概率: 对于 P_tr 依赖 k 的模型 (qwindow/fermi)
        它随 k 变化; 对于 P_tr 与 k 无关的模型 (tunneling/dwba/icf) 恒为 P(b)。
        e_star = Q_capture + E_rel(t-Th) 是准自由事件激发能。

        icf/fermi 覆写此方法以复用向量化 return_details 路径 (避免逐事件循环)。
        """
        k_mag, k_theta, k_phi = self.fermi_sampler.sample(n_samples)
        p = np.empty(n_samples)
        es = np.empty(n_samples)
        for i in range(n_samples):
            p[i] = self.probability(e_cm, b, k_vec=(k_mag[i], k_theta[i], k_phi[i]))
            es[i] = t_th_relative_energy(e_cm, k_mag[i], k_theta[i])[1]
        return k_mag, k_theta, p, es


# ============================================================
# 2. 指数隧穿模型 (最简单)
# ============================================================

class TunnelingModel(TransferModel):
    """基于指数隧穿的转移概率

    P_tr(D) = P₀ · exp(-2κ D)

    其中 κ = √(2μ_BE)/ħ 是衰减常数
    D = 最近接近距离

    适合: 低于库仑势垒的转移反应
    """

    def __init__(self, kappa: float = None, p0: float = 1.0):
        super().__init__("Tunneling")
        if kappa is None:
            # κ 由弹核内 α-t 团簇的分离能决定 (S_αt = 2.468 MeV)
            be = abs(config.system.q_breakup)
            self.kappa = np.sqrt(2.0 * config.system.mu_alpha_t * be) / config.HBARC
        else:
            self.kappa = kappa
        self.p0 = p0

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """P(D) = P₀ exp(-2κ D)"""
        d = config.distance_of_closest_approach(
            config.system.proj.Z, config.system.targ.Z, config.system.mu_proj_targ, e_cm, b
        )
        return self.p0 * np.exp(-2.0 * self.kappa * d)

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        """角度依赖的转移概率"""
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         config.system.proj.Z, config.system.targ.Z,
                                         config.system.mu_proj_targ)
        return self.probability(e_cm, b)


# ============================================================
# 3. Q 值窗口修正隧穿模型
# ============================================================

class QWindowTunnelingModel(TransferModel):
    """含 Q 值窗口的隧穿模型

    P_tr(D) = P₀ exp(-2κD) × exp(-(Q_eff - Q_opt)²/(2Γ_Q²))

    其中:
      Q_eff = Q_total + E*_internal (含费米运动贡献)
      Q_opt = 最优 Q 值 (来自运动学匹配)
      Γ_Q = Q 窗宽度

    最优 Q 值条件: 在转移点, 入射道与出射道的相对动量匹配
      k_i ≈ k_f  →  Q_opt ≈ (Z₁Z₂ - Z₃Z₄)e²/D
    """

    def __init__(self, kappa: float = None, p0: float = 1.0,
                 gamma_q: float = None):
        super().__init__("Q-window Tunneling")
        if kappa is None:
            # κ 由弹核内 α-t 团簇的分离能决定 (S_αt = 2.468 MeV)
            be = abs(config.system.q_breakup)
            self.kappa = np.sqrt(2.0 * config.system.mu_alpha_t * be) / config.HBARC
        else:
            self.kappa = kappa

        if gamma_q is None:
            # Q 窗宽度 ∼ 2-5 MeV (经验)
            self.gamma_q = 3.0
        else:
            self.gamma_q = gamma_q
        self.p0 = p0

    def q_opt(self, e_cm: float, d: float) -> float:
        """最优 Q 值 (半经典运动学匹配)

        Q_opt = (Z₃Z₄/Z₁Z₂ - 1) E_cm

        其中 1,2 是入射道, 3,4 是出口道
        """
        z_proj = config.system.proj.Z    # Z₁ = 3 (⁷Li)
        z_targ = config.system.targ.Z    # Z₂ = 90 (²³²Th)
        z_spec = config.system.spectator.Z  # Z₃ = 2 (α)
        z_prod = config.system.product.Z    # Z₄ = 91 (²³⁵Pa)
        # 在擦边处
        ratio = (z_spec * z_prod) / (z_proj * z_targ)
        return (ratio - 1.0) * e_cm

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """含 Q 窗的概率"""
        d = config.distance_of_closest_approach(
            config.system.proj.Z, config.system.targ.Z, config.system.mu_proj_targ, e_cm, b
        )

        # 隧穿因子
        p_tunnel = np.exp(-2.0 * self.kappa * d)

        # 最优激发能 (库仑匹配): E*_opt = Q_total − Q_opt
        q_opt = self.q_opt(e_cm, d)
        e_star_opt = config.system.q_total - q_opt

        # 事件激发能: 由准自由 t-Th 相对动能给出 E* = Q_capture + E_rel
        k_vec = kwargs.get('k_vec')
        if k_vec is not None:
            e_star = t_th_relative_energy(e_cm, k_vec[0], k_vec[1])[1]
        else:
            e_star = config.system.q_capture

        p_q = np.exp(-(e_star - e_star_opt)**2 / (2.0 * self.gamma_q**2))

        return self.p0 * p_tunnel * p_q

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         config.system.proj.Z, config.system.targ.Z,
                                         config.system.mu_proj_targ)
        return self.probability(e_cm, b, **kwargs)


# ============================================================
# 4. 半经典 DWBA-lite 转移模型
# ============================================================

class SemiclassicalTransferModel(TransferModel):
    """半经典 DWBA 转移模型

    转移振幅:
      a_fi(b) = (1/iħ) ∫₋∞^∞ dt F(R(t)) exp(i ω t + i γ(t))

    其中:
      ω = ΔE/ħ = (E_f - E_i)/ħ  (Q 值效应)
      γ(t) = 库仑相因子
      F(R) = 转移形状因子

    转移概率:
      P(b) = |a_fi(b)|²

    零程近似:
      F(R) = D₀ · φ_αt(R)

    实现: 使用定态相位近似 (SPA)
      P(b) = P₀ D₀² |φ_αt(D)|² × (2πħ²)/(ħv|κ_i - κ_f|)
    """

    def __init__(self, d0: float = None):
        super().__init__("Semiclassical DWBA")
        if d0 is None:
            self.d0 = config.model.d0_manual
        else:
            self.d0 = d0

    def form_factor(self, r: float) -> float:
        """零程形状因子 F(R) = D₀ φ_αt(R)

        使用 Yukawa 波函数近似:
          φ_αt(r) ≈ √(2κ/4π) · exp(-κr)/r
          κ = √(2μ_αt·BE)/ħ
        """
        be = abs(config.system.q_breakup)
        kappa = np.sqrt(2.0 * config.system.mu_alpha_t * be) / config.HBARC
        norm = np.sqrt(2.0 * kappa / (4.0 * np.pi))
        phi = norm * np.exp(-kappa * max(r, 1e-6)) / max(r, 1e-6)
        return self.d0 * phi

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """半经典转移概率 (定态相位近似)

        P(b) ≈ (2π/ħ) · D₀² |φ_αt(D)|² / [v · |d/dR(κ_i(R) - κ_f(R))|_D]
        """
        d = config.distance_of_closest_approach(
            config.system.proj.Z, config.system.targ.Z, config.system.mu_proj_targ, e_cm, b
        )

        # 形状因子在最近接近距离处
        form = self.form_factor(d)
        form_sq = form**2

        # 定态相位点处的速度
        v_rel = np.sqrt(2.0 * e_cm / config.system.mu_proj_targ)

        # 入射道 / 出口道局域动量差
        # κ_i² = 2μ_i (E - V_i(r))/ħ²
        kappa_i = self._local_wavenumber(e_cm, d, 'in')
        kappa_f = self._local_wavenumber(e_cm + config.system.q_total, d, 'out')

        # d/dR (κ_i - κ_f) 在 D 处
        dr = 0.1  # fm
        d_plus = d + dr
        dk_i_plus = self._local_wavenumber(e_cm, d_plus, 'in')
        dk_f_plus = self._local_wavenumber(e_cm + config.system.q_total, d_plus, 'out')
        dkappa_dr = (dk_i_plus - dk_f_plus - kappa_i + kappa_f) / dr

        if abs(dkappa_dr) < 1e-10:
            dkappa_dr = 1e-10

        # 概率
        prefactor = 2.0 * np.pi / config.HBARC
        prob = prefactor * form_sq / (v_rel * abs(dkappa_dr))

        # 限制在 [0, 1]
        return min(max(prob, 0.0), 1.0)

    def _local_wavenumber(self, e: float, r: float, channel: str = 'in') -> float:
        """局域波数 κ(r) = √(2μ(E - V(r)))/ħ

        如果 E - V(r) < 0 → 虚构波数 (隧穿区)
        """
        if channel == 'in':
            z1, z2 = config.system.proj.Z, config.system.targ.Z
            a1, a2 = config.system.proj.A, config.system.targ.A
            mu = config.system.mu_proj_targ
            # 核势 ~0 在擦边距离 (库仑主导)
        else:  # 'out'
            z1, z2 = config.system.spectator.Z, config.system.product.Z
            a1, a2 = config.system.spectator.A, config.system.product.A
            mu = config.system.mu_alpha_pa

        v_coul = z1 * z2 * config.E2 / max(r, 1e-6)
        diff = e - v_coul

        if diff >= 0:
            return np.sqrt(2.0 * mu * diff) / config.HBARC
        else:
            # 经典禁止区: 返回虚波数
            return -np.sqrt(2.0 * mu * abs(diff)) / config.HBARC

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         config.system.proj.Z, config.system.targ.Z,
                                         config.system.mu_proj_targ)
        return self.probability(e_cm, b, **kwargs)


# ============================================================
# 5. 包含费米动量积分的完整隧穿模型
# ============================================================

class FermiIntegratedModel(TransferModel):
    """完整模型: 费米动量抽样 + t-Th 俘获 + 几何截断

    物理图像:
      - 7Li 到达核表面 (入射道 Hill-Wheeler × 擦边几何截断)
      - 7Li 破裂, t 以相对动能 E_rel(t-Th) 接近 ²³²Th
      - t 被 Th 俘获的概率 = t-Th 势垒的 Hill-Wheeler 穿透 P_capture(E_rel)
      - E_rel 由费米动量分布抽样决定 → 激发能 E* = Q_capture + E_rel 展宽

    与 exp(−2κD) 隧穿模型不同: 绝对标度由 t-Th 俘获物理决定, 而非 α-t 束缚
    尾在重离子表面的指数抑制 (后者在擦边处给出 ~1e-7, 使绝对截面失效)。
    """

    def __init__(self, kappa: float = None, gamma_q: float = None,
                 use_numerov_wf: bool = False, f_icf: float = 0.25):
        super().__init__("Fermi-Integrated")
        if kappa is None:
            be = abs(config.system.q_breakup)
            self.kappa = np.sqrt(2.0 * config.system.mu_alpha_t * be) / config.HBARC
        else:
            self.kappa = kappa

        self.gamma_q = gamma_q if gamma_q is not None else 3.0
        self.f_icf = f_icf
        self.delta_b = None
        self._use_numerov = use_numerov_wf
        # 势垒缓存 (7Li+Th 入射道, t+Th 俘获道)
        self._rb = None
        self._vb = None
        self._hbar_omega = None
        self._rb_tth = None
        self._vb_tth = None
        self._hw_tth = None

    def _ensure_barriers(self):
        """计算 7Li+Th 与 t+Th 两个势垒 (缓存)"""
        if self._rb is not None:
            return
        from .potentials import total_potential, find_barrier, akyuz_winther_potential
        r_grid = np.linspace(0.5, 30.0, 2000)

        # 7Li+Th 入射道势垒
        v_tot = total_potential(r_grid, 1.0,
                                 config.system.proj.Z, config.system.proj.A,
                                 config.system.targ.Z, config.system.targ.A,
                                 config.model.v0_in, config.model.r0_in, config.model.a_in)
        rb, vb, curv = find_barrier(r_grid, v_tot)
        self._rb = rb
        self._vb = vb
        self._hbar_omega = config.HBARC * np.sqrt(max(abs(curv), 1e-6) / config.system.mu_proj_targ)

        # t+Th 俘获道势垒 (Akyüz-Winther 估算 t-Th 核势)
        v0_t, r0_t, a_t = akyuz_winther_potential(
            config.system.cluster.A, config.system.targ.Z, config.system.targ.A, config.system.cluster.Z)
        v_tth = total_potential(r_grid, 1.0,
                                 config.system.cluster.Z, config.system.cluster.A,
                                 config.system.targ.Z, config.system.targ.A,
                                 v0_t, r0_t, a_t)
        rb_t, vb_t, curv_t = find_barrier(r_grid, v_tth)
        self._rb_tth = rb_t
        self._vb_tth = vb_t
        self._hw_tth = config.HBARC * np.sqrt(max(abs(curv_t), 1e-6) / config.system.mu_t_th)

    def probability(self, e_cm: float, b: float,
                     n_fermi_samples: int = 5000,
                     return_details: bool = False):
        """完整转移概率 (含费米动量积分 + Q 窗口 + 角度依赖)

        算法:
          1. 计算卢瑟福轨道 → D(b)
          2. 蒙特卡洛抽样费米动量 k 矢量
          3. 对每个 k:
             a. 计算转移到 ²³²Th 后 t 的相对动能
             b. 计算 Q_eff = Q_total + E_fermi
             c. 计算隧穿概率 P_tunnel(D, k)
             d. 计算 Q 窗口因子
          4. 平均 → P(b)

        Parameters
        ----------
        e_cm : 质心系能量
        b : 碰撞参数
        n_fermi_samples : 费米动量抽样数
        return_details : 是否返回运动学细节

        Returns
        -------
        p_avg : 平均转移概率
        (可选) details : 包含 D, k 分布, E* 分布, 俘获概率等
        """
        self._ensure_barriers()

        # 入射道几何截断 (与 ICF 相同口径: 融合 b_g)
        if e_cm > self._vb:
            b_g = self._rb * np.sqrt(1.0 - self._vb / e_cm)
        else:
            b_g = 0.0
        b_g = max(b_g, 0.01)
        if self.delta_b is None:
            self.delta_b = config.model.a0 * 0.8
        p_geo = 1.0 / (1.0 + np.exp((b - b_g) / self.delta_b))

        # 入射道势垒穿透 (7Li+Th)
        t_entrance = 1.0 / (1.0 + np.exp(2.0 * np.pi * (self._vb - e_cm) / self._hbar_omega))

        p_base = self.f_icf * t_entrance * p_geo

        # n_fermi_samples <= 0: 仅几何概率 (供分波形状等快速调用)
        if n_fermi_samples <= 0:
            if return_details:
                return p_base, {'d': b_g, 'probabilities': np.array([p_base]),
                                'e_star': np.array([config.system.q_capture]),
                                'q_eff': np.array([config.system.q_capture]), 'q_opt': 0.0}
            return p_base

        # 费米动量抽样 → 每事件 t-Th 俘获概率
        k_mag, k_theta, k_phi = self.fermi_sampler.sample(n_fermi_samples)

        probabilities = np.zeros(n_fermi_samples)
        e_star_values = np.zeros(n_fermi_samples)
        q_eff_values = np.zeros(n_fermi_samples)

        for i in range(n_fermi_samples):
            e_rel, e_star = t_th_relative_energy(e_cm, k_mag[i], k_theta[i])
            e_star_values[i] = max(e_star, 0.0)
            q_eff_values[i] = e_star

            # t-Th 俘获: E_rel 高于 t-Th 势垒 → 俘获概率趋近 1
            p_cap = 1.0 / (1.0 + np.exp(2.0 * np.pi *
                                         (self._vb_tth - e_rel) / self._hw_tth))
            probabilities[i] = p_base * p_cap

        p_avg = np.mean(probabilities)

        if return_details:
            return p_avg, {
                'd': b_g,
                'k_mag': k_mag,
                'k_theta': k_theta,
                'k_phi': k_phi,
                'probabilities': probabilities,
                'e_star': e_star_values,
                'q_eff': q_eff_values,
                'q_opt': 0.0,
                'e_star_opt': self._vb_tth,
                'vb_tth': self._vb_tth,
            }
        return p_avg

    def event_distribution(self, e_cm: float, b: float, n_samples: int):
        """复用向量化费米积分路径 (概率已含 t-Th 俘获加权)"""
        _, details = self.probability(e_cm, b, n_fermi_samples=n_samples,
                                      return_details=True)
        return (np.asarray(details['k_mag']), np.asarray(details['k_theta']),
                np.asarray(details['probabilities']), np.asarray(details['e_star']))

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         config.system.proj.Z, config.system.targ.Z,
                                         config.system.mu_proj_targ)
        return self.probability(e_cm, b, **kwargs)


# ============================================================
# 6. ICF 占比校准模型 (推荐用于垒上能区)
# ============================================================

class ICFFractionModel(TransferModel):
    """基于 ICF 占比的半经典转移模型

    物理:
      P(b, E) = T(E) × f_ICF / [1 + exp((b − b_g(E))/Δb)]

    其中:
      - T(E) = 1/(1 + exp(2π(Vb−E_cm)/(ħω)))  Hill-Wheeler 势垒穿透
      - b_g(E) = Rb·√(1−Vb/E) for E > Vb (经典角动量截断)
      - b_g(E) = 0             for E ≤ Vb (无经典越垒分波)
      - f_ICF = 0.25 (Lei & Moro 2019)

    垒下: T(E) 主导指数衰减, b_g=0, 单势垒量子隧穿
    垒上: b_g 根据离心势截断增长, T(E)→1, 经典几何截面

    势垒参数 (Rb, Vb, ħω) 从 WS+Coulomb 数值有效势求取, 非经验值。
    """

    def __init__(self, f_icf: float = 0.25,
                 delta_b: float = None,
                 use_numerov_wf: bool = False):
        super().__init__("ICF-Fraction")
        self.f_icf = f_icf
        self.delta_b = delta_b  # None -> 自动
        self._use_numerov = use_numerov_wf
        # 缓存势垒参数 (只算一次)
        self._rb: float | None = None
        self._vb: float | None = None
        self._hbar_omega: float | None = None

    def _ensure_barrier(self):
        """计算 WS+Coulomb 势垒参数 (只算一次, 缓存)"""
        if self._rb is not None:
            return
        from .potentials import total_potential, find_barrier
        r_grid = np.linspace(0.5, 30.0, 2000)
        v_tot = total_potential(r_grid, 1.0,
                                 config.system.proj.Z, config.system.proj.A,
                                 config.system.targ.Z, config.system.targ.A,
                                 config.model.v0_in, config.model.r0_in, config.model.a_in)
        rb, vb, curv = find_barrier(r_grid, v_tot)
        self._rb = rb
        self._vb = vb
        self._hbar_omega = config.HBARC * np.sqrt(max(abs(curv), 1e-6) / config.system.mu_proj_targ)

    def probability(self, e_cm: float, b: float,
                     n_fermi_samples: int = 2000,
                     return_details: bool = False):
        """ICF 转移概率

        P(b,E) = T(E) × f_ICF / [1 + exp((b − b_g)/Δb)]

        n_fermi_samples=0 时只返回几何概率, 不做费米动量积分。
        """
        k = config.wavenumber(config.system.mu_proj_targ, e_cm)

        self._ensure_barrier()
        rb = self._rb
        vb = self._vb
        hbar_omega = self._hbar_omega

        if e_cm <= 0:
            t_barrier = 0.0
        else:
            t_barrier = 1.0 / (1.0 + np.exp(2.0 * np.pi * (vb - e_cm) / hbar_omega))

        if e_cm > vb:
            b_g = rb * np.sqrt(1.0 - vb / e_cm)
        else:
            b_g = 0.0
        b_g = max(b_g, 0.01)

        if self.delta_b is None:
            self.delta_b = config.model.a0 * 0.8

        p_geo = 1.0 / (1.0 + np.exp((b - b_g) / self.delta_b))
        p_base = t_barrier * self.f_icf * p_geo

        # 不做费米动量积分: 直接返回几何概率
        if n_fermi_samples <= 0:
            if return_details:
                return p_base, {'d': b_g, 'b_g': b_g, 'probabilities': np.array([p_base]),
                                 'e_star': np.array([config.system.q_capture]),
                                 'q_eff': np.array([config.system.q_total]), 'q_opt': 0.0}
            return p_base

        # 费米动量积分 (对激发能分布)
        k_mag, k_theta, k_phi = self.fermi_sampler.sample(n_fermi_samples)
        e_star_values = np.zeros(n_fermi_samples)
        q_eff_values = np.zeros(n_fermi_samples)

        for i in range(n_fermi_samples):
            e_rel, e_star = t_th_relative_energy(e_cm, k_mag[i], k_theta[i])
            e_star_values[i] = max(e_star, 0.0)
            q_eff_values[i] = e_star

        if return_details:
            return p_base, {
                'd': config.distance_of_closest_approach(
                    config.system.proj.Z, config.system.targ.Z, config.system.mu_proj_targ, e_cm, b),
                'b_g': b_g,
                'k_mag': k_mag,
                'k_theta': k_theta,
                'k_phi': k_phi,
                'probabilities': np.full(n_fermi_samples, p_base),
                'e_star': e_star_values,
                'q_eff': q_eff_values,
                'q_opt': 0.0,
            }
        return p_base

    def event_distribution(self, e_cm: float, b: float, n_samples: int):
        """复用向量化费米路径 (E* 谱, 概率与 k 无关 = p_base)"""
        _, details = self.probability(e_cm, b, n_fermi_samples=n_samples,
                                      return_details=True)
        return (np.asarray(details['k_mag']), np.asarray(details['k_theta']),
                np.asarray(details['probabilities']), np.asarray(details['e_star']))

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         config.system.proj.Z, config.system.targ.Z,
                                         config.system.mu_proj_targ)
        return self.probability(e_cm, b, **kwargs)


# ============================================================
# 7. 转移概率工厂函数
# ============================================================

def create_model(model_type: str = "icf", **kwargs) -> TransferModel:
    """创建转移概率模型

    Parameters
    ----------
    model_type : 模型类型
        "tunneling"  — 简单指数隧穿 (垒下)
        "qwindow"    — Q 值窗口 + 隧穿
        "dwba"       — 半经典 DWBA (定态相位)
        "fermi"      — 费米动量积分 + 隧穿
        "icf"        — ICF 占比校准 (推荐, 垒上能区)

    Returns
    -------
    TransferModel 子类实例
    """
    models = {
        "tunneling": TunnelingModel,
        "qwindow": QWindowTunnelingModel,
        "dwba": SemiclassicalTransferModel,
        "fermi": FermiIntegratedModel,
        "icf": ICFFractionModel,
    }
    if model_type not in models:
        raise ValueError(f"未知模型: {model_type}, 可选: {list(models.keys())}")
    return models[model_type](**kwargs)
