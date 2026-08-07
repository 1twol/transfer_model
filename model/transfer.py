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
            # 从 t+²³²Th 的束缚能估计
            be = abs(_sys.q_capture)  # 使用捕获 Q 值作为有效束缚能
            self.kappa = np.sqrt(2.0 * _sys.mu_t_th * be) / config.HBARC
        else:
            self.kappa = kappa
        self.p0 = p0

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """P(D) = P₀ exp(-2κ D)"""
        d = config.distance_of_closest_approach(
            _sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm, b
        )
        return self.p0 * np.exp(-2.0 * self.kappa * d)

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        """角度依赖的转移概率"""
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         _sys.proj.Z, _sys.targ.Z,
                                         _sys.mu_proj_targ)
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
            be = abs(_sys.q_capture)
            self.kappa = np.sqrt(2.0 * _sys.mu_t_th * be) / config.HBARC
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
        z_proj = _sys.proj.Z    # Z₁ = 3 (⁷Li)
        z_targ = _sys.targ.Z    # Z₂ = 90 (²³²Th)
        z_spec = _sys.spectator.Z  # Z₃ = 2 (α)
        z_prod = _sys.product.Z    # Z₄ = 91 (²³⁵Pa)
        # 在擦边处
        ratio = (z_spec * z_prod) / (z_proj * z_targ)
        return (ratio - 1.0) * e_cm

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """含 Q 窗的概率"""
        d = config.distance_of_closest_approach(
            _sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm, b
        )

        # 隧穿因子
        p_tunnel = np.exp(-2.0 * self.kappa * d)

        # Q 值窗口
        q_opt = self.q_opt(e_cm, d)
        q_total = _sys.q_total

        # 费米运动对 Q_eff 的修正
        if 'k_vec' in kwargs and kwargs['k_vec'] is not None:
            k_vec = kwargs['k_vec']
            k_mag = k_vec[0]
            e_fermi = config.HBARC**2 * k_mag**2 / (2.0 * _sys.cluster.mass_MeV)
            q_eff = q_total + e_fermi
        else:
            q_eff = q_total

        p_q = np.exp(-(q_eff - q_opt)**2 / (2.0 * self.gamma_q**2))

        return self.p0 * p_tunnel * p_q

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         _sys.proj.Z, _sys.targ.Z,
                                         _sys.mu_proj_targ)
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
            self.d0 = _mod.d0_manual
        else:
            self.d0 = d0

    def form_factor(self, r: float) -> float:
        """零程形状因子 F(R) = D₀ φ_αt(R)

        使用 Yukawa 波函数近似:
          φ_αt(r) ≈ √(2κ/4π) · exp(-κr)/r
          κ = √(2μ_αt·BE)/ħ
        """
        be = abs(_sys.q_breakup)
        kappa = np.sqrt(2.0 * _sys.mu_alpha_t * be) / config.HBARC
        norm = np.sqrt(2.0 * kappa / (4.0 * np.pi))
        phi = norm * np.exp(-kappa * max(r, 1e-6)) / max(r, 1e-6)
        return self.d0 * phi

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """半经典转移概率 (定态相位近似)

        P(b) ≈ (2π/ħ) · D₀² |φ_αt(D)|² / [v · |d/dR(κ_i(R) - κ_f(R))|_D]
        """
        d = config.distance_of_closest_approach(
            _sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm, b
        )

        # 形状因子在最近接近距离处
        form = self.form_factor(d)
        form_sq = form**2

        # 定态相位点处的速度
        v_rel = np.sqrt(2.0 * e_cm / _sys.mu_proj_targ)

        # 入射道 / 出口道局域动量差
        # κ_i² = 2μ_i (E - V_i(r))/ħ²
        kappa_i = self._local_wavenumber(e_cm, d, 'in')
        kappa_f = self._local_wavenumber(e_cm + _sys.q_total, d, 'out')

        # d/dR (κ_i - κ_f) 在 D 处
        dr = 0.1  # fm
        d_plus = d + dr
        dk_i_plus = self._local_wavenumber(e_cm, d_plus, 'in')
        dk_f_plus = self._local_wavenumber(e_cm + _sys.q_total, d_plus, 'out')
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
            z1, z2 = _sys.proj.Z, _sys.targ.Z
            a1, a2 = _sys.proj.A, _sys.targ.A
            mu = _sys.mu_proj_targ
            # 核势 ~0 在擦边距离 (库仑主导)
        else:  # 'out'
            z1, z2 = _sys.spectator.Z, _sys.product.Z
            a1, a2 = _sys.spectator.A, _sys.product.A
            mu = _sys.mu_alpha_pa

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
                                         _sys.proj.Z, _sys.targ.Z,
                                         _sys.mu_proj_targ)
        return self.probability(e_cm, b, **kwargs)


# ============================================================
# 5. 包含费米动量积分的完整隧穿模型
# ============================================================

class FermiIntegratedModel(TransferModel):
    """完整模型: 费米动量抽样 + 隧穿 + Q 窗口 + 后加速

    这是用于最终计算的推荐模型。
    """

    def __init__(self, kappa: float = None, gamma_q: float = None,
                 use_numerov_wf: bool = False):
        super().__init__("Fermi-Integrated")
        if kappa is None:
            be = abs(_sys.q_capture)
            self.kappa = np.sqrt(2.0 * _sys.mu_t_th * be) / config.HBARC
        else:
            self.kappa = kappa

        self.gamma_q = gamma_q if gamma_q is not None else 3.0
        self._use_numerov = use_numerov_wf

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
        (可选) details : 包含 D, k 分布, Q_eff, E* 分布等
        """
        d = config.distance_of_closest_approach(
            _sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm, b
        )

        # 费米动量抽样
        k_mag, k_theta, k_phi = self.fermi_sampler.sample(n_fermi_samples)

        # 对每个费米动量配置计算转移概率
        probabilities = np.zeros(n_fermi_samples)
        e_star_values = np.zeros(n_fermi_samples)
        q_eff_values = np.zeros(n_fermi_samples)

        for i in range(n_fermi_samples):
            # 费米动能
            e_fermi = config.HBARC**2 * k_mag[i]**2 / (2.0 * _sys.cluster.mass_MeV)

            # t 在 ⁷Li 内的方向余弦
            cos_theta_k = np.cos(k_theta[i])

            # 有效 Q 值 (含费米运动)
            q_eff = _sys.q_total + e_fermi

            # 隧穿因子 (D 被费米动量方向修正)
            # t 沿束流方向(cos_θ>0) → 有效距离更近 → 隧穿更大
            d_eff = d / (1.0 + 0.1 * cos_theta_k)  # 简单几何修正
            p_tunnel = np.exp(-2.0 * self.kappa * d_eff)

            # Q 窗口
            z_spec, z_prod = _sys.spectator.Z, _sys.product.Z
            z_proj, z_targ = _sys.proj.Z, _sys.targ.Z
            ratio = (z_spec * z_prod) / (z_proj * z_targ)
            q_opt = (ratio - 1.0) * e_cm
            p_q = np.exp(-(q_eff - q_opt)**2 / (2.0 * self.gamma_q**2))

            # t + ²³²Th 相对动能 → 激发 ²³⁵Pa
            # E* = Q_capture + E_rel(t-Th)
            v_cm = np.sqrt(2.0 * e_cm / _sys.mu_proj_targ)
            v_t_in_li = config.HBARC * k_mag[i] / _sys.cluster.mass_MeV

            # t 在 CM 系的速度 (简化的1D投影)
            v_t_cm_sq = (v_cm + v_t_in_li * cos_theta_k)**2 + \
                        (v_t_in_li * np.sin(k_theta[i]))**2
            e_rel_t_th = 0.5 * _sys.mu_t_th * v_t_cm_sq
            e_star = _sys.q_capture + e_rel_t_th

            probabilities[i] = p_tunnel * p_q
            e_star_values[i] = max(e_star, 0.0)
            q_eff_values[i] = q_eff

        p_avg = np.mean(probabilities)

        if return_details:
            return p_avg, {
                'd': d,
                'k_mag': k_mag,
                'k_theta': k_theta,
                'probabilities': probabilities,
                'e_star': e_star_values,
                'q_eff': q_eff_values,
                'q_opt': (ratio - 1.0) * e_cm,
            }
        return p_avg

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         _sys.proj.Z, _sys.targ.Z,
                                         _sys.mu_proj_targ)
        return self.probability(e_cm, b, **kwargs)


# ============================================================
# 6. ICF 占比校准模型 (推荐用于垒上能区)
# ============================================================

class ICFFractionModel(TransferModel):
    """基于 ICF 占比的实验校准模型

    物理图像:
      - 对于 b ≤ b_g (擦边碰撞参数以内): 融合概率 ~1
      - ICF (转移) 占比 f_ICF ≈ 25% (Lei & Moro 2019)
      - 对于 b > b_g: 转移概率指数衰减
      - 费米运动提供激发能分布

    转移概率:
      P_tr(b) = f_ICF / [1 + exp((b - b_g)/Δb)]

    这是半经验模型, 在擦边区域给出合理的转移截面,
    同时保留费米运动对激发能谱的影响。
    """

    def __init__(self, f_icf: float = 0.25,
                 delta_b: float = None,
                 use_numerov_wf: bool = False):
        super().__init__("ICF-Fraction")
        self.f_icf = f_icf
        self.delta_b = delta_b  # None → 自动
        self._use_numerov = use_numerov_wf

    def probability(self, e_cm: float, b: float,
                     n_fermi_samples: int = 2000,
                     return_details: bool = False):
        """ICF 校准转移概率

        P_tr(b) = f_ICF / [1 + exp((b - b_g)/Δb)]

        然后对费米动量求平均得到激发能分布。
        """
        r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A, _mod.r0)
        l_g = grazing_angular_momentum(e_cm, r_int,
                                        _sys.proj.Z, _sys.targ.Z)
        k = config.wavenumber(_sys.mu_proj_targ, e_cm)
        b_g = l_g / k if k > 0 else r_int

        if self.delta_b is None:
            self.delta_b = _mod.a0 * 0.8  # ~ 0.5 fm, 表面厚度

        # 平滑阶跃: Fermi 函数形式
        p_base = self.f_icf / (1.0 + np.exp((b - b_g) / self.delta_b))

        # 费米动量积分 (对激发能分布)
        k_mag, k_theta, k_phi = self.fermi_sampler.sample(n_fermi_samples)
        e_star_values = np.zeros(n_fermi_samples)
        q_eff_values = np.zeros(n_fermi_samples)

        for i in range(n_fermi_samples):
            e_fermi = config.HBARC**2 * k_mag[i]**2 / (2.0 * _sys.cluster.mass_MeV)
            cos_theta_k = np.cos(k_theta[i])

            v_cm = np.sqrt(2.0 * e_cm / _sys.mu_proj_targ)
            v_t_in_li = config.HBARC * k_mag[i] / _sys.cluster.mass_MeV

            v_t_cm_sq = (v_cm + v_t_in_li * cos_theta_k)**2 + \
                        (v_t_in_li * np.sin(k_theta[i]))**2
            e_rel_t_th = 0.5 * _sys.mu_t_th * v_t_cm_sq
            e_star_values[i] = max(_sys.q_capture + e_rel_t_th, 0.0)
            q_eff_values[i] = _sys.q_total + e_fermi

        if return_details:
            return p_base, {
                'd': config.distance_of_closest_approach(
                    _sys.proj.Z, _sys.targ.Z, _sys.mu_proj_targ, e_cm, b),
                'b_g': b_g,
                'k_mag': k_mag,
                'k_theta': k_theta,
                'probabilities': np.full(n_fermi_samples, p_base),
                'e_star': e_star_values,
                'q_eff': q_eff_values,
                'q_opt': 0.0,
            }
        return p_base

    def probability_angle(self, e_cm: float, theta_cm: float, **kwargs) -> float:
        from .kinematics import impact_parameter_from_angle
        b = impact_parameter_from_angle(theta_cm, e_cm,
                                         _sys.proj.Z, _sys.targ.Z,
                                         _sys.mu_proj_targ)
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
