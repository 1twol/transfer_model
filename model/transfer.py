"""
transfer.py — 转移概率 (ICF 占比校准模型)

核心方法:
  P(b, E) = T(E) × f_ICF / [1 + exp((b − b_g(E))/Δb)]

  - T(E)   = 1/[1 + exp(2π(Vb−E_cm)/ħω)]   Hill-Wheeler 势垒穿透
  - b_g(E) = Rb·√(1−Vb/E)                   经典角动量截断 (E > Vb)
  - f_ICF  = 0.25 (Lei & Moro 2019)
  - 势垒参数 (Rb, Vb, ħω) 从 WS+Coulomb 数值有效势求取

转移事件的运动学 (α 旁观者动能分布、E* 谱) 由 cross_section 统一处理:
  E* = E_cm + Q_total − E_α(∞) − E_Pa   (能量守恒, 含库仑后加速)
  俘获条件: E* ≥ Q_cap, 不满足的事件 (t 未被俘获) 剔除

参考:
  - Broglia & Winther, "Heavy Ion Reactions", Ch. 7
  - Lei & Moro, PRC 100, 014618 (2019)
"""

import numpy as np
from typing import Optional

from . import config
from .structure import FermiMomentumSampler

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

    def event_distribution(self, e_cm: float, b: float, n_samples: int):
        """抽样费米事件, 返回 (k_mag, k_theta, p) 数组

        p 是单个费米事件的转移概率。ICF 模型 P 与 k 无关, p 恒为 P(b)。
        E* 等运动学量由 cross_section._event_physics 统一计算 (能量守恒口径)。
        """
        raise NotImplementedError


# ============================================================
# 2. ICF 占比校准模型 (唯一模型)
# ============================================================

class ICFFractionModel(TransferModel):
    """基于 ICF 占比的半经典转移模型

    物理:
      P(b, E) = T(E) × f_ICF / [1 + exp((b − b_g(E))/Δb)]

    垒下: T(E) 主导指数衰减, b_g=0, 单势垒量子隧穿
    垒上: b_g 根据离心势截断增长, T(E)→1, 经典几何截面

    绝对标度由入口道势垒穿透 × ICF 占比决定 (而非 α-t 束缚尾在重离子
    表面的指数抑制)。α 旁观者的动能分布形状完全由运动学决定 (费米动量、
    近点切向速度、库仑后加速), 与 P(b) 的绝对标度解耦。
    """

    def __init__(self, f_icf: float = 0.25, delta_b: float = None):
        super().__init__("ICF-Fraction")
        self.f_icf = f_icf
        self.delta_b = delta_b  # None -> 自动
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

    def _p_base(self, e_cm: float, b: float) -> float:
        """P(b) = T(E) × f_ICF × 擦边几何截断"""
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
        return t_barrier * self.f_icf * p_geo

    def probability(self, e_cm: float, b: float, **kwargs) -> float:
        """ICF 转移概率 P(b,E) = T(E) × f_ICF / (1 + exp((b−b_g)/Δb))"""
        return self._p_base(e_cm, b)

    def event_distribution(self, e_cm: float, b: float, n_samples: int):
        """ICF 的 P 与费米 k 无关, 每个事件概率恒为 P(b)"""
        k_mag, k_theta, _ = self.fermi_sampler.sample(n_samples)
        p = np.full(n_samples, self._p_base(e_cm, b))
        return k_mag, k_theta, p


# ============================================================
# 3. 转移概率工厂函数
# ============================================================

def create_model(model_type: str = "icf", **kwargs) -> TransferModel:
    """创建转移概率模型

    Parameters
    ----------
    model_type : 模型类型
        "icf" — ICF 占比校准 (唯一模型; 费米动量仅作用于运动学分布)

    Returns
    -------
    TransferModel 子类实例
    """
    if model_type != "icf":
        raise ValueError(f"未知模型: {model_type}, 可选: ['icf']")
    return ICFFractionModel(**kwargs)
