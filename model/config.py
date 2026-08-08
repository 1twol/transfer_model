"""
config.py — 物理常数、体系参数、模型可调参数

所有量采用 MeV·fm·amu 单位制 (hc ≈ 197.327 MeV·fm)。
"""

from dataclasses import dataclass, field
import numpy as np

# ============================================================
# 1. 基本物理常数
# ============================================================

HBARC = 197.3269804          # MeV·fm
HBAR_SQ_OVER_2M_NUCLEON = 20.7355  # ħ²/(2m_N) MeV·fm², m_N ≈ 938.919 MeV/c²

# 精细结构常数相关
E2 = 1.43996448               # e² MeV·fm  (e²/(4πε₀))

# 核子质量 (MeV/c², AME2020)
M_N = 938.918746
M_P = 938.272088

# ============================================================
# 2. 核素质量 (u, AME2020)
# ============================================================

@dataclass
class Nuclide:
    """核素基本数据"""
    symbol: str
    Z: int
    A: int
    mass_u: float               # 原子质量 / u
    mass_MeV: float = 0.0       # m·c² / MeV (自动计算)

    def __post_init__(self):
        # 1 u = 931.49410242 MeV/c²
        self.mass_MeV = self.mass_u * 931.49410242

    @property
    def mass_number(self):
        return self.A

# AME2020 原子质量 (u)
NUCLIDES = {
    "n":   Nuclide("n",    0,   1,   1.00866491588),
    "p":   Nuclide("p",    1,   1,   1.00782503223),
    "d":   Nuclide("d",    1,   2,   2.01410177811),
    "t":   Nuclide("t",    1,   3,   3.01604928132),
    "He3": Nuclide("³He",  2,   3,   3.01602932212),
    "α":   Nuclide("α",    2,   4,   4.00260325413),
    "Li6": Nuclide("⁶Li",  3,   6,   6.01512288742),
    "Li7": Nuclide("⁷Li",  3,   7,   7.01600343666),
    "Th232": Nuclide("²³²Th", 90, 232, 232.038053608),
    "Pa234": Nuclide("²³⁴Pa", 91, 234, 234.04330556),
    "Pa235": Nuclide("²³⁵Pa", 91, 235, 235.04539900),
}


# ============================================================
# 3. 反应体系参数: ⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa*
# ============================================================

@dataclass
class SystemParams:
    """⁷Li + ²³²Th → α + ²³⁵Pa 体系参数"""

    # --- 弹核 (⁷Li) ---
    proj: Nuclide = field(default_factory=lambda: NUCLIDES["Li7"])
    # --- 靶核 (²³²Th) ---
    targ: Nuclide = field(default_factory=lambda: NUCLIDES["Th232"])
    # --- 转移团簇 (t) ---
    cluster: Nuclide = field(default_factory=lambda: NUCLIDES["t"])
    # --- 旁观者核 (α) ---
    spectator: Nuclide = field(default_factory=lambda: NUCLIDES["α"])
    # --- 产物核 (²³⁵Pa) ---
    product: Nuclide = field(default_factory=lambda: NUCLIDES["Pa235"])

    # --- Q 值 (MeV) ---
    # ⁷Li → α + t
    q_breakup: float = -2.467994       # MeV, Q = M_⁷Li - M_α - M_t (<0)
    # t + ²³²Th → ²³⁵Pa
    q_capture: float = 8.108           # MeV (放热)
    # ⁷Li + ²³²Th → α + ²³⁵Pa (净)
    q_total: float = 5.640            # MeV = q_breakup + q_capture

    # --- 约化质量 (MeV/c²) ---
    mu_proj_targ: float = 0.0         # ⁷Li-²³²Th
    mu_alpha_t: float = 0.0           # α-t (⁷Li 内部)
    mu_t_th: float = 0.0              # t-²³²Th
    mu_alpha_pa: float = 0.0          # α-²³⁵Pa (出口道)

    # --- 库仑势垒 (MeV) ---
    vb: float = 30.49                  # ⁷Li+²³²Th 裸势垒 (CCFULL)

    def __post_init__(self):
        m_proj = self.proj.mass_MeV
        m_targ = self.targ.mass_MeV
        m_cl = self.cluster.mass_MeV
        m_spec = self.spectator.mass_MeV
        m_prod = self.product.mass_MeV

        self.mu_proj_targ = (m_proj * m_targ) / (m_proj + m_targ)
        self.mu_alpha_t = (m_spec * m_cl) / (m_spec + m_cl)
        self.mu_t_th = (m_cl * m_targ) / (m_cl + m_targ)
        self.mu_alpha_pa = (m_spec * m_prod) / (m_spec + m_prod)


# ============================================================
# 4. 模型可调参数
# ============================================================

@dataclass
class ModelParams:
    """三体转移模型的可调参数"""

    # --- 几何参数 (fm) ---
    r0: float = 1.25                  # 半径参数
    a0: float = 0.65                  # 表面弥散 (fm)

    # --- α-t 团簇势 (⁷Li 内部) ---
    v0_alpha_t: float = 74.5          # WS 势阱深度 (MeV)
    r0_alpha_t: float = 1.15          # 半径参数
    a_alpha_t: float = 0.70           # 弥散
    n_alpha_t: int = 0                # 径向节点数 (ℓ=1 p-wave 基态 n=0)
    l_alpha_t: int = 1                # 轨道角动量 (ℓ=1, p-wave; 7Li alpha-t cluster)

    # --- t-²³²Th 束缚势 ---
    v0_t_th: float = -60.0            # 光学势实部深度 (MeV)
    # 注意: ²³⁵Pa 中 t 的束缚态参数应自动确定

    # --- 入射道核势 (⁷Li+²³²Th) ---
    # Akyüz-Winther 系统学: V0 ≈ 50-80 MeV
    v0_in: float = 65.0
    r0_in: float = 1.18
    a_in: float = 0.65

    # --- 出口道核势 (α+²³⁵Pa) ---
    v0_out: float = 70.0
    r0_out: float = 1.20
    a_out: float = 0.60

    # --- ⁷Li 内部费米动量 ---
    k_fermi: float = 0.0              # 费米动量 (fm⁻¹), 0=自动从WF计算
    sigma_k: float = 0.0              # 动量宽度 (fm⁻¹), 0=自动
    # 不自动时的手动值
    k_fermi_manual: float = 0.65      # fm⁻¹, 对应 ~ 128 MeV/c
    sigma_k_manual: float = 0.30      # fm⁻¹

    # --- 转移振幅 ---
    use_zero_range: bool = True        # True=零程DWBA, False=有限程
    d0: float = 0.0                   # 零程常数 (MeV·fm³/²), 0=从WF估算
    d0_manual: float = 150.0          # 手动 D₀ 值

    # --- 积分参数 ---
    n_b: int = 100                    # 碰撞参数网格点数
    b_max: float = 0.0                # 最大碰撞参数 (fm), 0=自动
    b_max_factor: float = 1.5         # b_max = factor * R_int
    n_k: int = 50000                  # 费米动量蒙特卡洛抽样数

    # --- 能量范围 ---
    e_lab_min: float = 20.0           # lab 能量下限 (MeV)
    e_lab_max: float = 40.0           # lab 能量上限 (MeV)
    e_lab_step: float = 2.0           # 步长 (MeV)

    # --- 角度范围 ---
    theta_min_deg: float = 5.0        # 最小散射角
    theta_max_deg: float = 90.0       # 最大散射角
    n_theta: int = 50                 # 角度网格点数

    def __post_init__(self):
        # 自动确定半径和势参数(如果留 0)
        if self.k_fermi != 0:
            self.k_fermi_manual = self.k_fermi
        if self.sigma_k != 0:
            self.sigma_k_manual = self.sigma_k
        if self.d0 != 0:
            self.d0_manual = self.d0


# ============================================================
# 5. 全局实例 (默认参数)
# ============================================================

system = SystemParams()
model = ModelParams()


# ============================================================
# 6. 工具函数
# ============================================================

def reduced_mass(m1_MeV: float, m2_MeV: float) -> float:
    """约化质量 (MeV/c²)"""
    return m1_MeV * m2_MeV / (m1_MeV + m2_MeV)


def wavenumber(mu_MeV: float, e_cm: float) -> float:
    """波数 k = √(2μE)/ħ (fm⁻¹)"""
    return np.sqrt(2.0 * mu_MeV * e_cm) / HBARC


def sommerfeld(z1: int, z2: int, mu_MeV: float, e_cm: float) -> float:
    """Sommerfeld 参数 η = Z₁Z₂e²/(ħv)"""
    v = HBARC * wavenumber(mu_MeV, e_cm) / mu_MeV
    # v = ħk/μ = √(2E/μ)
    # 但更直接: η = Z₁Z₂ e² / (ħ v) = Z₁Z₂ e² μ / (ħ² k)
    k = wavenumber(mu_MeV, e_cm)
    return z1 * z2 * E2 * mu_MeV / (HBARC * HBARC * k)


def e_lab_to_e_cm(e_lab: float, m_proj_MeV: float, m_targ_MeV: float) -> float:
    """实验室系 → 质心系能量"""
    return e_lab * m_targ_MeV / (m_proj_MeV + m_targ_MeV)


def e_cm_to_e_lab(e_cm: float, m_proj_MeV: float, m_targ_MeV: float) -> float:
    """质心系 → 实验室系能量"""
    return e_cm * (m_proj_MeV + m_targ_MeV) / m_targ_MeV


def distance_of_closest_approach(z1: int, z2: int, mu_MeV: float,
                                  e_cm: float, b: float) -> float:
    """纯库仑轨道最近接近距离 D = η/k + √[(η/k)² + b²] (fm)"""
    eta = sommerfeld(z1, z2, mu_MeV, e_cm)
    k = wavenumber(mu_MeV, e_cm)
    eta_over_k = eta / k
    return eta_over_k + np.sqrt(eta_over_k**2 + b**2)


def interaction_radius(a1: float, a2: float, r0: float = 1.25) -> float:
    """强吸收半径 R_int = r₀ (A₁^{1/3} + A₂^{1/3}) (fm)"""
    return r0 * (a1**(1.0/3.0) + a2**(1.0/3.0))


def coulomb_barrier(z1: int, z2: int, r_int: float) -> float:
    """点电荷库仑势垒 V_CB = Z₁Z₂ e² / R_int (MeV)"""
    return z1 * z2 * E2 / r_int
