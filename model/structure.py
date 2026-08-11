"""
structure.py — ⁷Li 团簇波函数与费米动量分布

模型:
  ⁷Li → α + t, 束缚能 BE = 2.468 MeV

方法:
  1. Numerov 数值求解 α-t 径向薛定谔方程 → 坐标空间波函数
  2. Fourier-Bessel 变换 → 动量空间波函数 ψ(k)
  3. 动量分布 P(k) = |ψ(k)|² 用于费米动量抽样

参考:
  - Brink, "Semiclassical transfer amplitudes" (1978)
  - Bertsch & Esbensen, "Cluster structure of light nuclei" (1991)
"""

import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from typing import Tuple, Optional

from . import config

# 复用全局配置
_sys = config.system
_mod = config.model


# ============================================================
# 1. Numerov 方法解径向薛定谔方程
# ============================================================

def woods_saxon_potential(r: np.ndarray, v0: float, r0: float, a: float) -> np.ndarray:
    """Woods-Saxon 势: V(r) = -V₀ / (1 + exp((r-R)/a))

    Parameters
    ----------
    r : array, 径向坐标 (fm)
    v0 : float, 势阱深度 (MeV), v0>0 为吸引
    r0 : float, 半径参数 (fm), R = r0 * (A₁^{1/3} + A₂^{1/3})
    a  : float, 弥散参数 (fm)
    """
    radius = r0 * (_sys.spectator.A**(1/3) + _sys.cluster.A**(1/3))
    return -v0 / (1.0 + np.exp((r - radius) / a))


def coulomb_potential(r: np.ndarray, z1: int, z2: int) -> np.ndarray:
    """点电荷库仑势 (MeV)

    均匀带电球近似:
    r >= R_c: V_c = Z₁Z₂ e² / r
    r <  R_c: V_c = Z₁Z₂ e²/(2R_c) * (3 - (r/R_c)²)
    """
    r = np.maximum(r, 1e-10)
    rc = 1.25 * (_sys.spectator.A**(1/3) + _sys.cluster.A**(1/3))
    v = np.where(r >= rc,
                 z1 * z2 * config.E2 / r,
                 z1 * z2 * config.E2 / (2 * rc) * (3 - (r / rc)**2))
    return v


def effective_potential(r: np.ndarray, v0: float, r0_ws: float, a_ws: float,
                         z1: int, z2: int, l_val: int, mu: float) -> np.ndarray:
    """有效势 (径向 Schrödinger 方程中的 V_eff)

    V_eff = V_WS + V_coul + ħ² ℓ(ℓ+1)/(2μ r²)
    """
    v_ws = woods_saxon_potential(r, v0, r0_ws, a_ws)
    v_coul = coulomb_potential(r, z1, z2)
    centrifugal = config.HBAR_SQ_OVER_2M_NUCLEON * l_val * (l_val + 1) / (r**2)
    # 修正: 使用实际约化质量而非 m_N
    centrifugal = centrifugal * (938.919 / mu)  # 近似修正因子
    return v_ws + v_coul + centrifugal


def numerov_wavefunction(r: np.ndarray, energy: float, v0: float,
                          r0_ws: float, a_ws: float, z1: int, z2: int,
                          l_val: int, mu: float) -> np.ndarray:
    """Numerov 方法积分径向波函数 u(r) = r·R(r)

    求解: d²u/dr² = -k²(r) u(r)
    k²(r) = 2μ/ħ² * (E - V_eff(r))

    Parameters
    ----------
    r : 径向网格 (均匀步长)
    energy : 试探能量 (MeV), E < 0 为束缚态
    v0, r0_ws, a_ws : Woods-Saxon 参数
    z1, z2 : 核电荷数
    l_val : 轨道角动量
    mu : 约化质量 (MeV/c²)

    Returns
    -------
    u : 径向波函数 (未归一化)
    """
    n = len(r)
    dr = r[1] - r[0]
    u = np.zeros(n)

    # 有效势与 k²
    v_eff = effective_potential(r, v0, r0_ws, a_ws, z1, z2, l_val, mu)
    k2 = 2.0 * mu / config.HBARC**2 * (energy - v_eff)

    # Numerov 系数: f = 1 + h²/12 k²
    f = 1.0 + dr**2 / 12.0 * k2

    # ---- 出发点: r → 0 ----
    # 小 r 渐近行为: u(r) ∼ r^{ℓ+1}
    u[0] = 0.0
    u[1] = dr**(l_val + 1)

    for i in range(1, n - 1):
        u[i + 1] = ((12.0 - 10.0 * f[i]) * u[i] - f[i - 1] * u[i - 1]) / f[i + 1]

        # 如果波函数指数增长，截断
        if abs(u[i + 1]) > 1e100:
            u[i + 1:] = 0.0
            break

    return u


def shooting_eigenvalue(r: np.ndarray, v0: float, r0_ws: float, a_ws: float,
                         z1: int, z2: int, l_val: int, mu: float,
                         e_guess: float = -2.0,
                         n_nodes_target: int = 1,
                         tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, np.ndarray, int]:
    """打靶法求束缚态能量本征值 (末点过零扫描 + 节点数匹配)

    束缚态边界条件 u(r_max) = 0。对固定 r_max (渐近区), u 在 r_max 处
    的符号随 E 扫过每个本征能量翻转一次, 所以:

      1. 扫描 E ∈ [-100, -0.01] 找 u(r_max) 符号翻转区间 (每个翻转 = 一个束缚态);
      2. 按翻转区间的波函数节点数匹配 n_nodes_target (节点数随 E 变浅而减少);
      3. 在匹配区间二分, 使 u(r_max) 精确过零。

    注意: 旧的"节点数+末点符号"判据对 ℓ≠0 会收敛到阱底以下的发散假解
    (E=-100, 波函数指数增长), 已被本算法替代。

    Parameters
    ----------
    r : 径向网格
    v0, r0_ws, a_ws : WS 势参数
    z1, z2 : 电荷数
    l_val : 轨道角动量
    mu : 约化质量
    e_guess : 初始能量猜测 (MeV, 束缚态 <0)
    n_nodes_target : 目标径向节点数
    tol : 收敛容差
    max_iter : 最大迭代次数

    Returns
    -------
    energy : 本征能量 (MeV)
    u : 波函数 (未归一化)
    n_nodes : 实际节点数 (V0 太浅无匹配态时 < n_nodes_target)
    """
    n = len(r)
    n_max = n - 1

    def _solve(e: float):
        """积分到能量 e, 返回 (u 在 r_max 的符号, u)。发散解 (numerov 爆炸被
        截断) 用最后一个非零点的符号代替 (衰减到 0 的解不会被截断)。"""
        u = numerov_wavefunction(r, e, v0, r0_ws, a_ws, z1, z2, l_val, mu)
        nz = np.count_nonzero(u)
        if nz >= n - 5:
            return np.sign(u[-1]), u
        return np.sign(u[nz - 1]), u

    def _count_nodes(u: np.ndarray) -> int:
        return int(np.sum(np.diff(np.sign(u[:-1] * u[1:])) < 0))

    # ---- 1. 扫描找符号翻转 (每个翻转 = 一个束缚态) ----
    e_scan = np.linspace(-100.0, -0.01, 160)
    signs = np.array([_solve(e)[0] for e in e_scan])
    flips = np.where(signs[:-1] != signs[1:])[0]

    # 束缚态集中在 E ≈ -BE 附近 (浅区); 粗扫不足时在浅区加密
    if len(flips) <= n_nodes_target:
        e_scan = np.linspace(-10.0, -0.01, 400)
        signs = np.array([_solve(e)[0] for e in e_scan])
        flips = np.where(signs[:-1] != signs[1:])[0]

    # ---- 2. 按节点数匹配目标态 (节点数随 E 变浅而减少) ----
    candidates = []
    for fi in flips:
        e_mid = 0.5 * (e_scan[fi] + e_scan[fi + 1])
        _, u_mid = _solve(e_mid)
        candidates.append((fi, _count_nodes(u_mid)))

    matched = [fi for fi, nn in candidates if nn == n_nodes_target]
    if not matched:
        # V0 太浅, 不存在 n_nodes_target 节点态: 返回近 0 能量 + 实际节点数
        u_hi = numerov_wavefunction(r, -0.01, v0, r0_ws, a_ws, z1, z2, l_val, mu)
        return -0.01, u_hi, _count_nodes(u_hi)

    # 同节点数若多个, 取最深 (扫描顺序最前)
    idx = matched[0]

    # ---- 3. 二分使 u(r_max) 过零 ----
    e_a, e_b = e_scan[idx], e_scan[idx + 1]
    s_a = signs[idx]
    for _ in range(max_iter):
        e_mid = 0.5 * (e_a + e_b)
        s_m, _ = _solve(e_mid)
        if s_m == s_a:
            e_a = e_mid
        else:
            e_b = e_mid
        if e_b - e_a < tol:
            break

    energy = 0.5 * (e_a + e_b)
    u_final = numerov_wavefunction(r, energy, v0, r0_ws, a_ws, z1, z2, l_val, mu)
    return energy, u_final, _count_nodes(u_final)


def solve_bound_state(r_grid: np.ndarray, v0: float, r0_ws: float, a_ws: float,
                       z1: int, z2: int, l_val: int, mu: float,
                       target_be: float = 2.468,
                       n_nodes: int = 1,
                       v0_tol: float = 0.01) -> Tuple[float, np.ndarray]:
    """求解束缚态: 调节 V₀ 使能量本征值匹配 target_be

    Parameters
    ----------
    r_grid : 径向网格 (fm)
    v0 : 初始 V₀ 猜测 (MeV)
    r0_ws, a_ws : WS 几何参数
    z1, z2 : 电荷数
    l_val : 轨道角动量
    mu : 约化质量
    target_be : 目标束缚能 (MeV, 正值)
    n_nodes : 期望径向节点数
    v0_tol : V₀ 容差 (MeV)

    Returns
    -------
    v0_final : 最优 V₀ (MeV)
    u_norm : 归一化径向波函数 u(r)
    """
    target_e = -target_be

    v0_low = 10.0
    v0_high = 200.0

    for _ in range(50):
        v0_mid = (v0_low + v0_high) / 2.0
        e_found, u, n = shooting_eigenvalue(r_grid, v0_mid, r0_ws, a_ws,
                                              z1, z2, l_val, mu,
                                              e_guess=target_e,
                                              n_nodes_target=n_nodes)

        # 节点数不足 -> 势阱太浅, 需要增大 V0
        if n < n_nodes:
            v0_low = v0_mid
        # 节点数过多 -> 势阱太深, 需要减小 V0
        elif n > n_nodes:
            v0_high = v0_mid
        else:
            # 节点数对了, 根据能量微调
            # e_found < target_e (更负) -> V0 偏大
            if e_found < target_e:
                v0_high = v0_mid
            else:
                v0_low = v0_mid

        if v0_high - v0_low < v0_tol:
            break

    v0_final = (v0_low + v0_high) / 2.0
    _, u_final, _ = shooting_eigenvalue(r_grid, v0_final, r0_ws, a_ws,
                                           z1, z2, l_val, mu,
                                           e_guess=target_e,
                                           n_nodes_target=n_nodes)

    # 归一化: ∫₀^∞ |u(r)|² dr = 1
    norm = np.sqrt(simpson(u_final**2, r_grid))
    if norm > 1e-30:
        u_final = u_final / norm

    return v0_final, u_final


# ============================================================
# 2. 坐标空间 → 动量空间变换
# ============================================================

def wavefunction_to_momentum(r: np.ndarray, u: np.ndarray, l_val: int,
                               k_grid: np.ndarray) -> np.ndarray:
    """Fourier-Bessel 变换: 径向波函数 → 动量空间波函数

    ψ_{ℓm}(k) = √(2/π) (-i)^ℓ ∫₀^∞ r² j_ℓ(kr) R(r) dr
    其中 R(r) = u(r)/r

    动量分布: P(k) = 4π k² |ψ(k)|² (对 m 求和, 各向同性)

    Parameters
    ----------
    r : 径向网格
    u : 径向波函数 u(r) = r·R(r)
    l_val : 轨道角动量
    k_grid : 动量网格 (fm⁻¹)

    Returns
    -------
    psi_k : 动量空间波函数 ψ(k) 在各 k_grid 上的值
    """
    dr = r[1] - r[0]
    psi_k = np.zeros_like(k_grid, dtype=complex)

    for i, k in enumerate(k_grid):
        jl = spherical_jn(l_val, k * r)
        # ψ(k) = ∫ r² j_ℓ(kr) [u(r)/r] dr = ∫ r j_ℓ(kr) u(r) dr
        integrand = r * jl * u
        psi_k[i] = simpson(integrand, dx=dr)

    # 归一化常数
    psi_k *= np.sqrt(2.0 / np.pi) * (-1j)**l_val

    return psi_k


def momentum_distribution(k_grid: np.ndarray, psi_k: np.ndarray) -> np.ndarray:
    """动量分布 P(k) = 4π k² |ψ(k)|²

    满足归一化: ∫₀^∞ P(k) dk = 1
    """
    pk = 4.0 * np.pi * k_grid**2 * np.abs(psi_k)**2
    # 归一化检查
    norm = simpson(pk, k_grid)
    if norm > 1e-30:
        pk = pk / norm
    return pk


# ============================================================
# 3. 高斯近似 (快速计算用)
# ============================================================

def gaussian_momentum_distribution(k: np.ndarray, sigma_k: float) -> np.ndarray:
    """高斯动量分布近似

    P(k) dk = 4π/(2πσ²)^{3/2} k² exp(-k²/(2σ²)) dk

    Parameters
    ----------
    k : 动量 (fm⁻¹)
    sigma_k : 宽度 (fm⁻¹)
    """
    pk = k**2 * np.exp(-k**2 / (2.0 * sigma_k**2))
    pk *= 4.0 * np.pi / (2.0 * np.pi * sigma_k**2)**1.5
    return pk


def estimate_sigma_k(be: float, mu: float) -> float:
    """从束缚能和约化质量估算动量宽度

    σ_k ≈ √(2μ·BE) / (2ħ)  (用 Yukawa 波函数渐近估计)
    """
    kappa = np.sqrt(2.0 * mu * be) / config.HBARC
    # 对于 Yukawa 波函数 ψ ∝ exp(-κr)/r, 动量空间半高宽 ≈ κ
    return kappa * 0.6  # 经验因子


# ============================================================
# 4. 费米动量蒙特卡洛抽样
# ============================================================

class FermiMomentumSampler:
    """⁷Li 内 t 的费米动量分布抽样器

    使用方法:
        sampler = FermiMomentumSampler()
        k = sampler.sample(n=10000)  # 抽样 10000 个动量矢量 (|k|, θ_k, φ_k)
        或
        k_mag, k_theta, k_phi = sampler.sample_components(n=10000)
    """

    def __init__(self, use_numerov: bool = True,
                 l_val: Optional[int] = None,
                 n_nodes: Optional[int] = None):
        """
        Parameters
        ----------
        use_numerov : 是否用 Numerov 精确求解 (False 则用高斯近似)
        l_val : α-t 相对轨道角动量 (None=ℓ=1, p-wave; 7Li 基态)
        n_nodes : 径向节点数 (None=自动)
        """
        self.use_numerov = use_numerov
        self.l_val = l_val if l_val is not None else 1  # 7Li 基态 alpha-t 为 p-wave (l=1)
        self.n_nodes = n_nodes if n_nodes is not None else _mod.n_alpha_t

        self._pk_interpolator = None
        self._k_grid = None
        self._pk_grid = None
        self._sigma_k = None
        self._setup_distribution()

    def _setup_distribution(self):
        """构建动量分布"""
        if self.use_numerov:
            self._setup_numerov_distribution()
        else:
            self._setup_gaussian_distribution()

    def _setup_numerov_distribution(self):
        """从 Numerov 解构建动量分布"""
        mu_alpha_t = _sys.mu_alpha_t
        be = abs(_sys.q_breakup)

        # 径向网格
        r_max = 40.0  # fm
        n_r = 2000
        r = np.linspace(0.01, r_max, n_r)

        # 求解束缚态
        v0_final, u = solve_bound_state(
            r, _mod.v0_alpha_t, _mod.r0_alpha_t, _mod.a_alpha_t,
            _sys.spectator.Z, _sys.cluster.Z,
            self.l_val, mu_alpha_t,
            target_be=be, n_nodes=self.n_nodes
        )

        print(f"  [structure] alpha-t bound state (l={self.l_val}): "
              f"V0={v0_final:.1f} MeV, BE={be:.3f} MeV")

        # Fourier 变换
        k_max = 3.0  # fm⁻¹
        n_k = 1000
        self._k_grid = np.linspace(0.001, k_max, n_k)
        psi_k = wavefunction_to_momentum(r, u, self.l_val, self._k_grid)
        self._pk_grid = momentum_distribution(self._k_grid, psi_k)

        # 构建插值器用于抽样
        from scipy.interpolate import interp1d
        # 累积分布函数
        cdf = np.cumsum(self._pk_grid) * (self._k_grid[1] - self._k_grid[0])
        cdf = cdf / cdf[-1]
        self._pk_interpolator = interp1d(self._k_grid, self._pk_grid,
                                          kind='linear', bounds_error=False,
                                          fill_value=0.0)
        self._cdf_interpolator = interp1d(cdf, self._k_grid,
                                           kind='linear', bounds_error=False,
                                           fill_value=(0.0, self._k_grid[-1]))
        self._cdf = cdf

    def _setup_gaussian_distribution(self):
        """高斯近似动量分布"""
        be = abs(_sys.q_breakup)
        self._sigma_k = estimate_sigma_k(be, _sys.mu_alpha_t)
        if self._sigma_k <= 0 or self._sigma_k == 0:
            self._sigma_k = _mod.sigma_k_manual
        print(f"  [structure] Gaussian momentum dist: sigma_k≈{self._sigma_k:.3f} fm^-1 "
              f"(BE={be:.3f} MeV)")

        # 预计算分布
        self._k_grid = np.linspace(0.001, 3.0, 1000)
        self._pk_grid = gaussian_momentum_distribution(self._k_grid, self._sigma_k)
        cdf = np.cumsum(self._pk_grid) * (self._k_grid[1] - self._k_grid[0])
        cdf = cdf / cdf[-1]
        self._cdf = cdf

        from scipy.interpolate import interp1d
        self._cdf_interpolator = interp1d(cdf, self._k_grid,
                                           kind='linear', bounds_error=False,
                                           fill_value=(0.0, self._k_grid[-1]))

    def sample_k_magnitude(self, n: int) -> np.ndarray:
        """逆变换抽样: 返回 |k| (fm⁻¹)"""
        u = np.random.uniform(0.001, 0.999, n)
        if self._cdf_interpolator is not None:
            return self._cdf_interpolator(u)
        else:
            # 无插值器时使用高斯直接抽样
            return np.abs(np.random.normal(0, self._sigma_k, n))

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """蒙特卡洛抽样费米动量矢量

        Returns
        -------
        k_mag : 动量大小 (fm⁻¹), shape (n,)
        theta : 极角 (弧度), shape (n,) — 相对于束流方向
        phi : 方位角 (弧度), shape (n,)
        """
        k_mag = self.sample_k_magnitude(n)
        # 各向同性分布: cosθ 均匀, φ 均匀
        cos_theta = np.random.uniform(-1.0, 1.0, n)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n)
        return k_mag, theta, phi

    def get_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (k_grid, P(k)) 用于画图"""
        return self._k_grid.copy(), self._pk_grid.copy()

    @property
    def sigma_k(self) -> float:
        return self._sigma_k


# ============================================================
# 4b. α-t 相对距离抽样器 (破裂点 D + r_αt)
# ============================================================

class AlphaTRadiusSampler:
    """⁷Li 内 α-t 相对距离 r_αt 抽样器

    α 在破裂点的库仑势能取决于 α 到靶核的距离, 约等于近点距离 D(b)
    加 α-t 内部间距 r_αt。r_αt 从束缚态坐标波函数的高斯近似抽样:

      3D 球壳分布 P(r) dr ∝ r²·e^{−r²/2σ_r²} dr

    σ_r = config.sigma_r_alpha_t (默认 2.5 fm, 束缚态尺度 ~1/κ≈2.2 fm,
    p-wave 峰在 1-2 fm → <r_αt> ≈ 3-4 fm)。库仑后加速增益 C₁/(D+r_αt)
    随 r_αt 涨落, 使 E_α 分布展宽。

    注: 该几何下的 Numerov 束缚态解 (v0_alpha_t/r0_alpha_t/a_alpha_t 组合)
    在浅阱区不可靠 (0 节点态缺失), 故用高斯近似。
    """

    def __init__(self):
        self._r_grid = None
        self._cdf_interpolator = None
        self._mean_r = None
        self._setup_distribution()

    def _setup_distribution(self):
        sigma_r = _mod.sigma_r_alpha_t

        r_max = 20.0
        n_r = 2000
        r = np.linspace(0.01, r_max, n_r)

        dist = r**2 * np.exp(-r**2 / (2.0 * sigma_r**2))
        # 截断 4σ: 束缚波函数尾部短, 极端 r_αt (对应 b 很大的切向动能放大)
        # 在物理上无贡献, 只污染分布轴
        dist = np.where(r <= 4.0 * sigma_r, dist, 0.0)

        cdf = np.cumsum(dist) * (r[1] - r[0])
        if cdf[-1] < 1e-30:
            raise RuntimeError("AlphaTRadiusSampler: 波函数分布积分为零")
        cdf = cdf / cdf[-1]

        self._r_grid = r
        self._cdf = cdf
        self._cdf_interpolator = interp1d(cdf, r, kind='linear',
                                          bounds_error=False,
                                          fill_value=(r[0], r[-1]))

        # 平均 α-t 间距 <r> (诊断用)
        dr = r[1] - r[0]
        self._mean_r = np.trapezoid(r * dist, r) / np.trapezoid(dist, r)

    def sample(self, n: int) -> np.ndarray:
        """逆变换抽样 r_αt 大小 (fm)"""
        u = np.random.uniform(0.001, 0.999, n)
        return self._cdf_interpolator(u)

    def sample_3d(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """抽样 r_αt 三维矢量: (r_mag, theta, phi), 方向各向同性

        t 相对 ⁷Li 质心位移 d_t = (m_α/M)·r⃗_αt, α 相对质心 d_α = −(m_t/M)·r⃗_αt,
        方向耦合 (t 朝 Th / α 朝 Th) 由调用方用 theta 与轨道径向比对得到。
        """
        r_mag = self.sample(n)
        cos_theta = np.random.uniform(-1.0, 1.0, n)
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n)
        return r_mag, theta, phi

    @property
    def mean_r(self) -> float:
        return self._mean_r


# ============================================================
# 5. 零程常数 D₀ 估算
# ============================================================

def estimate_d0(u: np.ndarray, r: np.ndarray, l_val: int,
                 kappa: float) -> float:
    """从波函数估算零程常数 D₀

    D₀ 定义为转移形状因子的零程极限:
      F(R) = D₀ · φ_αt(R)  (对于 t 转移)

    其中 D₀ ≈ √(4π) · ħ²/(2μ_αt) · [du/dr]|_{r→0}

    Parameters
    ----------
    u : 径向波函数 u(r)
    r : 径向网格
    l_val : 轨道角动量
    kappa : √(2μ_BE)/ħ

    Returns
    -------
    d0 : D₀ (MeV·fm³/²)
    """
    if l_val != 0:
        # 对 ℓ≠0, 零程近似修正
        return _mod.d0_manual

    # 外推 u(r)/r 到 r→0
    # 对于 s-wave: u(r) ≈ A·r, 所以 R(0) = u'(0) ≈ u(r_min)/r_min
    idx = max(1, np.argmax(r > 0.5))
    r_small = r[idx]
    u_small = abs(u[idx])
    if r_small < 1e-10:
        return _mod.d0_manual

    # 零程常数
    hbar_sq_over_2mu = config.HBARC**2 / (2.0 * _sys.mu_alpha_t)
    d0 = np.sqrt(4.0 * np.pi) * hbar_sq_over_2mu * u_small / r_small

    return abs(d0)
