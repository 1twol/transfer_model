# 三体转移反应模型 — 半经典计算框架

## 物理背景

### 反应
```
⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa*
```

本模型用于特洛伊木马方法（Trojan Horse Method）框架下计算 **t（氚）转移** 截面。⁷Li 作为"特洛伊木马"弹核，其 α + t 团簇结构允许 t 在擦边碰撞中转移到靶核 ²³²Th，形成激发的 ²³⁵Pa*。随后 ²³⁵Pa* 经 PACE4 蒸发计算得到 ²³⁴Pa 产生截面。

### 为什么需要三体模型？

现有 CCFULL+PACE4 流水线使用简化的 EEXCN 公式（`E* ≈ Q₀ − Q_opt`），未显式处理：
- ⁷Li 内部 t 的费米运动对激发能的展宽
- 转移概率的角度依赖
- 库仑场中的后加速效应
- 入射能、碰撞参数、费米动量的联合相空间

本模型从基础物理出发，建立可扩展的半经典框架。

---

## 物理模型

### 核心假设

1. **团簇图像**：⁷Li → α + t，束缚能 BE = 2.468 MeV
2. **二步过程**：t 在擦边距离从 α 场隧穿到 ²³²Th 场
3. **经典轨道**：入射道/出口道相对运动用卢瑟福双曲轨道描述
4. **突然近似**：转移瞬间发生，α 为旁观者

### 初态：三体 (α, t, ²³²Th)

```
⁷Li 波函数: Ψ_Li(r_αt) = φ_αt(r) · Y_lm(Ω)
                 ↓ Fourier 变换
费米动量分布: P(k) = 4πk²|ψ(k)|²
```

t 在 ⁷Li 内的动量分布由 α-t 相对波函数的 Fourier 变换给出。

可选两种描述：
- **Numerov 数值解**：Woods-Saxon 势阱 + 薛定谔方程精确解
- **高斯近似**（默认）：P(k) ∝ k² exp(−k²/2σ²)，σ ≈ 0.30 fm⁻¹

### 转移机制

转移概率在半经典图像中写为：

```
P_tr(b, E) = ∫ d³k P(k) · P_tunnel(D_eff) · P_Q−window(Q_eff)
```

其中：
- **隧穿因子**：P_tunnel = exp(−2κ D_eff)，κ ∝ √(μ_BE)
- **Q 值窗口**：P_Q = exp(−(Q_eff − Q_opt)²/(2Γ²))
- **最优 Q 值**：Q_opt = (Z_αZ_Pa/Z_LiZ_Th − 1) · E_cm

  4. **ICF-占比校准** (默认): P_tr(b) = f_ICF / (1 + exp((b-b_g)/Δb)), P_ICF≈25%

### 末态：两体 (α, ²³⁵Pa*)

转移后，α 与 ²³⁵Pa* 在库仑排斥下飞离：

```
E_α(∞) = E_α(transfer) + Z_α Z_Pa e² / D_transfer
```

²³⁵Pa 的激发能：

```
E*(²³⁵Pa) = Q_capture + E_rel(t−²³²Th at transfer)
           = 8.108 MeV + (费米运动贡献)
```

### 截面计算

```
σ_tr(E) = 2π ∫₀^∞ b db · P_tr(b, E)           [激发函数]

dσ/dΩ(θ) = (dσ/dΩ)_Ruth × P_tr(θ)            [角分布]

dσ/dE* = ∫ b db · dP/dE*(b)                   [激发能谱]
```

---

## 代码结构

```
transfer_model/
├── model/                  # 核心模块
│   ├── __init__.py
│   ├── config.py           # 物理常数、体系参数、可调参数
│   ├── structure.py        # ⁷Li 团簇波函数、费米动量抽样
│   ├── potentials.py       # 库仑势、核势 (WS)、形状因子
│   ├── kinematics.py       # 卢瑟福轨道、擦边角、坐标变换
│   ├── transfer.py         # 转移概率模型 (5种)
│   └── cross_section.py    # 相空间积分、截面计算
├── output/                 # 输出模块
│   └── __init__.py
├── tests/                  # 测试
│   └── test_kinematics.py
├── examples/               # 示例 (待扩展)
├── post_process.py         # 画图、PACE4 接口
├── li7_th232_main.py       # 主程序
└── README.md               # 本文档
```

---

## 使用方法

### 快速运行

```bash
# 完整计算 (默认 Fermi-Integrated 模型)
python li7_th232_main.py

# 快速测试 (简化参数, ~30秒)
python li7_th232_main.py --quick

# 指定模型
python li7_th232_main.py --model tunneling
python li7_th232_main.py --model qwindow
python li7_th232_main.py --model fermi      # 默认, 最完整

# 自定义能量范围
python li7_th232_main.py --energy 20 45 5

# 不画图, 不生成 PACE4
python li7_th232_main.py --no-plot --no-pace4
```

### 在 Python 脚本中使用

```python
from model.config import system, model
from model.transfer import create_model
from model.cross_section import compute_full

# 选择模型
tr_model = create_model("fermi")

# 完整计算
result = compute_full(tr_model, e_lab_range=[20, 24, 28, 32, 36, 40])

# 查看结果
print(result['excitation']['sigma'])  # 激发函数
print(result['angular']['dsigma_domega'])  # 角分布
print(result['e_star_spectrum']['e_star_mean'])  # 平均 E*
```

### 生成 PACE4 输入

```python
from post_process import generate_pace4_input

meta = generate_pace4_input(
    result['e_star_spectrum'],
    output_dir="./pace4_inputs",
    cascade_num=10000,
    facla_value=10.0
)
# → 生成按激发能 binned 的 .pace 文件
```

---

## 可调参数

所有参数在 `model/config.py` 的 `ModelParams` 中定义：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `r0` | 半径参数 | 1.25 fm |
| `a0` | 表面弥散 | 0.65 fm |
| `v0_alpha_t` | α-t WS 势深度 | 74.5 MeV |
| `k_fermi_manual` | 费米动量 | 0.65 fm⁻¹ |
| `sigma_k_manual` | 费米动量宽度 | 0.30 fm⁻¹ |
| `d0_manual` | 零程常数 D₀ | 150 MeV·fm³/² |
| `n_b` | b 网格点数 | 100 |
| `n_theta` | 角度网格点数 | 50 |
| `gamma_q` (QWindow) | Q 窗宽度 | 3.0 MeV |
| `kappa` (Tunneling) | 隧穿衰减常数 | 自动(从 BE) |

---

## 扩展方向

### 近期
- [ ] DWBA 双中心积分 (有限程)
- [ ] Numerov 束缚态波函数替代高斯近似
- [ ] ⁶Li 弹核 (α+d 团簇) 扩展
- [ ] 与 CCFULL 输出的 CDCC 结果比较

### 中期
- [ ] São Paulo 双折叠势
- [ ] 含张量力的转移形状因子
- [ ] 连续态离散化 (CDCC-light)
- [ ] 裂变竞争修正

### 远期
- [ ] 完全量子力学 CDCC 计算
- [ ] 多步转移过程
- [ ] 中子转移 vs 带电粒子转移的统一框架

---

## 参考文献

1. Brink, D.M., Phys. Lett. B 40, 37 (1972) — *半经典转移振幅*
2. Broglia, R.A. & Winther, A., "Heavy Ion Reactions" (2004) — *经典轨道与核势*
3. Satchler, G.R., "Direct Nuclear Reactions" (1983) — *DWBA 与转移形状因子*
4. Lei, J. & Moro, A.M., PRC 99, 044602 (2019) — *⁷Li+²⁰⁹Bi ICF 占比*
5. Hagino, K., Rowley, N., Kruppa, A.T., CPC 123, 143 (1999) — *CCFULL 程序*
6. Chamon, L.C. et al., PRL 79, 5218 (1997) — *São Paulo 双折叠势*

---

## 许可

本代码为大创项目 `基于特洛伊木马方法的 ²³⁴Pa 截面计算` 的一部分，仅用于学术研究。
