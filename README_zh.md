
<p align="center">
  <a href="./README.md">English</a>
  &nbsp;·&nbsp;
  <strong>简体中文</strong>
</p>

# transfer_model

特洛伊木马方法的半经典三体转移反应模型。

反应：**⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa\***

该模型显式处理 ⁷Li 的 α + t 团簇结构、t 转移过程以及末态两体库仑解体，计算转移截面、角分布和激发能谱。可输出 PACE4 输入文件用于统计蒸发计算。

---

## 安装

```bash
git clone https://github.com/1twol/transfer_model.git
cd transfer_model

# 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# 或
.venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
```

依赖：Python ≥ 3.9, NumPy, SciPy。Matplotlib 可选（用于画图）。

---

## 快速开始

```bash
# 快速测试（约 30 秒）
python li7_th232_main.py --quick

# 完整计算（ICF 模型，α 旁观者动能分布为核心输出）
python li7_th232_main.py

# 自定义能量范围
python li7_th232_main.py --energy 20 45 5

# 跳过画图或 PACE4 生成
python li7_th232_main.py --no-plot --no-pace4
```

### 交互式模式

用逐项提问代替命令行参数，走同一条计算链路：

```bash
python interactive.py
```

依次询问：转移模型（唯一：icf）、快速测试、能量范围、PACE4 模式、
费米动量抽样数、是否画图、输出目录——内部调用同一个 `li7_th232_main.main()`。

### 作为 Python 库使用

```python
from model.config import system, model
from model.transfer import create_model
from model.cross_section import compute_full

tr_model = create_model("icf")
result = compute_full(tr_model, e_lab_range=[20, 24, 28, 32, 36, 40])

print(result['excitation']['sigma'])            # 激发函数
print(result['alpha_energy']['e_alpha_mean'])   # 平均 α 动能 (核心输出)
print(result['angular']['dsigma_domega'])       # 角分布
print(result['e_star_spectrum']['e_star_mean']) # 平均激发能
```

### PACE4 接口

```python
from post_process import generate_pace4_from_spectrum

meta = generate_pace4_from_spectrum(
    result['e_star_spectrum'],
    e_cm=31.06,
    output_dir="./pace4_inputs",
    cascades=10000,
)
# → 生成正确固定列格式的 .pace 文件
```

同时输出 EEXCN 参考表，配合外部分波数据使用：

```bash
# 表格保存在 output_*/eexcn_table.txt
# 配合 CCFULL partial.dat 使用：
python generate_pace.py partial.dat --e-star <表中的 EEXCN 值>
```

---

## 转移模型

唯一模型为 **ICF 占比校准**（`icf`）：

```
P(b, E) = T(E) × f_ICF / [1 + exp((b − b_g)/Δb)]
  T(E)   = 1/[1 + exp(2π(Vb−E_cm)/ħω)]    Hill-Wheeler 势垒穿透
  b_g(E) = Rb·√(1−Vb/E)                    经典角动量截断
  f_ICF  = 0.25 (Lei & Moro 2019)
```

绝对标度由入口道势垒穿透 × ICF 占比决定。α 旁观者的动能分布形状由
运动学完全决定（费米动量、近点切向速度、库仑后加速），与 P(b) 解耦。

---

## 物理模型

### 核心假设

1. **团簇图像**：⁷Li → α + t，束缚能 BE = 2.468 MeV
2. **两步过程**：t 在擦边距离转移并被 ²³²Th 俘获，α 为旁观者
3. **经典轨道**：入射道采用卢瑟福双曲轨道，α 在近点以切向速度 + 费米速度出发
4. **突然近似**：转移瞬时发生，α 与 t 动量守恒分离

### 三体 → 两体

初态为三体体系：α、t（束缚于 ⁷Li 中）和 ²³²Th。t 团簇携带由 α-t 相对波函数导出的费米动量分布。可选描述方式：

- **高斯近似**（默认）：P(k) ∝ k² exp(−k²/2σ²)，σ ≈ 0.27 fm⁻¹
- **Numerov 数值解**（可开启）：完整 Woods-Saxon 束缚态 + Fourier-Bessel 变换

### 转移概率

```
P_tr(b, E) = T(E_cm)·f_ICF / (1 + exp((b − b_g)/Δb))，f_ICF ≈ 0.25
```

费米动量 MC 只作用于事件运动学（α 动能、出射角、E*），不进入 σ 标度。

### 出口道与激发能（能量守恒，全链路唯一口径）

转移后，α 在 ²³⁵Pa 库仑排斥场中传播到无穷远（含库仑后加速增益
C₁/(D+r_αt)，破裂点距离 = 库仑近点 D(b) + α-t 内部间距 r_αt，r_αt 从
束缚态坐标波函数的高斯近似抽样，σ_r=2.5 fm），每事件激发能由能量守恒
严格给出：

```
E* = E_cm + Q_total − E_α(∞) − E_Pa(反冲)      [能量守恒]
俘获条件: E* ≥ Q_capture = 8.108 MeV
非俘获事件 (t 未被俘获) 从 σ(E)/α 动能分布/E* 谱/角分布中剔除
```

Q_opt 单值法（E\*_opt 库仑匹配）仅作为图上参照虚线保留——实验表明
α 动能是分布而非固定值。

### 截面计算

```
σ(E)          = 2π ∫ b db · P_tr(b, E)            激发函数
dσ/dE_α       = Σ 事件 2π·b·P_tr(b)·δ(E_α−E_α(b,k))  α 旁观者动能分布 (核心输出)
d²σ/dE_α dΩ_α = 双微分热图 (THM 坐标系, 与实验对比)
dσ/dΩ_α       = α 角分布 (近点切向 v_near=b·v_∞/D + 费米 → 库仑传播)
dσ/dE*        = 按能量守恒从 α 动能映射的激发能谱 → PACE4 EEXCN
```

α 在近点（束流前方轴线）以切向速度出发；近点 D(b)<R_int 的近正碰
事件被剔除（融合吸收，不产生旁观 α）；σ(E)、E* 谱、α 分布共享同一
b 网格与俘获条件，截面积分严格一致。

---

## 代码结构

```
transfer_model/
├── model/                  # 核心物理模块
│   ├── config.py           # 物理常数、AME2020 质量、体系参数
│   ├── structure.py        # ⁷Li 团簇波函数、费米动量抽样
│   ├── potentials.py       # 库仑势 + Woods-Saxon 核势
│   ├── kinematics.py       # 卢瑟福轨道、擦边角、坐标系变换
│   ├── transfer.py         # 转移概率模型（ICF 占比校准）
│   └── cross_section.py    # 相空间积分 → 截面
├── post_process.py         # 画图工具 + PACE4 .pace 文件生成
├── li7_th232_main.py       # 命令行入口
├── tests/
│   └── test_kinematics.py  # 运动学单元测试
├── requirements.txt
├── README.md
└── README_zh.md
```

---

## 可调参数

所有参数定义在 `model/config.py` 中的 `ModelParams` 里：

| 参数 | 含义 | 默认值 |
|-----------|-------------|---------|
| `r0` | 半径参数 | 1.25 fm |
| `a0` | 表面弥散 | 0.65 fm |
| `k_fermi_manual` | 费米动量 | 0.65 fm⁻¹ |
| `sigma_k_manual` | 费米动量宽度 | 0.30 fm⁻¹ |
| `d0_manual` | 零程常数 D₀ | 150 MeV·fm³/² |
| `n_b` | 碰撞参数网格点数 | 100 |
| `n_theta` | 角度网格点数 | 50 |
| `f_icf` | ICF 占比（唯一标定参数） | 0.25 |
| `sigma_r_alpha_t` | α-t 相对距离宽度（破裂点 D+r_αt） | 2.5 fm |

---

## 扩展

模块化设计，各物理环节相互独立：

- **新弹核**：在 `config.py` 中添加质量/电荷数据，结构和运动学自动适配
- **更换核势**：替换 `potentials.py`——已预留 São Paulo 双折叠势接口
- **完整 DWBA**：在 `transfer.py` 的 `TransferModel` 基类上扩展有限程形状因子
- **CDCC**：用耦合道连续态离散化通道替换 `kinematics.py`

---

## 参考文献

1. Brink, D.M., *Phys. Lett. B* 40, 37 (1972) — 半经典转移振幅
2. Broglia, R.A. & Winther, A., *Heavy Ion Reactions* (2004) — 经典轨道与核势
3. Satchler, G.R., *Direct Nuclear Reactions* (1983) — DWBA 与转移形状因子
4. Lei, J. & Moro, A.M., *Phys. Rev. C* 99, 044602 (2019) — ⁷Li 诱发反应的 ICF 占比
5. Hagino, K., Rowley, N., Kruppa, A.T., *Comput. Phys. Commun.* 123, 143 (1999) — CCFULL
6. Chamon, L.C. et al., *Phys. Rev. Lett.* 79, 5218 (1997) — São Paulo 双折叠势

---

## License

MIT — 详见仓库。
