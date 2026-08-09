
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

# 使用默认 ICF 模型进行完整计算
python li7_th232_main.py

# 尝试不同的转移模型
python li7_th232_main.py --model fermi      # 费米动量积分
python li7_th232_main.py --model tunneling  # 简单指数隧穿
python li7_th232_main.py --model qwindow    # Q 值窗口 + 隧穿
python li7_th232_main.py --model dwba       # 半经典 DWBA

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

依次询问：转移模型、快速测试、能量范围、PACE4 模式、费米动量抽样数、
是否画图、输出目录——内部调用同一个 `li7_th232_main.main()`。
推荐模型（icf、fermi）列在最前；示意模型（tunneling、qwindow、dwba）
放在"其他示意模型"二级选择里，并提示其绝对标度未标定。

### 作为 Python 库使用

```python
from model.config import system, model
from model.transfer import create_model
from model.cross_section import compute_full

tr_model = create_model("icf")
result = compute_full(tr_model, e_lab_range=[20, 24, 28, 32, 36, 40])

print(result['excitation']['sigma'])            # 激发函数
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

| 模型 | CLI 参数 | 说明 |
|-------|----------|------|
| ICF 占比校准 | `icf` *(默认)* | 擦边角动量处的平滑 Fermi 台阶，乘以 f_ICF；费米 MC 只用于 E* 谱 |
| 费米动量积分 | `fermi` | 对 ⁷Li 内部动量做蒙特卡洛；每事件 t–Th 俘获 = 该事件 E_rel 处 t-Th 势垒的 Hill-Wheeler 穿透 × 入射势垒 × 几何截断 |
| Q 窗口隧穿 | `qwindow` | ⚠ *示意* — exp(−2κD) × 激发能匹配窗；绝对标度未标定 |
| 简单隧穿 | `tunneling` | ⚠ *示意* — P ∝ exp(−2κD)；绝对标度未标定 |
| 半经典 DWBA | `dwba` | ⚠ *示意* — 定态相位 + 零程；绝对标度未标定 |

> 五个模型现在都通过统一的费米动量抽样给出真实的激发能分布 dσ/dE* 和前向峰 α 角分布。
> 三个 ⚠ *示意* 模型只适合比较形状——其绝对 σ（P₀/D₀ 未标定，~1e-5–0.3 mb）不可信。
> 交互入口中它们被放在"其他示意模型"二级选择里。

---

## 物理模型

### 核心假设

1. **团簇图像**：⁷Li → α + t，束缚能 BE = 2.468 MeV
2. **两步过程**：t 在擦边距离从 α 场隧穿到靶核
3. **经典轨道**：入射道和出口道均采用卢瑟福双曲轨道
4. **突然近似**：转移瞬时发生，α 为旁观者

### 三体 → 两体

初态为三体体系：α、t（束缚于 ⁷Li 中）和 ²³²Th。t 团簇携带由 α-t 相对波函数导出的费米动量分布。可选描述方式：

- **高斯近似**（默认）：P(k) ∝ k² exp(−k²/2σ²)，σ ≈ 0.30 fm⁻¹
- **Numerov 数值解**（可开启）：完整 Woods-Saxon 束缚态 + Fourier-Bessel 变换

### 转移概率

```
P_tr(b, E) = ∫ d³k P(k) · P_tr(b, k)
```

各模型组成：

- **ICF 校准**（默认）：P_tr(b) = T(E_cm)·f_ICF / (1 + exp((b − b_g)/Δb))，f_ICF ≈ 0.25；
  费米 MC 只用于构建 E* 谱（不进入 σ 标度）。
- **费米积分**（`fermi`）：P_tr(b,k) = T(E_cm)·f_ICF·p_geo(b)·P_capture(E_rel(t−Th))，
  其中 P_capture 是 t-Th 势垒在事件 E_rel 处的 Hill-Wheeler 穿透。垒上标度与 `icf` 一致，
  垒下因双势垒（入射 + t-Th）抑制更强。
- **激发能匹配窗口**（`qwindow`）：P_Q = exp(−(E\*_event − E\*_opt)² / 2Γ²)，其中
  E\*_event = Q_capture + E_rel(t−Th)，E\*_opt = Q₀ − Q_opt。
- **简单隧穿**（`tunneling`）：P_tr = P₀·exp(−2κ D_eff)，κ ∝ √(μ_BE)——仅为示意。
- **最优 Q 值**：Q_opt = (Z_α·Z_Pa / Z_Li·Z_Th − 1) · E_cm

### 出口道

转移后，α 与 ²³⁵Pa\* 在库仑排斥下飞离，渐近动能由两体能量守恒
给出（T_rel(∞) = E_cm + Q₀ − E\*，已含库仑后加速的增益）：

```
E*(²³⁵Pa) = Q_capture + E_rel(t−²³²Th at transfer)
T_rel(∞)  = E_cm + Q₀ − E*
```

### 截面计算

```
σ(E)       = 2π ∫ b db · P_tr(b, E)           激发函数
dσ/dΩ_α    = Σ 事件 2π·b·P_tr(b,k)·δ(θ_α−θ_α(b,k))   α 角分布 (旁观者: α 在近点以近点切向速度 v_near=b·v_∞/D + 费米速度出发, 再经 ²³⁵Pa 库仑排斥传播到无穷远 → 渐近 θ_α, E_α; 近点落入核内的近正碰 b 事件被剔除——它们融合吸收, 不产生旁观 α)
dσ/dE*     = ∫ b db · (按样本 P_tr 加权的 E* 直方图)   激发能谱
```

---

## 代码结构

```
transfer_model/
├── model/                  # 核心物理模块
│   ├── config.py           # 物理常数、AME2020 质量、体系参数
│   ├── structure.py        # ⁷Li 团簇波函数、费米动量抽样
│   ├── potentials.py       # 库仑势 + Woods-Saxon 核势
│   ├── kinematics.py       # 卢瑟福轨道、擦边角、坐标系变换
│   ├── transfer.py         # 转移概率模型（5 种实现）
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
| `gamma_q` | Q 窗口宽度（QWindow 模型） | 3.0 MeV |

---

## 扩展

模块化设计，各物理环节相互独立：

- **新弹核**：在 `config.py` 中添加质量/电荷数据，结构和运动学自动适配
- **更换核势**：替换 `potentials.py`——已预留 São Paulo 双折叠势接口
- **完整 DWBA**：在 `transfer.py` 的 `SemiclassicalTransferModel` 基础上扩展有限程形状因子
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
