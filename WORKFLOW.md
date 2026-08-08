# transfer_model 运行流程与脚本说明

⁷Li + ²³²Th 三体转移模型（Trojan Horse 方法，目标产物 ²³⁴Pa）的运行链路与各脚本职责。

---

## 1. 整体调用链

```
                          用户
                     ┌──────┴──────┐
                     ▼             ▼
        ┌──────────────────┐  ┌──────────────────┐
        │ li7_th232_main.py│  │  interactive.py  │
        │ 命令行主入口       │  │ 交互式入口(逐项提问)│
        └────────┬─────────┘  └────────┬─────────┘
                 │  内部调用 main(args)  │
                 └──────────┬──────────┘
                            ▼
                 ┌──────────────────────┐
                 │  post_process.py      │  画图 + PACE4 .pace + EEXCN表
                 └──────────┬───────────┘
                            ▼
              ┌────────────────────────────────┐
              │         model/ 计算包           │
              │  cross_section.py ← 截面主入口   │
              │  transfer.py     ← 5种转移模型   │
              │  structure.py    ← 费米动量抽样  │
              │  kinematics.py   ← 轨道/运动学   │
              │  potentials.py   ← 势垒/势函数   │
              │  config.py       ← 常数/Q值/参数 │
              └────────────────────────────────┘
                            │
                            ▼
                    .pace 文件 (PACE4 输入)
                            │
                            ▼   (外部程序 PACE4 / LISE++)
                    PACE4 蒸发输出
                            │
                            ▼
              ┌──────────────────────────────┐
              │  plot_pace4_summary.py        │
              │  汇总 PACE4 输出 → heatmap/totals │
              └──────────────────────────────┘
```

---

## 2. 入口脚本的先后顺序

```
步骤1  选入口
       python li7_th232_main.py --model icf [选项]      # 命令行
       或  python interactive.py                        # 交互式, 问7个问题后调 main()
                │
步骤2  main() 内部依次做 7 件事:
       ① 打印体系信息 (config: 核素/Q值/约化质量)
       ② 打印运动学参考表 (kinematics: η/k/θ_g/L_g/b_g)
       ③ create_model() 建模型      (transfer 工厂)
       ④ compute_full() 三重计算    (cross_section)
              ├─ 激发函数 σ(E)   → 对每个能量、每个b调 model.probability(e_cm,b)
              ├─ α角分布 dσ/dΩ   → 费米事件算α角 → 直方图
              └─ E*激发能谱       → 费米MC → 加权直方图 → PACE4用
       ⑤ plot_all() 画三张图      (post_process)
       ⑥ 生成 PACE4 .pace + eexcn表 (post_process)
       ⑦ 写 result_summary.txt
                │
步骤3  外部: 把 .pace 喂给 PACE4 / LISE++ 跑蒸发
                │
步骤4  python plot_pace4_summary.py    # 汇总 PACE4 输出画图
```

---

## 3. 每个脚本干什么

| 脚本 | 角色 | 干什么 |
|---|---|---|
| **li7_th232_main.py** | 命令行主入口 | 解析参数 → 建模型 → 算截面/角分布/E*谱 → 画图 → 生成 PACE4 输入 |
| **interactive.py** | 交互式入口 | 逐项提问收集参数 → 复用 `li7_th232_main.main()`（不重复逻辑） |
| **post_process.py** | 后处理库 | 画图函数 + **生成 PACE4 .pace 文件**（含分波σ_L）+ EEXCN 汇总表 |
| **plot_pace4_summary.py** | PACE4 汇总脚本 | 读 PACE4 输出目录 → 汇总 → 画 heatmap/totals 图 |
| model/**config.py** | 数据/参数 | 物理常数(ħc,e²)、AME2020核素质量、Q值、约化质量、所有可调模型参数 |
| model/**structure.py** | ⁷Li 结构 | α+t 束缚态（Numerov 解 / 高斯近似）、动量空间波函数、费米动量蒙特卡洛抽样器 |
| model/**kinematics.py** | 运动学 | 卢瑟福轨道 b↔θ、最近接近距离 D、擦边角/角动量、t-Th 相对动能 E_rel、CM↔Lab 变换、库仑后加速工具 |
| model/**potentials.py** | 势函数 | 库仑势、Woods-Saxon 核势、复合势(V_coul+V_nuc+离心)、势垒搜索(R_b,V_b,ħω)、形状因子 |
| model/**transfer.py** | 转移概率 | 5 个模型：icf(默认) / fermi / tunneling / qwindow / dwba，每个给出 P_tr(b)；`create_model()` 工厂 + `event_distribution()` 统一费米事件抽样（E*/角分布共用）|
| model/**cross_section.py** | 截面积分 | σ(E)=2π∫b·P db；α角分布(PWIA旁观者)；E*谱(按样本概率加权)；`compute_full()` 汇总 |
| tests/**test_kinematics.py** | 测试 | 运动学模块单元测试（b↔θ 自洽、擦边角、CM↔Lab、Q值） |

---

## 4. 数据流：一个量怎么从参数走到产物

```
config.py 参数 (V0, r0, a, σ_k, D0, f_ICF, ...)
   │
   ▼
potentials.py → 势垒 V_b, R_b, ħω   ←────┐
   │                                      │
   ▼                                      │
transfer.py → P_tr(b) ──┐                 │
structure.py → 费米动量抽样 k ─────────────┤  (icf/fermi 模型)
   │                                      │
   ▼                                      │
cross_section.py:                        │
   σ(E) = 2π∫ b P_tr db  ────────────────┤
   E*谱 = Σ 事件 b·P_tr(k) 加权          │
   角分布 = Σ 事件 P_tr(k)·δ(θ_α)        │
   │                                      │
   ▼                                      ▼
post_process.py → .pace (E*, σ_L, total_σ)   CCFULL partial.dat (外部分波,可选)
   │
   ▼
PACE4 (外部) → 234Pa 截面
   │
   ▼
plot_pace4_summary.py → 激发函数图/热图
```
