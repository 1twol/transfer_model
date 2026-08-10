#!/usr/bin/env python3
"""
examples/prl_209bi_compare.py — 对照 PRL 122, 102501 (Cook et al. 2019)

⁷Li + ²⁰⁹Bi → α + ²¹²Bi 的 α 旁观者双微分/角分布, 与实验 Fig.1 对比。

实验条件 (PRL Fig.1): E_cm = 38.72 MeV, E_cm/Vb = 1.31 → Vb ≈ 29.56 MeV。
实验趋势: α 角分布宽锥、峰在 θ_grazing≈60°, E_α 峰 ~30 MeV、范围 20-40 MeV,
双微分峰在 θ≈40-80°, E_α≈20-40 MeV。

本脚本:
  1. 临时把 config.system 换成 ⁷Li+²⁰⁹Bi 体系 (make_system_bismuth)
  2. 用模型算 E_cm=38.72 (E_lab=40.02) 的 α 双微分 + 角分布
  3. 画图 + 打印数值, 与实验对比

注意:
  - 实验关注破裂 α 产额, 不关心 t 俘获产额, 故 ²⁰⁹Bi+t 吸热 Q 值不影响对比
  - 模型是半经典近似, 角峰比实验 ~60° 略靠后是正常偏差
  - 用完后恢复 config.system (不影响其他调用)

用法:
  python examples/prl_209bi_compare.py [--n-fermi 5000] [--outdir examples/output]
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 终端 (GBK 代码页) 下强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, ValueError):
    pass

from model import config
from model.config import make_system_bismuth
from model.transfer import create_model
from model.cross_section import (compute_alpha_double_differential,
                                 compute_angular_distribution,
                                 _alpha_b_min)

E_CM_PRL = 38.72      # MeV, PRL Fig.1
Vb_PRL = 29.56        # MeV, E_cm/Vb = 1.31


def main():
    parser = argparse.ArgumentParser(description="PRL 209Bi α 旁观者对比")
    parser.add_argument("--n-fermi", type=int, default=5000)
    parser.add_argument("--outdir", type=str, default=os.path.join("examples", "output"))
    args = parser.parse_args()

    # ---- 1. 临时切换体系 ----
    saved_system = config.system
    sys_bi = make_system_bismuth()
    config.system = sys_bi
    # cross_section 现在用 config.system 惰性读取, 无需 patch

    e_lab = E_CM_PRL * (sys_bi.proj.mass_MeV + sys_bi.targ.mass_MeV) / sys_bi.targ.mass_MeV

    print("=" * 62)
    print("  PRL 122, 102501 对照: ⁷Li + ²⁰⁹Bi (E_cm = %.2f MeV, E_cm/Vb = %.2f)" % (E_CM_PRL, E_CM_PRL / Vb_PRL))
    print("  E_lab = %.2f MeV, Vb = %.2f MeV" % (e_lab, Vb_PRL))
    print("  Q_total = %.2f MeV (t 俘获吸热, 不影响 α 分布对比)" % sys_bi.q_total)
    print("=" * 62)

    try:
        m = create_model("icf")
        b_min = _alpha_b_min(E_CM_PRL)
        print(f"\n  近正碰下界 b_min = {b_min:.2f} fm (D<R_int 的融合吸收事件剔除)")

        # ---- 2. α 双微分 ----
        print("\n  [α 双微分 d²σ/dE_α dΩ_α]")
        d2 = compute_alpha_double_differential(
            m, e_lab=e_lab, n_b=40, n_fermi=args.n_fermi, verbose=False)
        ds = d2['d2sigma']
        e_marg = ds.sum(axis=1)
        th_marg = ds.sum(axis=0)
        epk = np.argmax(e_marg)
        tpk = np.argmax(th_marg)
        print(f"    E_α 峰 = {d2['e_alpha'][epk]:.1f} MeV  (实验 ~30 MeV)")
        print(f"    E_α 范围 = [{d2['e_alpha'].min():.0f}, {d2['e_alpha'].max():.0f}] MeV  (实验 20-40)")
        print(f"    θ 峰 = {d2['theta_alpha_deg'][tpk]:.0f}°  (实验 40-80°)")
        print(f"    均值 <E_α> = {np.average(d2['e_alpha'], weights=e_marg):.1f} MeV")

        # ---- 3. 角分布 ----
        print("\n  [α 角分布 dσ/dΩ]")
        ang = compute_angular_distribution(m, e_lab, n_theta=37, n_fermi=args.n_fermi,
                                           verbose=False)
        pk = np.argmax(ang['dsigma_domega'])
        aint = np.trapezoid(ang['dsigma_domega'] * np.sin(ang['theta_cm']),
                            ang['theta_cm']) * 2 * np.pi
        print(f"    角分布峰 = {ang['theta_cm_deg'][pk]:.0f}°  (实验 ~60°)")
        print(f"    dσ/dΩ_peak = {ang['dsigma_domega'][pk]:.1f} mb/sr")
        print(f"    积分截面 = {aint:.1f} mb")
        print("\n    θ(deg):", " ".join("%.0f" % t for t in ang['theta_cm_deg'][::6]))
        print("    dσ/dΩ :", " ".join("%.1f" % d for d in ang['dsigma_domega'][::6]))

        # ---- 4. 画图 ----
        try:
            import matplotlib.pyplot as plt
            os.makedirs(args.outdir, exist_ok=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            # 双微分热图
            logz = np.log10(np.maximum(ds, 1e-30))
            mesh = ax1.pcolormesh(d2['theta_alpha_deg'], d2['e_alpha'], logz,
                                  shading='auto', cmap='viridis')
            fig.colorbar(mesh, ax=ax1, label="log10(d²σ/dE_α dΩ_α) [mb/sr/MeV]")
            ax1.set_xlabel("θ_α (deg)")
            ax1.set_ylabel("E_α (MeV)")
            ax1.set_title(f"Model α double-diff (E_cm={E_CM_PRL} MeV)")
            ax1.grid(alpha=0.2)

            # 角分布 (log)
            ax2.semilogy(ang['theta_cm_deg'], ang['dsigma_domega'], 'o-', color='C0', lw=1.5)
            ax2.axvline(60, color='red', ls=':', lw=1.5,
                        label="θ_grazing≈60° (exp)")
            ax2.set_xlabel("θ_α (deg)")
            ax2.set_ylabel("dσ/dΩ (mb/sr)")
            ax2.set_title("Model α angular distribution")
            ax2.grid(alpha=0.3)
            ax2.legend()

            fig.suptitle(f"⁷Li+²⁰⁹Bi comparison to PRL Fig.1 (E_cm={E_CM_PRL} MeV)")
            plt.tight_layout()
            out = os.path.join(args.outdir, "prl_209bi_compare.png")
            plt.savefig(out, dpi=150, bbox_inches='tight')
            print(f"\n  [plot] → {out}")
        except Exception as e:
            print(f"  [plot] 失败: {e}")

        print("\n  对比结论:")
        print("    实验: 角分布宽锥, 峰~60°, E_α 峰~30 MeV, 范围 20-40")
        print("    模型: 半经典近点 + 库仑后传播, 见上方数值")
    finally:
        config.system = saved_system
        print("\n  (config.system 已恢复为默认体系)")


if __name__ == "__main__":
    main()
