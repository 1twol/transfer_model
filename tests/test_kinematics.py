"""
test_kinematics.py — 运动学模块单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import model.config as config
from model.kinematics import (rutherford_trajectory,
                                grazing_angle,
                                grazing_angular_momentum,
                                impact_parameter_from_angle,
                                cm_to_lab)

_sys = config.system


def test_rutherford_trajectory():
    """测试卢瑟福轨道参数"""
    e_cm = 30.0  # MeV
    b = 10.0     # fm
    traj = rutherford_trajectory(e_cm, b,
                                   _sys.proj.Z, _sys.targ.Z,
                                   _sys.mu_proj_targ)
    print(f"E_cm={e_cm:.1f} MeV, b={b:.1f} fm")
    print(f"  η = {traj.eta:.2f}")
    print(f"  k = {traj.k:.3f} fm⁻¹")
    print(f"  D = {traj.d:.2f} fm")
    print(f"  θ_cm = {np.degrees(traj.theta_cm):.1f}°")
    print(f"  a = {traj.a_half:.2f} fm")
    print(f"  e = {traj.eccentricity:.2f}")

    # 一致性检查: b → θ → b
    b_recovered = impact_parameter_from_angle(traj.theta_cm, e_cm,
                                                _sys.proj.Z, _sys.targ.Z,
                                                _sys.mu_proj_targ)
    print(f"  b → θ → b: {b:.2f} → {np.degrees(traj.theta_cm):.1f}° → {b_recovered:.2f} fm")
    assert abs(b - b_recovered) < 0.1, f"b 恢复失败: {b} vs {b_recovered}"


def test_grazing_angle():
    """测试擦边角"""
    e_cm = 30.0
    r_int = config.interaction_radius(_sys.proj.A, _sys.targ.A)
    theta_g, theta_g_lab, l_g = grazing_angle(e_cm, r_int)
    print(f"E_cm={e_cm:.1f} MeV, R_int={r_int:.1f} fm")
    print(f"  θ_g (CM)  = {np.degrees(theta_g):.1f}°")
    print(f"  θ_g (Lab) = {np.degrees(theta_g_lab):.1f}°")
    print(f"  L_g = {l_g:.1f} ħ")

    l_g2 = grazing_angular_momentum(e_cm, r_int)
    print(f"  L_g (公式2) = {l_g2:.1f} ħ")
    # 不同公式可能略有差异, 但应同数量级
    assert abs(l_g - l_g2) < max(5.0, l_g * 0.3), f"L_g 差异过大: {l_g} vs {l_g2}"


def test_cm_lab_conversion():
    """测试 CM ↔ Lab 变换"""
    theta_cm = np.radians(30)
    e_cm = 30.0
    theta_lab, e_lab = cm_to_lab(theta_cm, e_cm,
                                     _sys.spectator.mass_MeV,
                                     _sys.product.mass_MeV,
                                     _sys.q_total)
    print(f"θ_cm={np.degrees(theta_cm):.1f}° → θ_lab={np.degrees(theta_lab):.1f}°")
    print(f"E_cm={e_cm:.1f} MeV → E_lab={e_lab:.1f} MeV")

    # 对于 m_α << m_Pa, θ_lab ≈ θ_cm (比值接近 1)
    ratio = np.degrees(theta_lab) / np.degrees(theta_cm)
    print(f"  θ_lab/θ_cm ratio = {ratio:.3f}")
    assert 0.90 < ratio < 1.10, f"角度变换异常: {ratio:.3f}"


def test_system_params():
    """验证体系参数合理性"""
    print(f"Q 值检查:")
    print(f"  ⁷Li → α + t: Q = {_sys.q_breakup:.3f} (应为 -2.468 MeV)")
    print(f"  t + ²³²Th → ²³⁵Pa: Q = {_sys.q_capture:.3f} (应为 +8.108 MeV)")
    print(f"  Net: {_sys.q_total:.3f} (应为 +5.640 MeV)")

    assert abs(_sys.q_breakup + 2.468) < 0.01
    assert abs(_sys.q_capture - 8.108) < 0.1
    assert abs(_sys.q_total - 5.640) < 0.1


if __name__ == '__main__':
    print("=" * 50)
    print("运动学模块测试")
    print("=" * 50)
    test_system_params()
    print()
    test_rutherford_trajectory()
    print()
    test_grazing_angle()
    print()
    test_cm_lab_conversion()
    print()
    print("所有测试通过 ✓")
