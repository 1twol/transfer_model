
<p align="center">
  <strong>English</strong>
  &nbsp;·&nbsp;
  <a href="./README_zh.md">简体中文</a>
</p>

# transfer_model

Semiclassical three-body transfer reaction model for the Trojan Horse Method.

Reaction: **⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa\***

Computes transfer cross sections, angular distributions, and excitation energy spectra by explicitly modeling ⁷Li's α + t cluster structure, the t-transfer step, and the final two-body Coulomb breakup. Outputs PACE4 input files for statistical evaporation calculations.

All computation is pure Python (NumPy / SciPy) — no external program calls.

---

## Installation

```bash
git clone https://github.com/1twol/transfer_model.git
cd transfer_model

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# or
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

Requirements: Python ≥ 3.9, NumPy, SciPy. Matplotlib is optional (for plotting).

---

## Quick start

```bash
# Fast test run (~30 seconds)
python li7_th232_main.py --quick

# Full calculation with the default ICF model
python li7_th232_main.py

# Try different transfer models
python li7_th232_main.py --model fermi      # Fermi-motion-integrated
python li7_th232_main.py --model tunneling  # Simple exponential tunneling
python li7_th232_main.py --model qwindow    # Q-value window + tunneling
python li7_th232_main.py --model dwba       # Semiclassical DWBA

# Custom energy range
python li7_th232_main.py --energy 20 45 5

# Suppress plots or PACE4 generation
python li7_th232_main.py --no-plot --no-pace4
```

### As a library

```python
from model.config import system, model
from model.transfer import create_model
from model.cross_section import compute_full

tr_model = create_model("icf")
result = compute_full(tr_model, e_lab_range=[20, 24, 28, 32, 36, 40])

print(result['excitation']['sigma'])           # excitation function
print(result['angular']['dsigma_domega'])      # angular distribution
print(result['e_star_spectrum']['e_star_mean']) # mean excitation energy
```

### PACE4 interface

```python
from post_process import generate_pace4_from_spectrum

meta = generate_pace4_from_spectrum(
    result['e_star_spectrum'],
    e_cm=31.06,
    output_dir="./pace4_inputs",
    cascades=10000,
)
# → writes .pace files in the correct fixed-column format
```

An EEXCN reference table is also emitted for use with external partial-wave data:

```bash
# The table is written to output_*/eexcn_table.txt
# Use with a CCFULL partial.dat:
python generate_pace.py partial.dat --e-star <value_from_table>
```

---

## Transfer models

| Model | CLI flag | Description |
|-------|----------|-------------|
| ICF fraction | `icf` *(default)* | Smooth Fermi-step at grazing angular momentum, scaled by f_ICF |
| Fermi-integrated | `fermi` | Monte Carlo integration over ⁷Li internal momentum distribution + tunneling + Q-window |
| Q-window tunneling | `qwindow` | Exponential tunneling with optimal Q-value matching |
| Simple tunneling | `tunneling` | P ∝ exp(−2κD) with distance of closest approach |
| Semiclassical DWBA | `dwba` | Stationary-phase approximation with zero-range form factor |

---

## Physical model

### Core assumptions

1. **Cluster picture**: ⁷Li → α + t, binding energy BE = 2.468 MeV
2. **Two-step process**: t tunnels from the α field to the target at grazing distance
3. **Classical trajectories**: Rutherford hyperbolae for entrance and exit channels
4. **Sudden approximation**: transfer is instantaneous; α acts as a spectator

### Three-body → two-body

The initial state has three bodies: α, t (bound in ⁷Li), and ²³²Th. The t cluster carries a Fermi momentum distribution derived from the α-t relative wave function. Available descriptions:

- **Gaussian approximation** (default): P(k) ∝ k² exp(−k²/2σ²), σ ≈ 0.30 fm⁻¹
- **Numerov solution** (opt-in): full Woods-Saxon bound state + Fourier-Bessel transform

### Transfer probability

```
P_tr(b, E) = ∫ d³k P(k) · P_tunnel(D_eff) · P_Q−window(Q_eff)
```

- **Tunneling**: P_tunnel = exp(−2κ D_eff), κ ∝ √(μ_BE)
- **Q-value window**: P_Q = exp(−(Q_eff − Q_opt)² / 2Γ²)
- **Optimal Q-value**: Q_opt = (Z_α·Z_Pa / Z_Li·Z_Th − 1) · E_cm
- **ICF calibration** (default): P_tr(b) = f_ICF / (1 + exp((b − b_g)/Δb)), f_ICF ≈ 0.25

### Exit channel

After transfer, α and ²³⁵Pa\* separate under their mutual Coulomb repulsion:

```
E_α(∞) = E_α(transfer) + Z_α·Z_Pa·e² / D_transfer
E*(²³⁵Pa) = Q_capture + E_rel(t−²³²Th at transfer)
```

### Cross sections

```
σ(E)      = 2π ∫ b db · P_tr(b, E)           excitation function
dσ/dΩ(θ)  = (dσ/dΩ)_Ruth × P_tr(θ)            angular distribution
dσ/dE*    = ∫ b db · dP(b)/dE*                excitation energy spectrum
```

---

## Code structure

```
transfer_model/
├── model/                  # Core physics
│   ├── config.py           # Physical constants, AME2020 masses, beam parameters
│   ├── structure.py        # ⁷Li cluster wave function, Fermi momentum sampling
│   ├── potentials.py       # Coulomb + Woods-Saxon nuclear potentials
│   ├── kinematics.py       # Rutherford trajectories, grazing angle, frame transforms
│   ├── transfer.py         # Transfer probability models (5 implementations)
│   └── cross_section.py    # Phase-space integration → cross sections
├── post_process.py         # Plotting utilities + PACE4 .pace file generation
├── li7_th232_main.py       # CLI entry point
├── tests/
│   └── test_kinematics.py  # Kinematics unit tests
├── requirements.txt
├── README.md
└── README_zh.md
```

---

## Parameters

All adjustable parameters live in `ModelParams` (in `model/config.py`):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `r0` | Radius parameter | 1.25 fm |
| `a0` | Surface diffuseness | 0.65 fm |
| `k_fermi_manual` | Fermi momentum | 0.65 fm⁻¹ |
| `sigma_k_manual` | Fermi momentum width | 0.30 fm⁻¹ |
| `d0_manual` | Zero-range constant D₀ | 150 MeV·fm³/² |
| `n_b` | Impact-parameter grid points | 100 |
| `n_theta` | Angular grid points | 50 |
| `gamma_q` | Q-window width (QWindow model) | 3.0 MeV |

---

## Extending

Built for modular extension. Each physical component is isolated:

- **New projectile**: add mass/charge data in `config.py`; structure and kinematics adapt automatically.
- **Different potential**: swap `potentials.py` — a São Paulo double-folding interface is already stubbed.
- **Full DWBA**: extend `SemiclassicalTransferModel` in `transfer.py` with finite-range form factors.
- **CDCC**: replace `kinematics.py` with coupled-discretized-continuum channels.

---

## References

1. Brink, D.M., *Phys. Lett. B* 40, 37 (1972) — semiclassical transfer amplitudes
2. Broglia, R.A. & Winther, A., *Heavy Ion Reactions* (2004) — classical orbits and nuclear potentials
3. Satchler, G.R., *Direct Nuclear Reactions* (1983) — DWBA and transfer form factors
4. Lei, J. & Moro, A.M., *Phys. Rev. C* 99, 044602 (2019) — ICF fraction for ⁷Li-induced reactions
5. Hagino, K., Rowley, N., Kruppa, A.T., *Comput. Phys. Commun.* 123, 143 (1999) — CCFULL
6. Chamon, L.C. et al., *Phys. Rev. Lett.* 79, 5218 (1997) — São Paulo double-folding potential

---

## License

MIT — see the repository for details.
