
<p align="center">
  <strong>English</strong>
  &nbsp;·&nbsp;
  <a href="./README_zh.md">简体中文</a>
</p>

# transfer_model

Semiclassical three-body transfer reaction model for the Trojan Horse Method.

Reaction: **⁷Li + ²³²Th → α + t + ²³²Th → α + ²³⁵Pa\***

Computes transfer cross sections, angular distributions, and excitation energy spectra by explicitly modeling ⁷Li's α + t cluster structure, the t-transfer step, and the final two-body Coulomb breakup. Outputs PACE4 input files for statistical evaporation calculations.

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

# Full calculation (ICF model; α spectator kinetic-energy distribution is the
# central output)
python li7_th232_main.py

# Custom energy range
python li7_th232_main.py --energy 20 45 5

# Suppress plots or PACE4 generation
python li7_th232_main.py --no-plot --no-pace4
```

### Interactive mode

Step through the same computation with prompts (instead of CLI flags):

```bash
python interactive.py
```

It asks for: transfer model (only icf), quick-test mode, energy range,
PACE4 mode, Fermi sampling count, plotting, and output directory — then calls
the same `li7_th232_main.main()` under the hood.

### As a library

```python
from model.config import system, model
from model.transfer import create_model
from model.cross_section import compute_full

tr_model = create_model("icf")
result = compute_full(tr_model, e_lab_range=[20, 24, 28, 32, 36, 40])

print(result['excitation']['sigma'])           # excitation function
print(result['alpha_energy']['e_alpha_mean'])  # mean α kinetic energy (central output)
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

## Transfer model

The only model is **ICF fraction** (`icf`):

```
P(b, E) = T(E) × f_ICF / [1 + exp((b − b_g)/Δb)]
  T(E)   = 1/[1 + exp(2π(Vb−E_cm)/ħω)]    Hill-Wheeler barrier transmission
  b_g(E) = Rb·√(1−Vb/E)                    classical angular-momentum cutoff
  f_ICF  = 0.25 (Lei & Moro 2019)
```

The absolute scale is set by the entrance-barrier transmission × ICF fraction.
The α spectator kinetic-energy distribution is fully determined by the
kinematics (Fermi momentum, near-point tangential velocity, Coulomb
post-acceleration) and decoupled from P(b).

---

## Physical model

### Core assumptions

1. **Cluster picture**: ⁷Li → α + t, binding energy BE = 2.468 MeV
2. **Two-step process**: t transfers at grazing distance and is captured by ²³²Th; α is the spectator
3. **Classical trajectories**: Rutherford hyperbola for the entrance channel; α departs from the near point with tangential velocity + Fermi
4. **Sudden approximation**: transfer is instantaneous; α and t separate with momentum conservation

### Three-body → two-body

The initial state has three bodies: α, t (bound in ⁷Li), and ²³²Th. The t cluster carries a Fermi momentum distribution derived from the α-t relative wave function. Available descriptions:

- **Gaussian approximation** (default): P(k) ∝ k² exp(−k²/2σ²), σ ≈ 0.27 fm⁻¹
- **Numerov solution** (opt-in): full Woods-Saxon bound state + Fourier-Bessel transform

### Transfer probability

```
P_tr(b, E) = T(E_cm)·f_ICF / (1 + exp((b − b_g)/Δb)), f_ICF ≈ 0.25
```

The Fermi-momentum MC only feeds the event kinematics (α energy, angle, E*),
not the σ scale.

### Exit channel and excitation energy (energy conservation, single convention)

After transfer, α Coulomb-propagates in the ²³⁵Pa field to infinity (including
the post-acceleration gain C₁/(D+r_αt); the breakup radius = Coulomb near point
D(b) + the α-t internal separation r_αt, sampled from a Gaussian approximation
of the bound-state coordinate wave function, σ_r=2.5 fm). Per-event excitation
energy is fixed by energy conservation:

```
E* = E_cm + Q_total − E_α(∞) − E_Pa(recoil)      [energy conservation]
Capture condition: E* ≥ Q_capture = 8.108 MeV
Non-capture events (t not captured) are removed from σ(E), dσ/dE_α, dσ/dE*, dσ/dΩ
```

The Q_opt single-value estimate (E\*_opt Coulomb matching) is kept only as a
dashed reference line on plots — experiments show the α kinetic energy is a
distribution, not a fixed value.

### Cross sections

```
σ(E)          = 2π ∫ b db · P_tr(b, E)             excitation function
dσ/dE_α       = Σ_events 2π·b·P_tr(b)·δ(E_α−E_α(b,k))   α spectator kinetic-energy distribution (central output)
d²σ/dE_α dΩ_α = double-differential heat map (THM frame, for experiment comparison)
dσ/dΩ_α       = α angular distribution (near-point tangential v_near=b·v_∞/D + Fermi → Coulomb propagation)
dσ/dE*        = excitation-energy spectrum mapped from the α energy by energy conservation → PACE4 EEXCN
```

α starts at the ⁷Li near point (ahead of the target on the beam axis) with
tangential velocity + Fermi; head-on b events whose near point D(b)<R_int fall
inside the nucleus and fuse, emitting no spectator α. σ(E), the E* spectrum and
the α distribution share the same b grid and capture condition, so their
integrals agree exactly.

---

## Code structure

```
transfer_model/
├── model/                  # Core physics
│   ├── config.py           # Physical constants, AME2020 masses, beam parameters
│   ├── structure.py        # ⁷Li cluster wave function, Fermi momentum sampling
│   ├── potentials.py       # Coulomb + Woods-Saxon nuclear potentials
│   ├── kinematics.py       # Rutherford trajectories, grazing angle, frame transforms
│   ├── transfer.py         # Transfer probability model (ICF fraction)
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
| `f_icf` | ICF fraction (the single calibration parameter) | 0.25 |
| `sigma_r_alpha_t` | α-t separation width (breakup at D+r_αt) | 2.5 fm |

---

## Extending

Built for modular extension. Each physical component is isolated:

- **New projectile**: add mass/charge data in `config.py`; structure and kinematics adapt automatically.
- **Different potential**: swap `potentials.py` — a São Paulo double-folding interface is already stubbed.
- **Full DWBA**: extend the `TransferModel` base class in `transfer.py` with finite-range form factors.
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
