#!/usr/bin/env python3
"""
Plot PACE4 ICF results from summary files.

Reads every `E=<beam>MeV/pace4_summary.txt` under a given output directory and produces:
1. Heatmap of compound-nucleus excitation energy (EEXCN) vs. beam energy,
   weighted by the partial cross section σ from PACE4.
2. Total ICF cross section vs. beam energy.
3. Mean excitation energy (from the weighted distribution) vs. beam energy.

Usage:
    python plot_pace4_summary.py
    python plot_pace4_summary.py --base_dir output_icf_20260807_221316/Li7+Th232_icf
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_summary(path: Path):
    """Parse one pace4_summary.txt file.

    Returns dict with keys:
        beam_energy (MeV), total_sigma (mb), mean_eexcn (MeV), std_eexcn (MeV),
        entries: list of (eexcn, sigma_mb, percent)
    """
    text = path.read_text()

    # Extract header info
    m_beam = re.search(r"E=(\d+)MeV", path.name)
    if not m_beam:
        m_beam = re.search(r"E\s*=\s*(\d+)\s*MeV", text)
    beam_energy = float(m_beam.group(1)) if m_beam else None

    m_total = re.search(r"Total\s+\S+\s*=\s*([0-9.eE+-]+)\s*mb", text, re.IGNORECASE)
    total_sigma = float(m_total.group(1)) if m_total else None

    m_mean = re.search(r"Mean\s+E\*\s*=\s*([0-9.eE+-]+)\s*MeV", text)
    mean_eexcn = float(m_mean.group(1)) if m_mean else None

    m_std = re.search(r"Std\s+E\*\s*=\s*([0-9.eE+-]+)\s*MeV", text)
    std_eexcn = float(m_std.group(1)) if m_std else None

    # Extract table rows
    entries = []
    for line in text.splitlines():
        # Look for lines that begin with optional whitespace and an integer index
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            try:
                idx = int(parts[0])
                eexcn = float(parts[1])
                sigma = float(parts[2])
                percent = float(parts[3])
                filename = parts[4]
                entries.append({
                    "idx": idx,
                    "eexcn": eexcn,
                    "sigma": sigma,
                    "percent": percent,
                    "file": filename,
                })
            except ValueError:
                continue

    return {
        "path": path,
        "beam_energy": beam_energy,
        "total_sigma": total_sigma,
        "mean_eexcn": mean_eexcn,
        "std_eexcn": std_eexcn,
        "entries": entries,
    }


def load_all_summaries(base_dir: Path):
    summaries = []
    for summary_path in sorted(base_dir.rglob("pace4_summary.txt")):
        s = parse_summary(summary_path)
        if s["beam_energy"] is not None:
            summaries.append(s)
    summaries.sort(key=lambda x: x["beam_energy"])
    return summaries


def build_weighted_grid(summaries, quantity="sigma"):
    """Build a dense grid suitable for pcolormesh.

    quantity: 'sigma' for partial cross section (mb), 'percent' for %.
    """
    all_beam = np.array([s["beam_energy"] for s in summaries])
    all_eexcn = np.concatenate([np.array([e["eexcn"] for e in s["entries"]]) for s in summaries])

    # Create regular 1-D grids
    beam_edges = np.linspace(all_beam.min() - 2, all_beam.max() + 2, 300)
    eexcn_edges = np.linspace(all_eexcn.min() - 1, all_eexcn.max() + 1, 300)
    Z = np.zeros((len(eexcn_edges) - 1, len(beam_edges) - 1), dtype=float)

    # Add each EEXCN bin weighted by sigma or percent
    for s in summaries:
        beam = s["beam_energy"]
        for e in s["entries"]:
            val = e[quantity]
            # Locate the cell whose center is closest to this (beam, eexcn) point
            b_idx = np.argmin(np.abs(0.5 * (beam_edges[:-1] + beam_edges[1:]) - beam))
            e_idx = np.argmin(np.abs(0.5 * (eexcn_edges[:-1] + eexcn_edges[1:]) - e["eexcn"]))
            Z[e_idx, b_idx] += val

    return beam_edges, eexcn_edges, Z


def plot_heatmap(beam_edges, eexcn_edges, Z, summaries, quantity_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(beam_edges, eexcn_edges, Z, shading="auto", cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(quantity_label)

    ax.set_xlabel("Beam energy $E_{\\mathrm{lab}}$ (MeV)")
    ax.set_ylabel("Compound-nucleus excitation energy $E^*$ (MeV)")
    ax.set_title("PACE4 ICF: $E^*$ distribution vs. beam energy")

    # Overlay mean E* points
    beams = [s["beam_energy"] for s in summaries]
    means = [s["mean_eexcn"] for s in summaries]
    ax.plot(beams, means, "w--o", markersize=5, linewidth=1.5, label="Mean $E^*$")
    ax.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_totals(summaries):
    fig, ax = plt.subplots(figsize=(7, 4))
    beams = [s["beam_energy"] for s in summaries]
    totals = [s["total_sigma"] for s in summaries]
    means = [s["mean_eexcn"] for s in summaries]

    ax.semilogy(beams, totals, "-o", label=r"Total ICF $\sigma$ (mb)")
    ax.set_xlabel("Beam energy $E_{\\mathrm{lab}}$ (MeV)")
    ax.set_ylabel("Total cross section (mb)")
    ax.set_title("PACE4 ICF total cross section vs. beam energy")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # Secondary axis for mean E*
    ax2 = ax.twinx()
    ax2.plot(beams, means, "s--", color="tab:orange", label="Mean $E^*$ (MeV)")
    ax2.set_ylabel("Mean $E^*$ (MeV)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    return fig


def print_table(summaries):
    print("\nBeam (MeV) | Total σ (mb) | Mean E* (MeV) | Std E* (MeV)")
    print("-" * 60)
    for s in summaries:
        print(
            f"{s['beam_energy']:<10.0f} | "
            f"{s['total_sigma']:<14.6e} | "
            f"{s['mean_eexcn']:<14.2f} | "
            f"{s['std_eexcn']:<14.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot PACE4 summary results.")
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("output_icf_20260807_221316/Li7+Th232_icf"),
        help="Directory containing E=<beam>MeV subfolders with pace4_summary.txt.",
    )
    parser.add_argument(
        "--quantity",
        choices=["sigma", "percent"],
        default="sigma",
        help="Quantity used to color the heatmap.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output file stem.")
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {args.base_dir}")

    summaries = load_all_summaries(args.base_dir)
    if not summaries:
        raise RuntimeError(f"No pace4_summary.txt files found under {args.base_dir}")

    print(f"Loaded {len(summaries)} energy points from {args.base_dir}")
    print_table(summaries)

    beam_edges, eexcn_edges, Z = build_weighted_grid(summaries, quantity=args.quantity)
    q_label = "Partial cross section σ (mb)" if args.quantity == "sigma" else "Weight (%)"

    fig_hm = plot_heatmap(beam_edges, eexcn_edges, Z, summaries, q_label)
    fig_tot = plot_totals(summaries)

    if args.out:
        heatmap_file = f"{args.out}_heatmap.png"
        totals_file = f"{args.out}_totals.png"
    else:
        heatmap_file = "pace4_heatmap.png"
        totals_file = "pace4_totals.png"

    fig_hm.savefig(heatmap_file, dpi=300)
    fig_tot.savefig(totals_file, dpi=300)
    print(f"\nSaved: {heatmap_file}, {totals_file}")


if __name__ == "__main__":
    main()
