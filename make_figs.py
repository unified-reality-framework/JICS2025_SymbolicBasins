#!/usr/bin/env python3
"""
Generate JCSER figures:
- figs/jcsre_fig_yield.pdf
- figs/jcsre_fig_resolution.pdf
Reproducible, matplotlib-only, no custom styles/colors.
to run:
python make_figs.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------- Output dir ----------
OUT = Path("figs")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Reproducibility ----------
rng = np.random.default_rng(42)

# ---------- Model parameters (edit as needed) ----------
# Yield scaling: Y/Y0 = 1 + alpha * (delta_n/n0)_max * (Te_hot/Te_cold)
alpha_mean, alpha_std = 0.15, 0.02
dnn0_mean, dnn0_std   = 0.18, 0.05    # peak relative cold-e enhancement
RT_mean, RT_std       = 10.0, 1.5     # temperature ratio Te_hot/Te_cold

# Clip bounds to keep physical ranges sensible
dnn0_bounds = (0.05, 0.35)
RT_bounds   = (5.0, 15.0)

N = 10_000

# ---------- Monte-Carlo (Yield) ----------
alpha = rng.normal(alpha_mean, alpha_std, size=N)
dnn0  = np.clip(rng.normal(dnn0_mean, dnn0_std, size=N), *dnn0_bounds)
RT    = np.clip(rng.normal(RT_mean, RT_std, size=N), *RT_bounds)

Y_ratio = 1.0 + alpha * dnn0 * RT          # Y/Y0
Y_boost = Y_ratio - 1.0                    # fractional boost

# Summary stats
Y_mu  = float(np.mean(Y_ratio))
Y_ci  = np.percentile(Y_ratio, [2.5, 97.5])
Y_pct = (Y_ratio - 1.0) * 100.0            # %
Y_pct_ci = np.percentile(Y_pct, [2.5, 97.5])

print(f"[Yield] Mean boost: {Y_mu-1:.3f} ({(Y_mu-1)*100:.1f} %)")
print(f"[Yield] 95% CI (ratio): [{Y_ci[0]:.3f}, {Y_ci[1]:.3f}]")
print(f"[Yield] 95% CI (percent): [{Y_pct_ci[0]:.1f}%, {Y_pct_ci[1]:.1f}%]")

# ---------- Plot: Yield distribution ----------
plt.figure()
plt.hist(Y_pct, bins=50, density=True)
plt.xlabel("Etch-yield enhancement (%)")
plt.ylabel("Density")
plt.title("Monte-Carlo yield distribution (N=10,000)")
plt.tight_layout()
plt.savefig(OUT / "jcsre_fig_yield.pdf")
plt.close()

# ---------- Virtual mask resolution ----------
# Dx_min = 2π / (eta * |l| * k_p)
eta_mean, eta_std = 0.87, 0.05
l_fixed           = 20             # harmonic index (use positive integer)
kp_mean, kp_std   = 1.0e6, 1.0e5   # m^-1

eta = rng.normal(eta_mean, eta_std, size=N)
kp  = rng.normal(kp_mean, kp_std, size=N)
# Keep parameters positive
eta = np.clip(eta, 0.5, 1.2)
kp  = np.clip(kp, 2.5e5, 2.0e6)

dx_m  = 2.0 * np.pi / (eta * abs(l_fixed) * kp)   # meters
dx_um = dx_m * 1e6

dx_mu = float(np.mean(dx_um))
dx_ci = np.percentile(dx_um, [2.5, 97.5])

print(f"[Resolution] Mean Δx_min: {dx_mu:.3f} µm")
print(f"[Resolution] 95% CI: [{dx_ci[0]:.3f}, {dx_ci[1]:.3f}] µm")

# ---------- Plot: Resolution distribution ----------
plt.figure()
plt.hist(dx_um, bins=50, density=True)
plt.xlabel("Minimum feature size Δx_min (µm)")
plt.ylabel("Density")
plt.title("Virtual mask resolution distribution (N=10,000)")
plt.tight_layout()
plt.savefig(OUT / "jcsre_fig_resolution.pdf")
plt.close()
