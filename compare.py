"""
Сравнение TCL2 и HEOM: наложение + разности.
=============================================
Все параметры берутся из params.py.
  python compare.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from params import *

# ══════════════════════════════════════════════════════════════════════
# 1. Решение TCL2
# ══════════════════════════════════════════════════════════════════════

print("[TCL2] Решение...")
import numerical as tcl2
t_tcl, sx_tcl, sy_tcl, sz_tcl, pur_tcl, N_t = tcl2.solve()
t_u, sx_u, sy_u, sz_u = tcl2.solve_unitary()

# ══════════════════════════════════════════════════════════════════════
# 2. Решение HEOM (выполняется при импорте — результаты уже готовы)
# ══════════════════════════════════════════════════════════════════════

print("[HEOM] Решение...")
import heom
t_arr    = heom.t_arr
sx_heom  = heom.sx_heom
sy_heom  = heom.sy_heom
sz_heom  = heom.sz_heom
pur_heom = np.array(heom.purity)

print("\nРасчёты завершены. Строю графики...")

# ══════════════════════════════════════════════════════════════════════
# 3. Графики: наложение (окно 1)
# ══════════════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
fig1.suptitle(
    f"TCL2 vs HEOM  |  $\\Lambda={LAMBDA}$, $\\lambda_s={lambda_s}$, "
    f"$\\gamma={GAMMA}$, $T={TEMP}$\n"
    f"$\\lambda_{{heom}}=\\Lambda^2\\lambda_s={LAMBDA**2*lambda_s:.4f}$",
    fontsize=13
)

for idx, (y_tcl, y_heom, y_unit, label, col) in enumerate([
    (sx_tcl, sx_heom, sx_u, r"$\langle\sigma_x\rangle$", "C0"),
    (sy_tcl, sy_heom, sy_u, r"$\langle\sigma_y\rangle$", "C1"),
    (sz_tcl, sz_heom, sz_u, r"$\langle\sigma_z\rangle$", "C2"),
]):
    ax = axes1[0, idx]
    ax.plot(t_u, y_unit, "--", color="grey", lw=1.0, label="Унитарная", alpha=0.6)
    ax.plot(t_arr, y_heom, "-", color=col, lw=2.0, label="HEOM")
    ax.plot(t_tcl, y_tcl, "--", color=col, lw=1.8, label="TCL2", alpha=0.85)
    ax.set_xlabel("$t$"); ax.set_ylabel(label)
    ax.set_title(label); ax.legend(fontsize=7); ax.grid(alpha=0.3)

axes1[1, 0].plot(t_arr, pur_heom, "-", color="C3", lw=2.0, label="HEOM")
axes1[1, 0].plot(t_tcl, pur_tcl, "--", color="C3", lw=1.8, label="TCL2", alpha=0.85)
axes1[1, 0].axhline(1.0, ls=":", color="black", lw=0.8)
axes1[1, 0].axhline(0.5, ls=":", color="grey", lw=0.8)
axes1[1, 0].set_xlabel("$t$"); axes1[1, 0].set_ylabel(r"$\mathrm{Tr}(\rho_S^2)$")
axes1[1, 0].set_title("Чистота"); axes1[1, 0].set_ylim(0, 1.05)
axes1[1, 0].legend(fontsize=7); axes1[1, 0].grid(alpha=0.3)

bl_heom = np.sqrt(sx_heom**2 + sy_heom**2 + sz_heom**2)
bl_tcl  = np.sqrt(sx_tcl**2 + sy_tcl**2 + sz_tcl**2)
axes1[1, 1].plot(t_arr, bl_heom, "-", color="C4", lw=2.0, label="HEOM")
axes1[1, 1].plot(t_tcl, bl_tcl, "--", color="C4", lw=1.8, label="TCL2", alpha=0.85)
axes1[1, 1].axhline(1.0, ls=":", color="grey", lw=0.8)
axes1[1, 1].set_xlabel("$t$"); axes1[1, 1].set_ylabel(r"$|\vec{\beta}(t)|$")
axes1[1, 1].set_title("Длина вектора Блоха")
axes1[1, 1].legend(fontsize=7); axes1[1, 1].grid(alpha=0.3)

w_pl = np.linspace(0.01, 10, 500)
J_full = LAMBDA**2 * lambda_s * GAMMA * w_pl / (w_pl**2 + GAMMA**2)
axes1[1, 2].plot(w_pl, J_full, color="C5", lw=2)
axes1[1, 2].set_xlabel(r"$\omega$"); axes1[1, 2].set_ylabel(r"$J_{full}(\omega)$")
axes1[1, 2].set_title(r"$J_{full}=\Lambda^2 J(\omega)$"); axes1[1, 2].grid(alpha=0.3)

fig1.tight_layout()
fig1.savefig("compare_overlay.png", dpi=150, bbox_inches="tight")
print("Сохранено: compare_overlay.png")

# ══════════════════════════════════════════════════════════════════════
# 4. Графики: разности (окно 2)
# ══════════════════════════════════════════════════════════════════════

delta_sx = sx_tcl - sx_heom
delta_sy = sy_tcl - sy_heom
delta_sz = sz_tcl - sz_heom
delta_norm = np.sqrt(delta_sx**2 + delta_sy**2 + delta_sz**2)
delta_pur = pur_tcl - pur_heom

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle(
    f"Разность TCL2 − HEOM  |  $\\Lambda={LAMBDA}$, $\\lambda_s={lambda_s}$",
    fontsize=13
)

axes2[0, 0].plot(t_arr, delta_sx, lw=1.8, color="C0", label=r"$\Delta\langle\sigma_x\rangle$")
axes2[0, 0].plot(t_arr, delta_sy, lw=1.8, color="C1", label=r"$\Delta\langle\sigma_y\rangle$")
axes2[0, 0].plot(t_arr, delta_sz, lw=1.8, color="C2", label=r"$\Delta\langle\sigma_z\rangle$")
axes2[0, 0].axhline(0, ls=":", color="black", lw=0.8)
axes2[0, 0].set_xlabel("$t$"); axes2[0, 0].set_ylabel(r"TCL2 $-$ HEOM")
axes2[0, 0].set_title(r"$\Delta\langle\sigma_\alpha\rangle$")
axes2[0, 0].legend(fontsize=8); axes2[0, 0].grid(alpha=0.3)

axes2[0, 1].plot(t_arr, delta_norm, lw=2, color="C7")
axes2[0, 1].set_xlabel("$t$")
axes2[0, 1].set_ylabel(r"$|\vec{\sigma}_{TCL2} - \vec{\sigma}_{HEOM}|$")
axes2[0, 1].set_title("Норма разности вектора Блоха")
axes2[0, 1].grid(alpha=0.3)

axes2[1, 0].plot(t_arr, delta_pur, lw=2, color="C3")
axes2[1, 0].axhline(0, ls=":", color="black", lw=0.8)
axes2[1, 0].set_xlabel("$t$")
axes2[1, 0].set_ylabel(r"$\mathrm{Tr}(\rho^2)_{TCL2} - \mathrm{Tr}(\rho^2)_{HEOM}$")
axes2[1, 0].set_title("Разность чистоты")
axes2[1, 0].grid(alpha=0.3)

axes2[1, 1].axis("off")
stats = (
    f"Параметры:\n"
    f"  LAMBDA = {LAMBDA},  lambda_s = {lambda_s}\n"
    f"  lam_heom = {LAMBDA**2*lambda_s:.6f}\n"
    f"  GAMMA = {GAMMA},  T = {TEMP}\n\n"
    f"Макс. отклонения:\n"
    f"  max|d<sx>| = {np.max(np.abs(delta_sx)):.6f}\n"
    f"  max|d<sy>| = {np.max(np.abs(delta_sy)):.6f}\n"
    f"  max|d<sz>| = {np.max(np.abs(delta_sz)):.6f}\n"
    f"  max|d<s>|  = {np.max(delta_norm):.6f}\n"
    f"  max|dPur|  = {np.max(np.abs(delta_pur)):.6f}\n\n"
    f"Финал (t={T_MAX}):\n"
    f"  <sz>_TCL2 = {sz_tcl[-1]:+.6f}\n"
    f"  <sz>_HEOM = {sz_heom[-1]:+.6f}\n"
    f"  Pur_TCL2  = {pur_tcl[-1]:.6f}\n"
    f"  Pur_HEOM  = {pur_heom[-1]:.6f}"
)
axes2[1, 1].text(0.05, 0.95, stats, transform=axes2[1, 1].transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig2.tight_layout()
fig2.savefig("compare_difference.png", dpi=150, bbox_inches="tight")
print("Сохранено: compare_difference.png")

print("\nЗакройте окна графиков для завершения.")
plt.show(block=True)