"""
run.py — управляющий файл
=========================
Режимы запуска:
  python run.py numeric   — только TCL2
  python run.py compare   — TCL2 + HEOM + сравнение
  python run.py           — интерактивный выбор

Логика сохранения файлов:
  <base_dir из .env>/<режим>/<UTC-timestamp>/
"""

import sys
import os
import shutil
import inspect
from datetime import datetime, timezone

# ══════════════════════════════════════════════════════════════════════
# Выбор режима
# ══════════════════════════════════════════════════════════════════════

VALID_MODES = {"numeric", "compare"}

def choose_mode() -> str:
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()
        if mode not in VALID_MODES:
            print(f"[run.py] Неизвестный режим '{mode}'. Допустимые: {VALID_MODES}")
            sys.exit(1)
        return mode

    print("Выберите режим запуска:")
    print("  1) numeric  — только TCL2")
    print("  2) compare  — TCL2 + HEOM + сравнение")
    while True:
        choice = input("Введите номер или название режима [1/2]: ").strip().lower()
        if choice in ("1", "numeric"):
            return "numeric"
        if choice in ("2", "compare"):
            return "compare"
        print("  Неверный ввод, попробуйте ещё раз.")

MODE = choose_mode()
print(f"\n[run.py] Режим: {MODE}\n")

# ══════════════════════════════════════════════════════════════════════
# Папка для сохранения результатов
# ══════════════════════════════════════════════════════════════════════

with open(".env", "r") as f:
    base_dir = f.readline().strip().split(": ", maxsplit=1)[-1]

time_stamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
out_dir = os.path.join(base_dir, MODE, time_stamp)
os.makedirs(out_dir, exist_ok=True)
print(f"[run.py] Результаты будут сохранены в: {out_dir}\n")

# ══════════════════════════════════════════════════════════════════════
# Режим: numeric — только TCL2
# ══════════════════════════════════════════════════════════════════════

if MODE == "numeric":
    print("[numeric] Запуск TCL2...")
    import numerical as tcl2

    # Сохранить спектральную плотность и N_I
    with open(os.path.join(out_dir, "functions.txt"), "w") as f:
        f.write(inspect.getsource(tcl2.J))
        f.write("\n\n")
        f.write(inspect.getsource(tcl2.N_I))

    shutil.copy("params.py", os.path.join(out_dir, "params.py"))

    t, sx_d, sy_d, sz_d, purity, N_t = tcl2.solve()
    t_u, sx_u, sy_u, sz_u = tcl2.solve_unitary()
    tcl2.plot_results(t, sx_d, sy_d, sz_d, purity, N_t,
                      t_u, sx_u, sy_u, sz_u, file_path=out_dir)

    print(f"\n[numeric] Финальные значения (t={t[-1]:.2f}):")
    print(f"  <sx> = {sx_d[-1]:+.6f}")
    print(f"  <sy> = {sy_d[-1]:+.6f}")
    print(f"  <sz> = {sz_d[-1]:+.6f}")
    print(f"  Tr(ρ²) = {purity[-1]:.6f}")

# ══════════════════════════════════════════════════════════════════════
# Режим: compare — TCL2 + HEOM
# В режиме compare спектральная плотность и N_I фиксируются такими,
# какими их ожидает HEOM: J(ω) = lambda_s·γ·ω/(ω²+γ²), N_I(ω) = тепловое.
# ══════════════════════════════════════════════════════════════════════

elif MODE == "compare":
    import numpy as np
    import matplotlib.pyplot as plt
    import params as pm

    # ------------------------------------------------------------------
    # Проверка/предупреждение: в compare режиме TCL2 должен использовать
    # ту же J и N_I, что и HEOM. Патчим модуль numerical до его импорта.
    # ------------------------------------------------------------------

    # Переопределяем J и N_I в пространстве имён numerical через monkey-patch
    # ДО того, как numerical начнёт строить массивы на уровне модуля.
    # Для этого временно подменяем params значениями, совместимыми с HEOM.

    import importlib
    import types

    # Создаём «прокси»-модуль params с перекрытыми функциями для TCL2
    # (сами числовые параметры берутся из оригинального params.py)

    print("[compare] Патчим J и N_I в numerical для совместимости с HEOM...")

    # Импортируем numerical — он сам импортирует params
    import numerical as tcl2

    # Теперь подменяем J и N_I на совместимые с HEOM версии
    # J_heom(ω) = LAMBDA² * lambda_s * γ * ω / (ω² + γ²)
    # N_I(ω)    = тепловое распределение Бозе-Эйнштейна
    def J_heom_compat(omega, t=0.0):
        """Спектральная плотность, совместимая с HEOM (лоренцевская).
        LAMBDA^2 не включается — он уже учтён внутри TCL2-ядра через rh_lambda.
        """
        return 2 * pm.lambda_s * pm.GAMMA * omega / (omega ** 2 + pm.GAMMA ** 2)

    def N_I_heom_compat(omega):
        """Тепловое распределение Бозе-Эйнштейна (как в HEOM)."""
        if pm.TEMP == 0:
            return 0.0
        x = omega / pm.TEMP
        if x > 700:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)

    # Патчим пространство имён модуля numerical
    tcl2.J   = J_heom_compat
    tcl2.N_I = N_I_heom_compat

    # Пересчитываем все предвычисленные массивы, зависящие от J и N_I
    print("[compare] Пересчёт предвычисленных массивов TCL2 с новым J...")
    tcl2.N0         = np.array([N_I_heom_compat(w) for w in tcl2.omega_arr])
    tcl2.g2_modes_t = np.array([J_heom_compat(tcl2.omega_arr, t) * tcl2.d_omega
                                 for t in tcl2.t_arr])
    tcl2.J_fine_t   = np.array([J_heom_compat(tcl2._w_fine, t) for t in tcl2.t_arr])

    # Пересчёт C0_mat
    print("[compare] Пересчёт C0_mat...")
    _dw_f       = tcl2._w_fine[1] - tcl2._w_fine[0]
    _tau_disc   = np.arange(pm.T_POINTS) * tcl2.dt
    _phase_disc = np.outer(tcl2._w_fine, _tau_disc)
    _ker_disc   = np.cos(_phase_disc) - 1j * np.sin(_phase_disc)
    _trap_w     = np.full(len(tcl2._w_fine), _dw_f)
    _trap_w[0]  = _dw_f / 2
    _trap_w[-1] = _dw_f / 2
    tcl2.C0_mat = (tcl2.J_fine_t * _trap_w[None, :]) @ _ker_disc
    del _phase_disc, _ker_disc
    print("[compare] C0_mat пересчитан.")

    # ------------------------------------------------------------------
    # Решение TCL2
    # ------------------------------------------------------------------
    print("\n[compare] Решение TCL2...")
    t_tcl, sx_tcl, sy_tcl, sz_tcl, pur_tcl, N_t_tcl = tcl2.solve()
    t_u, sx_u, sy_u, sz_u = tcl2.solve_unitary()

    # ------------------------------------------------------------------
    # Решение HEOM
    # ------------------------------------------------------------------
    print("\n[compare] Решение HEOM...")
    import heom
    t_arr    = heom.t_arr
    sx_heom  = heom.sx_heom
    sy_heom  = heom.sy_heom
    sz_heom  = heom.sz_heom
    pur_heom = np.array(heom.purity)

    print("\n[compare] Расчёты завершены. Строю графики...")

    # ------------------------------------------------------------------
    # Сохранение метаданных
    # ------------------------------------------------------------------
    with open(os.path.join(out_dir, "functions.txt"), "w") as f:
        f.write("# J и N_I, использованные в TCL2 для режима compare\n\n")
        f.write(inspect.getsource(J_heom_compat))
        f.write("\n\n")
        f.write(inspect.getsource(N_I_heom_compat))

    shutil.copy("params.py", os.path.join(out_dir, "params.py"))

    # ------------------------------------------------------------------
    # График 1: наложение
    # ------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle(
        f"TCL2 vs HEOM  |  $\\Lambda={pm.LAMBDA}$, $\\lambda_s={pm.lambda_s}$, "
        f"$\\gamma={pm.GAMMA}$, $T={pm.TEMP}$\n"
        r"$J(\omega)=\lambda_s\gamma\omega/(\omega^2+\gamma^2)$",
        fontsize=13
    )

    for idx, (y_tcl, y_heom, y_unit, label, col) in enumerate([
        (sx_tcl, sx_heom, sx_u, r"$\langle\sigma_x\rangle$", "C0"),
        (sy_tcl, sy_heom, sy_u, r"$\langle\sigma_y\rangle$", "C1"),
        (sz_tcl, sz_heom, sz_u, r"$\langle\sigma_z\rangle$", "C2"),
    ]):
        ax = axes1[0, idx]
        ax.plot(t_u,   y_unit, "--", color="grey", lw=1.0,  label="Унитарная", alpha=0.6)
        ax.plot(t_arr, y_heom, "-",  color=col,    lw=2.0,  label="HEOM")
        ax.plot(t_tcl, y_tcl,  "--", color=col,    lw=1.8,  label="TCL2", alpha=0.85)
        ax.set_xlabel("$t$"); ax.set_ylabel(label)
        ax.set_title(label); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    axes1[1, 0].plot(t_arr, pur_heom, "-",  color="C3", lw=2.0, label="HEOM")
    axes1[1, 0].plot(t_tcl, pur_tcl,  "--", color="C3", lw=1.8, label="TCL2", alpha=0.85)
    axes1[1, 0].axhline(1.0, ls=":", color="black", lw=0.8)
    axes1[1, 0].axhline(0.5, ls=":", color="grey",  lw=0.8)
    axes1[1, 0].set_xlabel("$t$"); axes1[1, 0].set_ylabel(r"$\mathrm{Tr}(\rho_S^2)$")
    axes1[1, 0].set_title("Чистота"); axes1[1, 0].set_ylim(0, 1.05)
    axes1[1, 0].legend(fontsize=7); axes1[1, 0].grid(alpha=0.3)

    bl_heom = np.sqrt(sx_heom**2 + sy_heom**2 + sz_heom**2)
    bl_tcl  = np.sqrt(sx_tcl**2  + sy_tcl**2  + sz_tcl**2)
    axes1[1, 1].plot(t_arr, bl_heom, "-",  color="C4", lw=2.0, label="HEOM")
    axes1[1, 1].plot(t_tcl, bl_tcl,  "--", color="C4", lw=1.8, label="TCL2", alpha=0.85)
    axes1[1, 1].axhline(1.0, ls=":", color="grey", lw=0.8)
    axes1[1, 1].set_xlabel("$t$"); axes1[1, 1].set_ylabel(r"$|\vec{\beta}(t)|$")
    axes1[1, 1].set_title("Длина вектора Блоха")
    axes1[1, 1].legend(fontsize=7); axes1[1, 1].grid(alpha=0.3)

    w_pl   = np.linspace(0.01, 10, 500)
    J_full = pm.lambda_s * pm.GAMMA * w_pl / (w_pl**2 + pm.GAMMA**2)
    axes1[1, 2].plot(w_pl, J_full, color="C5", lw=2)
    axes1[1, 2].set_xlabel(r"$\omega$"); axes1[1, 2].set_ylabel(r"$J(\omega)$")
    axes1[1, 2].set_title(r"$J(\omega)=\lambda_s\gamma\omega/(\omega^2+\gamma^2)$"); axes1[1, 2].grid(alpha=0.3)

    fig1.tight_layout()
    p1 = os.path.join(out_dir, "compare_overlay.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"[compare] Сохранено: {p1}")

    # ------------------------------------------------------------------
    # График 2: разности
    # ------------------------------------------------------------------
    delta_sx   = sx_tcl  - sx_heom
    delta_sy   = sy_tcl  - sy_heom
    delta_sz   = sz_tcl  - sz_heom
    delta_norm = np.sqrt(delta_sx**2 + delta_sy**2 + delta_sz**2)
    delta_pur  = pur_tcl - pur_heom

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
    fig2.suptitle(
        f"Разность TCL2 − HEOM  |  $\\Lambda={pm.LAMBDA}$, $\\lambda_s={pm.lambda_s}$",
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
        f"  LAMBDA   = {pm.LAMBDA},  lambda_s = {pm.lambda_s}\n"
        f"  GAMMA    = {pm.GAMMA},   T        = {pm.TEMP}\n\n"
        f"Макс. отклонения:\n"
        f"  max|d<sx>| = {np.max(np.abs(delta_sx)):.6f}\n"
        f"  max|d<sy>| = {np.max(np.abs(delta_sy)):.6f}\n"
        f"  max|d<sz>| = {np.max(np.abs(delta_sz)):.6f}\n"
        f"  max|d<s>|  = {np.max(delta_norm):.6f}\n"
        f"  max|dPur|  = {np.max(np.abs(delta_pur)):.6f}\n\n"
        f"Финал (t={pm.T_MAX}):\n"
        f"  <sz>_TCL2 = {sz_tcl[-1]:+.6f}\n"
        f"  <sz>_HEOM = {sz_heom[-1]:+.6f}\n"
        f"  Pur_TCL2  = {pur_tcl[-1]:.6f}\n"
        f"  Pur_HEOM  = {pur_heom[-1]:.6f}"
    )
    axes2[1, 1].text(0.05, 0.95, stats, transform=axes2[1, 1].transAxes,
                     fontsize=10, verticalalignment="top", fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig2.tight_layout()
    p2 = os.path.join(out_dir, "compare_difference.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"[compare] Сохранено: {p2}")

    print("\n[compare] Закройте окна графиков для завершения.")
    plt.show(block=True)