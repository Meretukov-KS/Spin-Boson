"""
Решение спин-бозонной модели через Heom
======================================
Точное решение

Работает только для спектральной плотности вида
J(ω) = λ_s · γ · ω / (ω² + γ²)
======================================
"""
import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath, DrudeLorentzPadeBath
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from params import *

#======================================
# Операторы
#======================================
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

H_S = -0.5 * DELTA * sx + 0.5 * H_FIELD * sz # гамильтониан системы

Q = sz                  # оператор взаимодействия с окружением

#======================================
# Начальное состояние  
#======================================
pauli_0 = PAULI_I                         # начальное состояние в виде вектора Паули
pauli_0 = pauli_0 / np.linalg.norm(pauli_0) # нормировка
rho0 = 0.5 * (qt.identity(2) 
              + pauli_0[0] * sx 
              + pauli_0[1] * sy 
              + pauli_0[2] * sz)

#======================================
# Баня: DrudeLorentzPadeBath
#======================================
bath = DrudeLorentzPadeBath(
    Q = Q,
    T=TEMP,
    lam=(LAMBDA ** 2) *lambda_s,
    gamma=GAMMA,
    Nk=N_MATSUBARA,
)

#======================================
# Решение HEOM
#======================================
solver = HEOMSolver(
    H=H_S,
    bath=bath,
    max_depth=N_DEPTH,
)

t_arr = np.linspace(0, T_MAX, T_POINTS) # массив времени

result = solver.run(rho0, t_arr, e_ops=[sx, sy, sz])

sx_heom = result.expect[0]
sy_heom = result.expect[1]
sz_heom = result.expect[2]
rho_heom = []
for i in range(len(t_arr)):
    rho_heom.append(0.5 * (qt.identity(2) 
                          + sx_heom[i] * sx 
                          + sy_heom[i] * sy 
                          + sz_heom[i] * sz))
purity  = [rho.purity() for rho in rho_heom]

#======================================
# Унитарная эволюция для сравнения
#======================================
result_u = qt.mesolve(H_S, rho0, t_arr, [], [sx, sy, sz])
sx_u = result_u.expect[0]
sy_u = result_u.expect[1]
sz_u = result_u.expect[2]

#======================================================================#
# Проверка сходимости по глубине иерархии
#======================================================================#
 
def check_convergence():
    fig, ax = plt.subplots(figsize=(8, 4))
    for depth in [1, 2, 3, 5]:
        s = HEOMSolver(H_S, bath, max_depth=depth)
        r = s.run(rho0, t_arr, e_ops=[sz])
        ax.plot(t_arr, r.expect[0], lw=1.8, label=f"N_depth={depth}")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\sigma_z\rangle$")
    ax.set_title("Сходимость HEOM по глубине иерархии")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("heom_convergence.png", dpi=150)
    plt.show()

#======================================
# Визуализация
#======================================
def plot_results():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"HEOM  |  $\\Delta={DELTA}$, $h={H_FIELD}$, "
        f"$\\lambda_s={lambda_s}$, $\\gamma={GAMMA}$, $T={TEMP}$, "
        f"$N_{{depth}}={N_DEPTH}$, $N_{{k}}={N_MATSUBARA}$, "
        f"Паде={'да' if USE_PADE else 'нет'}\n"
        r"$J(\omega)=\lambda_s\gamma\omega/(\omega^2+\gamma^2)$",
        fontsize=12
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
 
    for idx, (yd, yu, label, col) in enumerate([
        (sx_heom, sx_u, r"$\langle\sigma_x\rangle$", "C0"),
        (sy_heom, sy_u, r"$\langle\sigma_y\rangle$", "C1"),
        (sz_heom, sz_u, r"$\langle\sigma_z\rangle$", "C2"),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(t_arr, yu, "--", color="grey", lw=1.4, label="Без окружения")
        ax.plot(t_arr, yd, "-",  color=col,   lw=2.0, label="HEOM")
        ax.set_xlabel("$t$"); ax.set_ylabel(label)
        ax.set_title(label); ax.legend(fontsize=7); ax.grid(alpha=0.3)
 
    ax_pur = fig.add_subplot(gs[1, 0])
    ax_pur.plot(t_arr, purity, color="C3", lw=2)
    ax_pur.axhline(1.0, ls="--", color="black", lw=1, label="чистое")
    ax_pur.axhline(0.5, ls="--", color="grey",  lw=1, label="смешанное")
    ax_pur.set_xlabel("$t$"); ax_pur.set_ylabel(r"$\mathrm{Tr}(\rho_S^2)$")
    ax_pur.set_title("Чистота"); ax_pur.set_ylim(0, 1.05)
    ax_pur.legend(fontsize=7); ax_pur.grid(alpha=0.3)
 
    ax_bloch = fig.add_subplot(gs[1, 1])
    bloch_len = np.sqrt(sx_heom**2 + sy_heom**2 + sz_heom**2)
    ax_bloch.plot(t_arr, bloch_len, color="C4", lw=2)
    ax_bloch.axhline(1.0, ls="--", color="grey", lw=1, label="чистое состояние")
    ax_bloch.set_xlabel("$t$"); ax_bloch.set_ylabel(r"$|\vec{\beta}(t)|$")
    ax_bloch.set_title("Длина вектора Блоха")
    ax_bloch.legend(fontsize=7); ax_bloch.grid(alpha=0.3)
 
    ax_J = fig.add_subplot(gs[1, 2])
    w_pl = np.linspace(0.01, 10, 500)
    J_pl = lambda_s * GAMMA * w_pl / (w_pl**2 + GAMMA**2)
    ax_J.plot(w_pl, J_pl, color="C5", lw=2)
    ax_J.set_xlabel(r"$\omega$"); ax_J.set_ylabel(r"$J(\omega)$")
    ax_J.set_title("Спектральная плотность"); ax_J.grid(alpha=0.3)
 
    plt.savefig("heom_results.png", dpi=150, bbox_inches="tight")
    print("График сохранён: heom_results.png")
    plt.show()
 
#======================================
# Запуск
#======================================
if __name__ == "__main__":
    print(f"HEOM: N_depth={N_DEPTH}, Nk={N_MATSUBARA}, Паде={USE_PADE}")
    print(f"  Δ={DELTA}, h={H_FIELD}, λ_s={lambda_s}, γ={GAMMA}, T={TEMP}\n")
 
    plot_results()
 
    print(f"\nФинальные значения:")
    print(f"  <sx>(T)   = {sx_heom[-1]:.6f}")
    print(f"  <sy>(T)   = {sy_heom[-1]:.6f}")
    print(f"  <sz>(T)   = {sz_heom[-1]:.6f}")
    print(f"  Tr(ρ²)(T) = {purity[-1]:.6f}")
 
    check_convergence()
