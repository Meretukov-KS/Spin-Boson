"""
Спин-Бозонная модель (Spin-Boson Model)
========================================
Решение связанной системы уравнений (ур. 33, 34, 35/37 из документа):

  drho_S/dt        — уравнение на матрицу плотности спина           (ур. 33)
  d<sigma_a>/dt    — уравнение на средние значения спина            (ур. 34/36)
  d<N(omega,t)>/dt — уравнение на заселённость мод окружения        (ур. 35/37)

Спектральная плотность (лоренцевская):
    J(omega) = lambda * gamma * omega / (omega^2 + gamma^2)

Метод: TCL2 (time-convolutionless, 2-й порядок по lambda),
Метод численного счета: TR-BDF2 (трапециевидный метод Рунге-Кутта для систем с памятью),
Свойства:
  - A-устойчив (подходит для жёстких задач)
  - 2-й порядок точности
  - L-устойчив (подавляет высокочастотные осцилляции)
  - Для ρ_S: итерации сходятся за 3-5 шагов при λ_s << 1
"""

#======================================================================#
# импорт необходимых библиотек
#======================================================================#

import numpy as np
from scipy.integrate import quad, simpson
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import params as pm
import numba as nb
from datetime import datetime, timezone
import os
import inspect      # Для сохранения J и N_i
import shutil       # Для сохранения params


#======================================================================#
# параметры модели
#======================================================================#


# Операторы Паули (чистый numpy)
sx = np.array([[0, 1],  [1,  0]], dtype=complex)
sy = np.array([[0, -1j],[1j, 0]], dtype=complex)
sz = np.array([[1, 0],  [0, -1]], dtype=complex)
pauli = [sx, sy, sz]
H_S = -0.5 * pm.DELTA * sx + 0.5 * pm.H_FIELD * sz  # гамильтониан системы

# Тензор Леви-Чивита для спина
eps = np.zeros((3, 3, 3))
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1


def expect(op: np.ndarray, rho: np.ndarray) -> float:
    """Ожидаемое значение оператора op в состоянии rho: Re Tr(op @ rho)."""
    return np.real(np.trace(op @ rho))


def to_shrodinger(rho_int: np.ndarray, t: float) -> np.ndarray:
    """Преобразует матрицу плотности из представления взаимодействия в представление Шредингера."""
    U = expm(-1j * H_S * t)           # унитарный оператор эволюции для свободного спина
    return U @ rho_int @ U.conj().T


# Функция спектральной плотности
def J(omega: float, t: float = 0.0) -> float:
    k_shift     = 5 * pm.k + 1/2 * t
    w_shift     = pm.w0 + 0.2 * np.sin(sqrt_Omega * t)
    g_shift     = pm.g
    eta_shift   = 2 * pm.eta + 1/5 * t
    wc_shift    = pm.w0
    s_shift     = pm.s
    ohmic = eta_shift * (omega ** s_shift) * (wc_shift ** (1 - s_shift)) * np.exp(-omega / wc_shift)
    lorentz = g_shift ** 2 / np.pi * (k_shift) / ((omega - w_shift) ** 2 + (k_shift) **2)
    return ohmic + lorentz


# Начальное распределение для среды (тепловое распределение)
def N_I(omega: float) -> float:
    return 0
    # if pm.TEMP == 0:
    #     return 0.0
    # x = omega / pm.TEMP
    # if x > 700:  # избегаем переполнения экспоненты
    #     return 0.0
    # return 1.0 / (np.exp(x) - 1.0)

# Начальное состояние спина (двухуровневой системы)
pauli_i = np.array(pm.PAULI_I) / np.linalg.norm(pm.PAULI_I)     # нормируем вектор средних спинов
rho0 = 0.5 * (np.eye(2, dtype=complex)
              + pauli_i[0] * sx
              + pauli_i[1] * sy
              + pauli_i[2] * sz)     # начальное состояние

Omega       = pm.DELTA ** 2 + pm.H_FIELD ** 2  # частота спиновой прецессии
sqrt_Omega  = np.sqrt(Omega)

# Коэффициенты c_z^k(t) — разложение σ_z^lr(t) по σ_x, σ_y, σ_z (ур. 18 из файла)
# с учётом комплексной частоты √(-Ω) = i√Ω:
#   cosh(t√(-Ω)) → cos(t√Ω)
#   sinh(t√(-Ω)) → i·sin(t√Ω),  √(-Ω) → i√Ω
#
# c_z^x = (Δh·cosh(t√(-Ω)) - Δh) / Ω  →  Δh(cos(t√Ω) - 1) / Ω
# c_z^y = Δ·√(-Ω)·sinh(t√(-Ω)) / Ω   →  -Δ·sin(t√Ω) / √Ω
# c_z^z = (Δ²·cosh(t√(-Ω)) + h²) / Ω →  (Δ²·cos(t√Ω) + h²) / Ω
def c_coef(t: float) -> np.ndarray:
    """Вектор коэффициентов c_z^k(t) для k = x, y, z."""
    if Omega == 0.0:
        return np.array([0, 0, 1])  # в случае Ω=0, c_z^z(t) = 1, остальные 0

    return np.array([
        pm.DELTA * pm.H_FIELD * (np.cos(t * sqrt_Omega) - 1) / Omega,     # c_z^x
        -pm.DELTA * np.sin(t * sqrt_Omega) / sqrt_Omega,               # c_z^y
        (pm.DELTA**2 * np.cos(t * sqrt_Omega) + pm.H_FIELD**2) / Omega,   # c_z^z
    ])

#======================================================================#
# Корреляционная функция среды
#C(t) = ∫ dω J(ω) [(2N(ω)+1) cos(ωt) - i sin(ωt)]
#======================================================================#

t_arr       = np.linspace(0, pm.T_MAX, pm.T_POINTS)         # сетка по времени
dt          = t_arr[1] - t_arr[0]                           # шаг по времени
c_coef_arr  = np.array([c_coef(t) for t in t_arr])

# Сетка по частоте и предвычесленные значения J(ω) и N(ω) для всех точек сетки
omega_arr   = np.linspace(0.1, pm.OMEGA_MAX, pm.OMEGA_POINTS)
N0          = np.array([N_I(w) for w in omega_arr])
d_omega     = omega_arr[1] - omega_arr[0]       # шаг по частоте

# J принимает массив omega (все операции — numpy), поэтому вызываем один раз на шаг
# g2_modes_t shape: (T_POINTS, OMEGA_POINTS)
g2_modes_t  = np.array([J(omega_arr, t) * d_omega for t in t_arr])

_w_fine     = np.linspace(0.1, pm.OMEGA_MAX, 5000)

# J_fine_t shape: (T_POINTS, 5000) — тонкая сетка для корреляционного интеграла
J_fine_t    = np.array([J(_w_fine, t) for t in t_arr])

# ---- Предвычисление C0_mat[j, k] = ∫ J(ω,t_j)·(cos(ω·k·dt) − i·sin(ω·k·dt)) dω ----
# C(τ,N) = C0(τ) + 2·C_N(τ,N), где C_N считается на грубой сетке с актуальным N_vec.
# C0 считается один раз через одно матричное умножение BLAS: (T×5000)@(5000×T).
print("Precomputing C0_mat ...")
_dw_f         = _w_fine[1] - _w_fine[0]
_tau_disc     = np.arange(pm.T_POINTS) * dt                               # (T_POINTS,)
_phase_disc   = np.outer(_w_fine, _tau_disc)                               # (5000, T_POINTS)
_ker_disc     = np.cos(_phase_disc) - 1j * np.sin(_phase_disc)            # (5000, T_POINTS)
# Трапециевидные веса по ω
_trap_w       = np.full(len(_w_fine), _dw_f)
_trap_w[0]    = _dw_f / 2
_trap_w[-1]   = _dw_f / 2
# Одно умножение матриц (BLAS): (T_POINTS × 5000) @ (5000 × T_POINTS)
C0_mat        = (J_fine_t * _trap_w[None, :]) @ _ker_disc                 # (T_POINTS, T_POINTS)
del _phase_disc, _ker_disc                                                  # освобождаем ~80 МБ
print("C0_mat done.")


def C_corr(tau: float, N_arr: np.ndarray, step: int = 0) -> complex:
    """Корреляционная функция среды C(tau)."""
    N_interp = np.interp(_w_fine, omega_arr, N_arr)  # интерполяция N(ω) на более тонкую сетку
    cos_term = (2 * N_interp + 1) * np.cos(_w_fine * tau)
    sin_term = -1j * np.sin(_w_fine * tau)

    J_t = J_fine_t[step]
    integrand = J_t * (cos_term + sin_term)
    return simpson(integrand, _w_fine)

#======================================================================#
# Подинтегралное выражение для правой части уравнения для спина.
#======================================================================#
def S(t: float) -> np.ndarray:
    """Функция S(t) для уравнения Линблада."""
    c = c_coef(t)
    return c[0] * sx + c[1] * sy + c[2] * sz

S_arr = np.array([S(t) for t in t_arr])

rh_labmda = pm.LAMBDA   # Отдельная переменная для лямбды
@nb.njit(cache=True, parallel=True)
def _rhssg_kernel(St: np.ndarray, St1_arr: np.ndarray, Ct_arr: np.ndarray,
                  rho: np.ndarray, t1_arr: np.ndarray) -> np.ndarray:
    n = len(t1_arr)
    integrand = np.zeros((n, 4), dtype=np.complex128)
    # Параллельный цикл по истории — каждая итерация пишет в свою строку
    for i in nb.prange(n):
        comm1 = St @ St1_arr[i] @ rho - St1_arr[i] @ rho @ St
        comm2 = St @ rho @ St1_arr[i] - rho @ St1_arr[i] @ St
        m = -rh_labmda ** 2 * (Ct_arr[i] * comm1 - np.conj(Ct_arr[i]) * comm2)
        integrand[i] = m.ravel()
    result = np.zeros(4, dtype=np.complex128)
    for j in range(4):
        result[j] = np.trapz(integrand[:, j], t1_arr)
    return result.reshape(2, 2)


# def RHSsg(rho: qt.Qobj, t: float, t1_arr: np.array, ct_arr: np.array) -> qt.Qobj:
#     """Правая часть уравнения для спина."""
#     if len(t1_arr) < 2:
#         return qt.Qobj(np.zeros((2, 2)))  # если нет истории, возвращаем нулевую матрицу

#     St          = S(t)                                        # оператор S(t) для текущего времени
#     integrand   = np.zeros((len(t1_arr), 4), dtype=complex)   # вектор для хранения подинтегрального выражения для каждого t1

#     for i, t1 in enumerate(t1_arr):
#         St1 = S(t1)
#         Ct = ct_arr[i]  # C(t - t1) для текущего t1
#         comm1 = St * St1 * rho - St1 * rho * St
#         comm2 = St * rho * St1 - rho * St1 * St
#         integrand[i] = (-pm.LAMBDA ** 2 * (Ct * comm1 - np.conj(Ct) * comm2)).full().flatten()

#     return qt.Qobj(np.trapz(integrand, t1_arr, axis=0).reshape((2, 2)))  # численное интегрирование по t1


#======================================================================#
# Уравнение для N(omega, t) (ур. 37 из файла)
#======================================================================#

_lam2 = pm.LAMBDA ** 2   # λ² — константа для numba-ядра

@nb.njit(cache=True, parallel=True)
def _RHSn_kernel(N_vec: np.ndarray, sigma_vec: np.ndarray,
                 ct: np.ndarray, cl: np.ndarray,
                 t1_arr: np.ndarray, omega_arr: np.ndarray,
                 g2_t: np.ndarray, t_cur: float, lam2: float) -> np.ndarray:
    """
    JIT-ядро для правой части уравнения на N(ω).

    Замены vs numpy-версия:
      einsum 'l,nl->n'          → cl @ ct          (матрично-векторное)
      einsum 'nl,k,lkg,g->n'   → cl @ cross(ct,σ) (тензор Л.-Ч. = cross-product)
      np.outer + cos/sin        → скалярные cos/sin в параллельном цикле по ω
      np.trapz axis=1           → ручная трапеция внутри цикла по ω
    """
    n_omega = len(omega_arr)
    n       = len(t1_arr)

    # Эквивалент einsum'ов
    sum_cc  = cl @ ct                    # (n,)  ←  'l,nl->n'
    cross   = np.cross(ct, sigma_vec)    # (3,)  ← ε_{lkg} ct_k σ_g
    sum_eps = cl @ cross                 # (n,)

    # Параллельный цикл по модам ω — каждая независима
    result = np.zeros(n_omega)
    for k in nb.prange(n_omega):
        wk     = omega_arr[k]
        Nk2p1  = 2.0 * N_vec[k] + 1.0
        s      = 0.0
        for i in range(n - 1):
            tau0 = t_cur - t1_arr[i]
            tau1 = t_cur - t1_arr[i + 1]
            dt_  = t1_arr[i + 1] - t1_arr[i]
            f0   = (np.cos(wk * tau0) * sum_cc[i]
                    - np.sin(wk * tau0) * Nk2p1 * sum_eps[i])
            f1   = (np.cos(wk * tau1) * sum_cc[i + 1]
                    - np.sin(wk * tau1) * Nk2p1 * sum_eps[i + 1])
            s   += 0.5 * (f0 + f1) * dt_
        result[k] = s

    return 2.0 * lam2 * g2_t * result


def RHSn(N_vec: np.ndarray, sigma_vec: np.ndarray,
         t: float, t1_arr: np.ndarray, c_arr: np.ndarray) -> np.ndarray:
    """Правая часть уравнения для всех мод N(omega)."""
    if len(t1_arr) < 2:
        return np.zeros(pm.OMEGA_POINTS)
    ct       = c_coef(t)
    step_idx = np.argmin(np.abs(t_arr - t))
    return _RHSn_kernel(N_vec, sigma_vec, ct, c_arr,
                        t1_arr, omega_arr, g2_modes_t[step_idx], t, _lam2)


#======================================================================#
# Параметры TR-BDF2 для решения системы с памятью
#======================================================================#
# Параметр метода γ = 2 - √2
# Стадия 1: (трапеция до t + γ * dt)
#   ρ¹ = ρⁿ + (γ*dt/2) * (F(tⁿ, ρⁿ) + F(tⁿ + γ*dt, ρ¹))
# Стадия 2 (BDF2 до t + dt):
#   ρⁿ⁺¹ = (1/(γ*(2-γ))) * ρ¹ - ((1-γ)²/(γ*(2-γ))) * ρⁿ
#           + (dt*(1-γ)/(2-γ)) * F(tⁿ + dt, ρⁿ⁺¹)
# Неявные стадии решаются итерациями (fixed-point), которые
# сходятся при λ²*dt << 1.
#======================================================================#
GAMMA_TRBDF2    = 2 - np.sqrt(2)  # оптимальный параметр для TR-BDF2
MAX_ITER        = 10              # максимальное число итераций для решения неявных стадий
TOL             = 1e-6            # критерий сходимости для итераций

def normal_rho(rho: np.ndarray) -> np.ndarray:
    """Нормирует и симметризует матрицу плотности, гарантируя эрмитовость и единичную след."""
    rho_temp = (rho + rho.conj().T) / 2          # симметризация: (ρ + ρ†)/2
    rho_temp = rho_temp / np.trace(rho_temp)      # нормировка следа
    eigvals, eigvecs    = np.linalg.eigh(rho_temp)
    eigvals             = np.maximum(eigvals.real, 0)                                   # обнуляем отрицательные собственные значения
    eigvals             = eigvals / eigvals.sum()                                       # нормируем собственные значения
    rho_new             = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_new


#======================================================================#
# Численное решение системы уравнений для спина
#======================================================================#
def trbdf2_rho_step(rho: np.ndarray, step: int, dt: float, N_vec: np.ndarray) -> np.ndarray:
    """
    Один шаг метода TR-BDF2 для матрицы плотности ρ.
    step — индекс текущего момента t_n = t_arr[step].
    Возвращает ρ на следующем временном шаге t_{n+1} = t_arr[step+1].
    """
    g  = GAMMA_TRBDF2
    j2 = min(step + 1, pm.T_POINTS - 1)

    t1_arr_cur = t_arr[:step + 1]
    t1_mid     = np.append(t1_arr_cur, t_arr[step] + g * dt)
    t1_end     = np.append(t1_mid,     t_arr[step] + dt)

    # Веса J(ω,t)·N(ω,t)·dω на грубой сетке — несут всю физику обратной связи N→C
    gN_curr = g2_modes_t[step]                                 * N_vec  # (OMEGA_POINTS,)
    gN_mid  = ((1-g)*g2_modes_t[step] + g*g2_modes_t[j2])    * N_vec
    gN_end  = g2_modes_t[j2]                                   * N_vec

    def _batch_C(tau_arr: np.ndarray, c0_row: np.ndarray, gN: np.ndarray) -> np.ndarray:
        """
        C(τ) = C0(τ) + 2·C_N(τ, N).
        C0: O(n_tau)          — линейная интерп. из C0_mat (предвычислен).
        C_N: O(500 · n_tau)   — грубая сетка с текущим N_vec (физика сохранена).
        Итого ~10× быстрее исходного O(5000 · n_tau).
        """
        k_frac = np.clip(tau_arr / dt, 0.0, pm.T_POINTS - 1 - 1e-10)
        k0     = k_frac.astype(int)
        k1     = np.minimum(k0 + 1, pm.T_POINTS - 1)
        C0     = (1.0 - (k_frac - k0)) * c0_row[k0] + (k_frac - k0) * c0_row[k1]
        C_N    = 2.0 * (gN @ np.cos(np.outer(omega_arr, tau_arr)))   # (n_tau,)
        return C0 + C_N

    Ct_curr = _batch_C(t_arr[step]      - t1_arr_cur, C0_mat[step],                        gN_curr)
    Ct_mid  = _batch_C(t_arr[step]+g*dt - t1_mid,     (1-g)*C0_mat[step]+g*C0_mat[j2],    gN_mid)
    Ct_end  = _batch_C(t_arr[step]+dt   - t1_end,      C0_mat[j2],                         gN_end)

    # S-операторы
    St          = S_arr[step]
    St2         = S_arr[step + 1]
    St1         = (1 - g) * S_arr[step] + g * S_arr[step + 1]
    St1_arr_cur = S_arr[:step + 1]
    St1_mid     = np.concatenate([St1_arr_cur, St1.reshape(1, 2, 2)], axis=0)
    St1_end     = np.concatenate([St1_mid,     St2.reshape(1, 2, 2)], axis=0)

    F_cur = _rhssg_kernel(St, St1_arr_cur, Ct_curr, rho, t1_arr_cur)

    # --- Стадия 1: трапеция до t + γ·dt ---
    rho1 = rho + g * dt * F_cur
    for _ in range(MAX_ITER):
        F_mid    = _rhssg_kernel(St1, St1_mid, Ct_mid, rho1, t1_mid)
        rho1_new = normal_rho(rho + (g * dt / 2) * (F_cur + F_mid))
        if np.linalg.norm(rho1_new - rho1) < TOL:
            rho1 = rho1_new
            break
        rho1 = rho1_new

    # --- Стадия 2: BDF2 до t + dt ---
    a1 = 1.0 / (g * (2.0 - g))
    a2 = (1.0 - g) ** 2 / (g * (2.0 - g))
    a3 = dt * (1.0 - g) / (2.0 - g)

    rho_new = normal_rho(a1 * rho1 - a2 * rho)
    for _ in range(MAX_ITER):
        F_end        = _rhssg_kernel(St2, St1_end, Ct_end, rho_new, t1_end)
        rho_new_next = normal_rho(a1 * rho1 - a2 * rho + a3 * F_end)
        if np.linalg.norm(rho_new_next - rho_new) < TOL:
            rho_new  = rho_new_next
            break
        rho_new = rho_new_next

    return rho_new

#======================================================================#
# TR-BDF2 для N(ω) на мелкой подсетке
#======================================================================#
def trdf2_N_substep(N_vec: np.ndarray, sigma_vec: np.ndarray,
                     t_sub: float, dt_sub: float, t1_arr: np.ndarray, c_arr: np.ndarray) -> np.ndarray:
    """Один шаг метода TR-BDF2 для вектора N(ω).
    sigma_vec - интерполированные средние спины для текущего подшага.
    """
    g = GAMMA_TRBDF2
    t1 = t_sub + g * dt_sub
    t2 = t_sub + dt_sub

    t1_mid  = np.append(t1_arr, [t_sub, t1])                      # История до t_sub
    c_mid   = np.vstack((c_arr, c_coef(t_sub), c_coef(t1)))       # История c до t_sub
    t1_end  = np.append(t1_mid, t2)
    c_end   = np.vstack((c_mid, c_coef(t2))) # История c до t1

    F_cur = RHSn(N_vec, sigma_vec, t_sub, t1_arr, c_arr)

    # --- Стадия 1: трапеция относительно N_1 ---
    N1 = N_vec + g * dt_sub * F_cur  # Начальное приближение для N_1
    for _ in range(MAX_ITER):
        F_mid = RHSn(N1, sigma_vec, t1, t1_mid, c_mid)
        N1_new = np.maximum(N_vec + (g * dt_sub / 2) * (F_cur + F_mid), 0)  # гарантируем неотрицательность
        if np.max(np.abs(N1_new - N1)) < TOL:
            N1 = N1_new
            break
        N1 = N1_new

    # --- Стадия 2: BDF2 относительно N_{n+1} ---
    a1 = 1.0 / (g * (2.0 - g))
    a2 = (1.0 - g) ** 2 / (g * (2.0 - g))
    a3 = (1.0 - g) / (2.0 - g)

    N_new = np.maximum(a1 * N1 - a2 * N_vec, 0)  # Начальное приближение для N_{n+1}
    for _ in range(MAX_ITER):
        F_end = RHSn(N_new, sigma_vec, t2, t1_end, c_end)
        N_next = np.maximum(a1 * N1 - a2 * N_vec + a3 * dt_sub * F_end, 0)  # гарантируем неотрицательность
        if np.max(np.abs(N_next - N_new)) < TOL:
            N_new = N_next
            break
        N_new = N_next

    return N_new

def solve():
    # Подсетка для N(ω) с большим числом точек для лучшей точности интегрирования по ω
    dt_sub_max = np.pi / pm.OMEGA_MAX
    N_SUB = max(1, int(np.ceil(dt / dt_sub_max)) + 1)  # число подшагов для N(ω)
    dt_sub = dt / N_SUB

    print(f"  TR-BDF2: dt={dt:.4f}, dt_sub={dt_sub:.4f}, N_SUB={N_SUB}")
    print(f"  γ_TRBDF2={GAMMA_TRBDF2:.4f}, TOL={TOL}, MAX_ITER={MAX_ITER}\n")

    # начальные условия
    rho         = rho0.copy()
    N_vec       = N0.copy()
    sigma_vec   = np.array([expect(op, rho) for op in pauli])
    # c_coef_arr уже предвычислен — используем срезы, без vstack на каждом шаге

    # Текущие значения
    rho_shcr_0  = to_shrodinger(rho, 0)           # начальное состояние в представлении Шредингера
    sx_t        = [expect(sx, rho_shcr_0)]
    sy_t        = [expect(sy, rho_shcr_0)]
    sz_t        = [expect(sz, rho_shcr_0)]
    pur_t       = [np.real(np.trace(rho_shcr_0 @ rho_shcr_0))]
    N_t         = [N_vec.copy()]

    print("Начало решения...")
    print(f"  Δ={pm.DELTA}, h={pm.H_FIELD}, λ={pm.LAMBDA}, γ={pm.GAMMA}, T={pm.TEMP}")
    print(f"  Мод окружения: {pm.OMEGA_POINTS}, шагов по времени: {pm.T_POINTS}\n")

    for step in range(1, pm.T_POINTS):
        # step-1 — индекс текущего t_n, step — индекс t_{n+1}
        t = t_arr[step - 1]

        # --- Шаг TR-BDF2 для ρ_S (использует S_arr и t_arr через step) ---
        rho_new = trbdf2_rho_step(rho, step - 1, dt, N_vec)

        sigma_vec       = np.array([expect(op, rho)     for op in pauli])   # средние спины на текущем шаге
        sigma_vec_new   = np.array([expect(op, rho_new) for op in pauli])

        # --- Подшаг TR-BDF2 для N(ω) (работает на подсетке, float-based) ---
        N_sub    = N_vec.copy()
        t1_hist  = t_arr[:step]               # история времён (основная сетка до t_n)
        c_hist   = c_coef_arr[:step]          # соответствующие c-коэффициенты (срез, без vstack)

        for sub in range(N_SUB):
            t_sub = t + sub * dt_sub
            frac = (sub + 0.5) / N_SUB
            sigma_interp = (1 - frac) * sigma_vec + frac * sigma_vec_new
            N_sub = trdf2_N_substep(
                N_sub, sigma_interp,
                t_sub, dt_sub,
                t1_hist, c_hist
            )

        # --- Обновление значений ---
        rho   = rho_new
        N_vec = N_sub

        rho_schr = to_shrodinger(rho, t_arr[step])

        sx_t.append(expect(sx, rho_schr))
        sy_t.append(expect(sy, rho_schr))
        sz_t.append(expect(sz, rho_schr))
        pur_t.append(np.real(np.trace(rho_schr @ rho_schr)))
        N_t.append(N_vec.copy())

        if step % (pm.T_POINTS // 10) == 0:
            print(f"  t={t_arr[step]:.2f}/{pm.T_MAX}  "
                  f"<sz>={expect(sz, rho_schr):+.4f}  "
                  f"Tr(ρ²)={pur_t[-1]:.4f}")


    print("Решение завершено.")
    return (t_arr,
            np.array(sx_t), np.array(sy_t), np.array(sz_t),
            np.array(pur_t), np.array(N_t))


#======================================================================#
# унитарная динамика для сравнения
#======================================================================#
def solve_unitary():
    # H = -Δ/2·σx + h/2·σz  — константный гамильтониан
    # exp(-iHt) = cos(||H||t)·I - i·sin(||H||t)·H/||H||
    # ||H|| = sqrt_Omega/2
    H = -(pm.DELTA / 2) * sx + (pm.H_FIELD / 2) * sz
    pi_arr = np.array(pm.PAULI_I, dtype=float)
    pi_arr = pi_arr / np.linalg.norm(pi_arr)
    rho0_u = 0.5 * (np.eye(2, dtype=complex)
                    + pi_arr[0] * sx
                    + pi_arr[1] * sy
                    + pi_arr[2] * sz)
    t_u    = np.linspace(0, pm.T_MAX, pm.T_POINTS)
    norm_H = sqrt_Omega / 2                          # ||H||
    H_hat  = H / norm_H if norm_H > 0 else H        # единичное направление

    # Векторизованный расчёт exp(-iHt) для всех t сразу
    c_t = np.cos(norm_H * t_u)   # (T_POINTS,)
    s_t = np.sin(norm_H * t_u)   # (T_POINTS,)
    # U(t) = c(t)·I - i·s(t)·H_hat
    # rho(t) = U·rho0·U†  →  <σ> = Tr(σ·rho(t))
    sx_u_t = np.empty(pm.T_POINTS)
    sy_u_t = np.empty(pm.T_POINTS)
    sz_u_t = np.empty(pm.T_POINTS)
    for i in range(pm.T_POINTS):
        U     = c_t[i] * np.eye(2, dtype=complex) - 1j * s_t[i] * H_hat
        rho_t = U @ rho0_u @ U.conj().T
        sx_u_t[i] = expect(sx, rho_t)
        sy_u_t[i] = expect(sy, rho_t)
        sz_u_t[i] = expect(sz, rho_t)

    return t_u, sx_u_t, sy_u_t, sz_u_t

#======================================================================#
# запуск решения и построение графиков
#======================================================================#
def plot_results(t, sx_d, sy_d, sz_d, purity, N_t,
                 t_u, sx_u, sy_u, sz_u, file_path: str = "."):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"Spin-Boson Model  |  "
        f"$\\Delta={pm.DELTA}$, $h={pm.H_FIELD}$, $\\lambda={pm.LAMBDA}$, "
        f"$\\kappa={pm.k}$, $\\omega_0={pm.w0}$, $T_{{env}}={pm.TEMP}$\n"
        r"$J(\omega,t)=\eta\,\omega^s\,\omega_c^{1-s}e^{-\omega/\omega_c}"
        r"+\frac{g^2}{\pi}\frac{\kappa}{(\omega-\omega_0(t))^2+\kappa^2}$",
        fontsize=12
    )
    gs = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    # <sigma_x,y,z>
    for idx, (yd, yu, label, col) in enumerate([
        (sx_d, sx_u, r"$\langle\sigma_x\rangle$", "C0"),
        (sy_d, sy_u, r"$\langle\sigma_y\rangle$", "C1"),
        (sz_d, sz_u, r"$\langle\sigma_z\rangle$", "C2"),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(t_u, yu, "--", color="grey", lw=1.4, label="No bath")
        ax.plot(t,   yd, "-",  color=col,   lw=2.0, label="TCL2")
        ax.set_xlabel("$t$"); ax.set_ylabel(label)
        ax.set_xlim(0, pm.T_MAX)
        ax.set_title(label); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Purity
    ax_pur = fig.add_subplot(gs[1, 0])
    ax_pur.plot(t, purity, color="C3", lw=2)
    ax_pur.axhline(1.0, ls="--", color="black", lw=1, label="pure")
    ax_pur.axhline(0.5, ls="--", color="grey",  lw=1, label="mixed")
    ax_pur.set_xlabel("$t$"); ax_pur.set_ylabel(r"$\mathrm{Tr}(\rho_S^2)$")
    ax_pur.set_title("Purity"); ax_pur.set_xlim(0, pm.T_MAX); ax_pur.set_ylim(0, 1.05)
    ax_pur.legend(fontsize=7); ax_pur.grid(alpha=0.3)

    # J(omega) at t=0
    ax_J = fig.add_subplot(gs[1, 1])
    w_pl = np.linspace(0.01, pm.OMEGA_MAX, 500)
    ax_J.plot(w_pl, [J(w, 0.0) for w in w_pl], color="C4", lw=2)
    ax_J.set_xlabel(r"$\omega$"); ax_J.set_ylabel(r"$J(\omega,\,t{=}0)$")
    ax_J.set_title("Spectral density $J(\\omega, t{=}0)$"); ax_J.grid(alpha=0.3); ax_J.set_xlim(0, pm.OMEGA_MAX)
    ax_J.set_ylim(0, None)

    # Heatmap N(omega, t) — near the Lorentzian peak
    N_arr      = np.array(N_t).T   # (OMEGA_POINTS, T_POINTS)
    zoom_width = 1 * pm.k
    # Use the same w_shift formula as in J(omega, t)
    # w0t        = pm.w0 + 0.3 * np.sin(pm.DELTA * t_arr)
    w0t        = sqrt_Omega
    maxw       = max(np.max(w0t) + zoom_width, sqrt_Omega + zoom_width)
    minw       = max(min(abs(np.min(w0t) - zoom_width), abs(sqrt_Omega - zoom_width)), 0)
    mask       = (omega_arr >= minw) & (omega_arr <= maxw)
    ax_Nmap    = fig.add_subplot(gs[1, 2])
    im = ax_Nmap.pcolormesh(t, omega_arr[mask], np.log1p(N_arr[mask, :]),
                             shading="auto", cmap="inferno")
    fig.colorbar(im, ax=ax_Nmap, label=r"$\ln(1+\langle N(\omega,t)\rangle)$")
    # ax_Nmap.plot(t_arr, w0t, color="cyan", lw=1.2, ls="--",
    #              label=rf"$\omega_0(t)$, $\omega_0={pm.w0:.3f}$")
    ax_Nmap.axhline(sqrt_Omega, color="lime", lw=1.0, ls=":",
                    label=rf"$\sqrt{{\Omega}}={sqrt_Omega:.3f}$")
    ax_Nmap.set_xlabel("$t$"); ax_Nmap.set_ylabel(r"$\omega$")
    ax_Nmap.set_xlim(0, pm.T_MAX)
    ax_Nmap.set_title(rf"$\langle N(\omega,t)\rangle$ ($\pm1.5\kappa$ around $\omega_0$)")
    ax_Nmap.legend(fontsize=7)

    # N(omega_m, t) — modes near spin resonance
    ax_N = fig.add_subplot(gs[2, 0:2])
    idx_res = np.argmin(np.abs(omega_arr - sqrt_Omega))   # index closest to spin frequency
    half = 3                                               # ±3 modes around resonance
    idx_lo = max(0, idx_res - half)
    idx_hi = min(pm.OMEGA_POINTS - 1, idx_res + half + 1)
    selected = range(idx_lo, idx_hi)
    for m in selected:
        wm = omega_arr[m]
        lw = 2.4 if m == idx_res else 1.4
        ls = "-"  if m == idx_res else "--"
        ax_N.plot(t, N_arr[m], lw=lw, ls=ls,
                  label=rf"$\omega={wm:.3f}$, $N_0={N0[m]:.3f}$"
                        + (r" $\leftarrow$ res." if m == idx_res else ""))
    ax_N.set_xlabel("$t$"); ax_N.set_ylabel(r"$\langle N(\omega_m,t)\rangle$")
    ax_N.set_xlim(0, pm.T_MAX)
    ax_N.set_title(rf"Mode occupation near resonance $\sqrt{{\Omega}}={sqrt_Omega:.3f}$")
    ax_N.legend(fontsize=7); ax_N.grid(alpha=0.3)
    ax_N.set_ylim(0)

    # Bath correlation function C(tau)
    ax_C = fig.add_subplot(gs[2, 2])
    tau_pl = np.linspace(0, 10, 300)
    C_vals = np.array([C_corr(tau, N0, step=0) for tau in tau_pl])
    ax_C.plot(tau_pl, C_vals.real, label=r"Re $C(\tau)$", color="C5")
    ax_C.plot(tau_pl, C_vals.imag, label=r"Im $C(\tau)$", color="C6", ls="--")
    ax_C.set_xlabel(r"$\tau$"); ax_C.set_ylabel(r"$C(\tau)$")
    ax_C.set_title("Bath correlation function")
    ax_C.legend(fontsize=7); ax_C.grid(alpha=0.3); ax_C.set_xlim(0, tau_pl[-1])

    plt.savefig(os.path.join(file_path, "spin_boson_results.png"),
                dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {os.path.join(file_path, 'spin_boson_results.png')}")
    plt.show()

if __name__ == "__main__":
    #======================================================================#
    # Куда сохранять картинки
    #======================================================================#
    with open(".env", "r") as f:
        file_path = f.readline().strip().split(": ", maxsplit=1)[-1]

    time_stamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
    file_type = "numerical"
    file_path = os.path.join(file_path, file_type, time_stamp)
    os.makedirs(file_path, exist_ok=True)

    # Save of functions
    with open(os.path.join(file_path, "functions.txt"), "w") as f:
        f.write(inspect.getsource(J))
        f.write("\n\n")
        f.write(inspect.getsource(N_I))

    # Save params
    shutil.copy("params.py", os.path.join(file_path, "params.py"))


    t, sx_d, sy_d, sz_d, purity, N_t = solve()
    t_u, sx_u, sy_u, sz_u = solve_unitary()
    plot_results(t, sx_d, sy_d, sz_d, purity, N_t,
                 t_u, sx_u, sy_u, sz_u, file_path)
