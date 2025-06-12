import numpy as np
from math import sin, cos, pi, sqrt
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import CubicSpline
from scipy import integrate
import time

# Regla compuesta de integración (Simpson)
def intenumcomp(fun, a, b, N, regla="simpson"):
    S = 0.
    if regla.lower() == 'simpson':
        if N % 2 != 0:
            N += 1
        h = (b - a) / N
        xi0 = fun(a) + fun(b)
        xi1 = sum(fun(a + i * h) for i in range(1, N, 2))
        xi2 = sum(fun(a + i * h) for i in range(2, N, 2))
        S = (h / 3) * (xi0 + 4 * xi1 + 2 * xi2)
    return S

# Calcula coeficientes de mínimos cuadrados con LU
def least_squares_coefficients(log):
    a, b = 0, pi/2
    error_tol = 1e-5
    iterations = 0

    # Determina N tal que error < 10^{-5}
    def get_n_for_tol(fun_exact):
        nonlocal iterations
        n = 2
        while True:
            approx = intenumcomp(fun_exact, a, b, n, "simpson")
            exact, _ = integrate.quad(fun_exact, a, b)
            iterations += 1
            if abs(approx - exact) < error_tol:
                return n
            n += 2

    basis = [lambda x: 1, lambda x: x, lambda x: x**2]
    A = np.zeros((3, 3))
    b_vec = np.zeros(3)
    cycles = 0

    # Construye matriz A y vector b
    for i in range(3):
        for j in range(3):
            fun_ij = lambda x, fi=basis[i], fj=basis[j]: fi(x) * fj(x)
            n = get_n_for_tol(fun_ij)
            A[i, j] = intenumcomp(fun_ij, a, b, n, "simpson")
            cycles += 1
        fun_ib = lambda x, fi=basis[i]: fi(x) * sin(x)
        n = get_n_for_tol(fun_ib)
        b_vec[i] = intenumcomp(fun_ib, a, b, n, "simpson")
        cycles += 1

    lu, piv = lu_factor(A)
    coeffs = lu_solve((lu, piv), b_vec)

    log['lsq_iterations'] = iterations
    log['lsq_cycles'] = cycles
    return coeffs

# Grafica sin(x) y sus dos aproximaciones
def plot_approximations():
    log = {}
    start = time.time()

    a, b = 0, pi/2
    x_vals = np.linspace(a, b, 200)
    y_true = np.sin(x_vals)

    coeffs = least_squares_coefficients(log)
    y_ls = coeffs[0] + coeffs[1] * x_vals + coeffs[2] * x_vals**2

    # Interpolación spline con 5 puntos
    spline_start = time.time()
    x_spline = np.linspace(a, b, 5)
    y_spline = np.sin(x_spline)
    cs = CubicSpline(x_spline, y_spline)
    y_cubic = cs(x_vals)
    spline_time = time.time() - spline_start

    # Muestra todas las curvas
    plt.plot(x_vals, y_true, label='sin(x)', color='black')
    plt.plot(x_vals, y_ls, label='Min Cuadrados', linestyle='--')
    plt.plot(x_vals, y_cubic, label='Spline Cúbico', linestyle=':')
    plt.legend()
    plt.grid()
    plt.title(' sin(x) en [0, π/2]')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    end = time.time()

    # Informe final
    print("\n📊 Informe de Complejidad")
    print(f"- Tiempo total: {end - start:.4f} segundos")
    print(f"- Tiempo spline cúbico: {spline_time:.4f} segundos")
    print(f"- Iteraciones de integración (Simpson): {log['lsq_iterations']}")
    print(f"- Ciclos de construcción de A y b: {log['lsq_cycles']}")

# Ejecuta todo
plot_approximations()

