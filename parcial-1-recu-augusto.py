import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import time

# --------- 1. Serie de Taylor del seno ---------
def serie_seno(x):
    """Calcula los primeros 5 términos de la serie de Taylor de sin(x)"""
    return sum(((-1)**n) * (x**(2*n + 1)) / math.factorial(2*n + 1) for n in range(5))

# --------- Derivada conocida ---------
def derivada_serie_seno(x):
    """Derivada analítica de los primeros 5 términos de la serie de Taylor de sin(x)"""
    return sum(((-1)**n) * (2*n + 1) * (x**(2*n)) / np.math.factorial(2*n + 1) for n in range(5))

# --------- 2. Graficar f(x) = serie_seno(x) en [0, 6.4] ---------
def graficar_serie_seno():
    xs = np.arange(0, 6.4 + 0.01, 0.01)
    ys_serie = np.vectorize(serie_seno)(xs)
    ys_real = np.sin(xs)
    plt.plot(xs, ys_serie, label="Serie seno (5 términos)")
    plt.plot(xs, ys_real, '--', label="sin(x) real")
    plt.title("Comparación entre serie de Taylor y sin(x)")
    plt.xlabel("x")
    plt.ylabel("Valor de la función")
    plt.grid(True)
    plt.legend()
    plt.show()

# --------- 3. secante ---------
def rsecante(fun, x0, x1, err, mit):
    """metodo de la secante"""
    hx = [x0, x1]
    hf = [fun(x0), fun(x1)]
    for _ in range(mit):
        f0, f1 = hf[-2], hf[-1]
        if f1 - f0 == 0:
            break
        x2 = hx[-1] - f1 * (hx[-1] - hx[-2]) / (f1 - f0)
        fx2 = fun(x2)
        hx.append(x2)
        hf.append(fx2)
        if abs(fx2) < err:
            break
    return hx[2:], hf[2:] if len(hx) > 2 else (hx, hf)

# --------- 4. Newton ---------
def newton(fun, dfun, x0, err, mit):
    hx = [x0]
    hf = [fun(x0)]
    for _ in range(mit):
        fx = fun(x0)
        dfx = dfun(x0)
        if dfx == 0:
            break
        x1 = x0 - fx / dfx
        fx1 = fun(x1)
        hx.append(x1)
        hf.append(fx1)
        if abs(fx1) < err:
            break
        x0 = x1
    return hx, hf

# --------- 5. busqueda_ceros ---------
def busqueda_ceros(fun, x0, x1, err, mit):
    
    print("\n--- 🤖Buskando ceros🤖 ---")

    t0 = time.time()
    hx_newton, hf_newton = newton(fun, derivada_serie_seno, x0, err, mit)
    t1 = time.time()
    time_newton = t1 - t0

    t2 = time.time()
    hx_secante, hf_secante = rsecante(fun, x0, x1, err, mit)
    t3 = time.time()
    time_secante = t3 - t2
"""
    if len(hx_newton) > 0:
        print(f"Newton: raíz = {hx_newton[-1]:.6f}, iteraciones = {len(hx_newton)-1}, tiempo = {time_newton:.6f}s")
    else:
        print("Newton: no se encontro la raiz😠.")

    if len(hx_secante) > 0:
        print(f"Secante: raíz = {hx_secante[-1]:.6f}, iteraciones = {len(hx_secante)}, tiempo = {time_secante:.6f}s")
    else:
        print("Secante: no se encontro la raiz.")
    if len(hf_newton) > 0 and (len(hf_secante) == 0 or abs(hf_newton[-1]) < abs(hf_secante[-1])):
        return hx_newton[-1]
    elif len(hf_secante) > 0:
        return hx_secante[-1]
    else:
        return None
        """

# --------- 6. Pruebas finales ---------
if __name__ == "__main__":
    graficar_serie_seno()
    

    print("\nPrueba con puntos iniciales 3 y 6:")
    r1 = busqueda_ceros(serie_seno, 3, 6, 1e-5, 100)

    print("\nPrueba con punto inicial 4.5:")
    r2 = busqueda_ceros(serie_seno, 4.5, 4.5, 1e-5, 100)

    print("\nMejores raíces encontradas:")
    if r1 is not None:
        print(f"Raíz desde (3,6): {r1:.6f}")
 
    if r2 is not None:
        print(f"Raíz desde (4.5,4.5): {r2:.6f}")
