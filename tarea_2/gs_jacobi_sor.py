"""
Implementación en Python de los métodos iterativos:
 - Jacobi
 - Gauss-Seidel
 - SOR

Las fórmulas y el criterio de parada siguen las notas de clase (semana_4.pdf / semana_5.pdf):
Criterio: ||x^(k) - x^(k-1)||_∞ / ||x^(k)||_∞ <= tol

Funciones principales:
 - jacobi(...)
 - gauss_seidel(...)
 - sor(...)

Cada función devuelve una tupla: (x, info) donde info es un dict con keys:
 - 'converged' (bool)
 - 'iterations' (int)
 - 'residual_norm' (float) -> ||Ax - b||_∞

Autor: Emilio Porras Mejía <eporrasm@unal.edu.co>
"""

from typing import Tuple, Dict, Optional
import numpy as np

def _check_matrix(A: np.ndarray, b: np.ndarray):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser una matriz cuadrada (n x n).")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("b debe ser un vector de longitud n (compatible con A).")
    return A, b

def _stop_criterion(x_new: np.ndarray, x_old: np.ndarray, tol: float) -> bool:
    # Criterio de parada (Semana 5, pág. 2):
    # ||x^(k) - x^(k-1)||_∞ / ||x^(k)||_∞ <= tol
    num = np.linalg.norm(x_new - x_old, ord=np.inf)
    den = np.linalg.norm(x_new, ord=np.inf)
    if den == 0.0:
        return num <= tol
    return (num / den) <= tol

def jacobi(A: np.ndarray,
           b: np.ndarray,
           x0: Optional[np.ndarray] = None,
           tol: float = 1e-8,
           maxiter: int = 1000) -> Tuple[np.ndarray, Dict]:
    """
    Método de Jacobi (Semana 4, pág. 4):
      x_i^(k) = (1 / a_ii) * ( - Σ_{j≠i} a_ij x_j^(k-1) + b_i )
    """
    A, b = _check_matrix(A, b)
    n = A.shape[0]
    x_old = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if np.any(np.diag(A) == 0):
        raise ValueError("A tiene elementos diagonales nulos; Jacobi no es aplicable.")

    x_new = np.zeros_like(x_old)

    for k in range(1, maxiter + 1):
        for i in range(n):
            aii = A[i, i]
            # Cálculo de la suma Σ_{j≠i} a_ij x_j^(k-1)
            s = np.dot(A[i, :], x_old) - aii * x_old[i]
            x_new[i] = (-s + b[i]) / aii

        if _stop_criterion(x_new, x_old, tol):
            res_norm = np.linalg.norm(A.dot(x_new) - b, ord=np.inf)
            return x_new, {'converged': True, 'iterations': k, 'residual_norm': float(res_norm)}

        # Preparar siguiente iteración: x^(k-1) ← x^(k)
        x_old[:] = x_new

    res_norm = np.linalg.norm(A.dot(x_new) - b, ord=np.inf)
    return x_new, {'converged': False, 'iterations': maxiter, 'residual_norm': float(res_norm)}

def gauss_seidel(A: np.ndarray,
                 b: np.ndarray,
                 x0: Optional[np.ndarray] = None,
                 tol: float = 1e-8,
                 maxiter: int = 1000) -> Tuple[np.ndarray, Dict]:
    """
    Método de Gauss-Seidel (Semana 4, pág. 5):
      x_i^(k) = (1 / a_ii) * ( - Σ_{j=1}^{i-1} a_ij x_j^(k)
                               - Σ_{j=i+1}^{n} a_ij x_j^(k-1)
                               + b_i )
    """
    A, b = _check_matrix(A, b)
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if np.any(np.diag(A) == 0):
        raise ValueError("A tiene elementos diagonales nulos; Gauss-Seidel no es aplicable.")

    for k in range(1, maxiter + 1):
        x_old = x.copy()
        for i in range(n):
            aii = A[i, i]
            # Suma sobre j < i → usa los nuevos valores x_j^(k)
            s1 = np.dot(A[i, :i], x[:i]) if i > 0 else 0.0
            # Suma sobre j > i → usa los valores anteriores x_j^(k-1)
            s2 = np.dot(A[i, i+1:], x_old[i+1:]) if i < n-1 else 0.0
            x[i] = (- (s1 + s2) + b[i]) / aii

        if _stop_criterion(x, x_old, tol):
            res_norm = np.linalg.norm(A.dot(x) - b, ord=np.inf)
            return x, {'converged': True, 'iterations': k, 'residual_norm': float(res_norm)}

    res_norm = np.linalg.norm(A.dot(x) - b, ord=np.inf)
    return x, {'converged': False, 'iterations': maxiter, 'residual_norm': float(res_norm)}

def sor(A: np.ndarray,
        b: np.ndarray,
        omega: float,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-8,
        maxiter: int = 1000) -> Tuple[np.ndarray, Dict]:
    """
    Método SOR (Semana 5, pág. 12):
      x_i^(k) = (1 - ω)x_i^(k-1)
                + (ω / a_ii) * ( - Σ_{j=1}^{i-1} a_ij x_j^(k)
                                 - Σ_{j=i+1}^{n} a_ij x_j^(k-1)
                                 + b_i )
    """
    if omega <= 0 or omega >= 2:
        import warnings
        warnings.warn("omega fuera de (0,2). SOR puede no converger (ver notas de clase).")

    A, b = _check_matrix(A, b)
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if np.any(np.diag(A) == 0):
        raise ValueError("A tiene elementos diagonales nulos; SOR no es aplicable.")

    for k in range(1, maxiter + 1):
        x_old = x.copy()
        for i in range(n):
            aii = A[i, i]
            # Parte de Gauss-Seidel interna (usa x_j^(k) para j<i y x_j^(k-1) para j>i)
            s1 = np.dot(A[i, :i], x[:i]) if i > 0 else 0.0
            s2 = np.dot(A[i, i+1:], x_old[i+1:]) if i < n-1 else 0.0
            gs_update = (- (s1 + s2) + b[i]) / aii
            x[i] = (1.0 - omega) * x_old[i] + omega * gs_update

        if _stop_criterion(x, x_old, tol):
            res_norm = np.linalg.norm(A.dot(x) - b, ord=np.inf)
            return x, {'converged': True, 'iterations': k, 'residual_norm': float(res_norm)}

    res_norm = np.linalg.norm(A.dot(x) - b, ord=np.inf)
    return x, {'converged': False, 'iterations': maxiter, 'residual_norm': float(res_norm)}

if __name__ == "__main__":
    # ================================================================
    # CASO 1: Happy Path (Diagonalmente dominante → Convergencia rápida)
    # ================================================================
    A = np.array([[5.0, 2.0, 1.0],
                  [2.0, 6.0, 1.0],
                  [1.0, 1.0, 4.0]])
    b = np.array([12.0, 19.0, 10.0])

    print("=== CASO 1: SISTEMA DIAGONALMENTE DOMINANTE ===")
    print("A =\n", A)
    print("b =", b)
    print()

    x_ref = np.linalg.solve(A, b)
    print("Solución exacta (np.linalg.solve):", x_ref, "\n")

    x_j, info_j = jacobi(A, b, tol=1e-10, maxiter=5000)
    print("Jacobi ->", info_j)

    x_gs, info_gs = gauss_seidel(A, b, tol=1e-10, maxiter=5000)
    print("Gauss-Seidel ->", info_gs)

    x_sor1, info_sor1 = sor(A, b, omega=1.0, tol=1e-10, maxiter=5000)
    print("SOR ω=1.0 ->", info_sor1)

    x_sor12, info_sor12 = sor(A, b, omega=1.2, tol=1e-10, maxiter=5000)
    print("SOR ω=1.2 ->", info_sor12)

    print()
    print("||x_j - x_ref||_∞ =", np.linalg.norm(x_j - x_ref, ord=np.inf))
    print("||x_gs - x_ref||_∞ =", np.linalg.norm(x_gs - x_ref, ord=np.inf))
    print("||x_sor(1.2) - x_ref||_∞ =", np.linalg.norm(x_sor12 - x_ref, ord=np.inf))
    print("\n" + "="*70 + "\n")

    # ================================================================
    # CASO 2: No diagonalmente dominante → Divergencia esperada
    # ================================================================
    print("=== CASO 2: MATRIZ NO DIAGONALMENTE DOMINANTE ===")
    A = np.array([[2.0, 3.0],
                  [3.0, 2.0]])
    b = np.array([5.0, 5.0])
    print("A =\n", A)
    print("b =", b, "\n")

    for w in [0.8, 1.0, 1.5]:
        x, info = sor(A, b, omega=w, tol=1e-10, maxiter=100)
        print(f"SOR ω={w}: converged={info['converged']}, iterations={info['iterations']}")

    print("\n" + "="*70 + "\n")

    # ================================================================
    # CASO 3: Elemento diagonal nulo → Error detectado
    # ================================================================
    print("=== CASO 3: ELEMENTO DIAGONAL NULO ===")
    A = np.array([[0.0, 2.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    print("A =\n", A)
    print("b =", b)
    try:
        x, info = jacobi(A, b)
    except ValueError as e:
        print("Error detectado:", e)

    print("\n" + "="*70 + "\n")

    # ================================================================
    # CASO 4: ω fuera del rango → Advertencia + posible divergencia
    # ================================================================
    print("=== CASO 4: SOR CON ω FUERA DE (0, 2) ===")
    A = np.array([[4.0, 1.0, 1.0],
                  [1.0, 3.0, 0.0],
                  [1.0, 0.0, 2.0]])
    b = np.array([6.0, 5.0, 4.0])
    print("A =\n", A)
    print("b =", b)
    x, info = sor(A, b, omega=2.5, tol=1e-10, maxiter=100)
    print("SOR ω=2.5 ->", info)

    print("\n" + "="*70 + "\n")

    # ================================================================
    # CASO 5: Matriz casi singular → Convergencia muy lenta
    # ================================================================
    print("=== CASO 5: MATRIZ CASI SINGULAR (CONDICIONAMIENTO MALO) ===")
    A = np.array([[1.0, 0.99],
                  [0.99, 0.98]])
    b = np.array([1.99, 1.97])
    print("A =\n", A)
    print("b =", b)
    x, info = gauss_seidel(A, b, tol=1e-10, maxiter=5000)
    print("Gauss-Seidel ->", info)
    print("Resultado aproximado:", x)

