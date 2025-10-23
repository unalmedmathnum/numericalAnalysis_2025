"""
Implementación del método del gradiente descendente para resolver 
sistemas simétricos definidos positivos Ax = b.

El método del gradiente descendente es un algoritmo iterativo que minimiza
la función cuadrática f(x) = (1/2)x^T A x - b^T x, cuyo mínimo corresponde
a la solución del sistema lineal Ax = b cuando A es simétrica y definida positiva.

Autor: Grupo 2 - Análisis Numérico
"""

import numpy as np
from typing import Tuple, Optional


def gradiente_descendente(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False
) -> Tuple[np.ndarray, list, int]:
    """
    Resuelve el sistema Ax = b usando el método del gradiente descendente.
    
    Parámetros:
    -----------
    A : np.ndarray
        Matriz simétrica definida positiva de tamaño (n, n)
    b : np.ndarray
        Vector del lado derecho de tamaño (n,)
    x0 : np.ndarray, opcional
        Vector inicial. Si es None, se inicializa con ceros
    tol : float, opcional
        Tolerancia para el criterio de parada (norma del residuo)
    max_iter : int, opcional
        Número máximo de iteraciones
    verbose : bool, opcional
        Si es True, imprime información de progreso
        
    Retorna:
    --------
    x : np.ndarray
        Solución aproximada del sistema
    residuos : list
        Lista con la norma del residuo en cada iteración
    k : int
        Número de iteraciones realizadas
        
    Raises:
    -------
    ValueError
        Si A no es cuadrada o las dimensiones no son compatibles
    """
    
    # Validaciones
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")
    
    n = A.shape[0]
    
    if b.shape[0] != n:
        raise ValueError("Las dimensiones de A y b no son compatibles")
    
    # Inicialización
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    residuos = []
    
    # Iteraciones del gradiente descendente
    for k in range(max_iter):
        # Calcular el residuo: r = b - Ax
        r = b - A @ x
        
        # Calcular la norma del residuo
        norma_r = np.linalg.norm(r)
        residuos.append(norma_r)
        
        if verbose and k % 100 == 0:
            print(f"Iteración {k}: ||r|| = {norma_r:.6e}")
        
        # Criterio de parada
        if norma_r < tol:
            if verbose:
                print(f"\nConvergencia alcanzada en {k} iteraciones")
                print(f"Norma del residuo final: {norma_r:.6e}")
            return x, residuos, k
        
        # Calcular el tamaño de paso óptimo: alpha = (r^T r) / (r^T A r)
        Ar = A @ r
        alpha = (r.T @ r) / (r.T @ Ar)
        
        # Actualizar la solución: x = x + alpha * r
        x = x + alpha * r
    
    if verbose:
        print(f"\nSe alcanzó el número máximo de iteraciones ({max_iter})")
        print(f"Norma del residuo final: {residuos[-1]:.6e}")
    
    return x, residuos, max_iter


def gradiente_conjugado(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
    verbose: bool = False
) -> Tuple[np.ndarray, list, int]:
    """
    Resuelve el sistema Ax = b usando el método del gradiente conjugado.
    
    El gradiente conjugado es una mejora del gradiente descendente que utiliza
    direcciones conjugadas para acelerar la convergencia.
    
    Parámetros y retornos: iguales a gradiente_descendente()
    """
    
    # Validaciones
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada")
    
    n = A.shape[0]
    
    if b.shape[0] != n:
        raise ValueError("Las dimensiones de A y b no son compatibles")
    
    # Inicialización
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = b - A @ x  # Residuo inicial
    p = r.copy()    # Dirección de búsqueda inicial
    
    residuos = []
    
    # Iteraciones del gradiente conjugado
    for k in range(max_iter):
        # Calcular la norma del residuo
        norma_r = np.linalg.norm(r)
        residuos.append(norma_r)
        
        if verbose and k % 100 == 0:
            print(f"Iteración {k}: ||r|| = {norma_r:.6e}")
        
        # Criterio de parada
        if norma_r < tol:
            if verbose:
                print(f"\nConvergencia alcanzada en {k} iteraciones")
                print(f"Norma del residuo final: {norma_r:.6e}")
            return x, residuos, k
        
        # Calcular Ap
        Ap = A @ p
        
        # Calcular el tamaño de paso: alpha = (r^T r) / (p^T A p)
        r_dot_r = r.T @ r
        alpha = r_dot_r / (p.T @ Ap)
        
        # Actualizar la solución: x = x + alpha * p
        x = x + alpha * p
        
        # Actualizar el residuo: r_new = r - alpha * Ap
        r_new = r - alpha * Ap
        
        # Calcular beta para la nueva dirección conjugada
        beta = (r_new.T @ r_new) / r_dot_r
        
        # Actualizar la dirección de búsqueda: p = r_new + beta * p
        p = r_new + beta * p
        
        # Actualizar el residuo
        r = r_new
    
    if verbose:
        print(f"\nSe alcanzó el número máximo de iteraciones ({max_iter})")
        print(f"Norma del residuo final: {residuos[-1]:.6e}")
    
    return x, residuos, max_iter


def verificar_definida_positiva(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Verifica si una matriz es simétrica y definida positiva.
    
    Parámetros:
    -----------
    A : np.ndarray
        Matriz a verificar
    tol : float
        Tolerancia para verificar simetría
        
    Retorna:
    --------
    bool
        True si A es simétrica y definida positiva, False en caso contrario
    """
    # Verificar si es cuadrada
    if A.shape[0] != A.shape[1]:
        return False
    
    # Verificar simetría
    if not np.allclose(A, A.T, atol=tol):
        print("La matriz no es simétrica")
        return False
    
    # Verificar que todos los autovalores sean positivos
    try:
        eigenvalues = np.linalg.eigvalsh(A)
        if np.all(eigenvalues > 0):
            return True
        else:
            print(f"La matriz tiene autovalores no positivos: min(λ) = {np.min(eigenvalues)}")
            return False
    except np.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("MÉTODO DEL GRADIENTE DESCENDENTE")
    print("Resolución de sistemas simétricos definidos positivos")
    print("=" * 70)
    
    # Ejemplo 1: Sistema pequeño
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Sistema 3x3")
    print("=" * 70)
    
    # Definir una matriz simétrica definida positiva
    A1 = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0]
    ])
    
    b1 = np.array([1.0, 2.0, 3.0])
    
    print("\nMatriz A:")
    print(A1)
    print("\nVector b:")
    print(b1)
    
    # Verificar que A sea definida positiva
    print(f"\n¿A es simétrica y definida positiva? {verificar_definida_positiva(A1)}")
    
    # Resolver con gradiente descendente
    print("\n" + "-" * 70)
    print("Resolviendo con GRADIENTE DESCENDENTE:")
    print("-" * 70)
    x_gd, residuos_gd, iter_gd = gradiente_descendente(A1, b1, verbose=True)
    
    print("\nSolución obtenida:")
    print(x_gd)
    
    # Verificar la solución
    print("\nVerificación (Ax):")
    print(A1 @ x_gd)
    print("\nError ||Ax - b||:")
    print(np.linalg.norm(A1 @ x_gd - b1))
    
    # Comparar con solución exacta
    x_exacta = np.linalg.solve(A1, b1)
    print("\nSolución exacta (NumPy):")
    print(x_exacta)
    print("\nError ||x - x_exacta||:")
    print(np.linalg.norm(x_gd - x_exacta))
    
    # Resolver con gradiente conjugado
    print("\n" + "-" * 70)
    print("Resolviendo con GRADIENTE CONJUGADO:")
    print("-" * 70)
    x_gc, residuos_gc, iter_gc = gradiente_conjugado(A1, b1, verbose=True)
    
    print("\nSolución obtenida:")
    print(x_gc)
    print("\nError ||x - x_exacta||:")
    print(np.linalg.norm(x_gc - x_exacta))
    
    # Ejemplo 2: Sistema más grande
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Sistema 10x10")
    print("=" * 70)
    
    # Crear una matriz simétrica definida positiva de mayor tamaño
    n = 10
    np.random.seed(42)
    M = np.random.randn(n, n)
    A2 = M.T @ M + n * np.eye(n)  # A = M^T M + nI es definida positiva
    b2 = np.random.randn(n)
    
    print(f"\nMatriz A de tamaño {n}x{n}")
    print(f"Vector b de tamaño {n}")
    
    # Resolver con ambos métodos
    x_gd2, residuos_gd2, iter_gd2 = gradiente_descendente(
        A2, b2, tol=1e-8, max_iter=1000, verbose=False
    )
    x_gc2, residuos_gc2, iter_gc2 = gradiente_conjugado(
        A2, b2, tol=1e-8, max_iter=1000, verbose=False
    )
    x_exacta2 = np.linalg.solve(A2, b2)
    
    print("\nResultados:")
    print(f"Gradiente Descendente: {iter_gd2} iteraciones, error = {np.linalg.norm(x_gd2 - x_exacta2):.6e}")
    print(f"Gradiente Conjugado:   {iter_gc2} iteraciones, error = {np.linalg.norm(x_gc2 - x_exacta2):.6e}")
    
    # Graficar convergencia (opcional, requiere matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Comparación de convergencia
        ax1 = axes[0, 0]
        ax1.semilogy(residuos_gd2, 'b-', label='Gradiente Descendente', linewidth=2)
        ax1.semilogy(residuos_gc2, 'r--', label='Gradiente Conjugado', linewidth=2)
        ax1.set_xlabel('Iteración', fontsize=11)
        ax1.set_ylabel('Norma del residuo ||r||', fontsize=11)
        ax1.set_title('Convergencia: Gradiente Descendente vs Conjugado', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Convergencia del ejemplo 1 (3x3)
        ax2 = axes[0, 1]
        ax2.semilogy(residuos_gd, 'b-o', label='Gradiente Descendente', linewidth=2, markersize=4)
        ax2.semilogy(residuos_gc, 'r--s', label='Gradiente Conjugado', linewidth=2, markersize=4)
        ax2.set_xlabel('Iteración', fontsize=11)
        ax2.set_ylabel('Norma del residuo ||r||', fontsize=11)
        ax2.set_title('Convergencia: Sistema 3x3', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Razón de convergencia
        ax3 = axes[1, 0]
        if len(residuos_gd2) > 1:
            razon_gd = [residuos_gd2[i+1]/residuos_gd2[i] if residuos_gd2[i] > 0 else 0 
                        for i in range(len(residuos_gd2)-1)]
            ax3.plot(razon_gd, 'b-', label='Gradiente Descendente', linewidth=2)
        if len(residuos_gc2) > 1:
            razon_gc = [residuos_gc2[i+1]/residuos_gc2[i] if residuos_gc2[i] > 0 else 0 
                        for i in range(len(residuos_gc2)-1)]
            ax3.plot(razon_gc, 'r--', label='Gradiente Conjugado', linewidth=2)
        ax3.set_xlabel('Iteración', fontsize=11)
        ax3.set_ylabel('Razón ||r_{k+1}|| / ||r_k||', fontsize=11)
        ax3.set_title('Razón de Convergencia', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.2])
        
        # Subplot 4: Comparación de iteraciones vs tamaño del sistema
        ax4 = axes[1, 1]
        tamanios = [3, 5, 10, 15, 20]
        iters_gd = []
        iters_gc = []
        
        for n in tamanios:
            np.random.seed(42)
            M_test = np.random.randn(n, n)
            A_test = M_test.T @ M_test + n * np.eye(n)
            b_test = np.random.randn(n)
            
            _, _, iter_gd_test = gradiente_descendente(A_test, b_test, tol=1e-8, max_iter=1000, verbose=False)
            _, _, iter_gc_test = gradiente_conjugado(A_test, b_test, tol=1e-8, max_iter=1000, verbose=False)
            
            iters_gd.append(iter_gd_test)
            iters_gc.append(iter_gc_test)
        
        ax4.plot(tamanios, iters_gd, 'b-o', label='Gradiente Descendente', linewidth=2, markersize=8)
        ax4.plot(tamanios, iters_gc, 'r--s', label='Gradiente Conjugado', linewidth=2, markersize=8)
        ax4.set_xlabel('Tamaño del sistema (n)', fontsize=11)
        ax4.set_ylabel('Número de iteraciones', fontsize=11)
        ax4.set_title('Iteraciones vs Tamaño del Sistema', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('convergencia_gradiente.png', dpi=200, bbox_inches='tight')
        print("\nGráfica de convergencia guardada en: convergencia_gradiente.png")
        print("  - Subplot 1: Comparación de convergencia (sistema 10x10)")
        print("  - Subplot 2: Convergencia del sistema 3x3")
        print("  - Subplot 3: Razón de convergencia")
        print("  - Subplot 4: Iteraciones vs tamaño del sistema")
    except ImportError:
        print("\n(matplotlib no disponible para graficar)")
    
    print("\n" + "=" * 70)
    print("FIN DE LOS EJEMPLOS")
    print("=" * 70)

