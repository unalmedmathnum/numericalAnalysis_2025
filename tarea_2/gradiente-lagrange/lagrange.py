"""
Implementación del polinomio de interpolación de Lagrange.

El polinomio de Lagrange es un método de interpolación que dado un conjunto
de n+1 puntos (x_0, y_0), (x_1, y_1), ..., (x_n, y_n), construye un polinomio
de grado ≤ n que pasa exactamente por todos esos puntos.

La forma del polinomio de Lagrange es:
    P(x) = Σ_{i=0}^{n} y_i * L_i(x)

donde L_i(x) son los polinomios base de Lagrange:
    L_i(x) = Π_{j=0, j≠i}^{n} (x - x_j) / (x_i - x_j)

Autor: Grupo 2 - Análisis Numérico
"""

import numpy as np
from typing import Callable, Tuple


def polinomio_lagrange_base(x_puntos: np.ndarray, i: int) -> Callable:
    """
    Calcula el i-ésimo polinomio base de Lagrange L_i(x).
    
    Parámetros:
    -----------
    x_puntos : np.ndarray
        Array con las coordenadas x de los puntos de interpolación
    i : int
        Índice del polinomio base (0 <= i < len(x_puntos))
        
    Retorna:
    --------
    Callable
        Función que evalúa L_i(x) para cualquier valor x
    """
    n = len(x_puntos)
    
    def L_i(x):
        """Evalúa el polinomio base L_i en x"""
        resultado = 1.0
        for j in range(n):
            if j != i:
                resultado *= (x - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
        return resultado
    
    return L_i


def interpolacion_lagrange(
    x_puntos: np.ndarray,
    y_puntos: np.ndarray
) -> Callable:
    """
    Construye el polinomio de interpolación de Lagrange.
    
    Parámetros:
    -----------
    x_puntos : np.ndarray
        Array con las coordenadas x de los puntos de interpolación
    y_puntos : np.ndarray
        Array con las coordenadas y de los puntos de interpolación
        
    Retorna:
    --------
    Callable
        Función que evalúa el polinomio interpolador P(x) para cualquier valor x
        
    Raises:
    -------
    ValueError
        Si los arrays no tienen la misma longitud o si hay valores x repetidos
    """
    # Validaciones
    if len(x_puntos) != len(y_puntos):
        raise ValueError("x_puntos y y_puntos deben tener la misma longitud")
    
    if len(x_puntos) != len(np.unique(x_puntos)):
        raise ValueError("Los valores de x_puntos deben ser únicos")
    
    n = len(x_puntos)
    
    def P(x):
        """
        Evalúa el polinomio de Lagrange en x.
        
        Parámetros:
        -----------
        x : float o np.ndarray
            Valor(es) donde evaluar el polinomio
            
        Retorna:
        --------
        float o np.ndarray
            Valor(es) del polinomio en x
        """
        resultado = 0.0
        
        for i in range(n):
            # Calcular L_i(x)
            L_i = 1.0
            for j in range(n):
                if j != i:
                    L_i *= (x - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
            
            # Sumar y_i * L_i(x)
            resultado += y_puntos[i] * L_i
        
        return resultado
    
    return P


def interpolacion_lagrange_matriz(
    x_puntos: np.ndarray,
    y_puntos: np.ndarray,
    x_eval: np.ndarray
) -> np.ndarray:
    """
    Evalúa el polinomio de Lagrange en múltiples puntos usando forma matricial.
    
    Esta implementación es más eficiente para evaluar el polinomio en muchos puntos.
    
    Parámetros:
    -----------
    x_puntos : np.ndarray
        Array con las coordenadas x de los puntos de interpolación (tamaño n)
    y_puntos : np.ndarray
        Array con las coordenadas y de los puntos de interpolación (tamaño n)
    x_eval : np.ndarray
        Array con los puntos donde evaluar el polinomio (tamaño m)
        
    Retorna:
    --------
    np.ndarray
        Array con los valores del polinomio en x_eval (tamaño m)
    """
    # Validaciones
    if len(x_puntos) != len(y_puntos):
        raise ValueError("x_puntos y y_puntos deben tener la misma longitud")
    
    n = len(x_puntos)
    m = len(x_eval)
    
    # Matriz de evaluación (m x n)
    # L[k, i] = L_i(x_eval[k])
    L = np.ones((m, n))
    
    for i in range(n):
        for j in range(n):
            if j != i:
                L[:, i] *= (x_eval - x_puntos[j]) / (x_puntos[i] - x_puntos[j])
    
    # P(x_eval) = L @ y_puntos
    return L @ y_puntos


def calcular_error_interpolacion(
    x_puntos: np.ndarray,
    y_puntos: np.ndarray,
    funcion_real: Callable,
    x_eval: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Calcula el error de interpolación comparando con la función real.
    
    Parámetros:
    -----------
    x_puntos : np.ndarray
        Puntos de interpolación (x)
    y_puntos : np.ndarray
        Valores en los puntos de interpolación (y)
    funcion_real : Callable
        Función real f(x)
    x_eval : np.ndarray
        Puntos donde evaluar el error
        
    Retorna:
    --------
    errores : np.ndarray
        Error absoluto en cada punto de evaluación
    error_max : float
        Error máximo
    error_medio : float
        Error medio (norma L2)
    """
    # Evaluar el polinomio interpolador
    P = interpolacion_lagrange(x_puntos, y_puntos)
    y_interpolado = np.array([P(x) for x in x_eval])
    
    # Evaluar la función real
    y_real = np.array([funcion_real(x) for x in x_eval])
    
    # Calcular errores
    errores = np.abs(y_real - y_interpolado)
    error_max = np.max(errores)
    error_medio = np.sqrt(np.mean(errores**2))
    
    return errores, error_max, error_medio


if __name__ == "__main__":
    print("=" * 70)
    print("INTERPOLACIÓN DE LAGRANGE")
    print("=" * 70)
    
    # Ejemplo 1: Interpolación simple con 4 puntos
    print("\n" + "=" * 70)
    print("EJEMPLO 1: Interpolación con 4 puntos")
    print("=" * 70)
    
    x1 = np.array([0.0, 1.0, 2.0, 3.0])
    y1 = np.array([1.0, 2.0, 0.0, 3.0])
    
    print("\nPuntos de interpolación:")
    for i in range(len(x1)):
        print(f"  ({x1[i]:.1f}, {y1[i]:.1f})")
    
    # Construir el polinomio interpolador
    P1 = interpolacion_lagrange(x1, y1)
    
    # Evaluar en los puntos originales (debe dar exactamente los valores y)
    print("\nVerificación en los puntos originales:")
    for i in range(len(x1)):
        valor = P1(x1[i])
        print(f"  P({x1[i]:.1f}) = {valor:.6f}, y = {y1[i]:.1f}, error = {abs(valor - y1[i]):.2e}")
    
    # Evaluar en puntos intermedios
    print("\nEvaluación en puntos intermedios:")
    x_intermedios = np.array([0.5, 1.5, 2.5])
    for x in x_intermedios:
        print(f"  P({x:.1f}) = {P1(x):.6f}")
    
    # Ejemplo 2: Interpolación de una función conocida
    print("\n" + "=" * 70)
    print("EJEMPLO 2: Interpolación de f(x) = sin(x)")
    print("=" * 70)
    
    # Función a interpolar
    def f(x):
        return np.sin(x)
    
    # Puntos de interpolación
    n_puntos = 6
    x2 = np.linspace(0, 2*np.pi, n_puntos)
    y2 = f(x2)
    
    print(f"\nInterpolando con {n_puntos} puntos en [0, 2π]")
    
    # Construir interpolador
    P2 = interpolacion_lagrange(x2, y2)
    
    # Evaluar en muchos puntos para ver la aproximación
    x_eval = np.linspace(0, 2*np.pi, 100)
    y_real = f(x_eval)
    y_interpolado = np.array([P2(x) for x in x_eval])
    
    # Calcular error
    errores = np.abs(y_real - y_interpolado)
    error_max = np.max(errores)
    error_medio = np.sqrt(np.mean(errores**2))
    
    print(f"\nError máximo: {error_max:.6e}")
    print(f"Error medio (L2): {error_medio:.6e}")
    
    # Ejemplo 3: Comparación con diferentes números de puntos
    print("\n" + "=" * 70)
    print("EJEMPLO 3: Efecto del número de puntos en f(x) = e^x")
    print("=" * 70)
    
    def f3(x):
        return np.exp(x)
    
    x_eval3 = np.linspace(0, 1, 50)
    y_real3 = f3(x_eval3)
    
    print("\nInterpolando f(x) = e^x en [0, 1]:")
    print("\nNúmero de puntos | Error máximo | Error medio")
    print("-" * 50)
    
    for n in [3, 5, 7, 10]:
        x_puntos = np.linspace(0, 1, n)
        y_puntos = f3(x_puntos)
        
        # Interpolar
        y_interp = interpolacion_lagrange_matriz(x_puntos, y_puntos, x_eval3)
        
        # Calcular error
        error = np.abs(y_real3 - y_interp)
        error_max = np.max(error)
        error_medio = np.sqrt(np.mean(error**2))
        
        print(f"      {n:2d}          | {error_max:.6e} | {error_medio:.6e}")
    
    # Ejemplo 4: Fenómeno de Runge
    print("\n" + "=" * 70)
    print("EJEMPLO 4: Fenómeno de Runge con f(x) = 1/(1 + 25x²)")
    print("=" * 70)
    
    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)
    
    print("\nLa función de Runge muestra el fenómeno de Runge:")
    print("El error puede CRECER al aumentar el número de puntos")
    print("cuando se usan puntos equiespaciados.")
    
    x_eval4 = np.linspace(-1, 1, 100)
    y_real4 = runge(x_eval4)
    
    print("\nNúmero de puntos | Error máximo")
    print("-" * 40)
    
    for n in [5, 9, 13, 17]:
        x_puntos = np.linspace(-1, 1, n)
        y_puntos = runge(x_puntos)
        
        P = interpolacion_lagrange(x_puntos, y_puntos)
        y_interp = np.array([P(x) for x in x_eval4])
        
        error_max = np.max(np.abs(y_real4 - y_interp))
        print(f"      {n:2d}          | {error_max:.6f}")
    
    # Graficar ejemplos (opcional, requiere matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        # Crear figura con múltiples subplots mejorados
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Interpolación simple con polinomios base
        ax1 = plt.subplot(3, 3, 1)
        x_plot = np.linspace(x1[0], x1[-1], 200)
        y_plot = np.array([P1(x) for x in x_plot])
        ax1.plot(x_plot, y_plot, 'b-', linewidth=2.5, label='P(x) Lagrange')
        ax1.plot(x1, y1, 'ro', markersize=12, label='Puntos', zorder=5)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        ax1.set_xlabel('x', fontsize=10)
        ax1.set_ylabel('y', fontsize=10)
        ax1.set_title('Interpolación Simple (4 puntos)', fontsize=11, fontweight='bold')
        
        # Subplot 2: Polinomios base de Lagrange del ejemplo 1
        ax2 = plt.subplot(3, 3, 2)
        for i in range(len(x1)):
            L_i = polinomio_lagrange_base(x1, i)
            y_Li = np.array([L_i(x) for x in x_plot])
            ax2.plot(x_plot, y_Li, linewidth=2, label=f'L_{i}(x)')
        ax2.plot(x1, np.zeros_like(x1), 'ko', markersize=8)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_xlabel('x', fontsize=10)
        ax2.set_ylabel('L_i(x)', fontsize=10)
        ax2.set_title('Polinomios Base de Lagrange', fontsize=11, fontweight='bold')
        
        # Subplot 3: Error de interpolación del ejemplo 1
        ax3 = plt.subplot(3, 3, 3)
        x_intermedios_plot = np.linspace(x1[0], x1[-1], 100)
        y_interp_plot = np.array([P1(x) for x in x_intermedios_plot])
        # Para el error, comparamos con un polinomio suave artificial
        y_ref = np.interp(x_intermedios_plot, x1, y1)
        error_plot = np.abs(y_interp_plot - y_ref)
        ax3.plot(x_intermedios_plot, error_plot, 'r-', linewidth=2)
        ax3.fill_between(x_intermedios_plot, 0, error_plot, alpha=0.3)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('x', fontsize=10)
        ax3.set_ylabel('|Error|', fontsize=10)
        ax3.set_title('Error de Interpolación', fontsize=11, fontweight='bold')
        
        # Subplot 4: sin(x) - Función e interpolación
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(x_eval, y_real, 'g-', linewidth=3, label='sin(x) real', alpha=0.7)
        ax4.plot(x_eval, y_interpolado, 'b--', linewidth=2, label='P(x) Lagrange')
        ax4.plot(x2, y2, 'ro', markersize=10, label=f'{n_puntos} puntos', zorder=5)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9)
        ax4.set_xlabel('x', fontsize=10)
        ax4.set_ylabel('y', fontsize=10)
        ax4.set_title('Interpolación de sin(x)', fontsize=11, fontweight='bold')
        
        # Subplot 5: Error absoluto de sin(x)
        ax5 = plt.subplot(3, 3, 5)
        error_sin = np.abs(y_real - y_interpolado)
        ax5.plot(x_eval, error_sin, 'r-', linewidth=2)
        ax5.fill_between(x_eval, 0, error_sin, alpha=0.3, color='red')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel('x', fontsize=10)
        ax5.set_ylabel('|Error|', fontsize=10)
        ax5.set_title(f'Error: max={error_max:.2e}', fontsize=11, fontweight='bold')
        
        # Subplot 6: e^x con diferentes n
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(x_eval3, y_real3, 'k-', linewidth=3, label='e^x real', alpha=0.7)
        colors = ['blue', 'green', 'orange', 'purple']
        for idx, n in enumerate([3, 5, 7, 10]):
            x_puntos = np.linspace(0, 1, n)
            y_puntos = f3(x_puntos)
            y_interp = interpolacion_lagrange_matriz(x_puntos, y_puntos, x_eval3)
            ax6.plot(x_eval3, y_interp, '--', linewidth=1.5, label=f'n={n}', color=colors[idx])
        ax6.grid(True, alpha=0.3)
        ax6.legend(fontsize=8)
        ax6.set_xlabel('x', fontsize=10)
        ax6.set_ylabel('y', fontsize=10)
        ax6.set_title('Convergencia con e^x', fontsize=11, fontweight='bold')
        
        # Subplot 7: Error vs número de puntos para e^x
        ax7 = plt.subplot(3, 3, 7)
        n_vals = range(3, 16)
        errores_max = []
        for n in n_vals:
            x_pts = np.linspace(0, 1, n)
            y_pts = f3(x_pts)
            y_int = interpolacion_lagrange_matriz(x_pts, y_pts, x_eval3)
            err = np.max(np.abs(y_real3 - y_int))
            errores_max.append(err)
        ax7.semilogy(list(n_vals), errores_max, 'bo-', linewidth=2, markersize=6)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlabel('Número de puntos (n)', fontsize=10)
        ax7.set_ylabel('Error máximo', fontsize=10)
        ax7.set_title('Convergencia: e^x', fontsize=11, fontweight='bold')
        
        # Subplot 8: Fenómeno de Runge
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(x_eval4, y_real4, 'k-', linewidth=3, label='Runge 1/(1+25x²)', alpha=0.7)
        colors_runge = ['blue', 'orange', 'red']
        for idx, n in enumerate([5, 9, 13]):
            x_puntos = np.linspace(-1, 1, n)
            y_puntos = runge(x_puntos)
            P_r = interpolacion_lagrange(x_puntos, y_puntos)
            y_interp = np.array([P_r(x) for x in x_eval4])
            ax8.plot(x_eval4, y_interp, '--', linewidth=2, label=f'n={n}', color=colors_runge[idx])
            ax8.plot(x_puntos, y_puntos, 'o', markersize=5, color=colors_runge[idx])
        ax8.set_ylim([-0.5, 1.5])
        ax8.grid(True, alpha=0.3)
        ax8.legend(fontsize=8)
        ax8.set_xlabel('x', fontsize=10)
        ax8.set_ylabel('y', fontsize=10)
        ax8.set_title('Fenómeno de Runge', fontsize=11, fontweight='bold')
        
        # Subplot 9: Error vs n para fenómeno de Runge
        ax9 = plt.subplot(3, 3, 9)
        n_runge_vals = range(5, 22, 2)
        errores_runge = []
        for n in n_runge_vals:
            x_pts = np.linspace(-1, 1, n)
            y_pts = runge(x_pts)
            P_r = interpolacion_lagrange(x_pts, y_pts)
            y_int = np.array([P_r(x) for x in x_eval4])
            err = np.max(np.abs(y_real4 - y_int))
            errores_runge.append(err)
        ax9.semilogy(list(n_runge_vals), errores_runge, 'ro-', linewidth=2, markersize=6)
        ax9.grid(True, alpha=0.3)
        ax9.set_xlabel('Número de puntos (n)', fontsize=10)
        ax9.set_ylabel('Error máximo', fontsize=10)
        ax9.set_title('Fenómeno de Runge: Error ↑', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('interpolacion_lagrange.png', dpi=200, bbox_inches='tight')
        print("\n" + "=" * 70)
        print("Gráficas guardadas en: interpolacion_lagrange.png")
        print("  - Fila 1: Interpolación simple, polinomios base, error")
        print("  - Fila 2: sin(x), error de sin(x), convergencia e^x")
        print("  - Fila 3: Error vs n (e^x), fenómeno de Runge, error Runge")
        print("=" * 70)
        
    except ImportError:
        print("\n(matplotlib no disponible para graficar)")
    
    print("\n" + "=" * 70)
    print("FIN DE LOS EJEMPLOS")
    print("=" * 70)

