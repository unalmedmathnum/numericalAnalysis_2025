import math
import matplotlib.pyplot as plt
import numpy as np

def gauss_legendre_2p(f, a, b):
    """
    Implementa el método de cuadratura Gauss-Legendre de 2 puntos para aproximar
    la integral definida de una función f(x) en el intervalo [a, b].
    """
    # === PASO 1: Obtener nodos y pesos ===
    sqrt3 = math.sqrt(3)
    nodes = [-sqrt3/3, sqrt3/3]  # Nodos x_i en [-1, 1]
    weights = [1.0, 1.0]         # Pesos c_i correspondientes

    # === PASO 2: Transformar nodos al intervalo [a, b] ===
    transformed_nodes = []
    for x_i in nodes:
        x_transformed = 0.5 * ((b - a) * x_i + (b + a))
        transformed_nodes.append(x_transformed)

    # === PASO 3: Evaluar suma ponderada ===
    weighted_sum = 0.0
    for i in range(len(nodes)):
        weighted_sum += weights[i] * f(transformed_nodes[i])

    # === PASO 4: Retornar aproximación ===
    integral_approximation = 0.5 * (b - a) * weighted_sum
    return integral_approximation


# Ejemplo de uso centrado en comparación de error vs solución exacta
if __name__ == "__main__":
    # Definir la función a integrar
    def f(x):
        return math.exp(-x**2)  # Función gaussiana

    # Parámetros de integración
    a, b = 0.0, 2.0

    # Calcular aproximación con Gauss-Legendre
    approx = gauss_legendre_2p(f, a, b)

    # Calcular valor exacto usando scipy
    from scipy import integrate
    exact, _ = integrate.quad(f, a, b)

    # Calcular error
    error = abs(exact - approx)
    relative_error = error / abs(exact) if exact != 0 else error

    print(f"Función: e^(-x²) en el intervalo [{a}, {b}]")
    print(f"Aproximación (Gauss-Legendre): {approx:.8f}")
    print(f"Valor exacto: {exact:.8f}")
    print(f"Error absoluto: {error:.2e}")
    print(f"Error relativo: {relative_error:.2e}")

    # Crear gráfica comparativa centrada en error vs solución exacta
    plt.figure(figsize=(14, 5))

    # Subplot 1: Comparación de valores
    plt.subplot(1, 2, 1)
    methods = ['Gauss-Legendre', 'Valor exacto']
    values = [approx, exact]
    colors = ['lightcoral', 'lightgreen']

    bars = plt.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Comparación: Valor aproximado vs exacto', fontsize=14, fontweight='bold')
    plt.ylabel('Valor de la integral', fontsize=12)

    # Añadir valores en las barras
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.6f}', ha='center', va='bottom', fontweight='bold')

    # Añadir línea que marca el valor exacto para comparación visual
    plt.axhline(y=exact, color='red', linestyle='--', alpha=0.7, label='Valor exacto')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Análisis de errores
    plt.subplot(1, 2, 2)
    error_types = ['Error absoluto', 'Error relativo']
    error_values = [error, relative_error]
    colors_errors = ['orange', 'purple']

    bars_errors = plt.bar(error_types, error_values, color=colors_errors, alpha=0.8, edgecolor='black')
    plt.title('Análisis de errores', fontsize=14, fontweight='bold')
    plt.ylabel('Magnitud del error', fontsize=12)

    # Añadir valores en las barras de error
    for bar, value in zip(bars_errors, error_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{value:.2e}', ha='center', va='center', fontweight='bold', color='white')

    plt.grid(True, alpha=0.3)

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()

    # Análisis adicional: Comportamiento para diferentes funciones
    print("\n" + "="*70)
    print("ANÁLISIS COMPARATIVO PARA DIFERENTES FUNCIONES")
    print("="*70)

    # Definir varias funciones para comparar
    test_functions = [
        (lambda x: x**2, "x²", 8/3),  # ∫x² dx de 0 a 2 = 8/3
        (lambda x: math.sin(x), "sin(x)", 1.0),  # ∫sin(x) dx de 0 a π/2 = 1
        (lambda x: math.exp(x), "e^x", math.exp(2)-1),  # ∫e^x dx de 0 a 2 = e²-1
        (lambda x: 1/(1+x**2), "1/(1+x²)", math.atan(2))  # ∫1/(1+x²) dx de 0 a 2 = arctan(2)
    ]

    # Configurar la gráfica de comparación para múltiples funciones
    plt.figure(figsize=(12, 8))

    # Preparar datos para el gráfico
    function_names = []
    errors_abs = []
    errors_rel = []

    for f_test, f_name, exact_val in test_functions:
        # Calcular aproximación
        approx_test = gauss_legendre_2p(f_test, a, b)
        error_abs = abs(exact_val - approx_test)
        error_rel = error_abs / abs(exact_val) if exact_val != 0 else error_abs

        function_names.append(f_name)
        errors_abs.append(error_abs)
        errors_rel.append(error_rel)

        print(f"Función: {f_name:10} | Aprox: {approx_test:.6f} | Exacto: {exact_val:.6f} | Error: {error_abs:.2e}")

    # Gráfica de errores absolutos
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(function_names, errors_abs, color='coral', alpha=0.7)
    plt.title('Error absoluto por función', fontsize=14, fontweight='bold')
    plt.ylabel('Error absoluto', fontsize=12)
    plt.xticks(rotation=45)

    # Añadir valores en las barras
    for bar, error_val in zip(bars1, errors_abs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{error_val:.1e}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3)

    # Gráfica de errores relativos
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(function_names, errors_rel, color='lightseagreen', alpha=0.7)
    plt.title('Error relativo por función', fontsize=14, fontweight='bold')
    plt.ylabel('Error relativo', fontsize=12)
    plt.xticks(rotation=45)

    # Añadir valores en las barras
    for bar, error_val in zip(bars2, errors_rel):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{error_val:.1e}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()