from gauss_legendre_2p import gauss_legendre_2p
import math
import matplotlib.pyplot as plt
import numpy as np

def gauss_legendre_3p(f, a, b):
    """
    Implementa el método de cuadratura Gauss-Legendre de 3 puntos para aproximar
    la integral definida de una función f(x) en el intervalo [a, b].
    """
    # === PASO 1: Obtener nodos y pesos para 3 puntos ===
    # Nodos: raíces del polinomio de Legendre P₃(x)
    nodes = [-math.sqrt(3/5), 0, math.sqrt(3/5)]
    # Pesos correspondientes
    weights = [5/9, 8/9, 5/9]

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

# Ejemplo de uso con funciones que no tienen solución exacta simple
if __name__ == "__main__":
    a, b = 0.0, 2.0
    from scipy import integrate

    test_functions = [
        (lambda x: math.exp(-x**2), "e^(-x²)", "Gaussiana suave"),
        (lambda x: math.sin(x**2), "sin(x²)", "Oscilatoria rápida"),
        (lambda x: math.log(1 + abs(x-1)**3), "ln(1+|x-1|³)", "No suave en x=1"),
        (lambda x: math.sqrt(1 + math.sin(x)), "√(1+sin(x))", "Periódica suave"),
        (lambda x: math.exp(math.cos(2*x)), "e^(cos(2x))", "Oscilatoria suave")
    ]

    function_names = []
    errors_2pt = []
    errors_3pt = []
    reference_values = []
    percent_reduction = []

    print("="*70)
    print("COMPARACIÓN: GAUSS-LEGENDRE 2 PUNTOS vs 3 PUNTOS")
    print("="*70)

    for i, (f_test, f_name, f_desc) in enumerate(test_functions):
        reference, error_est = integrate.quad(f_test, a, b, limit=100)
        reference_values.append(reference)

        approx_2pt = gauss_legendre_2p(f_test, a, b)
        approx_3pt = gauss_legendre_3p(f_test, a, b)

        error_2pt = abs(reference - approx_2pt)
        error_3pt = abs(reference - approx_3pt)

        function_names.append(f_name)
        errors_2pt.append(error_2pt)
        errors_3pt.append(error_3pt)

        # Calcular porcentaje de reducción
        if error_2pt > 1e-18:
            red = 100.0 * (error_2pt - error_3pt) / error_2pt
        else:
            red = float('nan')
        percent_reduction.append(red)

        print(f"\nFUNCIÓN: {f_name}")
        print(f"Descripción: {f_desc}")
        print(f"Referencia (scipy): {reference:.10f}")
        print(f"Aprox. 2 puntos:    {approx_2pt:.10f} | Error: {error_2pt:.2e}")
        print(f"Aprox. 3 puntos:    {approx_3pt:.10f} | Error: {error_3pt:.2e}")
        if not math.isnan(red):
            print(f"% Reducción (2pt -> 3pt): {red:.2f}%")
        else:
            print(" % Reducción: N/A (error 2pt ~ 0)")


    plt.figure(figsize=(20, 8))
    x_pos = np.arange(len(function_names))
    width = 0.35

    # Gráfica 1: Errores absolutos
    ax1 = plt.subplot(1, 3, 1)
    bars1 = ax1.bar(x_pos - width/2, errors_2pt, width, label='2 puntos',
                    color='coral', alpha=0.85, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, errors_3pt, width, label='3 puntos',
                    color='steelblue', alpha=0.85, edgecolor='black')
    ax1.set_title('Errores absolutos (escala log)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Error absoluto', fontsize=13)
    ax1.set_xlabel('Función', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(function_names, rotation=45, ha='right', fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=11)
    # etiquetas simples
    for bar, error in zip(bars1 + bars2, errors_2pt + errors_3pt):
        if error > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, error, f'{error:.1e}',
                     ha='center', va='bottom', fontsize=9)

    # Gráfica 2
    ax2 = plt.subplot(1, 3, 2)
    colors = ['green' if (not math.isnan(r) and r > 0) else 'red' for r in percent_reduction]
    bars_red = ax2.bar(x_pos, percent_reduction, color=colors, alpha=0.85, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.6)
    ax2.axhline(50, color='grey', linestyle=':', alpha=0.5)
    ax2.set_title('% Reducción de error (2pt → 3pt)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Porcentaje de reducción (%)', fontsize=13)
    ax2.set_xlabel('Función', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(function_names, rotation=45, ha='right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Anotar valores en barras
    for bar, r in zip(bars_red, percent_reduction):
        label = f'{r:.1f}%' if not math.isnan(r) else 'N/A'
        va = 'bottom' if (not math.isnan(r) and r >= 0) else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, (r if not math.isnan(r) else 0),
                 label, ha='center', va=va, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Gráfica 3: Funciones y puntos de evaluación
    ax3 = plt.subplot(1, 3, 3)
    x_vals = np.linspace(a, b, 1000)
    functions_to_plot = min(2, len(test_functions))
    colors_plot = ['red', 'blue']
    for i in range(functions_to_plot):
        f_test, f_name, f_desc = test_functions[i]
        y_vals = [f_test(x) for x in x_vals]
        ax3.plot(x_vals, y_vals, label=f_name, linewidth=2.5, alpha=0.95)

        # Puntos 2-pt
        nodes_2pt = [-math.sqrt(1/3), math.sqrt(1/3)]
        transformed_2pt = [0.5 * ((b - a) * x_i + (b + a)) for x_i in nodes_2pt]
        y_2pt = [f_test(x) for x in transformed_2pt]

        # Puntos 3-pt
        nodes_3pt = [-math.sqrt(3/5), 0.0, math.sqrt(3/5)]
        transformed_3pt = [0.5 * ((b - a) * x_i + (b + a)) for x_i in nodes_3pt]
        y_3pt = [f_test(x) for x in transformed_3pt]

        ax3.plot(transformed_2pt, y_2pt, 's', markersize=9, color=colors_plot[i],
                 markeredgecolor='black', label=f'2pt ({f_name})' if i == 0 else "")
        ax3.plot(transformed_3pt, y_3pt, 'o', markersize=9, color=colors_plot[i],
                 markeredgecolor='black', label=f'3pt ({f_name})' if i == 0 else "")

    ax3.set_title('Funciones y puntos de evaluación (solo 2 funciones)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('x', fontsize=13)
    ax3.set_ylabel('f(x)', fontsize=13)
    ax3.legend(ncol=2, fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.annotate('□ = 2 puntos   ○ = 3 puntos',
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=11, ha='left', va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()