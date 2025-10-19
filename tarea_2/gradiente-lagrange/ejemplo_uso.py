"""
Ejemplo de uso de los módulos gradiente_descendente y lagrange

Este script demuestra cómo importar y usar las funciones de los módulos
implementados para la Tarea 2 del curso de Análisis Numérico.
"""

import numpy as np
from gradiente_descendente import gradiente_descendente, gradiente_conjugado, verificar_definida_positiva
from lagrange import interpolacion_lagrange, interpolacion_lagrange_matriz

print("=" * 70)
print("EJEMPLO DE USO DE LOS MÓDULOS IMPLEMENTADOS")
print("=" * 70)

# ============================================================================
# EJEMPLO 1: Gradiente Descendente
# ============================================================================
print("\n" + "=" * 70)
print("1. MÉTODO DEL GRADIENTE DESCENDENTE")
print("=" * 70)

# Crear un sistema Ax = b
A = np.array([
    [5.0, 1.0, 0.0],
    [1.0, 4.0, 1.0],
    [0.0, 1.0, 3.0]
])

b = np.array([6.0, 5.0, 4.0])

print("\nSistema a resolver: Ax = b")
print("\nMatriz A:")
print(A)
print("\nVector b:")
print(b)

# Verificar que A sea definida positiva
es_definida_positiva = verificar_definida_positiva(A)
print(f"\n¿A es simétrica y definida positiva? {es_definida_positiva}")

if es_definida_positiva:
    # Resolver con gradiente descendente
    x_gd, residuos_gd, iter_gd = gradiente_descendente(A, b, tol=1e-8, verbose=False)
    
    print(f"\nSolución por gradiente descendente ({iter_gd} iteraciones):")
    print(x_gd)
    
    # Resolver con gradiente conjugado
    x_gc, residuos_gc, iter_gc = gradiente_conjugado(A, b, tol=1e-8, verbose=False)
    
    print(f"\nSolución por gradiente conjugado ({iter_gc} iteraciones):")
    print(x_gc)
    
    # Solución exacta
    x_exacta = np.linalg.solve(A, b)
    print(f"\nSolución exacta (NumPy):")
    print(x_exacta)
    
    # Comparar errores
    error_gd = np.linalg.norm(x_gd - x_exacta)
    error_gc = np.linalg.norm(x_gc - x_exacta)
    
    print(f"\nError gradiente descendente: {error_gd:.6e}")
    print(f"Error gradiente conjugado:   {error_gc:.6e}")
    
    # Verificación
    print(f"\n||Ax - b|| con gradiente descendente: {np.linalg.norm(A @ x_gd - b):.6e}")
    print(f"||Ax - b|| con gradiente conjugado:   {np.linalg.norm(A @ x_gc - b):.6e}")

# ============================================================================
# EJEMPLO 2: Interpolación de Lagrange
# ============================================================================
print("\n" + "=" * 70)
print("2. INTERPOLACIÓN DE LAGRANGE")
print("=" * 70)

# Definir puntos de interpolación
x_puntos = np.array([-1.0, 0.0, 1.0, 2.0])
y_puntos = np.array([0.0, 1.0, 0.0, -3.0])

print("\nPuntos de interpolación:")
for i in range(len(x_puntos)):
    print(f"  ({x_puntos[i]:5.1f}, {y_puntos[i]:5.1f})")

# Construir el polinomio interpolador
P = interpolacion_lagrange(x_puntos, y_puntos)

# Evaluar en los puntos originales (verificación)
print("\nVerificación - el polinomio debe pasar por los puntos originales:")
for i in range(len(x_puntos)):
    valor = P(x_puntos[i])
    error = abs(valor - y_puntos[i])
    print(f"  P({x_puntos[i]:5.1f}) = {valor:8.6f}, y = {y_puntos[i]:5.1f}, error = {error:.2e}")

# Evaluar en puntos nuevos
x_nuevos = np.array([-0.5, 0.5, 1.5])
print("\nEvaluación en puntos nuevos:")
for x in x_nuevos:
    print(f"  P({x:5.1f}) = {P(x):8.6f}")

# Evaluar en muchos puntos usando la versión matricial
x_eval = np.linspace(-1, 2, 50)
y_eval = interpolacion_lagrange_matriz(x_puntos, y_puntos, x_eval)

print(f"\nEvaluación matricial en {len(x_eval)} puntos completada")
print(f"  Rango de valores: [{y_eval.min():.3f}, {y_eval.max():.3f}]")

# ============================================================================
# EJEMPLO 3: Interpolación de una función conocida
# ============================================================================
print("\n" + "=" * 70)
print("3. INTERPOLACIÓN DE f(x) = cos(x)")
print("=" * 70)

# Función a interpolar
def f(x):
    return np.cos(x)

# Puntos de interpolación
n_puntos = 5
x_interp = np.linspace(0, np.pi, n_puntos)
y_interp = f(x_interp)

print(f"\nInterpolando cos(x) con {n_puntos} puntos en [0, π]")

# Construir interpolador
P_cos = interpolacion_lagrange(x_interp, y_interp)

# Evaluar y comparar con la función real
x_test = np.array([np.pi/4, np.pi/2, 3*np.pi/4])

print("\nComparación P(x) vs cos(x):")
print(f"{'x':^10} | {'P(x)':^12} | {'cos(x)':^12} | {'error':^12}")
print("-" * 52)
for x in x_test:
    p_val = P_cos(x)
    f_val = f(x)
    error = abs(p_val - f_val)
    print(f"{x:10.6f} | {p_val:12.8f} | {f_val:12.8f} | {error:12.8f}")

# Calcular error en muchos puntos
x_eval_cos = np.linspace(0, np.pi, 100)
y_real = f(x_eval_cos)
y_interp_cos = np.array([P_cos(x) for x in x_eval_cos])
errores = np.abs(y_real - y_interp_cos)

print(f"\nError máximo en [0, π]: {np.max(errores):.6e}")
print(f"Error medio (RMS):      {np.sqrt(np.mean(errores**2)):.6e}")

# ============================================================================
# Resumen final
# ============================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print("\nAmbos módulos fueron importados y utilizados exitosamente:")
print("  ✓ gradiente_descendente.py - Resuelve sistemas lineales")
print("  ✓ lagrange.py - Interpola funciones usando polinomios de Lagrange")
print("\nCada módulo incluye el bloque 'if __name__ == \"__main__\"' con")
print("ejemplos detallados que se ejecutan solo al correr el archivo directamente.")
print("=" * 70)

