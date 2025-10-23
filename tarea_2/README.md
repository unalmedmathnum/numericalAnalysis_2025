# Tarea Grupo 2 - Análisis Numérico

Este repositorio contiene la implementación de métodos numéricos como parte de la Tarea 2 del curso de Análisis Numérico.

## Archivos implementados

### 1. `gradiente_descendente.py`
Implementación del método del gradiente descendente para resolver sistemas simétricos definidos positivos **Ax = b**.

**Características:**
- Implementa el método del **gradiente descendente** clásico
- Implementa el método del **gradiente conjugado** (más eficiente)
- Función para verificar si una matriz es simétrica y definida positiva
- Ejemplos de uso con sistemas pequeños y grandes
- Generación de gráficas de convergencia

**Funciones principales:**
- `gradiente_descendente(A, b, x0, tol, max_iter, verbose)`: Resuelve Ax = b por gradiente descendente
- `gradiente_conjugado(A, b, x0, tol, max_iter, verbose)`: Resuelve Ax = b por gradiente conjugado
- `verificar_definida_positiva(A, tol)`: Verifica si A es simétrica y definida positiva

**Uso:**
```python
import numpy as np
from gradiente_descendente import gradiente_descendente

# Definir sistema Ax = b
A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])

# Resolver
x, residuos, iter = gradiente_descendente(A, b, verbose=True)
print("Solución:", x)
```

**Ejemplo de ejecución:**
```bash
python3 gradiente_descendente.py
```

### 2. `lagrange.py`
Implementación del polinomio de interpolación de Lagrange.

**Características:**
- Construcción de polinomios base de Lagrange L_i(x)
- Interpolación completa con el polinomio P(x) = Σ y_i * L_i(x)
- Evaluación eficiente en múltiples puntos usando forma matricial
- Cálculo de errores de interpolación
- Ejemplos con funciones clásicas (sin(x), e^x, función de Runge)
- Demostración del fenómeno de Runge

**Funciones principales:**
- `polinomio_lagrange_base(x_puntos, i)`: Calcula el i-ésimo polinomio base L_i(x)
- `interpolacion_lagrange(x_puntos, y_puntos)`: Construye el polinomio interpolador completo
- `interpolacion_lagrange_matriz(x_puntos, y_puntos, x_eval)`: Evaluación matricial eficiente
- `calcular_error_interpolacion(x_puntos, y_puntos, funcion_real, x_eval)`: Calcula errores

**Uso:**
```python
import numpy as np
from lagrange import interpolacion_lagrange

# Definir puntos de interpolación
x_puntos = np.array([0.0, 1.0, 2.0, 3.0])
y_puntos = np.array([1.0, 2.0, 0.0, 3.0])

# Construir interpolador
P = interpolacion_lagrange(x_puntos, y_puntos)

# Evaluar en un punto
valor = P(1.5)
print(f"P(1.5) = {valor}")
```

**Ejemplo de ejecución:**
```bash
python3 lagrange.py
```

## Requisitos

Los archivos requieren las siguientes dependencias:

```
numpy>=1.21.0
matplotlib>=3.4.0
```

### Instalación de dependencias

```bash
# Opción 1: usando pip
pip install -r requirements.txt

# Opción 2: instalación manual desde PyPI público
python3 -m pip install --user --index-url https://pypi.org/simple/ numpy matplotlib
```

## Estructura del código

Ambos archivos siguen las mejores prácticas de programación en Python:

1. **Documentación completa**: Cada función tiene docstrings detallados con descripción, parámetros y valores de retorno
2. **Type hints**: Uso de anotaciones de tipo para mayor claridad
3. **Validación de entradas**: Verificación de dimensiones y valores válidos
4. **Bloque `if __name__ == "__main__"`**: Cada archivo incluye ejemplos de uso que se ejecutan solo cuando el archivo se corre directamente
5. **Comentarios explicativos**: El código está bien comentado para facilitar su comprensión

## Ejemplos de salida

### Gradiente Descendente

El script genera:
- Soluciones de sistemas lineales con comparación con la solución exacta
- Comparación entre gradiente descendente y gradiente conjugado
- **Gráfica de convergencia: `convergencia_gradiente.png`** (2x2 subplots):
  1. Convergencia GD vs GC (sistema 10x10)
  2. Convergencia del sistema 3x3
  3. Razón de convergencia ||r_{k+1}||/||r_k||
  4. Iteraciones vs tamaño del sistema (n=3 a 20)

### Interpolación de Lagrange

El script genera:
- Interpolación de puntos arbitrarios
- Interpolación de funciones conocidas (sin(x), e^x)
- Demostración del fenómeno de Runge
- **Gráficas de todas las interpolaciones: `interpolacion_lagrange.png`** (3x3 subplots):
  1. Interpolación simple con 4 puntos
  2. Polinomios base de Lagrange L_i(x)
  3. Error de interpolación
  4. Interpolación de sin(x)
  5. Error absoluto de sin(x)
  6. Convergencia con e^x (diferentes valores de n)
  7. Error máximo vs número de puntos (e^x)
  8. Fenómeno de Runge con función 1/(1+25x²)
  9. Error máximo vs n mostrando crecimiento del error
