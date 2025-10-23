import numpy as np
import matplotlib.pyplot as plt

def diferencias_divididas(lx, ly):
    n = len(lx)

    # Tabla de memoización (matriz triangular).
    DD = [[None] * n for _ in range(n)]

    def calculo(i, k):
        #Función recursiva para calcular DD(i, k).

        if k == 0:
            return ly[i]

        #Revisar Memoización: Si ya se calculó, retornarlo.
        if DD[i][k] is not None:
            return DD[i][k]

        #Aplicar Recurrencia: DD(i, k) = [DD(i+1, k-1) - DD(i, k-1)] / (x_{i+k} - x_i)

        dd_sup = calculo(i + 1, k - 1)
        dd_inf = calculo(i, k - 1)
        resultado = (dd_sup - dd_inf) / (lx[i + k] - lx[i])

        #Memoización

        DD[i][k] = resultado
        return resultado

    #Obtener los coeficientes: son los elementos de la primera fila de la tabla DD(0, k).
    coeficientes = [calculo(0, k) for k in range(n)]

    return coeficientes


def polinomio_newton(x, lx, coeficientes):
    n = len(coeficientes)
    resultado = coeficientes[0]
    for i in range(1, n):
        termino = coeficientes[i]
        for j in range(i):
            termino *= (x - lx[j])
        resultado += termino
    return resultado

if __name__ == "__main__":
  #Aproximación de la función f(x) = x^2 en el intervalo [1, 4]  
  lx = np.array([1.0, 2.0, 3.0, 4.0])
  ly = np.array([1.0, 4.0, 9.0, 16.0])

  b_coeficientes = diferencias_divididas(lx, ly)

  print(f"--- Coeficientes de Newton (b_i) Ejemplo 1: ---\n{[float(x) for x in b_coeficientes]}")
  def func_original(x):
    return x**2
  x_rango = np.linspace(lx.min() - 0.5, lx.max() + 0.5, 200)

  y_polinomio = [polinomio_newton(x_val, lx, b_coeficientes) for x_val in x_rango]
  y_original = [func_original(x_val) for x_val in x_rango]
  plt.figure(figsize=(10, 6))

  plt.plot(lx, ly, 'o', color='red', markersize=8, label='Nodos')

  plt.plot(x_rango, y_polinomio, '--', color='blue', label='Polinomio de Newton')

  plt.title('Interpolación de Newton, Ejemplo 1')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.axhline(0, color='black', linewidth=0.5)
  plt.axvline(0, color='black', linewidth=0.5)
  plt.legend()
  plt.show()

  error_absoluto = np.abs(np.array(y_original) - np.array(y_polinomio))
  error_cuadratico_medio = np.mean(error_absoluto**2)
  max_error = np.max(error_absoluto)
  punto_max_error = x_rango[np.argmax(error_absoluto)]
  print(f"MSE: {error_cuadratico_medio}")
  print(f"Error max: {max_error}")
  print(f"Punto de max error: {punto_max_error}")

  #Aproximación de la función f(x) = sin(x) en el intervalo [0, 2π]  
  lx2 = np.array([
    0.0,
    0.7853981633974483,
    1.5707963267948966,
    2.356194490192345,
    3.141592653589793,
    3.9269908169872414,
    4.71238898038469,
    5.497787143782138,
    6.283185307179586
  ])
  ly2 = np.array([
      0.0,
      0.7071067811865476,
      1.0,
      0.7071067811865476,
      0.0,
      -0.7071067811865476,
      -1.0,
      -0.7071067811865476,
      0.0
  ])

  b_coeficientes = diferencias_divididas(lx2, ly2)
  print(f"--- Coeficientes de Newton (b_i) Ejemplo 2: ---\n{[float(x) for x in b_coeficientes]}")
  def func_original2(x):
    return np.sin(x)

  x_rango = np.linspace(lx2.min(), lx2.max(), 200)

  y_spline = [polinomio_newton(x_val, lx2, b_coeficientes) for x_val in x_rango]
  y_original = [func_original2(x_val) for x_val in x_rango]
  plt.figure(figsize=(10, 6))

  plt.plot(lx2, ly2, 'o', color='red', markersize=8, label='Nodos')

  plt.plot(x_rango, y_spline, '--', color='blue', label='Polinomio de Newton')


  plt.title('Interpolación de Newton, Ejemplo 2')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)
  plt.axhline(0, color='black', linewidth=0.5)
  plt.axvline(0, color='black', linewidth=0.5)
  plt.legend()
  plt.show()

  error_absoluto = np.abs(np.array(y_original) - np.array(y_spline))
  error_cuadratico_medio = np.mean(error_absoluto**2)
  max_error = np.max(error_absoluto)
  punto_max_error = x_rango[np.argmax(error_absoluto)]
  print(f"MSE: {error_cuadratico_medio}")
  print(f"Error max: {max_error}")
  print(f"Punto de max error: {punto_max_error}")