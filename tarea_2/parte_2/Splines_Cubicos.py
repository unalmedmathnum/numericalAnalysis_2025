import numpy as np
import matplotlib.pyplot as plt

def splines_cubicos(lx, ly):

    n = len(lx)-1
    h = np.diff(lx) # h_j = x_{j+1} - x_j

    # Vector alpha (lado derecho del sistema de ecuaciones)
    # alpha_j = 3/h_j * (a_{j+1} - a_j) - 3/h_{j-1} * (a_j - a_{j-1})
    alpha = np.zeros(n+1)
    for j in range(1, n):
        alpha[j] = (3.0 / h[j]) * (ly[j + 1] - ly[j]) - (3.0 / h[j - 1]) * (ly[j] - ly[j - 1])

    #Matriz tridiagonal A:
    matriz_A = np.zeros((n + 1, n + 1))

    for i in range(n + 1):

        # Subdiagonal
        if i > 0 and i<n:
            matriz_A[i, i-1] = h[i-1]

        # Diagonal principal
        if  i == 0 or i == n:
          matriz_A[i, i] = 1
        else:
          matriz_A[i, i] = 2.0 * (h[i-1] + h[i])

        # Superdiagonal
        if i > 0 and i < n:
            matriz_A[i, i+1] = h[i]

    try:
        c_sol = np.linalg.solve(matriz_A, alpha) #Solución del sistema
    except np.linalg.LinAlgError:
        raise ValueError("Error: La matriz tridiagonal es singular.")

    #Cálculo final de coeficientes faltantes (a_j, b_j, d_j)
    polinomios_coefs = []


    for j in range(n): # n polinomios S_j(x)
        h_j = h[j]
        c_j = c_sol[j]
        c_j1 = c_sol[j+1]
        a_j = ly[j]
        a_j1 = ly[j+1]

        d_j = (c_j1 - c_j) / (3.0 * h_j)

        b_j = (a_j1 - a_j) / h_j - (h_j / 3.0) * (2.0 * c_j + c_j1)

        polinomios_coefs.append({
            'intervalo': (lx[j], lx[j+1]),
            'x_s': lx[j],
            'a': a_j,
            'b': b_j,
            'c': c_j,
            'd': d_j
        })

    return polinomios_coefs

def evaluar_spline(lcoeficientes, x):
    for coefs in lcoeficientes:
        x_s = coefs['x_s']

        if coefs['intervalo'][0] <= x <= coefs['intervalo'][1]:

            # El polinomio es S_j(x) = a_j + b_j(x-x_j) + c_j(x-x_j)^2 + d_j(x-x_j)^3
            delta_x = x - x_s

            resultado = (coefs['a'] + coefs['b'] * delta_x + coefs['c'] * delta_x**2 + coefs['d'] * delta_x**3)
            return resultado

    return np.nan

if __name__ == "__main__":
  #Aproximación de la función f(x) = x^2 en el intervalo [1, 4]
  lx = np.array([1.0, 2.0, 3.0, 4.0])
  ly = np.array([1.0, 4.0, 9.0, 16.0])

  coefs_ = splines_cubicos(lx, ly)

  if coefs_:
      print("--- Coeficientes de Spline Cúbico Ejemplo 1: ---")
      for j, coef in enumerate(coefs_):
          print(f"S{j}(x) en {coef['intervalo']}:")
          print(f"  a={coef['a']:.6f}, b={coef['b']:.6f}, c={coef['c']:.6f}, d={coef['d']:.6f}")

  def func_original(x):
    return x**2

  x_rango = np.linspace(lx.min(), lx.max(), 200)

  y_spline = [evaluar_spline(coefs_, x_val) for x_val in x_rango]
  y_original = [func_original(x_val) for x_val in x_rango]
  plt.figure(figsize=(10, 6))

  plt.plot(lx, ly, 'o', color='red', markersize=8, label='Nodos')

  plt.plot(x_rango, y_spline, '--', color='blue', label='Aprox Spline')


  plt.title('Splines, Ejemplo 1')
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

  coefs_ = splines_cubicos(lx2, ly2)

  if coefs_:
      print("--- Coeficientes de Spline Cúbico Ejemplo 2: ---")
      for j, coef in enumerate(coefs_):
          print(f"S{j}(x) en {coef['intervalo']}:")
          print(f"  a={coef['a']:.6f}, b={coef['b']:.6f}, c={coef['c']:.6f}, d={coef['d']:.6f}")

  def func_original2(x):
    return np.sin(x)

  x_rango = np.linspace(lx2.min(), lx2.max(), 200)

  y_spline = [evaluar_spline(coefs_, x_val) for x_val in x_rango]
  y_original = [func_original2(x_val) for x_val in x_rango]
  plt.figure(figsize=(10, 6))

  plt.plot(lx2, ly2, 'o', color='red', markersize=8, label='Nodos')

  plt.plot(x_rango, y_spline, '--', color='blue', label='Aprox Spline')


  plt.title('Splines, Ejemplo 2')
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