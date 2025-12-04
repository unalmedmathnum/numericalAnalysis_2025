import numpy as np
import matplotlib.pyplot as plt

def euler_mejorado_heun(f, t0, w0, h, N):
    """
    Implementa el método de Heun.

    Parámetros:
    -----------
    f : función
        Función f(t, w) de la EDO w' = f(t, w)
    t0 : float
        Valor inicial de t
    w0 : float
        Valor inicial w(t0) = α
    h : float
        Tamaño del paso
    N : int
        Número de iteraciones

    Retorna:
    --------
    t : array
        Vector de tiempos
    w : array
        Vector de soluciones aproximadas
    """
    t = np.zeros(N + 1)
    w = np.zeros(N + 1)

    t[0] = t0
    w[0] = w0

    for i in range(N):
        # Cálculo según la fórmula de Heun
        k1 = f(t[i], w[i])
        k2 = f(t[i] + 2*h/3, w[i] + (2*h/3)*f(t[i]+h/3,w[i]+(h/3)*k1))

        w[i+1] = w[i] + (h/4) * (k1 + 3*k2)
        t[i+1] = t[i] + h

    return t, w

if __name__ == "__main__":
  # Ejemplo del problema: y' = y - t² + 1, y(0) = 0.5
  def f(t, y):
      return y - t**2 + 1

  # Solución exacta: y(t) = (t + 1)² - 0.5e^t
  def solucion_exacta(t):
      return (t + 1)**2 - 0.5 * np.exp(t)


  # Parámetros según el problema
  t0 = 0.0
  w0 = 0.5
  h = 0.2
  N = 10
  t_final = N * h  # t_final = 2.0

  # Aplicar el método de Heun
  t_heun, w_heun = euler_mejorado_heun(f, t0, w0, h, N)

  # Calcular solución exacta y error
  y_exacta = solucion_exacta(t_heun)
  error = np.abs(w_heun - y_exacta)

  # Mostrar resultados en tabla
  print("="*70)
  print("MÉTODO DE HEUN")
  print("="*70)
  print(f"Ecuación: y' = y - t² + 1")
  print(f"Condición inicial: y(0) = {w0}")
  print(f"Intervalo: 0 ≤ t ≤ {t_final}")
  print(f"Paso: h = {h}")
  print(f"Número de iteraciones: N = {N}")
  print("="*70)
  print()
  print(f"{'i':^5} | {'t_i':^8} | {'w_i (Heun)':^15} | {'y_i (Exacta)':^15} | {'Error':^12}")
  print("-"*80)

  for i in range(N + 1):
      print(f"{i:^5} | {t_heun[i]:^8.1f} | {w_heun[i]:^15.8f} | {y_exacta[i]:^15.8f} | {error[i]:^15.8f}")

  print("="*80)
  print(f"\nSolución aproximada en t = {t_final}: y({t_final}) ≈ {w_heun[-1]:.8f}")
  print(f"Solución exacta en t = {t_final}:     y({t_final}) = {y_exacta[-1]:.8f}")
  print(f"Error absoluto final: {error[-1]:.8e}")
  print(f"Error máximo: {np.max(error):.8e}")
  print()

  # Graficar la solución
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

  # Gráfico 1: Solución aproximada vs exacta
  t_continuo = np.linspace(0, t_final, 200)
  y_continuo = solucion_exacta(t_continuo)

  ax1.plot(t_continuo, y_continuo, 'k-', linewidth=2, label='Solución exacta')
  ax1.plot(t_heun, w_heun, 'bo-', linewidth=2, markersize=8, label='Método de Heun')
  ax1.set_xlabel('t', fontsize=12)
  ax1.set_ylabel('y(t)', fontsize=12)
  ax1.set_title("Método de Heun: y' = y - t² + 1, y(0) = 0.5", fontsize=13)
  ax1.grid(True, alpha=0.3)
  ax1.legend(fontsize=11)
  ax1.set_xlim(-0.1, 2.1)

  # Gráfico 2: Error absoluto por iteración
  ax2.semilogy(t_heun, error, 'ro-', linewidth=2, markersize=8, label='Error absoluto')
  ax2.set_xlabel('t', fontsize=12)
  ax2.set_ylabel('Error absoluto', fontsize=12)
  ax2.set_title('Error en cada iteración', fontsize=13)
  ax2.legend(fontsize=11)
  ax2.grid(True, alpha=0.3)
  ax2.set_xlim(-0.1, 2.1)

  plt.tight_layout()
  plt.show()
