# boltzmann v1
import numpy as np

# pesos y sesgos del ejercicio 4.3
PESOS = np.array([
    [0.0, 0.8, -0.4],
    [0.8, 0.0, 0.6],
    [-0.4, 0.6, 0.0],
])
SESGOS = np.array([0.2, -0.1, 0.3])


def validar_pesos(pesos, tolerancia=1e-12):
    # la matriz tiene que ser cuadrada, simetrica y con ceros en la diagonal
    pesos = np.asarray(pesos, dtype=float)

    if pesos.ndim != 2 or pesos.shape[0] != pesos.shape[1]:
        raise ValueError("La matriz de pesos debe ser cuadrada.")

    # simetrica = igual a su transpuesta
    if not np.allclose(pesos, pesos.T, atol=tolerancia):
        raise ValueError("La matriz de pesos debe ser simétrica (w_ij = w_ji).")

    # una neurona no se conecta consigo misma
    if not np.allclose(np.diag(pesos), 0.0, atol=tolerancia):
        raise ValueError("La diagonal de la matriz de pesos debe ser cero.")


def energia(estado, pesos, sesgos):
    # E = -1/2 * suma(w_ij * s_i * s_j) - suma(b_i * s_i)
    # el 1/2 es porque cada par de neuronas se cuenta dos veces
    s = np.asarray(estado, dtype=float)
    parte_pesos = -0.5 * (s @ pesos @ s)
    parte_sesgos = -(sesgos @ s)
    # el + 0.0 es para que el estado 000 no salga como -0.0
    return float(parte_pesos + parte_sesgos) + 0.0


def estado_desde_numero(numero, n):
    # pasa un numero a binario, por ejemplo 5 -> [1, 0, 1]
    binario = format(numero, f"0{n}b")
    return [int(c) for c in binario]


if __name__ == "__main__":
    validar_pesos(PESOS)
    n = len(SESGOS)

    # energia de los 8 estados
    print("Estado   E(s)")
    for numero in range(2 ** n):
        estado = estado_desde_numero(numero, n)
        etiqueta = "".join(str(v) for v in estado)
        print(f"{etiqueta}     {energia(estado, PESOS, SESGOS):+.4f}")