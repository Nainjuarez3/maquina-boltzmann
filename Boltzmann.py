import csv
import numpy as np

SEMILLA = 42  # para que siempre salgan los mismos numeros aleatorios
ITERACIONES = 50000  # iteraciones que si se cuentan
DESCARTE = 1000  # iteraciones del inicio que no se cuentan

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
        raise ValueError("La matriz de pesos debe ser simetrica (w_ij = w_ji).")

    # una neurona no se conecta con ella misma
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


def probabilidad_activacion(indice, estado, pesos, sesgos, temperatura):
    # delta_e = cuanto baja la energia si la neurona se enciende
    # delta_e = suma(w_ij * s_j) + b_i
    # (como la diagonal es cero, la neurona no se cuenta a si misma)
    s = np.asarray(estado, dtype=float)
    delta_e = pesos[indice] @ s + sesgos[indice]
    # sigmoide: P(s_i = 1) = 1 / (1 + e^(-delta_e / T))
    return float(1.0 / (1.0 + np.exp(-delta_e / temperatura)))


def actualizar_unidad(indice, estado, pesos, sesgos, temperatura, generador):
    # la neurona se enciende con probabilidad p, si no, se apaga
    p = probabilidad_activacion(indice, estado, pesos, sesgos, temperatura)
    nuevo = list(estado)
    if generador.random() < p:
        nuevo[indice] = 1
    else:
        nuevo[indice] = 0
    return nuevo


def simular(estado_inicial, pesos, sesgos, temperatura, iteraciones, semilla):
    # una iteracion = actualizar todas las neuronas, una por una
    # con la misma semilla siempre sale la misma secuencia
    validar_pesos(pesos)
    generador = np.random.default_rng(semilla)
    estado = list(estado_inicial)

    estados = []
    energias = []
    for _ in range(iteraciones):
        for i in range(len(estado)):
            estado = actualizar_unidad(i, estado, pesos, sesgos, temperatura, generador)
        # se guarda como quedo la red al terminar la iteracion
        estados.append(estado)
        energias.append(energia(estado, pesos, sesgos))
    return estados, energias


def estado_desde_numero(numero, n):
    # pasa un numero a binario, por ejemplo 5 -> [1, 0, 1]
    binario = format(numero, f"0{n}b")
    return [int(c) for c in binario]


def etiqueta(estado):
    # [1, 0, 1] -> "101"
    return "".join(str(v) for v in estado)


def distribucion_teorica(pesos, sesgos, temperatura):
    # recorre todos los estados posibles y calcula E, e^(-E/T), Z y P
    n = len(sesgos)
    estados = [estado_desde_numero(k, n) for k in range(2 ** n)]
    energias = [energia(s, pesos, sesgos) for s in estados]
    factores = [float(np.exp(-e / temperatura)) for e in energias]
    z = sum(factores)  # funcion de particion
    probabilidades = [f / z for f in factores]
    return estados, energias, factores, z, probabilidades


def contar_frecuencias(visitados, n):
    # fraccion de las iteraciones que la red paso en cada estado
    conteo = [0] * (2 ** n)
    for s in visitados:
        numero = int(etiqueta(s), 2)  # "101" -> 5
        conteo[numero] += 1
    return [c / len(visitados) for c in conteo]


def guardar_csv(nombre, filas):
    encabezado = ["temperatura", "estado", "energia", "prob_teorica",
                  "frecuencia", "error_abs"]
    with open(nombre, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


if __name__ == "__main__":
    validar_pesos(PESOS)
    n = len(SESGOS)
    temperatura = 1.0

    # parte teorica
    estados, energias, factores, z, probs = distribucion_teorica(PESOS, SESGOS, temperatura)

    # parte experimental: se simula y se tiran las primeras iteraciones
    visitados, _ = simular([0, 0, 0], PESOS, SESGOS, temperatura,
                           DESCARTE + ITERACIONES, SEMILLA)
    frecuencias = contar_frecuencias(visitados[DESCARTE:], n)

    print(f"T = {temperatura}, semilla = {SEMILLA}, "
          f"iteraciones = {ITERACIONES}, descarte = {DESCARTE}\n")
    print("Estado    E(s)     e^(-E)   P teorica  Frecuencia  Error")
    filas = []
    for k in range(2 ** n):
        error = abs(probs[k] - frecuencias[k])
        print(f"{etiqueta(estados[k])}     {energias[k]:+.4f}   {factores[k]:.4f}   "
              f"{probs[k]:.4f}     {frecuencias[k]:.4f}      {error:.4f}")
        filas.append([temperatura, etiqueta(estados[k]), round(energias[k], 6),
                      round(probs[k], 6), round(frecuencias[k], 6), round(error, 6)])

    mas_probable = probs.index(max(probs))
    menor_energia = energias.index(min(energias))
    print(f"\nZ = {z:.4f}")
    print(f"Suma de probabilidades = {sum(probs):.10f}")
    print(f"Estado mas probable: {etiqueta(estados[mas_probable])}")
    print(f"Estado de menor energia: {etiqueta(estados[menor_energia])}")

    guardar_csv("resultados.csv", filas)
    print("\nSe guardo resultados.csv")