import csv

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 42  # para que siempre salgan los mismos numeros aleatorios
ITERACIONES = 50000  # iteraciones que si se cuentan
DESCARTE = 1000  # iteraciones del inicio que no se cuentan
TEMPERATURAS = [0.2, 1.0, 5.0]
VENTANA = 200  # iteraciones que se ven en la grafica de energia

AZUL = "#2a78d6"
NARANJA = "#eb6834"

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
    encabezado = ["experimento", "temperatura", "estado", "energia",
                  "prob_teorica", "frecuencia", "error_abs"]
    with open(nombre, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


def correr(pesos, sesgos, temperatura):
    # parte teorica + simulacion para una red y una temperatura
    # siempre se empieza en 000 y con la misma semilla
    n = len(sesgos)
    estados, energias, factores, z, probs = distribucion_teorica(pesos, sesgos, temperatura)
    visitados, energias_sim = simular([0] * n, pesos, sesgos, temperatura,
                                      DESCARTE + ITERACIONES, SEMILLA)
    return {
        "temperatura": temperatura,
        "estados": estados,
        "energias": energias,
        "factores": factores,
        "z": z,
        "probs": probs,
        "visitados": visitados,
        "energias_sim": energias_sim,
        "frecuencias": contar_frecuencias(visitados[DESCARTE:], n),
    }


def filas_csv(nombre, r):
    filas = []
    for k in range(len(r["estados"])):
        error = abs(r["probs"][k] - r["frecuencias"][k])
        filas.append([nombre, r["temperatura"], etiqueta(r["estados"][k]),
                      round(r["energias"][k], 6), round(r["probs"][k], 6),
                      round(r["frecuencias"][k], 6), round(error, 6)])
    return filas


def graficar_energia(resultados):
    # una grafica por temperatura, con el mismo eje para poder comparar
    fig, ejes = plt.subplots(len(resultados), 1, figsize=(8, 7), sharex=True, sharey=True)
    for eje, r in zip(ejes, resultados):
        eje.plot(range(1, VENTANA + 1), r["energias_sim"][:VENTANA], color=AZUL, linewidth=1.2)
        eje.set_title(f"T = {r['temperatura']}")
        eje.set_ylabel("Energia E(s)")
        eje.grid(alpha=0.3)
    ejes[-1].set_xlabel("Iteracion")
    fig.suptitle(f"Energia de la red en las primeras {VENTANA} iteraciones")
    fig.tight_layout()
    fig.savefig("energia_temperaturas.png", dpi=150)
    plt.close(fig)


def graficar_histogramas(resultados):
    # barras = lo que salio en la simulacion, puntos = probabilidad teorica
    fig, ejes = plt.subplots(1, len(resultados), figsize=(11, 4), sharey=True)
    for eje, r in zip(ejes, resultados):
        nombres = [etiqueta(s) for s in r["estados"]]
        eje.bar(nombres, r["frecuencias"], color=AZUL, label="Simulacion")
        eje.plot(nombres, r["probs"], "o", color="black", markersize=4, label="Teorica")
        eje.set_title(f"T = {r['temperatura']}")
        eje.set_xlabel("Estado")
        eje.grid(axis="y", alpha=0.3)
    ejes[0].set_ylabel("Frecuencia")
    ejes[-1].legend()
    fig.suptitle("Frecuencia de cada estado segun la temperatura")
    fig.tight_layout()
    fig.savefig("histograma_temperaturas.png", dpi=150)
    plt.close(fig)


def graficar_pesos(original, modificado):
    nombres = [etiqueta(s) for s in original["estados"]]
    x = np.arange(len(nombres))
    ancho = 0.38
    fig, eje = plt.subplots(figsize=(8, 4.5))
    eje.bar(x - 0.2, original["frecuencias"], ancho, color=AZUL, label="Original (w12 = 0.8)")
    eje.bar(x + 0.2, modificado["frecuencias"], ancho, color=NARANJA, label="Modificado (w12 = -0.8)")
    eje.set_xticks(x)
    eje.set_xticklabels(nombres)
    eje.set_xlabel("Estado")
    eje.set_ylabel("Frecuencia")
    eje.set_title("Efecto de cambiar el signo de w12 (T = 1)")
    eje.grid(axis="y", alpha=0.3)
    eje.legend()
    fig.tight_layout()
    fig.savefig("comparacion_pesos.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    validar_pesos(PESOS)
    filas = []

    # ---- 8.1 efecto de la temperatura ----
    resultados = [correr(PESOS, SESGOS, t) for t in TEMPERATURAS]

    print("EFECTO DE LA TEMPERATURA")
    print(f"semilla = {SEMILLA}, iteraciones = {ITERACIONES}, descarte = {DESCARTE}\n")
    print(f"T     Mas frecuente  Energia media  Desv. energia  "
          f"Visitados en {VENTANA}  Visitados en total")
    for r in resultados:
        mas_frecuente = r["frecuencias"].index(max(r["frecuencias"]))
        contadas = r["energias_sim"][DESCARTE:]
        en_ventana = len(set(etiqueta(s) for s in r["visitados"][:VENTANA]))
        en_total = sum(1 for f in r["frecuencias"] if f > 0)
        print(f"{r['temperatura']:<5} {etiqueta(r['estados'][mas_frecuente]):<14} "
              f"{np.mean(contadas):+.4f}        {np.std(contadas):.4f}         "
              f"{en_ventana:<17} {en_total}")
        filas += filas_csv("original", r)

    # ---- 8.3 teorico vs experimental con T = 1 ----
    r = resultados[1]
    print("\nTEORICO VS EXPERIMENTAL (T = 1)")
    print("Estado    E(s)     e^(-E)   P teorica  Frecuencia  Error")
    for k in range(len(r["estados"])):
        error = abs(r["probs"][k] - r["frecuencias"][k])
        print(f"{etiqueta(r['estados'][k])}     {r['energias'][k]:+.4f}   {r['factores'][k]:.4f}   "
              f"{r['probs'][k]:.4f}     {r['frecuencias'][k]:.4f}      {error:.4f}")
    print(f"Z = {r['z']:.4f}")
    print(f"Suma de probabilidades = {sum(r['probs']):.10f}")

    # ---- 8.2 efecto de los pesos: se cambia el signo de w12 ----
    pesos_mod = PESOS.copy()
    pesos_mod[0, 1] = -0.8
    pesos_mod[1, 0] = -0.8  # hay que cambiar los dos para que siga simetrica
    modificado = correr(pesos_mod, SESGOS, 1.0)

    print("\nEFECTO DE LOS PESOS (w12 de 0.8 a -0.8, T = 1)")
    print("Estado  E original  E modificada  Frec. original  Frec. modificada")
    for k in range(len(r["estados"])):
        print(f"{etiqueta(r['estados'][k])}     {r['energias'][k]:+.4f}     {modificado['energias'][k]:+.4f}       "
              f"{r['frecuencias'][k]:.4f}          {modificado['frecuencias'][k]:.4f}")
    filas += filas_csv("w12_negativo", modificado)

    # ---- archivos de salida ----
    guardar_csv("resultados.csv", filas)
    graficar_energia(resultados)
    graficar_histogramas(resultados)
    graficar_pesos(r, modificado)
    print("\nSe guardaron resultados.csv y las tres graficas")