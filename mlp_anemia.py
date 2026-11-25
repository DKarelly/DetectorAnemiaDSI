# En este archivo programo mi Perceptrón Multicapa (MLP)
# No uso TensorFlow, Keras, Scikit-Learn, Pandas ni NumPy: solo librerías nativas de Python.

import csv
import json
import math
import operator
import random
import time
from pathlib import Path


NOMBRES_CLASES = ["sin_anemia", "anemia_leve", "anemia_moderada", "anemia_severa"]
COLUMNA_OBJETIVO = "target_anemia_codigo"
SEMILLA = 42


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS Y PREPROCESAMIENTO PARA INFERENCIA (SIN PANDAS NI SKLEARN)
# ---------------------------------------------------------------------------

def cargar_dataset(ruta_csv):
    # Abro el CSV consolidado de ENDES y separo las columnas de entrada (X) de la etiqueta (y).
    entradas = []
    etiquetas = []
    with open(ruta_csv, mode="r", encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        columnas_entrada = [
            col for col in lector.fieldnames
            if col not in ("target_anemia_nivel", COLUMNA_OBJETIVO)
        ]
        for fila in lector:
            vector = [float(fila[col]) for col in columnas_entrada]
            clase = int(fila[COLUMNA_OBJETIVO])
            entradas.append(vector)
            etiquetas.append(clase)
    return entradas, etiquetas, columnas_entrada


def cargar_preprocesador(ruta_json):
    # Cargo las medias, desviaciones estándar y categorías que guardé del conjunto de entrenamiento
    # para transformar los datos de un niño nuevo exactamente igual que el dataset.
    with open(ruta_json, mode="r", encoding="utf-8") as archivo:
        return json.load(archivo)


def transformar_datos_usuario(datos_usuario, meta):
    # Convierto los datos reales ingresados por el usuario (edad, peso, talla, departamento, etc.)
    # al vector numérico estandarizado y codificado en One-Hot que espera la capa de entrada del MLP.
    vector = []

    # Primero estandarizo las 10 variables numéricas usando (valor - media) / desviación.
    for i, col in enumerate(meta["numeric_columns"]):
        valor = datos_usuario.get(col)
        if valor is None or valor == "":
            valor = meta["numeric_medians"][i]
        else:
            valor = float(valor)
        media = meta["numeric_means"][i]
        desv = meta["numeric_stds"][i]
        z = (valor - media) / desv if desv > 0 else 0.0
        vector.append(z)

    # Luego codifico las 17 variables categóricas con One-Hot (1 en la categoría elegida, 0 en el resto).
    for i, col in enumerate(meta["categorical_columns"]):
        valor = datos_usuario.get(col)
        if valor is None or str(valor).strip() == "":
            valor = meta["categorical_modes"][i]
        else:
            valor = str(valor).strip()
        for categoria in meta["categorical_categories"][i]:
            vector.append(1.0 if valor == categoria else 0.0)

    return vector


# ---------------------------------------------------------------------------
# 2. PARTICIÓN ESTRATIFICADA, VALIDACIÓN CRUZADA Y BALANCEO
# ---------------------------------------------------------------------------

def division_estratificada(entradas, etiquetas, proporcion_prueba=0.20, semilla=SEMILLA):
    # Agrupo los índices por clase para que entrenamiento (80%) y prueba (20%)
    # conserven la misma proporción de cada nivel de anemia.
    generador = random.Random(semilla)
    indices_por_clase = {}
    for idx, clase in enumerate(etiquetas):
        indices_por_clase.setdefault(clase, []).append(idx)

    idx_entrenamiento = []
    idx_prueba = []
    for clase in sorted(indices_por_clase):
        indices = indices_por_clase[clase][:]
        generador.shuffle(indices)
        n_prueba = max(1, round(len(indices) * proporcion_prueba))
        if len(indices) - n_prueba < 1:
            n_prueba = max(0, len(indices) - 1)
        idx_prueba.extend(indices[:n_prueba])
        idx_entrenamiento.extend(indices[n_prueba:])

    generador.shuffle(idx_entrenamiento)
    generador.shuffle(idx_prueba)

    x_train = [entradas[i] for i in idx_entrenamiento]
    y_train = [etiquetas[i] for i in idx_entrenamiento]
    x_test = [entradas[i] for i in idx_prueba]
    y_test = [etiquetas[i] for i in idx_prueba]
    return x_train, x_test, y_train, y_test


def crear_folds_estratificados(etiquetas, k_folds=5, semilla=SEMILLA):
    # Reparto los índices de cada clase en k particiones de forma equilibrada
    # para ejecutar validación cruzada estratificada sin usar scikit-learn.
    generador = random.Random(semilla)
    indices_por_clase = {}
    for idx, clase in enumerate(etiquetas):
        indices_por_clase.setdefault(clase, []).append(idx)

    folds = [[] for _ in range(k_folds)]
    for clase in sorted(indices_por_clase):
        indices = indices_por_clase[clase][:]
        generador.shuffle(indices)
        for pos, idx in enumerate(indices):
            folds[pos % k_folds].append(idx)

    for fold in folds:
        generador.shuffle(fold)
    return folds


def calcular_pesos_clase(etiquetas, num_clases=4):
    # Calculo pesos inversamente proporcionales a la frecuencia de cada clase:
    # n_total / (num_clases * n_clase). Así penalizo más los errores en clases minoritarias.
    total = len(etiquetas)
    conteos = [0] * num_clases
    for c in etiquetas:
        conteos[c] += 1
    pesos = []
    for conteo in conteos:
        if conteo > 0:
            pesos.append(total / (num_clases * conteo))
        else:
            pesos.append(1.0)
    return pesos


def distancia_euclidiana_cuadrada(a, b):
    # Calculo la distancia al cuadrado entre dos vectores para encontrar vecinos cercanos en SMOTE.
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def balancear_entrenamiento(x_train, y_train, objetivos_por_clase=None, semilla=SEMILLA):
    # Como en los datos reales hay muchos más niños sin anemia que con anemia severa,
    # balanceo SOLO el conjunto de entrenamiento (nunca la validación ni la prueba):
    # - Aplico submuestreo en la clase mayoritaria (sin_anemia).
    # - Aplico SMOTE (interpolación entre vecinos reales) en clases minoritarias como anemia_severa.
    if objetivos_por_clase is None:
        objetivos_por_clase = {0: 800, 1: 600, 2: 500, 3: 150}

    generador = random.Random(semilla)
    muestras_por_clase = {}
    for x, y in zip(x_train, y_train):
        muestras_por_clase.setdefault(y, []).append(x)

    x_bal = []
    y_bal = []

    for clase in sorted(muestras_por_clase):
        muestras = muestras_por_clase[clase]
        meta_n = objetivos_por_clase.get(clase, len(muestras))

        if len(muestras) >= meta_n:
            # Si sobran registros en esta clase, selecciono una muestra aleatoria representativa.
            elegidas = generador.sample(muestras, meta_n)
            x_bal.extend(elegidas)
            y_bal.extend([clase] * len(elegidas))
        else:
            # Conservo todos los registros reales y genero los faltantes con SMOTE.
            x_bal.extend(muestras)
            y_bal.extend([clase] * len(muestras))
            faltantes = meta_n - len(muestras)
            n_reales = len(muestras)

            if n_reales == 1:
                for _ in range(faltantes):
                    x_bal.append(muestras[0][:])
                    y_bal.append(clase)
                continue

            # Para cada muestra real, busco sus k vecinos más cercanos dentro de la misma clase.
            k_vecinos = min(5, n_reales - 1)
            vecinos_por_muestra = []
            for i in range(n_reales):
                distancias = [
                    (distancia_euclidiana_cuadrada(muestras[i], muestras[j]), j)
                    for j in range(n_reales) if j != i
                ]
                distancias.sort(key=lambda par: par[0])
                vecinos_por_muestra.append([j for _, j in distancias[:k_vecinos]])

            # Creo cada ejemplo sintético interpolando entre un punto real y uno de sus vecinos.
            for _ in range(faltantes):
                i = generador.randrange(n_reales)
                vecino_idx = generador.choice(vecinos_por_muestra[i])
                alfa = generador.random()
                base = muestras[i]
                vecino = muestras[vecino_idx]
                sintetico = [b + alfa * (v - b) for b, v in zip(base, vecino)]
                x_bal.append(sintetico)
                y_bal.append(clase)

    # Mezclo el conjunto balanceado para que los mini-batches tengan variedad de clases.
    orden = list(range(len(y_bal)))
    generador.shuffle(orden)
    return [x_bal[i] for i in orden], [y_bal[i] for i in orden]


# ---------------------------------------------------------------------------
# 3. ARQUITECTURA DEL PERCEPTRÓN MULTICAPA (MLP) Y OPTIMIZADOR ADAM DESDE CERO
# ---------------------------------------------------------------------------

def producto_punto(a, b):
    # Uso operator.mul y sum (nativos de Python) porque multiplican listas mucho más rápido que un bucle manual.
    return sum(map(operator.mul, a, b))


def softmax(logits):
    # Resto el máximo antes de aplicar exp() para evitar desbordamientos numéricos (overflow).
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    suma = sum(exps)
    return [e / suma for e in exps]


class RedNeuronalMLP:
    # Implemento una red neuronal feedforward con dos capas ocultas (64 y 32 neuronas),
    # activación ReLU en capas ocultas, Softmax en salida, Dropout, penalización L2 y optimizador Adam.

    def __init__(
        self,
        n_entradas,
        capas_ocultas=(64, 32),
        n_salidas=4,
        tasas_dropout=(0.3, 0.4),
        l2_lambda=1e-4,
        tasa_aprendizaje=0.001,
        semilla=SEMILLA,
    ):
        self.n_entradas = n_entradas
        self.capas_ocultas = list(capas_ocultas)
        self.n_salidas = n_salidas
        self.tasas_dropout = list(tasas_dropout)
        self.l2_lambda = l2_lambda
        self.tasa_aprendizaje = tasa_aprendizaje
        self.generador = random.Random(semilla)

        # Defino los tamaños de cada capa: [97, 64, 32, 4].
        self.tamanos = [n_entradas] + self.capas_ocultas + [n_salidas]
        self.num_capas = len(self.tamanos) - 1

        # Inicializo pesos y sesgos:
        # - En capas ReLU uso inicialización He: sqrt(2 / n_in).
        # - En la capa de salida Softmax uso Xavier: sqrt(1 / n_in).
        self.pesos = []
        self.sesgos = []
        for idx in range(self.num_capas):
            n_in = self.tamanos[idx]
            n_out = self.tamanos[idx + 1]
            es_salida = (idx == self.num_capas - 1)
            escala = math.sqrt((1.0 if es_salida else 2.0) / n_in)
            matriz_w = [
                [self.generador.gauss(0.0, escala) for _ in range(n_in)]
                for _ in range(n_out)
            ]
            vector_b = [0.0] * n_out
            self.pesos.append(matriz_w)
            self.sesgos.append(vector_b)

        # Inicializo los momentos de primer orden (m) y segundo orden (v) para Adam.
        self.m_w = [[[0.0] * self.tamanos[i] for _ in range(self.tamanos[i + 1])] for i in range(self.num_capas)]
        self.v_w = [[[0.0] * self.tamanos[i] for _ in range(self.tamanos[i + 1])] for i in range(self.num_capas)]
        self.m_b = [[0.0] * self.tamanos[i + 1] for i in range(self.num_capas)]
        self.v_b = [[0.0] * self.tamanos[i + 1] for i in range(self.num_capas)]
        self.paso_adam = 0

    def propagacion_adelante(self, x, entrenando=False):
        # Paso el vector de entrada capa por capa (forward pass) y guardo las activaciones
        # porque las voy a necesitar durante la retropropagación.
        activaciones = [x]
        mascaras_dropout = []
        actual = x

        for idx in range(self.num_capas):
            matriz_w = self.pesos[idx]
            vector_b = self.sesgos[idx]
            es_salida = (idx == self.num_capas - 1)

            z = [producto_punto(fila_w, actual) + b for fila_w, b in zip(matriz_w, vector_b)]

            if not es_salida:
                # Aplico ReLU: f(z) = max(0, z).
                a = [val if val > 0.0 else 0.0 for val in z]
                tasa_drop = self.tasas_dropout[idx] if idx < len(self.tasas_dropout) else 0.0
                if entrenando and tasa_drop > 0.0:
                    # Aplico Dropout invertido: apago neuronas al azar y escalo las activas.
                    escala_keep = 1.0 / (1.0 - tasa_drop)
                    mascara = [
                        escala_keep if self.generador.random() >= tasa_drop else 0.0
                        for _ in range(len(a))
                    ]
                    a = [val * m for val, m in zip(a, mascara)]
                    mascaras_dropout.append(mascara)
                else:
                    mascaras_dropout.append(None)
                activaciones.append(a)
                actual = a
            else:
                # En la última capa aplico Softmax para obtener probabilidades por clase.
                probs = softmax(z)
                activaciones.append(probs)
                mascaras_dropout.append(None)
                actual = probs

        return actual, activaciones, mascaras_dropout

    def evaluar_perdida_y_exactitud(self, entradas, etiquetas, pesos_clase=None):
        # En una sola pasada calculo tanto la pérdida Cross-Entropy (+ L2) como el Accuracy,
        # evitando repetir la propagación hacia adelante dos veces por época.
        total = len(etiquetas)
        suma_ce = 0.0
        aciertos = 0
        for x, y_real in zip(entradas, etiquetas):
            probs, _, _ = self.propagacion_adelante(x, entrenando=False)
            p = max(probs[y_real], 1e-12)
            peso = pesos_clase[y_real] if pesos_clase else 1.0
            suma_ce += -peso * math.log(p)
            if max(range(len(probs)), key=probs.__getitem__) == y_real:
                aciertos += 1

        perdida_datos = suma_ce / total
        suma_l2 = 0.0
        if self.l2_lambda > 0.0:
            for matriz_w in self.pesos:
                for fila in matriz_w:
                    suma_l2 += producto_punto(fila, fila)
        return perdida_datos + 0.5 * self.l2_lambda * suma_l2, aciertos / total

    def actualizar_con_adam(self, grad_w, grad_b, beta1=0.9, beta2=0.999, eps=1e-8):
        # Implemento el optimizador Adam: combina momento (beta1) y escala adaptativa (beta2)
        # con corrección de sesgo para converger rápido y estable.
        self.paso_adam += 1
        correccion_m = 1.0 - (beta1 ** self.paso_adam)
        correccion_v = 1.0 - (beta2 ** self.paso_adam)
        lr = self.tasa_aprendizaje

        for idx in range(self.num_capas):
            matriz_w = self.pesos[idx]
            mw = self.m_w[idx]
            vw = self.v_w[idx]
            gw = grad_w[idx]

            for i in range(len(matriz_w)):
                fila_w = matriz_w[i]
                fila_m = mw[i]
                fila_v = vw[i]
                fila_g = gw[i]
                for j in range(len(fila_w)):
                    g = fila_g[j]
                    m = beta1 * fila_m[j] + (1.0 - beta1) * g
                    v = beta2 * fila_v[j] + (1.0 - beta2) * (g * g)
                    fila_m[j] = m
                    fila_v[j] = v
                    m_hat = m / correccion_m
                    v_hat = v / correccion_v
                    fila_w[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

            vec_b = self.sesgos[idx]
            mb = self.m_b[idx]
            vb = self.v_b[idx]
            gb = grad_b[idx]
            for i in range(len(vec_b)):
                g = gb[i]
                m = beta1 * mb[i] + (1.0 - beta1) * g
                v = beta2 * vb[i] + (1.0 - beta2) * (g * g)
                mb[i] = m
                vb[i] = v
                m_hat = m / correccion_m
                v_hat = v / correccion_v
                vec_b[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

    def entrenar_minilote(self, lote_x, lote_y, pesos_clase=None):
        # Acumulo los gradientes del mini-lote mediante backpropagation y luego actualizo pesos con Adam.
        tam_lote = len(lote_y)
        grad_w = [[[0.0] * self.tamanos[i] for _ in range(self.tamanos[i + 1])] for i in range(self.num_capas)]
        grad_b = [[0.0] * self.tamanos[i + 1] for i in range(self.num_capas)]
        suma_ce_lote = 0.0

        for x, y_real in zip(lote_x, lote_y):
            probs, activaciones, mascaras = self.propagacion_adelante(x, entrenando=True)
            peso_muestra = pesos_clase[y_real] if pesos_clase else 1.0
            suma_ce_lote += -peso_muestra * math.log(max(probs[y_real], 1e-12))

            # Gradiente de Softmax + Cross-Entropy en la capa de salida: (probs - one_hot(y)) * peso.
            delta = [p * peso_muestra for p in probs]
            delta[y_real] -= peso_muestra

            # Propago el error desde la última capa hasta la primera (backpropagation).
            for idx in range(self.num_capas - 1, -1, -1):
                entrada_capa = activaciones[idx]
                gw_capa = grad_w[idx]
                gb_capa = grad_b[idx]

                for i, d_i in enumerate(delta):
                    if d_i == 0.0:
                        continue
                    gb_capa[i] += d_i
                    fila_gw = gw_capa[i]
                    for j, a_j in enumerate(entrada_capa):
                        if a_j != 0.0:
                            fila_gw[j] += d_i * a_j

                # Si no llegué a la capa de entrada, calculo el delta de la capa anterior.
                if idx > 0:
                    matriz_w = self.pesos[idx]
                    n_prev = self.tamanos[idx]
                    delta_prev = [0.0] * n_prev
                    for i, d_i in enumerate(delta):
                        if d_i == 0.0:
                            continue
                        fila_w = matriz_w[i]
                        for j in range(n_prev):
                            delta_prev[j] += fila_w[j] * d_i

                    # Derivada de ReLU (1 si activación > 0, si no 0) + máscara de Dropout.
                    act_prev = activaciones[idx]
                    mascara = mascaras[idx - 1]
                    if mascara is not None:
                        delta = [
                            dp * m if a_val > 0.0 else 0.0
                            for dp, a_val, m in zip(delta_prev, act_prev, mascara)
                        ]
                    else:
                        delta = [
                            dp if a_val > 0.0 else 0.0
                            for dp, a_val in zip(delta_prev, act_prev)
                        ]

        # Promedio los gradientes del mini-lote y agrego el término de regularización L2.
        inv_lote = 1.0 / tam_lote
        for idx in range(self.num_capas):
            matriz_w = self.pesos[idx]
            gw_capa = grad_w[idx]
            gb_capa = grad_b[idx]
            for i in range(len(matriz_w)):
                gb_capa[i] *= inv_lote
                fila_w = matriz_w[i]
                fila_gw = gw_capa[i]
                for j in range(len(fila_w)):
                    fila_gw[j] = fila_gw[j] * inv_lote + self.l2_lambda * fila_w[j]

        self.actualizar_con_adam(grad_w, grad_b)
        return suma_ce_lote

    def clonar_pesos(self):
        # Hago una copia profunda de los pesos actuales para poder restaurar la mejor época (Early Stopping).
        return (
            [[fila[:] for fila in matriz] for matriz in self.pesos],
            [vec[:] for vec in self.sesgos],
        )

    def restaurar_pesos(self, copia):
        pesos_copia, sesgos_copia = copia
        self.pesos = [[fila[:] for fila in matriz] for matriz in pesos_copia]
        self.sesgos = [vec[:] for vec in sesgos_copia]

    def entrenar(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        epocas=40,
        tam_lote=32,
        paciencia=7,
        pesos_clase=None,
        callback_progreso=None,
    ):
        # Entreno la red por mini-lotes monitoreando la pérdida en validación para detenerme a tiempo.
        historial = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        mejor_val_loss = float("inf")
        mejores_pesos = self.clonar_pesos()
        epocas_sin_mejora = 0
        indices = list(range(len(y_train)))

        for epoca in range(1, epocas + 1):
            self.generador.shuffle(indices)
            suma_train = 0.0
            for inicio in range(0, len(indices), tam_lote):
                lote_idx = indices[inicio:inicio + tam_lote]
                lote_x = [x_train[i] for i in lote_idx]
                lote_y = [y_train[i] for i in lote_idx]
                suma_train += self.entrenar_minilote(lote_x, lote_y, pesos_clase=pesos_clase)

            perdida_train = suma_train / len(y_train)
            perdida_val, acc_val = self.evaluar_perdida_y_exactitud(x_val, y_val, pesos_clase=None)

            historial["train_loss"].append(perdida_train)
            historial["val_loss"].append(perdida_val)
            historial["val_accuracy"].append(acc_val)

            if callback_progreso:
                callback_progreso(epoca, epocas, perdida_train, perdida_val, acc_val)

            # Aplico Early Stopping: guardo pesos si mejora la validación; si no mejora, cuento paciencia.
            if perdida_val < mejor_val_loss - 1e-4:
                mejor_val_loss = perdida_val
                mejores_pesos = self.clonar_pesos()
                epocas_sin_mejora = 0
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= paciencia:
                    break

        self.restaurar_pesos(mejores_pesos)
        return historial

    def predecir_probabilidades(self, entradas):
        # Devuelvo las 4 probabilidades Softmax para cada registro de entrada.
        return [self.propagacion_adelante(x, entrenando=False)[0] for x in entradas]

    def predecir_clases(self, entradas):
        # Elijo la clase con mayor probabilidad para cada registro.
        resultados = []
        for x in entradas:
            probs, _, _ = self.propagacion_adelante(x, entrenando=False)
            mejor_clase = max(range(len(probs)), key=lambda i: probs[i])
            resultados.append(mejor_clase)
        return resultados

    def guardar(self, ruta_json):
        # Guardo la arquitectura y todos los pesos entrenados en un JSON liviano.
        datos = {
            "n_entradas": self.n_entradas,
            "capas_ocultas": self.capas_ocultas,
            "n_salidas": self.n_salidas,
            "tasas_dropout": self.tasas_dropout,
            "l2_lambda": self.l2_lambda,
            "tasa_aprendizaje": self.tasa_aprendizaje,
            "pesos": self.pesos,
            "sesgos": self.sesgos,
        }
        Path(ruta_json).write_text(json.dumps(datos), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta_json):
        # Reconstruyo el modelo entrenado leyendo sus pesos desde el archivo JSON.
        datos = json.loads(Path(ruta_json).read_text(encoding="utf-8"))
        modelo = cls(
            n_entradas=datos["n_entradas"],
            capas_ocultas=datos["capas_ocultas"],
            n_salidas=datos["n_salidas"],
            tasas_dropout=datos["tasas_dropout"],
            l2_lambda=datos["l2_lambda"],
            tasa_aprendizaje=datos["tasa_aprendizaje"],
        )
        modelo.pesos = datos["pesos"]
        modelo.sesgos = datos["sesgos"]
        return modelo


# ---------------------------------------------------------------------------
# 4. MÉTRICAS DE EVALUACIÓN MULTICLASE 
# ---------------------------------------------------------------------------

def evaluar_predicciones(y_real, y_pred, nombres_clases=NOMBRES_CLASES):
    # Construyo la matriz de confusión 4x4 y calculo Accuracy, Precision, Recall y F1-score desde cero.
    n_clases = len(nombres_clases)
    matriz = [[0] * n_clases for _ in range(n_clases)]
    for real, pred in zip(y_real, y_pred):
        matriz[real][pred] += 1

    total = len(y_real)
    aciertos = sum(matriz[i][i] for i in range(n_clases))
    accuracy = aciertos / total if total > 0 else 0.0

    reporte_por_clase = {}
    precisiones = []
    recalls = []
    f1s = []

    for i, nombre in enumerate(nombres_clases):
        tp = matriz[i][i]
        fp = sum(matriz[fila][i] for fila in range(n_clases)) - tp
        fn = sum(matriz[i][col] for col in range(n_clases)) - tp
        soporte = sum(matriz[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        reporte_por_clase[nombre] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "soporte": soporte,
        }
        precisiones.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": sum(precisiones) / n_clases,
        "recall_macro": sum(recalls) / n_clases,
        "f1_macro": sum(f1s) / n_clases,
        "reporte_por_clase": reporte_por_clase,
        "matriz_confusion": matriz,
    }


# ---------------------------------------------------------------------------
# 5. FLUJO COMPLETO DE ENTRENAMIENTO, VALIDACIÓN Y PREDICCIÓN DE UN NIÑO
# ---------------------------------------------------------------------------

def ejecutar_entrenamiento(
    ruta_csv="ENDES_2025_MLP_macrorregion_norte.csv",
    carpeta_salida="artifacts",
    k_folds=5,
    epocas=25,
    epocas_cv=12,
    tam_lote=32,
    paciencia=5,
    ejecutar_cv=True,
    callback_estado=None,
):
    # Coordino todo el pipeline: carga, división 80/20, validación cruzada opcional,
    # entrenamiento del modelo final y guardado de artefactos.
    inicio = time.time()
    carpeta = Path(carpeta_salida)
    carpeta.mkdir(parents=True, exist_ok=True)

    if callback_estado:
        callback_estado("Cargando dataset consolidado ENDES 2023–2025...")
    x, y, columnas = cargar_dataset(ruta_csv)
    x_train, x_test, y_train, y_test = division_estratificada(x, y, proporcion_prueba=0.20, semilla=SEMILLA)

    # Validación cruzada estratificada de 5 particiones sobre el conjunto de entrenamiento.
    metricas_folds = []
    if ejecutar_cv:
        folds = crear_folds_estratificados(y_train, k_folds=k_folds, semilla=SEMILLA)
        for num_fold in range(k_folds):
            if callback_estado:
                callback_estado(f"Entrenando validación cruzada: Fold {num_fold + 1}/{k_folds}...")
            idx_val = set(folds[num_fold])
            xf_tr = [x_train[i] for i in range(len(y_train)) if i not in idx_val]
            yf_tr = [y_train[i] for i in range(len(y_train)) if i not in idx_val]
            xf_va = [x_train[i] for i in folds[num_fold]]
            yf_va = [y_train[i] for i in folds[num_fold]]

            xf_tr_bal, yf_tr_bal = balancear_entrenamiento(xf_tr, yf_tr, semilla=SEMILLA + num_fold)
            pesos_fold = calcular_pesos_clase(yf_tr_bal)

            red_fold = RedNeuronalMLP(n_entradas=len(columnas), semilla=SEMILLA + num_fold)
            hist_fold = red_fold.entrenar(
                xf_tr_bal, yf_tr_bal, xf_va, yf_va,
                epocas=epocas_cv, tam_lote=tam_lote, paciencia=4,
                pesos_clase=pesos_fold,
            )
            preds_fold = red_fold.predecir_clases(xf_va)
            m_fold = evaluar_predicciones(yf_va, preds_fold)
            m_fold["fold"] = num_fold + 1
            m_fold["epocas_entrenadas"] = len(hist_fold["train_loss"])
            metricas_folds.append(m_fold)

    # Entreno el modelo final separando un 20% interno de validación para Early Stopping.
    if callback_estado:
        callback_estado("Entrenando modelo MLP final (97 → 64 → 32 → 4)...")
    xf_tr, xf_va, yf_tr, yf_va = division_estratificada(x_train, y_train, proporcion_prueba=0.20, semilla=SEMILLA)
    xf_tr_bal, yf_tr_bal = balancear_entrenamiento(xf_tr, yf_tr, semilla=SEMILLA)
    pesos_final = calcular_pesos_clase(yf_tr_bal)

    modelo_final = RedNeuronalMLP(n_entradas=len(columnas), semilla=SEMILLA)

    def _progreso(ep, total_ep, tr_l, va_l, va_acc):
        if callback_estado:
            callback_estado(
                f"Modelo final — Época {ep}/{total_ep} | Loss train: {tr_l:.4f} | Loss val: {va_l:.4f} | Acc val: {va_acc:.2%}"
            )

    historial_final = modelo_final.entrenar(
        xf_tr_bal, yf_tr_bal, xf_va, yf_va,
        epocas=epocas, tam_lote=tam_lote, paciencia=paciencia,
        pesos_clase=pesos_final,
        callback_progreso=_progreso,
    )

    # Evalúo de forma ciega sobre el 20% de prueba real (nunca tocado ni balanceado).
    preds_test = modelo_final.predecir_clases(x_test)
    metricas_test = evaluar_predicciones(y_test, preds_test)
    metricas_test["epocas_entrenadas"] = len(historial_final["train_loss"])

    # Guardo el modelo, la matriz de confusión y el reporte completo de métricas.
    modelo_final.guardar(carpeta / "modelo_mlp.json")
    with open(carpeta / "matriz_confusion.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f)
        escritor.writerow(["real\\predicho"] + NOMBRES_CLASES)
        for nombre, fila in zip(NOMBRES_CLASES, metricas_test["matriz_confusion"]):
            escritor.writerow([nombre] + fila)

    resultados = {
        "dataset": str(ruta_csv),
        "semilla": SEMILLA,
        "arquitectura": [len(columnas), 64, 32, len(NOMBRES_CLASES)],
        "registros_totales": len(y),
        "registros_entrenamiento": len(y_train),
        "registros_entrenamiento_balanceado": len(yf_tr_bal),
        "registros_prueba": len(y_test),
        "pesos_clase_entrenamiento": {NOMBRES_CLASES[i]: round(w, 4) for i, w in enumerate(pesos_final)},
        "tiempo_segundos": round(time.time() - inicio, 2),
        "historial_final": historial_final,
        "validacion_cruzada_5_folds": metricas_folds,
        "metricas_prueba": metricas_test,
    }
    (carpeta / "metricas.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return modelo_final, resultados


def predecir_nino(modelo, meta, datos_usuario):
    # Transformo los datos del formulario y obtengo la clase estimada y sus 4 probabilidades.
    vector = transformar_datos_usuario(datos_usuario, meta)
    probs = modelo.predecir_probabilidades([vector])[0]
    clase_idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "clase_codigo": clase_idx,
        "clase_nombre": NOMBRES_CLASES[clase_idx],
        "probabilidades": {NOMBRES_CLASES[i]: probs[i] for i in range(len(NOMBRES_CLASES))},
    }


if __name__ == "__main__":
    print("=== Entrenamiento del MLP para Clasificación de Anemia Infantil ===")
    _, res = ejecutar_entrenamiento(
        callback_estado=lambda msg: print("•", msg)
    )
    m = res["metricas_prueba"]
    print("\n=== Resultados en el conjunto de prueba real (20%) ===")
    print(f"• Tiempo total: {res['tiempo_segundos']} s")
    print(f"• Accuracy:        {m['accuracy']:.4f}")
    print(f"• Precision macro: {m['precision_macro']:.4f}")
    print(f"• Recall macro:    {m['recall_macro']:.4f}")
    print(f"• F1-score macro:  {m['f1_macro']:.4f}")
    print("\n• Matriz de confusión (filas=real, columnas=predicho):")
    for nombre, fila in zip(NOMBRES_CLASES, m["matriz_confusion"]):
        print(f"  {nombre:16s}: {fila}")
