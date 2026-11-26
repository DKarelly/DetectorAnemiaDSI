# En este archivo construyo la interfaz gráfica de mi sistema inteligente usando únicamente
# tkinter (librería nativa de Python). Toda la lógica de la red neuronal está separada en mlp_anemia.py.

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from mlp_anemia import (
    NOMBRES_CLASES,
    RedNeuronalMLP,
    cargar_preprocesador,
    ejecutar_entrenamiento,
    predecir_nino,
)


ETIQUETAS_BONITAS = {
    "sin_anemia": "Sin Anemia",
    "anemia_leve": "Anemia Leve",
    "anemia_moderada": "Anemia Moderada",
}

COLORES_CLASE = {
    "sin_anemia": "#16a34a",
    "anemia_leve": "#d97706",
    "anemia_moderada": "#ea580c",
}

# Opciones legibles para que el usuario no tenga que memorizar los códigos numéricos de ENDES.
EDUCACION_OPCIONES = {
    "Sin educación": "0",
    "Primaria": "1",
    "Secundaria": "2",
    "Superior": "3",
}

QUINTIL_OPCIONES = {
    "1 - Más pobre": "1",
    "2 - Pobre": "2",
    "3 - Medio": "3",
    "4 - Rico": "4",
    "5 - Más rico": "5",
}

AGUA_OPCIONES = {
    "Red pública dentro de la vivienda": "11",
    "Red pública fuera de la vivienda": "12",
    "Pilón de uso público": "13",
    "Pozo en la vivienda/patio": "21",
    "Pozo público": "22",
    "Manantial / puquio": "41",
    "Río / acequia / lago": "43",
    "Agua de lluvia": "51",
    "Camión cisterna": "61",
    "Agua embotellada": "71",
    "Otro": "96",
}

SANEAMIENTO_OPCIONES = {
    "Red pública dentro de la vivienda": "11",
    "Red pública fuera de la vivienda": "12",
    "Letrina ventilada": "21",
    "Pozo séptico": "22",
    "Pozo ciego / negro": "23",
    "Río / acequia / canal": "31",
    "Sin servicio / campo abierto": "32",
    "Otro": "96",
}


class AplicacionAnemia(tk.Tk):
    # Defino la ventana principal de mi aplicación con tres pestañas:
    # evaluación de un niño, métricas/matriz de confusión y curva de aprendizaje.

    def __init__(self):
        super().__init__()
        self.title("Sistema Inteligente MLP — Clasificación de Anemia Infantil (ENDES 2023–2025)")
        self.geometry("1080x720")
        self.minsize(980, 660)
        self.configure(bg="#f1f5f9")

        # Estas rutas son relativas a la carpeta del proyecto, desde donde se ejecuta la app.
        self.ruta_csv = Path("ENDES_2025_MLP_macrorregion_norte.csv")
        self.ruta_meta = Path("preprocesador_endes.json")
        self.ruta_modelo = Path("artifacts/modelo_mlp.json")
        self.ruta_metricas = Path("artifacts/metricas.json")

        self.modelo = None
        self.meta = None
        self.metricas = None
        self.entrenando = False

        self._configurar_estilos()
        self._construir_encabezado()
        self._construir_pestanas()
        self._cargar_artefactos_iniciales()

    def _configurar_estilos(self):
        # Uso el tema clam para que los controles tengan colores y tamaños uniformes.
        estilo = ttk.Style(self)
        if "clam" in estilo.theme_names():
            estilo.theme_use("clam")
        estilo.configure("TNotebook", background="#f1f5f9", borderwidth=0)
        estilo.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"))
        estilo.configure("Card.TFrame", background="#ffffff", relief="flat")
        estilo.configure("TLabel", background="#ffffff", foreground="#1e293b", font=("Segoe UI", 9))
        estilo.configure("Titulo.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 11, "bold"))
        estilo.configure("Primario.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))

    def _construir_encabezado(self):
        cabecera = tk.Frame(self, bg="#0f172a", padx=20, pady=14)
        cabecera.pack(fill="x")
        tk.Label(
            cabecera,
            text="Clasificador de Anemia Infantil (6 a 59 meses) — Perceptrón Multicapa (MLP)",
            bg="#0f172a", fg="#ffffff", font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        tk.Label(
            cabecera,
            text="Macrorregión Norte del Perú (ENDES 2023–2025)  |  Algoritmo 100% Python Puro (sin librerías externas de IA)",
            bg="#0f172a", fg="#94a3b8", font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

    def _construir_pestanas(self):
        # Separo la predicción, la revisión de métricas y el entrenamiento en vistas independientes.
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=12)

        self.tab_prediccion = tk.Frame(self.notebook, bg="#f1f5f9")
        self.tab_metricas = tk.Frame(self.notebook, bg="#f1f5f9")
        self.tab_entrenar = tk.Frame(self.notebook, bg="#f1f5f9")

        self.notebook.add(self.tab_prediccion, text="  Evaluar Niño (Predicción)  ")
        self.notebook.add(self.tab_metricas, text="  Métricas y Matriz de Confusión  ")
        self.notebook.add(self.tab_entrenar, text="  Curva de Pérdida y Entrenamiento  ")

        self._construir_tab_prediccion()
        self._construir_tab_metricas()
        self._construir_tab_entrenar()

    @staticmethod
    def _validar_numero(valor, permitir_decimal):
        # Permito dejar el campo vacío o escribir un decimal incompleto mientras se edita;
        # rechazo letras, signos, comas y más de un punto decimal.
        if not valor:
            return True
        if permitir_decimal and valor.count(".") > 1:
            return False
        return all(
            "0" <= caracter <= "9" or (permitir_decimal and caracter == ".")
            for caracter in valor
        )

    # ------------------------------------------------------------------
    # PESTAÑA 1: FORMULARIO DE EVALUACIÓN Y BARRAS DE PROBABILIDAD
    # ------------------------------------------------------------------
    def _construir_tab_prediccion(self):
        contenedor = tk.Frame(self.tab_prediccion, bg="#f1f5f9")
        contenedor.pack(fill="both", expand=True, padx=6, pady=6)
        contenedor.columnconfigure(0, weight=3)
        contenedor.columnconfigure(1, weight=2)
        contenedor.rowconfigure(0, weight=1)

        # Panel izquierdo: ingreso de variables del menor y hogar.
        panel_izq = tk.Frame(contenedor, bg="#ffffff", bd=1, relief="solid", padx=16, pady=14)
        panel_izq.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(
            panel_izq, text="Datos del Niño (6 a 59 meses) y del Hogar",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        self.vars = {}
        campos_numericos = [
            ("edad_meses", "Edad (meses, 6–59):", "18"),
            ("peso_kg", "Peso (kg):", "10.5"),
            ("talla_cm", "Talla (cm):", "79.0"),
            ("altitud_m", "Altitud (m s.n.m.):", "213"),
            ("personas_hogar", "Personas en hogar:", "4"),
            ("ninos_menores_5", "Niños < 5 años:", "1"),
        ]

        for idx, (clave, etiqueta, defecto) in enumerate(campos_numericos):
            fila = 1 + (idx // 2)
            col = (idx % 2) * 2
            tk.Label(panel_izq, text=etiqueta, bg="#ffffff", fg="#334155", font=("Segoe UI", 9)).grid(
                row=fila, column=col, sticky="w", padx=(0, 6), pady=5
            )
            var = tk.StringVar(value=defecto)
            self.vars[clave] = var
            # Edad y cantidades son enteros; peso, talla y altitud admiten un punto decimal.
            permitir_decimal = clave in {"peso_kg", "talla_cm", "altitud_m"}
            validar = self.register(
                lambda valor, decimal=permitir_decimal: self._validar_numero(valor, decimal)
            )
            ttk.Entry(
                panel_izq,
                textvariable=var,
                width=16,
                validate="key",
                validatecommand=(validar, "%P"),
            ).grid(
                row=fila, column=col + 1, sticky="ew", padx=(0, 14), pady=5
            )

        departamentos = [
            "Amazonas", "Áncash", "Cajamarca", "La Libertad",
            "Lambayeque", "Piura", "San Martín", "Tumbes",
        ]
        # Mantengo las opciones categóricas cerradas para evitar valores que el modelo no conoce.
        campos_combo = [
            ("sexo", "Sexo:", ["Hombre", "Mujer"], "Hombre"),
            ("department", "Departamento:", departamentos, "Lambayeque"),
            ("area_residencia", "Área de residencia:", ["Urbano", "Rural"], "Urbano"),
            ("quintil_riqueza", "Quintil de riqueza:", list(QUINTIL_OPCIONES.keys()), "2 - Pobre"),
            ("educacion_madre", "Educación madre:", list(EDUCACION_OPCIONES.keys()), "Secundaria"),
            ("fuente_agua", "Fuente de agua:", list(AGUA_OPCIONES.keys()), "Red pública dentro de la vivienda"),
            ("servicio_higienico", "Servicio higiénico:", list(SANEAMIENTO_OPCIONES.keys()), "Red pública dentro de la vivienda"),
            ("diarrea_14d", "Diarrea (últ. 14d):", ["No", "Sí"], "No"),
            ("fiebre_14d", "Fiebre (últ. 14d):", ["No", "Sí"], "No"),
            ("tos_14d", "Tos (últ. 14d):", ["No", "Sí"], "No"),
            ("vitamina_a_6m", "Vitamina A (6m):", ["No", "Sí"], "No"),
            ("hierro_12m", "Suplem. hierro (12m):", ["Sí", "No"], "Sí"),
        ]

        fila_base = 4
        for idx, (clave, etiqueta, opciones, defecto) in enumerate(campos_combo):
            fila = fila_base + (idx // 2)
            col = (idx % 2) * 2
            tk.Label(panel_izq, text=etiqueta, bg="#ffffff", fg="#334155", font=("Segoe UI", 9)).grid(
                row=fila, column=col, sticky="w", padx=(0, 6), pady=5
            )
            var = tk.StringVar(value=defecto)
            self.vars[clave] = var
            ttk.Combobox(panel_izq, textvariable=var, values=opciones, state="readonly", width=22).grid(
                row=fila, column=col + 1, sticky="ew", padx=(0, 14), pady=5
            )

        panel_izq.columnconfigure(1, weight=1)
        panel_izq.columnconfigure(3, weight=1)

        # Botones de acción en la parte inferior del formulario.
        barra_botones = tk.Frame(panel_izq, bg="#ffffff")
        barra_botones.grid(row=12, column=0, columnspan=4, sticky="ew", pady=(16, 4))

        ttk.Button(
            barra_botones, text="Predecir Nivel de Anemia",
            style="Primario.TButton", command=self._ejecutar_prediccion,
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            barra_botones, text="Restaurar valores ejemplo",
            command=self._restaurar_ejemplo,
        ).pack(side="left")

        # Panel derecho: resultado de la predicción y gráfico de probabilidades.
        panel_der = tk.Frame(contenedor, bg="#ffffff", bd=1, relief="solid", padx=16, pady=14)
        panel_der.grid(row=0, column=1, sticky="nsew")

        tk.Label(
            panel_der, text="Resultado de la Red Neuronal (MLP)",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        self.caja_resultado = tk.Frame(panel_der, bg="#f8fafc", bd=1, relief="solid", padx=12, pady=14)
        self.caja_resultado.pack(fill="x", pady=(12, 14))

        self.lbl_clase_titulo = tk.Label(
            self.caja_resultado, text="Nivel estimado:", bg="#f8fafc", fg="#475569", font=("Segoe UI", 9)
        )
        self.lbl_clase_titulo.pack(anchor="w")

        self.lbl_clase_valor = tk.Label(
            self.caja_resultado, text="Presiona 'Predecir' para evaluar",
            bg="#f8fafc", fg="#0f172a", font=("Segoe UI", 16, "bold"),
        )
        self.lbl_clase_valor.pack(anchor="w", pady=(2, 0))

        tk.Label(
            panel_der, text="Distribución de probabilidades (Softmax):",
            bg="#ffffff", fg="#334155", font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(4, 6))

        self.canvas_probs = tk.Canvas(panel_der, width=360, height=220, bg="#ffffff", highlightthickness=0)
        self.canvas_probs.pack(fill="x", pady=(0, 12))

        tk.Label(
            panel_der,
            text="Nota académica: Este sistema es una herramienta educativa de apoyo\n"
                 "predictivo basada en datos de la ENDES 2023–2025.\n"
                 "No sustituye un examen de hemoglobina ni un diagnóstico médico.",
            bg="#fefce8", fg="#854d0e", font=("Segoe UI", 8),
            justify="left", padx=10, pady=8, relief="solid", bd=1,
        ).pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # PESTAÑA 2: MÉTRICAS Y MATRIZ DE CONFUSIÓN VISUAL
    # ------------------------------------------------------------------
    def _construir_tab_metricas(self):
        # Esta vista resume el test separado y permite comparar clases con la matriz de confusión.
        marco = tk.Frame(self.tab_metricas, bg="#f1f5f9")
        marco.pack(fill="both", expand=True, padx=6, pady=6)
        marco.columnconfigure(0, weight=1)
        marco.columnconfigure(1, weight=1)
        marco.rowconfigure(1, weight=1)

        # Tarjeta superior con indicadores globales.
        self.frame_kpis = tk.Frame(marco, bg="#ffffff", bd=1, relief="solid", padx=16, pady=12)
        self.frame_kpis.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.lbl_resumen_modelo = tk.Label(
            self.frame_kpis, text="Aún no se cargan métricas.",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 10), justify="left",
        )
        self.lbl_resumen_modelo.pack(anchor="w")

        # Izquierda: tabla de métricas por clase.
        panel_tabla = tk.Frame(marco, bg="#ffffff", bd=1, relief="solid", padx=14, pady=12)
        panel_tabla.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(
            panel_tabla, text="Desempeño por Clase (Conjunto de Prueba Real 20%)",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        columnas = ("clase", "precision", "recall", "f1", "soporte")
        self.tabla_clases = ttk.Treeview(panel_tabla, columns=columnas, show="headings", height=8)
        for col, texto, ancho in [
            ("clase", "Nivel de Anemia", 145),
            ("precision", "Precision", 80),
            ("recall", "Recall", 80),
            ("f1", "F1-Score", 80),
            ("soporte", "Soporte", 75),
        ]:
            self.tabla_clases.heading(col, text=texto)
            self.tabla_clases.column(col, width=ancho, anchor="center")
        self.tabla_clases.pack(fill="x", pady=(0, 12))

        self.lbl_cv_resumen = tk.Label(
            panel_tabla, text="", bg="#f8fafc", fg="#334155",
            font=("Segoe UI", 9), justify="left", padx=10, pady=8, relief="solid", bd=1,
        )
        self.lbl_cv_resumen.pack(fill="x")

        # Derecha: matriz de confusión dibujada en Canvas.
        panel_cm = tk.Frame(marco, bg="#ffffff", bd=1, relief="solid", padx=14, pady=12)
        panel_cm.grid(row=1, column=1, sticky="nsew")

        tk.Label(
            panel_cm, text="Matriz de Confusión (Filas = Real, Columnas = Predicho)",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.canvas_cm = tk.Canvas(panel_cm, width=440, height=340, bg="#ffffff", highlightthickness=0)
        self.canvas_cm.pack(expand=True)

    # ------------------------------------------------------------------
    # PESTAÑA 3: CURVA DE PÉRDIDA Y RE-ENTRENAMIENTO EN VIVO
    # ------------------------------------------------------------------
    def _construir_tab_entrenar(self):
        # La validación cruzada es opcional porque requiere entrenar cinco modelos adicionales.
        marco = tk.Frame(self.tab_entrenar, bg="#f1f5f9")
        marco.pack(fill="both", expand=True, padx=6, pady=6)
        marco.rowconfigure(1, weight=1)
        marco.columnconfigure(0, weight=1)

        controles = tk.Frame(marco, bg="#ffffff", bd=1, relief="solid", padx=16, pady=12)
        controles.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.var_incluir_cv = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controles,
            text="Incluir Validación Cruzada 5-Fold (toma más tiempo)",
            variable=self.var_incluir_cv,
        ).pack(side="left", padx=(0, 16))

        self.btn_reentrenar = ttk.Button(
            controles, text="Re-entrenar MLP desde cero",
            style="Primario.TButton", command=self._iniciar_entrenamiento,
        )
        self.btn_reentrenar.pack(side="left")

        self.lbl_estado_train = tk.Label(
            controles, text="Estado: Listo", bg="#ffffff", fg="#0f766e", font=("Segoe UI", 9, "bold")
        )
        self.lbl_estado_train.pack(side="right")

        panel_grafico = tk.Frame(marco, bg="#ffffff", bd=1, relief="solid", padx=16, pady=14)
        panel_grafico.grid(row=1, column=0, sticky="nsew")

        tk.Label(
            panel_grafico, text="Curva de Pérdida (Cross-Entropy) por Época — Entrenamiento vs Validación",
            bg="#ffffff", fg="#0f172a", font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        self.canvas_loss = tk.Canvas(panel_grafico, width=960, height=360, bg="#ffffff", highlightthickness=0)
        self.canvas_loss.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # CARGA DE MODELO, PREDICCIÓN Y DIBUJO DE GRÁFICOS NATIVOS
    # ------------------------------------------------------------------
    def _cargar_artefactos_iniciales(self):
        # Cargo los archivos existentes y descarto resultados que no coincidan con las tres clases actuales.
        if self.ruta_meta.exists():
            self.meta = cargar_preprocesador(self.ruta_meta)
        if self.ruta_modelo.exists():
            modelo_guardado = RedNeuronalMLP.cargar(self.ruta_modelo)
            if modelo_guardado.n_salidas == len(NOMBRES_CLASES):
                self.modelo = modelo_guardado
            else:
                self.lbl_estado_train.config(
                    text="Estado: Modelo desactualizado; vuelve a entrenar."
                )
        if self.ruta_metricas.exists():
            metricas_guardadas = json.loads(self.ruta_metricas.read_text(encoding="utf-8"))
            metricas_prueba = metricas_guardadas.get("metricas_prueba", {})
            matriz = metricas_prueba.get("matriz_confusion", [])
            if len(matriz) == len(NOMBRES_CLASES) and all(
                len(fila) == len(NOMBRES_CLASES) for fila in matriz
            ):
                self.metricas = metricas_guardadas
                self._actualizar_vistas_metricas()
            else:
                self.lbl_estado_train.config(
                    text="Estado: Métricas desactualizadas; vuelve a entrenar."
                )
        self._dibujar_barras_probabilidad({c: 0.0 for c in NOMBRES_CLASES})

    def _restaurar_ejemplo(self):
        # Repongo valores conocidos y válidos para que el usuario pueda probar una predicción de inmediato.
        valores = {
            "edad_meses": "18", "peso_kg": "10.5", "talla_cm": "79.0",
            "altitud_m": "213", "personas_hogar": "4", "ninos_menores_5": "1",
            "sexo": "Hombre", "department": "Lambayeque", "area_residencia": "Urbano",
            "quintil_riqueza": "2 - Pobre", "educacion_madre": "Secundaria",
            "fuente_agua": "Red pública dentro de la vivienda",
            "servicio_higienico": "Red pública dentro de la vivienda",
            "diarrea_14d": "No", "fiebre_14d": "No", "tos_14d": "No",
            "vitamina_a_6m": "No", "hierro_12m": "Sí",
        }
        for k, v in valores.items():
            self.vars[k].set(v)

    def _ejecutar_prediccion(self):
        if self.modelo is None or self.meta is None:
            messagebox.showwarning(
                "Modelo no entrenado",
                "Primero entrena el modelo ejecutando mlp_anemia.py o desde la pestaña de Entrenamiento.",
            )
            return

        # Convierto las entradas numéricas a float y muestro un aviso si quedó un valor vacío o incompleto.
        try:
            edad = float(self.vars["edad_meses"].get())
            peso = float(self.vars["peso_kg"].get())
            talla = float(self.vars["talla_cm"].get())
            altitud = float(self.vars["altitud_m"].get())
            personas = float(self.vars["personas_hogar"].get())
            menores5 = float(self.vars["ninos_menores_5"].get())
        except ValueError:
            messagebox.showerror("Dato inválido", "Por favor ingresa solo números válidos en los campos numéricos.")
            return

        if not (6 <= edad <= 59):
            messagebox.showerror("Rango de edad", "La edad del niño debe estar entre 6 y 59 meses.")
            return

        # Traduzco las etiquetas legibles del formulario a los códigos que se usaron al entrenar.
        datos_usuario = {
            "edad_meses": edad,
            "peso_kg": peso,
            "talla_cm": talla,
            "altitud_m": altitud,
            "personas_hogar": personas,
            "ninos_menores_5": menores5,
            "sexo": self.vars["sexo"].get(),
            "department": self.vars["department"].get(),
            "area_residencia": self.vars["area_residencia"].get(),
            "quintil_riqueza": QUINTIL_OPCIONES[self.vars["quintil_riqueza"].get()],
            "educacion_madre": EDUCACION_OPCIONES[self.vars["educacion_madre"].get()],
            "fuente_agua": AGUA_OPCIONES[self.vars["fuente_agua"].get()],
            "servicio_higienico": SANEAMIENTO_OPCIONES[self.vars["servicio_higienico"].get()],
            "diarrea_14d": self.vars["diarrea_14d"].get(),
            "fiebre_14d": self.vars["fiebre_14d"].get(),
            "tos_14d": self.vars["tos_14d"].get(),
            "vitamina_a_6m": self.vars["vitamina_a_6m"].get(),
            "hierro_12m": self.vars["hierro_12m"].get(),
        }

        resultado = predecir_nino(self.modelo, self.meta, datos_usuario)
        clase = resultado["clase_nombre"]
        color = COLORES_CLASE[clase]

        self.lbl_clase_valor.config(text=ETIQUETAS_BONITAS[clase], fg=color)
        self._dibujar_barras_probabilidad(resultado["probabilidades"])

    def _dibujar_barras_probabilidad(self, probabilidades):
        c = self.canvas_probs
        c.delete("all")
        # Dibujo una barra por clase: una barra más larga representa mayor probabilidad Softmax.
        ancho_barra_max = 210
        x_inicio = 125
        y = 18

        for clave in NOMBRES_CLASES:
            prob = probabilidades.get(clave, 0.0)
            etiqueta = ETIQUETAS_BONITAS[clave]
            color = COLORES_CLASE[clave]

            c.create_text(10, y + 14, text=etiqueta, anchor="w", font=("Segoe UI", 9, "bold"), fill="#1e293b")
            c.create_rectangle(x_inicio, y + 2, x_inicio + ancho_barra_max, y + 26, fill="#e2e8f0", outline="")
            ancho = max(2, int(ancho_barra_max * prob)) if prob > 0 else 0
            if ancho > 0:
                c.create_rectangle(x_inicio, y + 2, x_inicio + ancho, y + 26, fill=color, outline="")
            c.create_text(
                x_inicio + ancho_barra_max + 10, y + 14,
                text=f"{prob * 100:5.1f}%", anchor="w", font=("Segoe UI", 9, "bold"), fill="#0f172a",
            )
            y += 48

    def _actualizar_vistas_metricas(self):
        if not self.metricas:
            return
        # Reflejo el reporte del test, los promedios de validación cruzada y las curvas guardadas.
        m = self.metricas["metricas_prueba"]
        arq = " → ".join(str(n) for n in self.metricas["arquitectura"])
        self.lbl_resumen_modelo.config(
            text=(
                f"Arquitectura MLP: {arq}   |   Registros totales: {self.metricas['registros_totales']:,} "
                f"(Train: {self.metricas['registros_entrenamiento']:,} · Test real: {self.metricas['registros_prueba']:,})\n"
                f"Accuracy: {m['accuracy']:.2%}   |   Precision macro: {m['precision_macro']:.4f}   |   "
                f"Recall macro: {m['recall_macro']:.4f}   |   F1-score macro: {m['f1_macro']:.4f}   |   "
                f"Épocas: {m['epocas_entrenadas']}   |   Tiempo: {self.metricas['tiempo_segundos']} s"
            )
        )

        # Relleno la tabla por clase.
        for item in self.tabla_clases.get_children():
            self.tabla_clases.delete(item)
        for clave in NOMBRES_CLASES:
            info = m["reporte_por_clase"][clave]
            self.tabla_clases.insert(
                "", "end",
                values=(
                    ETIQUETAS_BONITAS[clave],
                    f"{info['precision']:.4f}",
                    f"{info['recall']:.4f}",
                    f"{info['f1_score']:.4f}",
                    info["soporte"],
                ),
            )

        # Muestro el resumen de validación cruzada 5-Fold si fue ejecutada.
        folds = self.metricas.get("validacion_cruzada_5_folds", [])
        if folds:
            acc_prom = sum(f["accuracy"] for f in folds) / len(folds)
            f1_prom = sum(f["f1_macro"] for f in folds) / len(folds)
            rec_prom = sum(f["recall_macro"] for f in folds) / len(folds)
            self.lbl_cv_resumen.config(
                text=(
                    f"Validación Cruzada Estratificada ({len(folds)}-Fold):\n"
                    f"• Accuracy promedio: {acc_prom:.2%}\n"
                    f"• Recall macro promedio: {rec_prom:.4f}\n"
                    f"• F1-score macro promedio: {f1_prom:.4f}"
                )
            )
        else:
            self.lbl_cv_resumen.config(text="Validación cruzada 5-Fold no ejecutada en la última corrida.")

        self._dibujar_matriz_confusion(m["matriz_confusion"])
        self._dibujar_curva_perdida(self.metricas.get("historial_final", {}))

    def _dibujar_matriz_confusion(self, matriz):
        c = self.canvas_cm
        c.delete("all")
        etiquetas_cortas = ["Sin\nanemia", "Leve", "Moderada"]
        x0, y0 = 90, 55
        celda = 66
        # Normalizo el color por el mayor conteo para distinguir visualmente valores altos y bajos.
        max_val = max(max(fila) for fila in matriz) or 1

        for j, texto in enumerate(etiquetas_cortas):
            c.create_text(x0 + j * celda + celda // 2, y0 - 22, text=texto, font=("Segoe UI", 8, "bold"), fill="#334155")

        for i, fila in enumerate(matriz):
            c.create_text(x0 - 12, y0 + i * celda + celda // 2, text=etiquetas_cortas[i], anchor="e", font=("Segoe UI", 8, "bold"), fill="#334155")
            for j, valor in enumerate(fila):
                intensidad = valor / max_val
                # Calculo un tono azul suave proporcional al valor de cada celda.
                r = int(241 - 190 * intensidad)
                g = int(245 - 140 * intensidad)
                b = int(249 - 40 * intensidad)
                color_fondo = f"#{r:02x}{g:02x}{b:02x}"
                color_texto = "#ffffff" if intensidad > 0.45 else "#0f172a"

                x1 = x0 + j * celda
                y1 = y0 + i * celda
                c.create_rectangle(x1, y1, x1 + celda, y1 + celda, fill=color_fondo, outline="#cbd5e1")
                c.create_text(x1 + celda // 2, y1 + celda // 2, text=str(valor), font=("Segoe UI", 10, "bold"), fill=color_texto)

    def _dibujar_curva_perdida(self, historial):
        c = self.canvas_loss
        c.delete("all")
        train_loss = historial.get("train_loss", [])
        val_loss = historial.get("val_loss", [])
        if not train_loss:
            c.create_text(450, 160, text="Sin historial de entrenamiento disponible.", fill="#64748b", font=("Segoe UI", 10))
            return

        w, h = 900, 320
        izq, der, arr, abj = 70, 40, 30, 50
        ancho = w - izq - der
        alto = h - arr - abj

        max_loss = max(max(train_loss), max(val_loss)) * 1.05
        min_loss = min(min(train_loss), min(val_loss)) * 0.95
        if max_loss - min_loss < 1e-6:
            max_loss += 0.5

        # Uso el rango observado en las pérdidas para convertirlas a coordenadas verticales del Canvas.
        # Dibujo ejes y cuadrícula.
        c.create_line(izq, arr, izq, arr + alto, fill="#334155", width=2)
        c.create_line(izq, arr + alto, izq + ancho, arr + alto, fill="#334155", width=2)

        for paso in range(5):
            frac = paso / 4.0
            y = arr + alto - frac * alto
            val = min_loss + frac * (max_loss - min_loss)
            c.create_line(izq, y, izq + ancho, y, fill="#e2e8f0", dash=(3, 3))
            c.create_text(izq - 8, y, text=f"{val:.2f}", anchor="e", font=("Segoe UI", 8), fill="#475569")

        n = len(train_loss)

        def puntos(serie):
            # Mapeo cada época al eje X y su pérdida al eje Y para dibujar la línea.
            pts = []
            for idx, val in enumerate(serie):
                x = izq + (idx / max(1, n - 1)) * ancho
                y = arr + alto - ((val - min_loss) / (max_loss - min_loss)) * alto
                pts.extend([x, y])
            return pts

        if n > 1:
            c.create_line(*puntos(train_loss), fill="#2563eb", width=2.5, smooth=False)
            c.create_line(*puntos(val_loss), fill="#dc2626", width=2.5, smooth=False)

        c.create_text(izq + ancho // 2, arr + alto + 30, text=f"Épocas (1 a {n})", font=("Segoe UI", 9, "bold"), fill="#334155")

        # Leyenda superior derecha.
        c.create_line(izq + ancho - 230, arr + 10, izq + ancho - 200, arr + 10, fill="#2563eb", width=3)
        c.create_text(izq + ancho - 194, arr + 10, text="Pérdida Entrenamiento", anchor="w", font=("Segoe UI", 9), fill="#1e293b")
        c.create_line(izq + ancho - 230, arr + 30, izq + ancho - 200, arr + 30, fill="#dc2626", width=3)
        c.create_text(izq + ancho - 194, arr + 30, text="Pérdida Validación", anchor="w", font=("Segoe UI", 9), fill="#1e293b")

    def _iniciar_entrenamiento(self):
        if self.entrenando:
            return
        # Ejecuto el cálculo en segundo plano y reservo self.after para actualizar widgets desde Tk.
        self.entrenando = True
        self.btn_reentrenar.config(state="disabled")
        self.lbl_estado_train.config(text="Estado: Entrenando en segundo plano...")

        incluir_cv = self.var_incluir_cv.get()

        def tarea():
            # Este hilo solo calcula; los cambios de interfaz se programan en el hilo principal.
            try:
                modelo, resultados = ejecutar_entrenamiento(
                    ruta_csv=self.ruta_csv,
                    carpeta_salida="artifacts",
                    ejecutar_cv=incluir_cv,
                    callback_estado=lambda msg: self.after(0, lambda m=msg: self.lbl_estado_train.config(text=m)),
                )
                self.modelo = modelo
                self.metricas = resultados
                self.after(0, self._fin_entrenamiento_ok)
            except Exception as err:
                self.after(0, lambda e=err: self._fin_entrenamiento_error(e))

        threading.Thread(target=tarea, daemon=True).start()

    def _fin_entrenamiento_ok(self):
        # Reactivo el botón, actualizo los gráficos y confirmo que se guardaron los resultados.
        self.entrenando = False
        self.btn_reentrenar.config(state="normal")
        self.lbl_estado_train.config(text="Estado: Entrenamiento completado.")
        self._actualizar_vistas_metricas()
        messagebox.showinfo("Entrenamiento finalizado", "El modelo MLP se entrenó y guardó correctamente.")

    def _fin_entrenamiento_error(self, err):
        # Restauro la interfaz y presento el error recibido desde el hilo de entrenamiento.
        self.entrenando = False
        self.btn_reentrenar.config(state="normal")
        self.lbl_estado_train.config(text="Estado: Error durante el entrenamiento.")
        messagebox.showerror("Error", f"Ocurrió un problema al entrenar:\n{err}")


if __name__ == "__main__":
    app = AplicacionAnemia()
    app.mainloop()
