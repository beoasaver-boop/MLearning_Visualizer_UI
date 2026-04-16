# MLearning_Visualizer_UI

Aplicación de escritorio en **Python** para **machine learning** (clasificación y regresión) con aprendizaje incremental y **visualización en tiempo real**. Permite cargar tablas (Excel o CSV), elegir variables predictoras y objetivo, y entrenar un modelo mientras se observan métricas en vivo mediante 6 gráficas que se actualizan por épocas.

## Características principales

- **Cuatro tipos de modelos**:
  - Regresión Logística (clasificación binaria/multiclase)
  - Regresión Lineal Simple (1 variable independiente)
  - Regresión Lineal Múltiple (múltiples variables independientes)
  - Random Forest (clasificación)

- **6 gráficas en tiempo real**:
  - Pérdida (Loss)
  - Precisión/R² Score
  - Evolución de coeficientes
  - Importancia de características
  - Residuos/Matriz de confusión
  - Sobreajuste (overfitting)


## Estructura del proyecto (NUEVA)

```
MLearning_Visualizer_UI/
├── config/               # Configuración de temas
├── ml_gui/              # Interfaz gráfica (separada en módulos)
├── analytics/           # Lógica de ML (modelos)
├── utils/               # Funciones auxiliares
├── main.py              # Punto de entrada
├── menu_principal.py    # Menú de selección
└── requirements.txt
```


## Requisitos
> [!IMPORTANT]
>- **Python 3.8+** (recomendado 3.10 o superior).
>- **Tkinter**: en Windows suele venir con la instalación oficial de Python. En Linux puede requerirse >el paquete del sistema (`python3-tk`). La línea `tkinter>=3.0.0` en `requirements.txt` no siempre se >instala vía `pip`; si falla, omítela e instala Tk desde el sistema.
>- Dependencias Python: ver [`requirements.txt`](requirements.txt) (`pandas`, `numpy`, `scikit-learn`, >`matplotlib`, `openpyxl`, `joblib`, `tkinter`).

## Instalación en Windows

```bash
# Instalando la interfaz en tu windows:
git clone https://github.com/beoasaver-boop/MLearning_Visualizer_UI.git
cd MLearning_Visualizer_UI
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```
# Instalación en Linux / MacOS

```bash
# Linux/macOS:
sudo apt update
sudo apt install python3 python3-tk
git clone https://github.com/beoasaver-boop/MLearning_Visualizer_UI.git
cd MLearning_Visualizer_UI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si `pip` no encuentra el paquete `tkinter`, instálalo solo a nivel de sistema o usa el Python de python.org que incluye Tk.

## Uso

Desde la raíz del proyecto:

```bash
python main.py
```


### Pipeline de Procesamiento

El pipeline automatizado incluye los siguientes pasos para cada modelo:

1. **Carga y Validación de Datos**: Soporta archivos Excel (.xlsx, .xls) y CSV. Valida formato y estructura básica.

2. **Limpieza de Datos**: 
   - Eliminación de filas con valores nulos (opción 1)
   - Imputación con mediana/moda (opción 2)
   - Relleno con cero (opción 3)

3. **Codificación de Variables Categóricas**: Transforma variables no numéricas usando Label Encoding.

4. **Estandarización**: Escala las características numéricas usando StandardScaler.

5. **División Train/Test**: Separa los datos según el porcentaje especificado (default 30% test).

6. **Entrenamiento Incremental**: 
   - Regresión Logística: SGDClassifier con pérdida log
   - Regresión Lineal: SGDRegressor para simple/múltiple
   - Random Forest: RandomForestClassifier con crecimiento incremental de árboles

7. **Visualización en Tiempo Real**: Actualiza 6 gráficas por época durante el entrenamiento.

8. **Evaluación Final**: Calcula métricas finales y genera reportes detallados.

### Flujo en la Interfaz

> [!TIP]
> 1. **Seleccionar Modelo**: Elige entre Regresión Logística, Lineal Simple, Lineal Múltiple o Random Forest desde el menú principal.
> 2. **Cargar datos** — Excel (`.xlsx`, `.xls`) o CSV (`.csv`).
> 3. **Seleccionar variables** — Elige las variables predictoras y objetivo de la lista disponible. Para regresión simple, selecciona exactamente 1 feature.
> 4. **Parámetros** — Ajusta:
>   - **Tamaño de prueba** — Proporción del conjunto de test (por defecto `0.3`).
>   - **Número de epochs** — Pasadas de entrenamiento incremental (por defecto `100`).
>   - **Manejo de nulos** — `1` eliminar filas con nulos; `2` imputar mediana (numéricas) / moda (categóricas); `3` rellenar con `0`.
>   - **Learning Rate** (solo para regresión lineal) — Tasa de aprendizaje para SGD.
> 5. **Iniciar entrenamiento** — El proceso corre en un **hilo en segundo plano** para no bloquear la ventana.

### Pestañas

- **Entrenamiento en vivo** — Curvas de **pérdida** (train/test) y **precisión** (train/test) actualizadas en cada epoch. La cuadrícula incluye espacio adicional para otros gráficos (títulos reservados en la UI).
- **Resultados finales** — Mejor y última accuracy, mejor epoch, **importancia de características** (valor absoluto de coeficientes), **matriz de confusión** y **reporte de clasificación** (precisión, recall, F1, soporte).

## Estructura del proyecto

| Archivo / módulo | Rol |
|------------------|-----|
| [`main.py`](main.py) | Punto de entrada: menú principal para seleccionar tipo de modelo. |
| [`menu_principal.py`](menu_principal.py) | Interfaz de selección de modelo (Regresión Logística, Lineal, Random Forest). |
| `ml_gui/` | Módulos de interfaz gráfica. |
| ├── [`app.py`](ml_gui/app.py) | Clase principal de la aplicación GUI (`MLVisualizerApp`). |
| ├── [`callbacks.py`](ml_gui/callbacks.py) | Callbacks para eventos de la interfaz (carga, selección, entrenamiento). |
| ├── [`widgets.py`](ml_gui/widgets.py) | Definición de widgets y layouts. |
| ├── [`plots.py`](ml_gui/plots.py) | Gestión de gráficas en tiempo real. |
| ├── [`results.py`](ml_gui/results.py) | Visualización de resultados finales. |
| ├── [`eda_viewer.py`](ml_gui/eda_viewer.py) | Exploración de datos. |
| ├── [`rf_plots.py`](ml_gui/rf_plots.py) | Gráficas específicas para Random Forest. |
| ├── [`tooltips.py`](ml_gui/tooltips.py) | Tooltips y ayuda. |
| `analytics/` | Lógica de machine learning. |
| ├── [`automl_core.py`](analytics/automl_core.py) | Regresión Logística (`AutoMLVisualizer`). |
| ├── [`linear_regression_core.py`](analytics/linear_regression_core.py) | Regresión Lineal (`LinearRegressionVisualizer`). |
| ├── [`random_forest_core.py`](analytics/random_forest_core.py) | Random Forest (`RandomForestVisualizer`). |
| `config/` | Configuración. |
| ├── [`theme.py`](config/theme.py) | Tema oscuro para Tkinter y estilos de Matplotlib. |
| `utils/` | Utilidades. |
| ├── [`helpers.py`](utils/helpers.py) | Funciones auxiliares y validaciones. |
| [`requirements.txt`](requirements.txt) | Dependencias Python. |


## Limitaciones y notas

- Es una herramienta orientada a **exploración y docencia**; no sustituye a pipelines de producción sin validación adicional (hiperparámetros, validación cruzada, etc.).


> [!TIP]
> **LICENSE:** This project contains code under multiple third-part licenses: [Numpy: https://numpy.org], [Pandas: https://pandas.pydata.org/], [tkinter: https://docs.python.org/es/3/library/tkinter.html], [Matplotlib: https://matplotlib.org/], [sklearn: https://scikit-learn.org/stable/index.html], [joblib: https://joblib.readthedocs.io/en/stable/], [openpyxl: https://openpyxl.readthedocs.io/en/stable/], specially thanks to all those libraries and their incredible job to the community. 
> The current codebase includes components licensed under the MLreaningVisualizer License with an additional requirement to preserve the "MLearning_Visualizer_UI" branding, as well as prior contributions under their respective original licenses. For a detailed record of license changes and the applicable terms for each section of the code, please refer to LICENSE in source code. 

