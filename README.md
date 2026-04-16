# MLearning_Visualizer_UI

Aplicación de escritorio en **Python** para **machine learning** (clasificación y regresión) con aprendizaje incremental y **visualización en tiempo real**. Permite cargar tablas (Excel o CSV), elegir variables predictoras y objetivo, y entrenar un modelo mientras se observan métricas en vivo mediante 6 gráficas que se actualizan por épocas.

## ✨ Características principales

- 🎯 **Tres tipos de modelos**:
  - Regresión Logística (clasificación binaria/multiclase)
  - Regresión Lineal Simple (1 variable independiente)
  - Regresión Lineal Múltiple (múltiples variables independientes)

- 📊 **6 gráficas en tiempo real**:
  - Pérdida (Loss)
  - Precisión/R² Score
  - Evolución de coeficientes
  - Importancia de características
  - Residuos/Matriz de confusión
  - Sobreajuste (overfitting)


## 📁 Estructura del proyecto (NUEVA)

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

- **Python 3.8+** (recomendado 3.10 o superior).
- **Tkinter**: en Windows suele venir con la instalación oficial de Python. En Linux puede requerirse el paquete del sistema (`python3-tk`). La línea `tkinter>=3.0.0` en `requirements.txt` no siempre se instala vía `pip`; si falla, omítela e instala Tk desde el sistema.
- Dependencias Python: ver [`requirements.txt`](requirements.txt) (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `openpyxl`, `joblib`, `tkinter`).

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

Opcionalmente, coloca un archivo **`icon.ico`** junto a `main.py` para el icono de la ventana.

### Flujo en la interfaz

1. **Cargar datos** — Excel (`.xlsx`, `.xls`) o CSV (`.csv`).
2. **Seleccionar variables** — En el cuadro de texto, indica las **features** separadas por comas (nombres exactos de columnas) y la **variable objetivo** en el campo correspondiente. Pulsa **Confirmar variables**.
3. **Parámetros** — Ajusta:
   - **Tamaño de prueba** — Proporción del conjunto de test (por defecto `0.3`).
   - **Número de epochs** — Pasadas de entrenamiento incremental (por defecto `100`).
   - **Manejo de nulos** — `1` eliminar filas con nulos; `2` imputar mediana (numéricas) / moda (categóricas); `3` rellenar con `0`.
4. **Iniciar entrenamiento** — El proceso corre en un **hilo en segundo plano** para no bloquear la ventana.

### Pestañas

- **Entrenamiento en vivo** — Curvas de **pérdida** (train/test) y **precisión** (train/test) actualizadas en cada epoch. La cuadrícula incluye espacio adicional para otros gráficos (títulos reservados en la UI).
- **Resultados finales** — Mejor y última accuracy, mejor epoch, **importancia de características** (valor absoluto de coeficientes), **matriz de confusión** y **reporte de clasificación** (precisión, recall, F1, soporte).

## Estructura del proyecto

| Archivo / módulo | Rol |
|------------------|-----|
| [`main.py`](main.py) | Punto de entrada: ventana Tk centrada (~1400×900), arranca `MLVisualizerApp`. |
| [`ml_gui.py`](ml_gui.py) | Interfaz gráfica (`MLVisualizerApp`): carga de archivo, formularios, hilos de entrenamiento, integración con Matplotlib. |
| [`automl_core.py`](automl_core.py) | Lógica ML (`AutoMLVisualizer`): carga, limpieza, codificación, escalado, división train/test, entrenamiento y métricas. |
| [`styles.py`](styles.py) | Colores del tema oscuro para Tkinter y parámetros de estilo de Matplotlib. |

## Detalles del modelo y del pipeline

- **Algoritmo**: `sklearn.linear_model.SGDClassifier` con `loss='log_loss'` (equivalente práctico a regresión logística con descenso por gradiente estocástico), `warm_start=True` y `max_iter=1` por epoch para simular entrenamiento por épocas.
- **Preprocesado**: `LabelEncoder` en categóricas y en el target si es texto; `StandardScaler` sobre las features; partición estratificada con `train_test_split`.
- **Clases**: Soporta **binaria** y **multicategoría**; las métricas de pérdida se adaptan según el número de clases.

## Limitaciones y notas

- Es una herramienta orientada a **exploración y docencia**; no sustituye a pipelines de producción sin validación adicional (hiperparámetros, validación cruzada, etc.).
- `joblib` está incluido en dependencias e importado en el núcleo, pero **no hay guardado de modelo en disco** implementado en el código actual; se puede añadir serialización con `joblib.dump` si lo necesitas.


> [!TIP]
> **LICENSE:** This project contains code under multiple third-part licenses: [Numpy: https://numpy.org], [Pandas: https://pandas.pydata.org/], [tkinter: https://docs.python.org/es/3/library/tkinter.html], [Matplotlib: https://matplotlib.org/], [sklearn: https://scikit-learn.org/stable/index.html]. The current codebase includes components licensed under the Open WebUI License with an additional requirement to preserve the "MLearning_Visualizer_UI" branding, as well as prior contributions under their respective original licenses. For a detailed record of license changes and the applicable terms for each section of the code, please refer to LICENSE in source code. 

