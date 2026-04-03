# MLearning_Visualizer_UI

Aplicación de escritorio en **Python** para **clasificación** con aprendizaje incremental y **visualización en tiempo real**. Permite cargar tablas (Excel o CSV), elegir variables predictoras y objetivo, y entrenar un modelo tipo **regresión logística** (mediante `SGDClassifier` con pérdida logarítmica, equivalente a regresión logística por descenso de gradiente), mientras se observan métricas en vivo por medio de 6 gráficas que se actualizan por épocas.

## Requisitos

- **Python 3.8+** (recomendado 3.10 o superior).
- **Tkinter**: en Windows suele venir con la instalación oficial de Python. En Linux puede requerirse el paquete del sistema (`python3-tk`). La línea `tkinter>=3.0.0` en `requirements.txt` no siempre se instala vía `pip`; si falla, omítela e instala Tk desde el sistema.
- Dependencias Python: ver [`requirements.txt`](requirements.txt) (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `openpyxl`, `joblib`, `tkinter`).

## Instalación

```bash
cd MLearning_Visualizer_UI
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

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

## Licencia

Si el repositorio no define licencia, aclárala según tu caso antes de redistribuir el código.
