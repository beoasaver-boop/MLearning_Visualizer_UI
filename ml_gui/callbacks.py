from tkinter import messagebox, filedialog
import threading
import numpy as np
from sklearn.metrics import accuracy_score

from analytics import AutoMLVisualizer, LinearRegressionVisualizer, RandomForestVisualizer
from ml_gui.eda_viewer import EDAViewer

class DataLoadingCallbacks:
    def __init__(self, app):
        self.app = app

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        self.app.file_path = file_path
        self.app.file_label.configure(text=f"{file_path.split('/')[-1]}")
        self.app.update_status(f"Archivo cargado: {file_path}")

        model_type = self.app.model_type
        if model_type == 'logistic':
            self.app.automl = AutoMLVisualizer(
                status_callback=self.app.update_status,
                plot_callback=self.app.update_training_plots
            )
        elif model_type in ['simple', 'multiple']:
            self.app.automl = LinearRegressionVisualizer(
                status_callback=self.app.update_status,
                plot_callback=self.app.update_training_plots
            )
        elif model_type == 'random_forest':
            self.app.automl = RandomForestVisualizer(
                status_callback=self.app.update_status,
                plot_callback=self.app.update_training_plots
            )
        else:
            messagebox.showerror("Error", f"Tipo de modelo no reconocido: {model_type}")
            return

        try:
            self.app.automl.load_data(file_path)
            df = self.app.automl.df
            self.app.update_status(f"Columnas disponibles: {self.app.automl.get_columns()}")

            self._show_eda(df, file_path.split('/')[-1])

            self.app.populate_feature_checkbuttons()
            self.app.confirm_vars_btn.configure(state="normal")
        except Exception as e:
            self.app.update_status(f"Error al cargar: {str(e)}")
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")

    def _show_eda(self, df, filename):
        try:
            EDAViewer(self.app.root, df, filename)
        except Exception as e:
            self.app.update_status(f"No se pudo mostrar EDA: {str(e)}")

class VariableSelectionCallbacks:
    def __init__(self, app):
        self.app = app

    def confirm_variables(self):
        features = [col for col, var in self.app.feature_vars.items() if var.get() == "on"]
        target = self.app.target_var.get().strip()

        if self.app.model_type == 'simple':
            if len(features) != 1:
                messagebox.showwarning(
                    "Advertencia",
                    "Regresion Lineal Simple requiere EXACTAMENTE 1 variable independiente."
                )
                return
        else:
            if len(features) == 0:
                messagebox.showwarning(
                    "Advertencia",
                    "Por favor seleccione al menos una variable independiente."
                )
                return

        if not target:
            messagebox.showwarning("Advertencia", "Por favor seleccione la variable dependiente")
            return

        available_cols = self.app.automl.get_columns()
        if target not in available_cols:
            messagebox.showerror("Error", f"Variable objetivo '{target}' no encontrada\n"
                                          f"Columnas disponibles: {available_cols}")
            return

        if target in features:
            messagebox.showerror("Error", "La variable objetivo no puede ser tambien una variable independiente")
            return

        self.app.automl.set_variables(features, target)

        model_desc = self.app.get_model_name()
        self.app.update_status(f"{model_desc} configurada:")
        self.app.update_status(f"   Features ({len(features)}): {features}")
        self.app.update_status(f"   Target: {target}")

        self.app.train_btn.configure(state="normal")

    def select_all(self):
        for var in self.app.feature_vars.values():
            var.set("on")
        if self.app.model_type == 'simple' and self.app.feature_vars:
            first = list(self.app.feature_vars.keys())[0]
            for col, var in self.app.feature_vars.items():
                var.set("on" if col == first else "off")

    def deselect_all(self):
        for var in self.app.feature_vars.values():
            var.set("off")

class TrainingCallbacks:
    def __init__(self, app):
        self.app = app

    def start_training(self):
        if self.app.is_training:
            return

        try:
            test_size = float(self.app.test_size_var.get())

            nulls_map = {
                "1 - Eliminar filas": "1",
                "2 - Mediana/Moda": "2",
                "3 - Rellenar con 0": "3"
            }
            nulls_raw = self.app.nulls_var.get()
            nulls_handling = nulls_map.get(nulls_raw, "2")

            epochs_raw = self.app.epochs_var.get().strip()
            if epochs_raw.endswith("e"):
                epochs_raw = epochs_raw[:-1]
            n_epochs = int(float(epochs_raw))

            learning_rate = None
            if hasattr(self.app, 'learning_rate_var') and self.app.learning_rate_var and self.app.model_type in ['simple', 'multiple']:
                learning_rate = float(self.app.learning_rate_var.get())
        except ValueError:
            messagebox.showerror("Error", "Parametros invalidos")
            return

        self.app.is_training = True
        self.app.train_btn.configure(state="disabled", text="Entrenando...")
        self.app.confirm_vars_btn.configure(state="disabled")
        self.app.load_btn.configure(state="disabled")
        if self.app.progress_bar:
            self.app.progress_bar.set(0)

        self.app.training_thread = threading.Thread(
            target=self._training_worker,
            args=(test_size, n_epochs, nulls_handling, learning_rate),
            daemon=True
        )
        self.app.training_thread.start()

    def _training_worker(self, test_size, n_epochs, nulls_handling, learning_rate=None):
        try:
            self.app.automl.clean_data(handle_nulls=nulls_handling)
            self.app.automl.split_data(test_size=test_size)

            if self.app.model_type in ['simple', 'multiple'] and learning_rate:
                results = self.app.automl.train_and_visualize(n_epochs=n_epochs, learning_rate=learning_rate)
            else:
                results = self.app.automl.train_and_visualize(n_epochs=n_epochs)

            self.app.root.after(0, self.app.show_final_results, results)
        except Exception as e:
            self.app.update_status(f"Error durante entrenamiento: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.app.is_training = False
            self.app.root.after(0, self._enable_buttons)

    def _enable_buttons(self):
        self.app.train_btn.configure(state="normal", text="Iniciar Entrenamiento")
        self.app.confirm_vars_btn.configure(state="normal")
        self.app.load_btn.configure(state="normal")
