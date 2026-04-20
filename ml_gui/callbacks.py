import tkinter as tk
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
        self.app.file_label.config(text=f"✅ {file_path.split('/')[-1]}")
        self.app.update_status(f"📁 Archivo cargado: {file_path}")
        
        # Crear instancia según tipo de modelo
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
            self.app.update_status(f"✅ Columnas disponibles: {self.app.automl.get_columns()}")
            
            # Mostrar ventana de análisis exploratorio (EDA)
            self._show_eda(df, file_path.split('/')[-1])
            
            self.app.populate_feature_checkbuttons()
            self.app.confirm_vars_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.app.update_status(f"❌ Error al cargar: {str(e)}")
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def _show_eda(self, df, filename):
        """Muestra ventana de análisis exploratorio de datos"""
        try:
            EDAViewer(self.app.root, df, filename)
        except Exception as e:
            self.app.update_status(f"⚠️ No se pudo mostrar EDA: {str(e)}")

class VariableSelectionCallbacks:
    def __init__(self, app):
        self.app = app
    
    def confirm_variables(self):
        # Obtener selección del listbox
        selected_indices = self.app.features_listbox.curselection()
        features = [self.app.features_listbox.get(i) for i in selected_indices]
        target = self.app.target_var.get().strip()
        
        # Validaciones según tipo de modelo
        if self.app.model_type == 'simple':
            if len(features) != 1:
                messagebox.showwarning(
                    "Advertencia", 
                    "Regresión Lineal Simple requiere EXACTAMENTE 1 variable independiente.\n"
                    "Use Ctrl+Click para seleccionar/deseleccionar."
                )
                return
        else:
            if len(features) == 0:
                messagebox.showwarning(
                    "Advertencia", 
                    "Por favor seleccione al menos una variable independiente.\n"
                    "Use Ctrl+Click para seleccionar múltiples."
                )
                return
        
        if not target:
            messagebox.showwarning("Advertencia", "Por favor ingrese la variable dependiente")
            return
        
        available_cols = self.app.automl.get_columns()
        if target not in available_cols:
            messagebox.showerror("Error", f"Variable objetivo '{target}' no encontrada\n"
                                          f"Columnas disponibles: {available_cols}")
            return
        
        if target in features:
            messagebox.showerror("Error", "La variable objetivo no puede ser también una variable independiente")
            return
        
        # Establecer variables
        self.app.automl.set_variables(features, target)
        
        model_desc = self.app.get_model_name()
        self.app.update_status(f"✅ {model_desc} configurada:")
        self.app.update_status(f"   Features ({len(features)}): {features}")
        self.app.update_status(f"   Target: {target}")
        
        # Habilitar botón de entrenamiento
        self.app.train_btn.config(state=tk.NORMAL)
    
    def select_all(self):
        if self.app.features_listbox:
            self.app.features_listbox.selection_set(0, tk.END)
            if self.app.model_type == 'simple' and self.app.features_listbox.size() > 0:
                self.app.features_listbox.selection_clear(1, tk.END)
    
    def deselect_all(self):
        if self.app.features_listbox:
            self.app.features_listbox.selection_clear(0, tk.END)

class TrainingCallbacks:
    def __init__(self, app):
        self.app = app
    
    def start_training(self):
        if self.app.is_training:
            return
        
        try:
            test_size = float(self.app.test_size_var.get())
            n_epochs = int(self.app.epochs_var.get())
            nulls_handling = self.app.nulls_var.get()
            learning_rate = None
            if hasattr(self.app, 'learning_rate_var') and self.app.learning_rate_var and self.app.model_type in ['simple', 'multiple']:
                learning_rate = float(self.app.learning_rate_var.get())
        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos")
            return
        
        self.app.is_training = True
        self.app.train_btn.config(state=tk.DISABLED, text="⏳ ENTRENANDO...")
        self.app.confirm_vars_btn.config(state=tk.DISABLED)
        self.app.load_btn.config(state=tk.DISABLED)
        
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
            self.app.update_status(f"❌ Error durante entrenamiento: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.app.is_training = False
            self.app.root.after(0, self._enable_buttons)
    
    def _enable_buttons(self):
        self.app.train_btn.config(state=tk.NORMAL, text="🚀 INICIAR ENTRENAMIENTO")
        self.app.confirm_vars_btn.config(state=tk.NORMAL)
        self.app.load_btn.config(state=tk.NORMAL)
