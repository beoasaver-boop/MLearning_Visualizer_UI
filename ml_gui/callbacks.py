"""
Manejadores de eventos y callbacks de la interfaz gráfica
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from utils import helpers


class DataLoadingCallbacks:
    """Callbacks para carga de datos"""
    
    def __init__(self, app):
        self.app = app
    
    def load_file(self):
        """Cargar archivo Excel/CSV"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.app.file_path = file_path
            self.app.left_panel_widgets['file_label'].config(text=f"✅ {file_path.split('/')[-1]}")
            self.app.update_status(f"📁 Archivo cargado: {file_path}")
            
            # Crear instancia según tipo de modelo
            if self.app.model_type == 'logistic':
                from analytics import AutoMLVisualizer
                self.app.automl = AutoMLVisualizer(
                    status_callback=self.app.update_status,
                    plot_callback=self.app.update_training_plots
                )
            else:
                from analytics import LinearRegressionVisualizer
                self.app.automl = LinearRegressionVisualizer(
                    status_callback=self.app.update_status,
                    plot_callback=self.app.update_training_plots
                )
            
            try:
                self.app.automl.load_data(file_path)
                self.app.update_status(f"✅ Columnas disponibles: {self.app.automl.get_columns()}")
                
                # Poblar checkbuttons
                self.app.populate_feature_checkbuttons()
                self.app.left_panel_widgets['confirm_vars_btn'].config(state=tk.NORMAL)
                
            except Exception as e:
                self.app.update_status(f"❌ Error al cargar: {str(e)}")
                messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")


class VariableSelectionCallbacks:
    """Callbacks para selección de variables"""
    
    def __init__(self, app):
        self.app = app
    
    def select_all(self):
        """Seleccionar todas las variables"""
        features_vars = self.app.left_panel_widgets['features_vars']
        for var in features_vars.values():
            var.set(True)
    
    def deselect_all(self):
        """Deseleccionar todas las variables"""
        features_vars = self.app.left_panel_widgets['features_vars']
        for var in features_vars.values():
            var.set(False)
    
    def confirm_variables(self):
        """Confirmar variables seleccionadas"""
        features_vars = self.app.left_panel_widgets['features_vars']
        target_var = self.app.left_panel_widgets['target_var']
        
        features = [col for col, var in features_vars.items() if var.get()]
        target = target_var.get().strip()
        
        # Validaciones
        if self.app.model_type == 'simple' and len(features) != 1:
            messagebox.showwarning("Advertencia", 
                                  "Regresión Lineal Simple requiere EXACTAMENTE 1 variable independiente")
            return
        
        # Validar usando helpers
        is_valid, error_msg = helpers.validate_variables(features, target, 
                                                         self.app.automl.get_columns())
        if not is_valid:
            messagebox.showwarning("Advertencia", error_msg)
            return
        
        # Establecer variables
        self.app.automl.set_variables(features, target)
        
        model_names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple'
        }
        
        self.app.update_status(f"✅ {model_names.get(self.app.model_type, 'Modelo')} configurada:")
        self.app.update_status(f"   Features ({len(features)}): {features}")
        self.app.update_status(f"   Target: {target}")
        
        # Habilitar entrenamiento
        self.app.left_panel_widgets['train_btn'].config(state=tk.NORMAL)


class TrainingCallbacks:
    """Callbacks para entrenamiento del modelo"""
    
    def __init__(self, app):
        self.app = app
    
    def start_training(self):
        """Iniciar entrenamiento en hilo separado"""
        if self.app.is_training:
            return
        
        # Obtener parámetros
        test_size_str = self.app.left_panel_widgets['test_size_var'].get()
        n_epochs_str = self.app.left_panel_widgets['epochs_var'].get()
        nulls_str = self.app.left_panel_widgets['nulls_var'].get()
        
        is_valid, error_msg = helpers.validate_parameters(test_size_str, n_epochs_str, nulls_str)
        if not is_valid:
            messagebox.showerror("Error", error_msg)
            return
        
        # Deshabilitar botones
        self.app.is_training = True
        self.app.left_panel_widgets['train_btn'].config(state=tk.DISABLED, text="⏳ ENTRENANDO...")
        self.app.left_panel_widgets['confirm_vars_btn'].config(state=tk.DISABLED)
        self.app.left_panel_widgets['load_btn'].config(state=tk.DISABLED)
        
        # Iniciar hilo
        self.app.training_thread = threading.Thread(
            target=self._training_worker,
            args=(float(test_size_str), int(n_epochs_str), nulls_str),
            daemon=True
        )
        self.app.training_thread.start()
    
    def _training_worker(self, test_size, n_epochs, nulls_handling):
        """Worker para entrenamiento en hilo separado"""
        try:
            self.app.automl.clean_data(handle_nulls=nulls_handling)
            self.app.automl.split_data(test_size=test_size)
            results = self.app.automl.train_and_visualize(n_epochs=n_epochs)
            
            # Mostrar resultados en el hilo principal
            self.app.root.after(0, self.app.show_final_results, results)
            
        except Exception as e:
            self.app.update_status(f"❌ Error durante entrenamiento: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.app.is_training = False
            self.app.root.after(0, self._enable_buttons)
    
    def _enable_buttons(self):
        """Rehabilitar botones después del entrenamiento"""
        self.app.left_panel_widgets['train_btn'].config(state=tk.NORMAL, 
                                                        text="🚀 INICIAR ENTRENAMIENTO")
        self.app.left_panel_widgets['confirm_vars_btn'].config(state=tk.NORMAL)
        self.app.left_panel_widgets['load_btn'].config(state=tk.NORMAL)
