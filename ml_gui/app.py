"""
Aplicación principal de ML Visualizer
Clase MLVisualizerApp - Coordinadora de toda la interfaz gráfica
"""
import tkinter as tk
import matplotlib.pyplot as plt
from config.theme import DARK_THEME, MATPLOTLIB_DARK_STYLE
from ml_gui.widgets import LeftPanelBuilder, RightPanelBuilder, setup_ttk_styles
from ml_gui.plots import TrainingPlotsManager
from ml_gui.callbacks import DataLoadingCallbacks, VariableSelectionCallbacks, TrainingCallbacks
from ml_gui.results import ResultsDisplayer
from utils import helpers

# Configurar matplotlib con tema oscuro
plt.rcParams.update(MATPLOTLIB_DARK_STYLE)


class MLVisualizerApp:
    """Aplicación principal de visualización de ML"""
    
    def __init__(self, root, model_type="logistic"):
        self.root = root
        self.model_type = model_type
        
        # Configurar ventana
        self.root.title(f"ML Visualizer - {self.get_model_name()}")
        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_THEME['bg'])
        helpers.center_window(self.root, 1400, 900)
        
        # Estado
        self.file_path = None
        self.automl = None
        self.is_training = False
        self.training_thread = None
        
        # Configurar estilos
        setup_ttk_styles(DARK_THEME)
        
        # Crear widgets
        self._create_ui()
        
        # Inicializar callbacks
        self._setup_callbacks()
    
    def get_model_name(self):
        """Retorna nombre del modelo según tipo"""
        names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple'
        }
        return names.get(self.model_type, 'Desconocido')
    
    def _create_ui(self):
        """Crear interfaz principal"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg=DARK_THEME['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo
        left_builder = LeftPanelBuilder(self.root, self.model_type, DARK_THEME)
        left_builder.build(main_frame)
        self.left_panel_widgets = left_builder.widgets
        
        # Panel derecho
        right_builder = RightPanelBuilder(DARK_THEME)
        right_builder.build(main_frame)
        self.right_panel_widgets = right_builder.widgets
        
        # Gestor de gráficas
        self.plots_manager = TrainingPlotsManager(
            self.right_panel_widgets['plots_frame'],
            DARK_THEME
        )
    
    def _setup_callbacks(self):
        """Configurar todos los callbacks de eventos"""
        # Data loading
        data_callbacks = DataLoadingCallbacks(self)
        self.left_panel_widgets['load_btn'].config(command=data_callbacks.load_file)
        
        # Variable selection
        var_callbacks = VariableSelectionCallbacks(self)
        self.left_panel_widgets['select_all_btn'].config(command=var_callbacks.select_all)
        self.left_panel_widgets['deselect_all_btn'].config(command=var_callbacks.deselect_all)
        self.left_panel_widgets['confirm_vars_btn'].config(command=var_callbacks.confirm_variables)
        
        # Training
        train_callbacks = TrainingCallbacks(self)
        self.left_panel_widgets['train_btn'].config(command=train_callbacks.start_training)
    
    def populate_feature_checkbuttons(self):
        """Poblar checkbuttons con columnas del dataset"""
        scroll_frame = self.left_panel_widgets['features_scroll_frame']
        features_vars = self.left_panel_widgets['features_vars']
        
        # Limpiar
        for widget in scroll_frame.winfo_children():
            widget.destroy()
        features_vars.clear()
        
        # Crear checkbuttons
        columns = self.automl.get_columns()
        for col in columns:
            var = tk.BooleanVar()
            features_vars[col] = var
            cb = tk.Checkbutton(
                scroll_frame, text=col, variable=var,
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                selectcolor=DARK_THEME['select_bg'],
                activebackground=DARK_THEME['frame_bg'],
                activeforeground=DARK_THEME['fg'],
                font=('Arial', 9)
            )
            cb.pack(anchor=tk.W, padx=5, pady=2)
            
            # Para regresión simple, limitar selección
            if self.model_type == 'simple':
                var.trace_add('write', lambda *args, c=col: self._limit_simple_selection(c))
        
        self.update_status(f"✅ Checkbuttons creados para {len(columns)} columnas")
    
    def _limit_simple_selection(self, selected_col):
        """Limitar a una sola selección en regresión simple"""
        features_vars = self.left_panel_widgets['features_vars']
        for col, var in features_vars.items():
            if col != selected_col and var.get():
                var.set(False)
    
    def update_status(self, message):
        """Actualizar área de estado"""
        status_text = self.left_panel_widgets['status_text']
        status_text.insert(tk.END, f"{message}\n")
        status_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_training_plots(self, epoch, n_epochs, train_losses, test_losses,
                             train_accuracies, test_accuracies, coef_history=None,
                             is_regression=False, is_simple=False, X_test=None,
                             y_test=None, model=None):
        """Actualizar gráficas en tiempo real"""
        
        # Actualizar pérdida y precisión
        self.plots_manager.update_loss_and_accuracy(
            train_losses, test_losses,
            train_accuracies, test_accuracies,
            is_regression
        )
        
        # Actualizar coeficientes
        feature_names = (self.automl.feature_names 
                        if hasattr(self.automl, 'feature_names') else None)
        self.plots_manager.update_coefficients(coef_history, feature_names)
        
        # Actualizar importancia de características
        importance = (self.automl.get_feature_importance()
                     if hasattr(self.automl, 'get_feature_importance') else None)
        self.plots_manager.update_feature_importance(
            importance, X_test, y_test, model, is_simple
        )
        
        # Actualizar residuos
        self.plots_manager.update_residuals(X_test, y_test, model, epoch, is_regression)
        
        # Actualizar sobreajuste
        self.plots_manager.update_overfitting(train_accuracies, test_accuracies)
        
        # Redibujar
        self.plots_manager.draw()
    
    def show_final_results(self, results):
        """Mostrar resultados finales"""
        displayer = ResultsDisplayer(
            self.right_panel_widgets['results_display'],
            self.model_type,
            self.automl,
            DARK_THEME
        )
        displayer.display(results)
        
        # Cambiar a pestaña de resultados
        self.right_panel_widgets['notebook'].select(1)
