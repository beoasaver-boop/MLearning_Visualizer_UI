# UI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from styles import DARK_THEME, MATPLOTLIB_DARK_STYLE

# Análisis
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
from automl_core import AutoMLVisualizer
from linear_regression_core import LinearRegressionVisualizer


plt.rcParams.update(MATPLOTLIB_DARK_STYLE)

class MLVisualizerApp:
    def __init__(self, root, model_type="logistic"):
        self.root = root
        self.model_type = model_type
        self.root.title(f"ML Visualizer - {self.get_model_name()}")
        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_THEME['bg'])
        
        # Variables
        self.file_path = None
        self.df = None
        self.automl = None
        self.is_training = False
        self.training_thread = None
        
        # Configuración de estilos tkinter
        self.setup_styles()
        
        # Crear interfaz tkinter
        self.create_widgets()
        
        # Variables para gráficas
        self.figures = {}
        self.canvases = {}
        self.coef_history = []
        self.colorbar_shown = False
        

    def get_model_name(self):
        """Retorna nombre del modelo según tipo"""
        names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple'
        }
        return names.get(self.model_type, 'Desconocido')
        
    def setup_styles(self):
        """Configurar estilos para ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores para diferentes widgets
        style.configure('Dark.TFrame', background=DARK_THEME['bg'])
        style.configure('Dark.TLabel', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])
        style.configure('Dark.TButton', background=DARK_THEME['button_bg'], 
                       foreground=DARK_THEME['button_fg'], borderwidth=0, focuscolor='none')
        style.map('Dark.TButton',
                 background=[('active', DARK_THEME['button_active'])])
        
        style.configure('Success.TButton', background=DARK_THEME['success'])
        style.configure('Info.TButton', background=DARK_THEME['info'])
        
        style.configure('Dark.TEntry', fieldbackground=DARK_THEME['entry_bg'],
                       foreground=DARK_THEME['entry_fg'])
        
    def create_widgets(self):
        """Crear todos los widgets de la interfaz"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo - Configuración
        left_panel = ttk.Frame(main_frame, style='Dark.TFrame', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Panel derecho - Visualización
        right_panel = ttk.Frame(main_frame, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ========== PANEL IZQUIERDO ==========
        
        # Título
        title_label = tk.Label(left_panel, text=f"ML Visualizer - {self.get_model_name()}", 
                            font=('Arial', 18, 'bold'),
                            bg=DARK_THEME['bg'], fg=DARK_THEME['highlight'])
        title_label.pack(pady=10)
        
        # Frame de archivo
        file_frame = tk.LabelFrame(left_panel, text="1. Cargar Datos", 
                                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                font=('Arial', 11, 'bold'))
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.file_label = tk.Label(file_frame, text="No hay archivo seleccionado",
                                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'])
        self.file_label.pack(padx=10, pady=5)
        
        self.load_btn = tk.Button(file_frame, text="📁 Cargar Archivo (Excel/CSV)",
                                command=self.load_file,
                                bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                                font=('Arial', 10), cursor='hand2')
        self.load_btn.pack(padx=10, pady=5)
        
        # Frame de variables
        vars_frame = tk.LabelFrame(left_panel, text="2. Seleccionar Variables",
                                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                font=('Arial', 11, 'bold'))
        vars_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables independientes con Checkbuttons
        tk.Label(vars_frame, text="Variables Independientes (Features):",
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        
        # ==== ESTO ES LO QUE FALTABA - Frame con scroll para checkbuttons ====
        features_container = tk.Frame(vars_frame, bg=DARK_THEME['frame_bg'])
        features_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas y scrollbar para scroll vertical
        features_canvas = tk.Canvas(features_container, bg=DARK_THEME['frame_bg'], 
                                    height=120, highlightthickness=0)
        scrollbar = tk.Scrollbar(features_container, orient="vertical", 
                                command=features_canvas.yview)
        self.features_scroll_frame = tk.Frame(features_canvas, bg=DARK_THEME['frame_bg'])
        
        # Configurar scroll
        scrollbar_frame = tk.Frame(features_container, bg=DARK_THEME['frame_bg'])
        scrollbar_frame.pack(side="right", fill="y")
        
        features_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(in_=scrollbar_frame, side="right", fill="y")
        
        # Crear ventana en canvas
        canvas_window = features_canvas.create_window((0, 0), window=self.features_scroll_frame, 
                                                        anchor="nw", width=features_canvas.winfo_width())
        
        def configure_scroll_region(event):
            features_canvas.configure(scrollregion=features_canvas.bbox("all"))
        
        def configure_canvas_width(event):
            features_canvas.itemconfig(canvas_window, width=event.width)
        
        self.features_scroll_frame.bind("<Configure>", configure_scroll_region)
        features_canvas.bind("<Configure>", configure_canvas_width)
        features_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Scroll con rueda del mouse
        def _on_mousewheel(event):
            features_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        features_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.features_scroll_frame.bind("<MouseWheel>", _on_mousewheel)        
        
        # Diccionario para almacenar las variables de los checkbuttons
        self.features_vars = {}
        
        # Botón para seleccionar/deseleccionar todas
        select_all_frame = tk.Frame(vars_frame, bg=DARK_THEME['frame_bg'])
        select_all_frame.pack(pady=5)
        
        def select_all():
            for var in self.features_vars.values():
                var.set(True)
            # Para regresión simple, corregir después
            if self.model_type == 'simple' and len(self.features_vars) > 0:
                # Dejar solo la primera seleccionada
                first = True
                for var in self.features_vars.values():
                    if first:
                        var.set(True)
                        first = False
                    else:
                        var.set(False)
        
        def deselect_all():
            for var in self.features_vars.values():
                var.set(False)
        
        tk.Button(select_all_frame, text="✓ Seleccionar Todas", command=select_all,
                bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                font=('Arial', 8), cursor='hand2').pack(side=tk.LEFT, padx=5)
        tk.Button(select_all_frame, text="✗ Deseleccionar Todas", command=deselect_all,
                bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                font=('Arial', 8), cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        # Variable dependiente
        tk.Label(vars_frame, text="Variable Dependiente (Target):",
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        
        self.target_var = tk.StringVar()
        self.target_entry = tk.Entry(vars_frame, textvariable=self.target_var,
                                    width=35, bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg'],
                                    insertbackground=DARK_THEME['fg'])
        self.target_entry.pack(padx=10, pady=5)
        
        # Botón confirmar variables
        self.confirm_vars_btn = tk.Button(vars_frame, text="✓ Confirmar Variables",
                                        command=self.confirm_variables,
                                        bg=DARK_THEME['success'], fg='white',
                                        font=('Arial', 10, 'bold'), cursor='hand2',
                                        state=tk.DISABLED)
        self.confirm_vars_btn.pack(padx=10, pady=10)
        
        # Frame de parámetros
        params_frame = tk.LabelFrame(left_panel, text="3. Parámetros de Entrenamiento",
                                    bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                    font=('Arial', 11, 'bold'))
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(params_frame, text="Tamaño de Prueba (0-1):",
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        self.test_size_var = tk.StringVar(value="0.3")
        tk.Entry(params_frame, textvariable=self.test_size_var, width=10,
                bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        tk.Label(params_frame, text="Número de Epochs:",
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        self.epochs_var = tk.StringVar(value="100")
        tk.Entry(params_frame, textvariable=self.epochs_var, width=10,
                bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        # Parámetro adicional para regresión (tasa de aprendizaje)
        if self.model_type in ['simple', 'multiple']:
            tk.Label(params_frame, text="Tasa de Aprendizaje (0.001-0.1):",
                    bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
            self.learning_rate_var = tk.StringVar(value="0.01")
            tk.Entry(params_frame, textvariable=self.learning_rate_var, width=10,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        tk.Label(params_frame, text="Manejo de Nulos:\n1=Eliminar, 2=Mediana/Moda, 3=0",
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        self.nulls_var = tk.StringVar(value="2")
        tk.Entry(params_frame, textvariable=self.nulls_var, width=10,
                bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        # Botón entrenar
        self.train_btn = tk.Button(left_panel, text="🚀 INICIAR ENTRENAMIENTO",
                                command=self.start_training,
                                bg=DARK_THEME['highlight'], fg='white',
                                font=('Arial', 12, 'bold'), cursor='hand2',
                                state=tk.DISABLED)
        self.train_btn.pack(padx=10, pady=20, fill=tk.X)
        
        # Área de estado
        status_frame = tk.LabelFrame(left_panel, text="Estado",
                                    bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                    font=('Arial', 11, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.status_text = tk.Text(status_frame, height=8, width=40,
                                bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg'],
                                wrap=tk.WORD, font=('Consolas', 9))
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar para status
        scrollbar_status = tk.Scrollbar(self.status_text)
        scrollbar_status.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar_status.set)
        scrollbar_status.config(command=self.status_text.yview)
        
        # ========== PANEL DERECHO ==========
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de gráficas de entrenamiento
        self.training_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.training_frame, text="📈 Entrenamiento en Vivo")
        
        # Frame para gráficas de entrenamiento
        self.plots_frame = tk.Frame(self.training_frame, bg=DARK_THEME['bg'])
        self.plots_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear subplots
        self.create_training_plots()
        
        # Pestaña de resultados
        self.results_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.results_frame, text="📊 Resultados Finales")
        
        # Frame para resultados
        self.results_display = tk.Frame(self.results_frame, bg=DARK_THEME['bg'])
        self.results_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def populate_feature_checkbuttons(self):
        """Poblar checkbuttons con columnas del dataset"""
        # Limpiar checkbuttons existentes
        for widget in self.features_scroll_frame.winfo_children():
            widget.destroy()
        
        self.features_vars.clear()
        
        # Crear checkbuttons para cada columna
        columns = self.automl.get_columns()
        for col in columns:
            var = tk.BooleanVar()
            self.features_vars[col] = var
            cb = tk.Checkbutton(
                self.features_scroll_frame, text=col, variable=var,
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                selectcolor=DARK_THEME['select_bg'],
                activebackground=DARK_THEME['frame_bg'],
                activeforeground=DARK_THEME['fg'],
                font=('Arial', 9)
            )
            cb.pack(anchor=tk.W, padx=5, pady=2)
            
            # Para regresión lineal simple, limitar a una selección
            if self.model_type == 'simple':
                var.trace_add('write', lambda *args, c=col: self.limit_simple_selection(c))
        
        self.update_status(f"✅ Checkbuttons creados para {len(columns)} columnas")
        
    def create_training_plots(self):
        """Crear las 6 gráficas de entrenamiento"""
        # Crear figura con 2x3 subplots
        self.fig = Figure(figsize=(10, 8), facecolor=DARK_THEME['bg'])
        
        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax4 = self.fig.add_subplot(2, 3, 4)
        self.ax5 = self.fig.add_subplot(2, 3, 5)
        self.ax6 = self.fig.add_subplot(2, 3, 6)
        
        # Configurar colores de ejes
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_facecolor(DARK_THEME['frame_bg'])
            ax.tick_params(colors=DARK_THEME['fg'])
            ax.xaxis.label.set_color(DARK_THEME['fg'])
            ax.yaxis.label.set_color(DARK_THEME['fg'])
            ax.title.set_color(DARK_THEME['fg'])
            ax.grid(True, alpha=0.3, color=DARK_THEME.get('grid_color', '#404040'))
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar líneas vacías para ax1 y ax2
        self.train_loss_line, = self.ax1.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.test_loss_line, = self.ax1.plot([], [], 'r-', label='Test Loss', linewidth=2)
        self.train_acc_line, = self.ax2.plot([], [], 'b-', label='Train Acc', linewidth=2)
        self.test_acc_line, = self.ax2.plot([], [], 'r-', label='Test Acc', linewidth=2)
        
        # Configurar límites iniciales para ax1 y ax2
        self.ax1.set_xlim(0, 100)  # Rango de epochs
        self.ax1.set_ylim(0, 1.1)   # Rango de loss (0 a 1)
        self.ax2.set_xlim(0, 100)
        self.ax2.set_ylim(0, 1.05)
        
        # Configurar ax3 (coeficientes)
        self.ax3.set_xlim(0, 100)
        self.ax3.set_ylim(-1, 1)
        
        # Configurar ax4 (importancia) - se actualizará después
        self.ax4.set_xlim(0, 1)
        
        # Configurar ax5 (matriz de confusión)
        self.ax5.set_xlim(-0.5, 1.5)
        self.ax5.set_ylim(-0.5, 1.5)
        
        # Configurar ax6 (sobreajuste)
        self.ax6.set_xlim(0, 100)
        self.ax6.set_ylim(-0.5, 0.5)
        self.ax6.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # Leyendas
        self.ax1.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        self.ax2.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        
        # Títulos
        self.ax1.set_title('Pérdida (Loss)')
        self.ax2.set_title('Precisión (Accuracy)')
        self.ax3.set_title('Evolución de Coeficientes')
        self.ax4.set_title('Importancia de Características')
        self.ax5.set_title('Matriz de Confusión')
        self.ax6.set_title('Sobreajuste')
        
        # Etiquetas de ejes
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Coeficiente')
        self.ax4.set_xlabel('Importancia')
        self.ax6.set_xlabel('Epoch')
        self.ax6.set_ylabel('Gap (Train - Test)')
        self.fig.tight_layout() 
       
    def update_training_plots(self, epoch, n_epochs, train_losses, test_losses,
                            train_accuracies, test_accuracies, coef_history=None,
                            is_regression=False, is_simple=False, X_test=None,
                            y_test=None, model=None):
        """Actualizar gráficas en tiempo real (adaptado para regresión)"""
        
        # === Gráfica 1: Pérdida (MSE para regresión, LogLoss para clasificación) ===
        self.train_loss_line.set_data(range(len(train_losses)), train_losses)
        self.test_loss_line.set_data(range(len(test_losses)), test_losses)
        
        if len(train_losses) > 1:
            max_loss = max(max(train_losses[-10:]), max(test_losses[-10:])) if train_losses else 1
            self.ax1.set_xlim(0, max(10, len(train_losses)))
            self.ax1.set_ylim(0, min(max_loss * 1.1, 10))
        
        loss_label = "MSE" if is_regression else "Log Loss"
        self.ax1.set_ylabel(loss_label)
        self.ax1.set_title(f'Pérdida ({loss_label})')
        
        # === Gráfica 2: Métrica principal (R² para regresión, Accuracy para clasificación) ===
        self.train_acc_line.set_data(range(len(train_accuracies)), train_accuracies)
        self.test_acc_line.set_data(range(len(test_accuracies)), test_accuracies)
        
        if len(train_accuracies) > 1:
            self.ax2.set_xlim(0, max(10, len(train_accuracies)))
            if is_regression:
                self.ax2.set_ylim(-1, 1)  # R² puede ser negativo
            else:
                self.ax2.set_ylim(0, 1.05)
        
        metric_label = "R² Score" if is_regression else "Accuracy"
        current_metric = test_accuracies[-1] if test_accuracies else 0
        self.ax2.set_title(f'{metric_label} (Epoch {epoch+1}/{n_epochs} - {metric_label}: {current_metric:.3f})')
        self.ax2.set_ylabel(metric_label)
        
        # === Gráfica 3: Coeficientes ===
        self.ax3.clear()
        self.ax3.set_facecolor(DARK_THEME['frame_bg'])
        if coef_history and len(coef_history) > 0:
            coef_array = np.array(coef_history)
            n_features = min(5, coef_array.shape[1])
            for i in range(n_features):
                label = self.automl.feature_names[i] if hasattr(self.automl, 'feature_names') and i < len(self.automl.feature_names) else f'Coef {i}'
                self.ax3.plot(range(len(coef_array[:, i])), coef_array[:, i], linewidth=1.5, label=label[:15])
            self.ax3.set_xlim(0, max(10, len(coef_array)))
            self.ax3.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'], fontsize=8)
        else:
            self.ax3.text(0.5, 0.5, 'Actualizando en\ntiempo real...', 
                        transform=self.ax3.transAxes, ha='center', va='center',
                        color=DARK_THEME['fg'], alpha=0.7)
        
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('Coeficiente')
        self.ax3.set_title('Evolución de Coeficientes')
        self.ax3.grid(True, alpha=0.3, color=DARK_THEME.get('grid_color', '#404040'))
        
        # === Gráfica 4: Importancia de Características (o Predicciones para regresión simple) ===
        self.ax4.clear()
        self.ax4.set_facecolor(DARK_THEME['frame_bg'])
        
        if is_regression and is_simple and X_test is not None and y_test is not None and model is not None:
            # Para regresión simple, mostrar scatter plot de predicciones
            y_pred = model.predict(X_test)
            self.ax4.scatter(y_test, y_pred, alpha=0.5, c=DARK_THEME['highlight'], s=10)
            
            # Línea ideal y=x
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            self.ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal')
            
            self.ax4.set_xlabel('Valores Reales')
            self.ax4.set_ylabel('Predicciones')
            self.ax4.set_title('Predicciones vs Reales')
            self.ax4.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
            
        elif hasattr(self.automl, 'get_feature_importance'):
            importance = self.automl.get_feature_importance()
            if importance:
                features = list(importance.keys())
                values = list(importance.values())
                indices = np.argsort(values)
                
                bars = self.ax4.barh(range(len(features)), [values[i] for i in indices])
                self.ax4.set_yticks(range(len(features)))
                self.ax4.set_yticklabels([features[i] for i in indices], fontsize=8)
                self.ax4.set_xlabel('Importancia (|Coeficiente|)' if not is_regression else '|Coeficiente|')
                self.ax4.set_title('Importancia de Características')
                
                # Colorear barras
                for i, bar in enumerate(bars):
                    bar.set_color(DARK_THEME['highlight'])
            else:
                self.ax4.text(0.5, 0.5, 'Se mostrará al\nfinal del entrenamiento', 
                            transform=self.ax4.transAxes, ha='center', va='center',
                            color=DARK_THEME['fg'], alpha=0.7)
        else:
            self.ax4.text(0.5, 0.5, 'Se mostrará al\nfinal del entrenamiento', 
                        transform=self.ax4.transAxes, ha='center', va='center',
                        color=DARK_THEME['fg'], alpha=0.7)
        
        self.ax4.grid(True, alpha=0.3, color=DARK_THEME.get('grid_color', '#404040'))
        
        # === Gráfica 5: Residuos (solo regresión) o Matriz de Confusión ===
        self.ax5.clear()
        self.ax5.set_facecolor(DARK_THEME['frame_bg'])
        
        if is_regression and X_test is not None and model is not None and epoch % 20 == 0:
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            self.ax5.hist(residuals, bins=30, color=DARK_THEME['highlight'], alpha=0.7, edgecolor='white')
            self.ax5.set_xlabel('Residuos')
            self.ax5.set_ylabel('Frecuencia')
            self.ax5.set_title('Distribución de Residuos')
            self.ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        else:
            self.ax5.text(0.5, 0.5, 'Se actualizará\nperiódicamente', 
                        transform=self.ax5.transAxes, ha='center', va='center',
                        color=DARK_THEME['fg'], alpha=0.7)
            self.ax5.set_title('Distribución de Residuos' if is_regression else 'Matriz de Confusión')
        
        self.ax5.grid(True, alpha=0.3, color=DARK_THEME.get('grid_color', '#404040'))
        
        # === Gráfica 6: Gap de sobreajuste ===
        self.ax6.clear()
        if len(train_accuracies) > 1:
            acc_gap = np.array(train_accuracies) - np.array(test_accuracies)
            self.ax6.plot(range(len(acc_gap)), acc_gap, 'g-', linewidth=2, label='Gap')
            self.ax6.set_xlim(0, max(10, len(acc_gap)))
            max_gap = max(abs(max(acc_gap)), abs(min(acc_gap))) if len(acc_gap) > 0 else 0.5
            self.ax6.set_ylim(-max_gap * 1.1, max_gap * 1.1)
        else:
            self.ax6.set_xlim(0, 100)
            self.ax6.set_ylim(-0.5, 0.5)
        
        self.ax6.set_facecolor(DARK_THEME['frame_bg'])
        self.ax6.set_xlabel('Epoch')
        metric_name = "R²" if is_regression else "Accuracy"
        self.ax6.set_ylabel(f'Gap ({metric_name} Train - Test)')
        self.ax6.set_title('Sobreajuste (gap positivo = sobreajuste)')
        self.ax6.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        self.ax6.grid(True, alpha=0.3, color=DARK_THEME.get('grid_color', '#404040'))
        self.ax6.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.canvas.flush_events()
    
    def update_status(self, message):
        """Actualizar área de estado"""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def load_file(self):
        """Cargar archivo Excel/CSV"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=f"✅ {file_path.split('/')[-1]}")
            self.update_status(f"📁 Archivo cargado: {file_path}")
            
        # Crear instancia según tipo de modelo
        if self.model_type == 'logistic':
            from automl_core import AutoMLVisualizer
            self.automl = AutoMLVisualizer(
                status_callback=self.update_status,
                plot_callback=self.update_training_plots
            )
        else:
            self.automl = LinearRegressionVisualizer(
                status_callback=self.update_status,
                plot_callback=self.update_training_plots
            )
            
        try:
            self.automl.load_data(file_path)
            self.update_status(f"✅ Columnas disponibles: {self.automl.get_columns()}")
            
            # Poblar checkbuttons después de cargar
            self.populate_feature_checkbuttons()
            
            self.confirm_vars_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.update_status(f"❌ Error al cargar: {str(e)}")
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")


    def limit_simple_selection(self, selected_col):
        """Limitar a una sola selección en regresión simple"""
        for col, var in self.features_vars.items():
            if col != selected_col and var.get():
                var.set(False)

    def limit_simple_selection(self, selected_col):
        """Limitar a una sola selección en regresión simple"""
        for col, var in self.features_vars.items():
            if col != selected_col and var.get():
                var.set(False)
                
    def confirm_variables(self):
        """Confirmar variables seleccionadas desde Checkbuttons"""
        # Obtener features seleccionadas
        features = [col for col, var in self.features_vars.items() if var.get()]
        target = self.target_var.get().strip()
        
        # Validaciones según tipo de modelo
        if self.model_type == 'simple':
            if len(features) != 1:
                messagebox.showwarning("Advertencia", "Regresión Lineal Simple requiere EXACTAMENTE 1 variable independiente")
                return
        else:
            if not features:
                messagebox.showwarning("Advertencia", "Por favor seleccione al menos una variable independiente")
                return
        
        if not target:
            messagebox.showwarning("Advertencia", "Por favor ingrese la variable dependiente")
            return
        
        # Verificar que las columnas existen
        available_cols = self.automl.get_columns()
        
        if target not in available_cols:
            messagebox.showerror("Error", f"Variable objetivo '{target}' no encontrada")
            return
        
        # Establecer variables
        self.automl.set_variables(features, target)
        
        model_desc = "Regresión Lineal Simple" if self.model_type == 'simple' else \
                    "Regresión Lineal Múltiple" if self.model_type == 'multiple' else \
                    "Regresión Logística"
        
        self.update_status(f"✅ {model_desc} configurada:")
        self.update_status(f"   Features ({len(features)}): {features}")
        self.update_status(f"   Target: {target}")
        
        # Habilitar botón de entrenamiento
        self.train_btn.config(state=tk.NORMAL)
        
    def start_training(self):
        """Iniciar entrenamiento en hilo separado"""
        if self.is_training:
            return
        
        # Obtener parámetros
        try:
            test_size = float(self.test_size_var.get())
            n_epochs = int(self.epochs_var.get())
            nulls_handling = self.nulls_var.get()
        except ValueError:
            messagebox.showerror("Error", "Parámetros inválidos")
            return
        
        # Deshabilitar botones durante entrenamiento
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED, text="⏳ ENTRENANDO...")
        self.confirm_vars_btn.config(state=tk.DISABLED)
        self.load_btn.config(state=tk.DISABLED)
        
        # Iniciar hilo de entrenamiento
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(test_size, n_epochs, nulls_handling),
            daemon=True
        )
        self.training_thread.start()
        
    def _training_worker(self, test_size, n_epochs, nulls_handling):
        """Worker para entrenamiento en hilo separado"""
        try:
            # Limpiar datos
            self.automl.clean_data(handle_nulls=nulls_handling)
            
            # Dividir datos
            self.automl.split_data(test_size=test_size)
            
            # Entrenar y visualizar
            results = self.automl.train_and_visualize(n_epochs=n_epochs)
            
            # Mostrar resultados finales
            self.root.after(0, self.show_final_results, results)
            
        except Exception as e:
            self.update_status(f"❌ Error durante entrenamiento: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_training = False
            self.root.after(0, self._enable_buttons)
            
    def _enable_buttons(self):
        """Rehabilitar botones después del entrenamiento"""
        self.train_btn.config(state=tk.NORMAL, text="🚀 INICIAR ENTRENAMIENTO")
        self.confirm_vars_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        
    def show_final_results(self, results):
        """Mostrar resultados finales adaptados al tipo de modelo"""
        
        # Limpiar frame de resultados
        for widget in self.results_display.winfo_children():
            widget.destroy()
        
        # Crear scrollable frame
        canvas = tk.Canvas(self.results_display, bg=DARK_THEME['bg'], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.results_display, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=DARK_THEME['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ========== TÍTULO ==========
        model_names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple'
        }
        model_name = model_names.get(self.model_type, 'Modelo')
        
        title = tk.Label(scrollable_frame, text=f"📊 RESULTADOS DEL MODELO - {model_name}",
                        font=('Arial', 16, 'bold'),
                        bg=DARK_THEME['bg'], fg=DARK_THEME['highlight'])
        title.pack(pady=10)
        
        # ========== MÉTRICAS PRINCIPALES ==========
        metrics_frame = tk.Frame(scrollable_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if self.model_type == 'logistic':
            # Métricas para clasificación
            tk.Label(metrics_frame, text=f"🎯 Mejor Accuracy: {results['best_accuracy']:.4f}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📈 Accuracy Final: {results['final_accuracy']:.4f}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"🏆 Mejor Epoch: {results['best_epoch']}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['info']).pack(pady=5)
        else:
            # Métricas para regresión
            tk.Label(metrics_frame, text=f"🎯 Mejor R² Score: {results['best_r2']:.4f}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📈 R² Score Final: {results['final_r2']:.4f}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📉 MSE Final: {results['final_mse']:.4f}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📊 MAE Final: {results['final_mae']:.4f}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"🏆 Mejor Epoch: {results['best_epoch']}",
                    font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['info']).pack(pady=5)
        
        # ========== ECUACIÓN DEL MODELO ==========
        if self.model_type != 'logistic' and 'coefficients' in results:
            eq_frame = tk.LabelFrame(scrollable_frame, text="📐 Ecuación del Modelo",
                                    bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                    font=('Arial', 12, 'bold'))
            eq_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Construir ecuación
            intercept = results['intercept']
            coefs = results['coefficients']
            
            equation = f"y = {intercept[0]:.4f}"
            for i, (feature, coef) in enumerate(coefs.items()):
                sign = " + " if coef >= 0 else " - "
                equation += f"{sign}{abs(coef):.4f} * {feature}"
            
            tk.Label(eq_frame, text=equation, wraplength=800,
                    font=('Consolas', 10), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['highlight'],
                    justify=tk.LEFT).pack(padx=10, pady=10)
        
        # ========== COEFICIENTES / IMPORTANCIA ==========
        if hasattr(self.automl, 'get_feature_importance'):
            importance = self.automl.get_feature_importance()
            if importance:
                imp_frame = tk.LabelFrame(scrollable_frame, text="📊 Coeficientes del Modelo",
                                        bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                        font=('Arial', 12, 'bold'))
                imp_frame.pack(fill=tk.X, padx=20, pady=10)
                
                # Ordenar por valor absoluto
                sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
                
                for feature, coef in sorted_items:
                    bar_frame = tk.Frame(imp_frame, bg=DARK_THEME['frame_bg'])
                    bar_frame.pack(fill=tk.X, padx=10, pady=2)
                    
                    # Nombre de la característica
                    tk.Label(bar_frame, text=f"{feature}:", width=25, anchor=tk.W,
                            bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).pack(side=tk.LEFT)
                    
                    # Valor del coeficiente
                    coef_color = DARK_THEME['success'] if coef > 0 else DARK_THEME['error']
                    tk.Label(bar_frame, text=f"{coef:+.4f}", width=12, anchor=tk.W,
                            bg=DARK_THEME['frame_bg'], fg=coef_color,
                            font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
                    
                    # Barra de magnitud (normalizada)
                    max_abs = max(abs(v) for v in importance.values()) if importance else 1
                    bar_width = int((abs(coef) / max_abs) * 200)
                    bar = tk.Canvas(bar_frame, width=200, height=18, 
                                bg=DARK_THEME['entry_bg'], highlightthickness=0)
                    bar.pack(side=tk.LEFT, padx=5)
                    bar_color = DARK_THEME['highlight'] if coef > 0 else DARK_THEME['warning']
                    bar.create_rectangle(0, 0, bar_width, 18, fill=bar_color, outline='')
                    
                    # Valor absoluto
                    tk.Label(bar_frame, text=f"|{abs(coef):.4f}|", width=10, anchor=tk.W,
                            bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).pack(side=tk.LEFT)
        
        # ========== MATRIZ DE CONFUSIÓN (solo clasificación) ==========
        if self.model_type == 'logistic' and hasattr(self.automl, 'get_confusion_matrix'):
            cm = self.automl.get_confusion_matrix()
            cm_frame = tk.LabelFrame(scrollable_frame, text="📊 Matriz de Confusión",
                                    bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                    font=('Arial', 12, 'bold'))
            cm_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Tabla de matriz
            table_frame = tk.Frame(cm_frame, bg=DARK_THEME['frame_bg'])
            table_frame.pack(pady=10)
            
            # Headers
            tk.Label(table_frame, text="", width=12, bg=DARK_THEME['frame_bg']).grid(row=0, column=0)
            tk.Label(table_frame, text="Predicho Negativo", width=15, relief=tk.RIDGE,
                    bg=DARK_THEME['highlight'], fg='white').grid(row=0, column=1, padx=1, pady=1)
            tk.Label(table_frame, text="Predicho Positivo", width=15, relief=tk.RIDGE,
                    bg=DARK_THEME['highlight'], fg='white').grid(row=0, column=2, padx=1, pady=1)
            
            # Filas
            tk.Label(table_frame, text="Real Negativo", width=12, relief=tk.RIDGE,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=1, column=0, padx=1, pady=1)
            tk.Label(table_frame, text=str(cm[0, 0]), width=15, relief=tk.RIDGE,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=1, column=1, padx=1, pady=1)
            tk.Label(table_frame, text=str(cm[0, 1]), width=15, relief=tk.RIDGE,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=1, column=2, padx=1, pady=1)
            
            tk.Label(table_frame, text="Real Positivo", width=12, relief=tk.RIDGE,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=2, column=0, padx=1, pady=1)
            tk.Label(table_frame, text=str(cm[1, 0]), width=15, relief=tk.RIDGE,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=2, column=1, padx=1, pady=1)
            tk.Label(table_frame, text=str(cm[1, 1]), width=15, relief=tk.RIDGE,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=2, column=2, padx=1, pady=1)
            
            # Métricas derivadas
            tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_table = tk.Frame(cm_frame, bg=DARK_THEME['frame_bg'])
            metrics_table.pack(pady=10)
            
            tk.Label(metrics_table, text=f"Accuracy: {accuracy:.4f}", width=15,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=5)
            tk.Label(metrics_table, text=f"Precision: {precision:.4f}", width=15,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=5)
            tk.Label(metrics_table, text=f"Recall: {recall:.4f}", width=15,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=5)
            tk.Label(metrics_table, text=f"F1-Score: {f1:.4f}", width=15,
                    bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=5)
        
        # ========== PREDICCIONES VS REALES (solo regresión simple) ==========
        if self.model_type == 'simple' and hasattr(self.automl, 'get_predictions'):
            preds = self.automl.get_predictions()
            if preds is not None:
                pred_frame = tk.LabelFrame(scrollable_frame, text="📈 Predicciones vs Valores Reales",
                                        bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                        font=('Arial', 12, 'bold'))
                pred_frame.pack(fill=tk.X, padx=20, pady=10)
                
                # Crear figura para scatter plot
                
                fig = Figure(figsize=(6, 4), facecolor=DARK_THEME['frame_bg'])
                ax = fig.add_subplot(111)
                ax.set_facecolor(DARK_THEME['entry_bg'])
                
                # Scatter plot
                ax.scatter(self.automl.y_test, preds, alpha=0.5, c=DARK_THEME['highlight'], s=20)
                
                # Línea ideal
                min_val = min(self.automl.y_test.min(), preds.min())
                max_val = max(self.automl.y_test.max(), preds.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal')
                
                ax.set_xlabel('Valores Reales', color=DARK_THEME['fg'])
                ax.set_ylabel('Predicciones', color=DARK_THEME['fg'])
                ax.set_title('Predicciones vs Reales', color=DARK_THEME['fg'])
                ax.tick_params(colors=DARK_THEME['fg'])
                ax.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
                ax.grid(True, alpha=0.3)
                
                canvas_plot = FigureCanvasTkAgg(fig, master=pred_frame)
                canvas_plot.draw()
                canvas_plot.get_tk_widget().pack(padx=10, pady=10)
        
        # ========== REPORTE DE CLASIFICACIÓN (solo logística) ==========
        if self.model_type == 'logistic' and hasattr(self.automl, 'get_classification_report'):
            report = self.automl.get_classification_report()
            report_frame = tk.LabelFrame(scrollable_frame, text="📋 Reporte de Clasificación",
                                        bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                        font=('Arial', 12, 'bold'))
            report_frame.pack(fill=tk.X, padx=20, pady=10)
            
            # Headers
            headers_frame = tk.Frame(report_frame, bg=DARK_THEME['frame_bg'])
            headers_frame.pack(pady=5)
            
            headers = ["Clase", "Precisión", "Recall", "F1-Score", "Soporte"]
            for i, header in enumerate(headers):
                tk.Label(headers_frame, text=header, width=12, relief=tk.RIDGE,
                        bg=DARK_THEME['highlight'], fg='white',
                        font=('Arial', 9, 'bold')).grid(row=0, column=i, padx=1, pady=1)
            
            # Datos por clase
            row = 1
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                    tk.Label(headers_frame, text=str(class_name)[:12], width=12, relief=tk.RIDGE,
                            bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).grid(row=row, column=0, padx=1, pady=1)
                    tk.Label(headers_frame, text=f"{metrics['precision']:.3f}", width=12, relief=tk.RIDGE,
                            bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).grid(row=row, column=1, padx=1, pady=1)
                    tk.Label(headers_frame, text=f"{metrics['recall']:.3f}", width=12, relief=tk.RIDGE,
                            bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).grid(row=row, column=2, padx=1, pady=1)
                    tk.Label(headers_frame, text=f"{metrics['f1-score']:.3f}", width=12, relief=tk.RIDGE,
                            bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).grid(row=row, column=3, padx=1, pady=1)
                    tk.Label(headers_frame, text=f"{metrics['support']:.0f}", width=12, relief=tk.RIDGE,
                            bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg'],
                            font=('Arial', 9)).grid(row=row, column=4, padx=1, pady=1)
                    row += 1
            
            # Accuracy global
            if 'accuracy' in report:
                acc_frame = tk.Frame(report_frame, bg=DARK_THEME['frame_bg'])
                acc_frame.pack(pady=5)
                tk.Label(acc_frame, text=f"Accuracy Global: {report['accuracy']:.4f}",
                        font=('Arial', 10, 'bold'),
                        bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack()
        
        # ========== DIAGNÓSTICO DEL MODELO ==========
        diag_frame = tk.LabelFrame(scrollable_frame, text="💡 Diagnóstico del Modelo",
                                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                font=('Arial', 12, 'bold'))
        diag_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if self.model_type == 'logistic':
            final_acc = results['final_accuracy']
            if final_acc >= 0.9:
                diag_text = "✅ Excelente modelo! Muy alta precisión."
            elif final_acc >= 0.8:
                diag_text = "👍 Buen modelo. Precisión aceptable."
            elif final_acc >= 0.7:
                diag_text = "⚠️ Modelo aceptable. Se puede mejorar."
            else:
                diag_text = "❌ Modelo mejorable. Considere más datos o diferentes features."
        else:
            final_r2 = results['final_r2']
            if final_r2 >= 0.9:
                diag_text = "✅ Excelente modelo! Muy alto poder predictivo."
            elif final_r2 >= 0.7:
                diag_text = "👍 Buen modelo. El modelo explica bien la varianza."
            elif final_r2 >= 0.5:
                diag_text = "⚠️ Modelo moderado. Podría mejorar con más features."
            else:
                diag_text = "❌ Modelo débil. El R² es bajo, considere transformaciones o más datos."
        
        tk.Label(diag_frame, text=diag_text, wraplength=800,
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                font=('Arial', 10)).pack(padx=10, pady=10)