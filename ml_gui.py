import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from automl_core import AutoMLVisualizer
from styles import DARK_THEME, MATPLOTLIB_DARK_STYLE

# Aplicar estilo oscuro de matplotlib
plt.rcParams.update(MATPLOTLIB_DARK_STYLE)

class MLVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Visualizer - Entrenamiento Automático")
        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_THEME['bg'])
        
        # Variables
        self.file_path = None
        self.df = None
        self.automl = None
        self.is_training = False
        self.training_thread = None
        
        # Configurar estilos ttk
        self.setup_styles()
        
        # Crear interfaz
        self.create_widgets()
        
        # Variables para gráficas
        self.figures = {}
        self.canvases = {}
        
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
        title_label = tk.Label(left_panel, text="ML Visualizer", 
                               font=('Arial', 20, 'bold'),
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
        
        self.load_btn = tk.Button(file_frame, text="Cargar Archivo (Excel/CSV)",
                                  command=self.load_file,
                                  bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                                  font=('Arial', 10), cursor='hand2')
        self.load_btn.pack(padx=10, pady=5)
        
        # Frame de variables
        vars_frame = tk.LabelFrame(left_panel, text="2. Seleccionar Variables",
                                   bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                   font=('Arial', 11, 'bold'))
        vars_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Variables independientes
        tk.Label(vars_frame, text="Variables Independientes (Features):",
                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        
        self.features_text = tk.Text(vars_frame, height=5, width=35,
                                     bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg'],
                                     insertbackground=DARK_THEME['fg'])
        self.features_text.pack(padx=10, pady=5)
        self.features_text.insert('1.0', 'Ejemplo: tenure, age, address, income')
        # Borrar el texto incial
        self.features_text.bind("<FocusIn>", lambda e: self.features_text.delete('1.0', tk.END) if self.features_text.get('1.0', 'end-1c') == 'Ejemplo: tenure, age, address, income' else None)
        
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
        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)
        
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
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar líneas vacías
        self.train_loss_line, = self.ax1.plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.test_loss_line, = self.ax1.plot([], [], 'r-', label='Test Loss', linewidth=2)
        self.train_acc_line, = self.ax2.plot([], [], 'b-', label='Train Acc', linewidth=2)
        self.test_acc_line, = self.ax2.plot([], [], 'r-', label='Test Acc', linewidth=2)
        
        self.ax1.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        self.ax2.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        self.ax1.grid(True, alpha=0.3, color=DARK_THEME['grid_color'])
        self.ax2.grid(True, alpha=0.3, color=DARK_THEME['grid_color'])
        self.ax1.set_ylim(0, 1)
        self.ax2.set_ylim(0, 1)
        
        self.ax1.set_title('Pérdida (Loss)')
        self.ax2.set_title('Precisión (Accuracy)')
        self.ax3.set_title('Evolución de Coeficientes')
        self.ax4.set_title('Importancia de Características')
        self.ax5.set_title('Matriz de Confusión')
        self.ax6.set_title('Sobreajuste')
        
        self.fig.tight_layout()
        
    def update_training_plots(self, epoch, n_epochs, train_losses, test_losses, 
                              train_accuracies, test_accuracies):
        """Actualizar gráficas en tiempo real"""
        # Actualizar líneas
        self.train_loss_line.set_data(range(len(train_losses)), train_losses)
        self.test_loss_line.set_data(range(len(test_losses)), test_losses)
        self.train_acc_line.set_data(range(len(train_accuracies)), train_accuracies)
        self.test_acc_line.set_data(range(len(test_accuracies)), test_accuracies)
        
        # Ajustar límites
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Actualizar título con epoch actual
        self.ax2.set_title(f'Precisión (Epoch {epoch+1}/{n_epochs})')
        
        self.canvas.draw()
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
            
            # Crear instancia de AutoML
            self.automl = AutoMLVisualizer(
                status_callback=self.update_status,
                plot_callback=self.update_training_plots
            )
            
            try:
                self.automl.load_data(file_path)
                self.update_status(f"✅ Columnas disponibles: {self.automl.get_columns()}")
                self.confirm_vars_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.update_status(f"❌ Error al cargar: {str(e)}")
                messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
                
    def confirm_variables(self):
        """Confirmar variables seleccionadas"""
        features_text = self.features_text.get('1.0', tk.END).strip()
        target = self.target_var.get().strip()
        
        if not features_text or not target:
            messagebox.showwarning("Advertencia", "Por favor ingrese las variables")
            return
        
        # Parsear features
        features = [f.strip() for f in features_text.split(',')]
        
        # Verificar que las columnas existen
        available_cols = self.automl.get_columns()
        missing_features = [f for f in features if f not in available_cols]
        
        if missing_features:
            messagebox.showerror("Error", f"Columnas no encontradas: {missing_features}")
            return
        
        if target not in available_cols:
            messagebox.showerror("Error", f"Variable objetivo '{target}' no encontrada")
            return
        
        # Establecer variables
        self.automl.set_variables(features, target)
        self.update_status(f"✅ Variables confirmadas: Features={features}, Target={target}")
        
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
        """Mostrar resultados finales en la pestaña de resultados"""
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
        
        # Título
        title = tk.Label(scrollable_frame, text="📊 RESULTADOS DEL MODELO",
                        font=('Arial', 16, 'bold'),
                        bg=DARK_THEME['bg'], fg=DARK_THEME['highlight'])
        title.pack(pady=10)
        
        # Métricas principales
        metrics_frame = tk.Frame(scrollable_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(metrics_frame, text=f"🎯 Mejor Accuracy: {results['best_accuracy']:.4f}",
                font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack(pady=5)
        tk.Label(metrics_frame, text=f"📈 Accuracy Final: {results['final_accuracy']:.4f}",
                font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
        tk.Label(metrics_frame, text=f"🏆 Mejor Epoch: {results['best_epoch']}",
                font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['info']).pack(pady=5)
        
        # Importancia de características
        importance = self.automl.get_feature_importance()
        if importance:
            imp_frame = tk.LabelFrame(scrollable_frame, text="📊 Importancia de Características",
                                     bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                     font=('Arial', 12, 'bold'))
            imp_frame.pack(fill=tk.X, padx=20, pady=10)
            
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                bar_frame = tk.Frame(imp_frame, bg=DARK_THEME['frame_bg'])
                bar_frame.pack(fill=tk.X, padx=10, pady=2)
                
                tk.Label(bar_frame, text=f"{feature}:", width=20, anchor=tk.W,
                        bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT)
                
                # Barra de progreso manual
                bar_width = int(imp * 200)
                bar = tk.Canvas(bar_frame, width=200, height=20, bg=DARK_THEME['entry_bg'], highlightthickness=0)
                bar.pack(side=tk.LEFT, padx=5)
                bar.create_rectangle(0, 0, bar_width, 20, fill=DARK_THEME['highlight'], outline='')
                
                tk.Label(bar_frame, text=f"{imp:.4f}", width=10, anchor=tk.W,
                        bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack(side=tk.LEFT)
        
        # Matriz de confusión
        cm = self.automl.get_confusion_matrix()
        cm_frame = tk.LabelFrame(scrollable_frame, text="📊 Matriz de Confusión",
                                bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                font=('Arial', 12, 'bold'))
        cm_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Mostrar matriz como tabla
        for i, row in enumerate(cm):
            row_frame = tk.Frame(cm_frame, bg=DARK_THEME['frame_bg'])
            row_frame.pack()
            for j, val in enumerate(row):
                tk.Label(row_frame, text=f"{val}", width=10, relief=tk.RIDGE,
                        bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=2, pady=2)
        
        # Reporte de clasificación
        report = self.automl.get_classification_report()
        report_frame = tk.LabelFrame(scrollable_frame, text="📋 Reporte de Clasificación",
                                    bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                    font=('Arial', 12, 'bold'))
        report_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Headers
        headers_frame = tk.Frame(report_frame, bg=DARK_THEME['frame_bg'])
        headers_frame.pack()
        tk.Label(headers_frame, text="Clase", width=15, relief=tk.RIDGE,
                bg=DARK_THEME['highlight'], fg='white').pack(side=tk.LEFT, padx=1, pady=1)
        tk.Label(headers_frame, text="Precisión", width=12, relief=tk.RIDGE,
                bg=DARK_THEME['highlight'], fg='white').pack(side=tk.LEFT, padx=1, pady=1)
        tk.Label(headers_frame, text="Recall", width=12, relief=tk.RIDGE,
                bg=DARK_THEME['highlight'], fg='white').pack(side=tk.LEFT, padx=1, pady=1)
        tk.Label(headers_frame, text="F1-Score", width=12, relief=tk.RIDGE,
                bg=DARK_THEME['highlight'], fg='white').pack(side=tk.LEFT, padx=1, pady=1)
        tk.Label(headers_frame, text="Soporte", width=10, relief=tk.RIDGE,
                bg=DARK_THEME['highlight'], fg='white').pack(side=tk.LEFT, padx=1, pady=1)
        
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                row_frame = tk.Frame(report_frame, bg=DARK_THEME['frame_bg'])
                row_frame.pack()
                tk.Label(row_frame, text=class_name[:15], width=15, relief=tk.RIDGE,
                        bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=1, pady=1)
                tk.Label(row_frame, text=f"{metrics['precision']:.3f}", width=12, relief=tk.RIDGE,
                        bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=1, pady=1)
                tk.Label(row_frame, text=f"{metrics['recall']:.3f}", width=12, relief=tk.RIDGE,
                        bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=1, pady=1)
                tk.Label(row_frame, text=f"{metrics['f1-score']:.3f}", width=12, relief=tk.RIDGE,
                        bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=1, pady=1)
                tk.Label(row_frame, text=f"{metrics['support']:.0f}", width=10, relief=tk.RIDGE,
                        bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT, padx=1, pady=1)