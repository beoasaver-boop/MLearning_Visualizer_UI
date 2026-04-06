"""
Construcción y gestión de widgets de la interfaz gráfica
"""
import tkinter as tk
from tkinter import ttk
from config.theme import DARK_THEME


class LeftPanelBuilder:
    """Constructor del panel izquierdo de configuración"""
    
    def __init__(self, root, model_type, dark_theme=None):
        self.root = root
        self.model_type = model_type
        self.theme = dark_theme or DARK_THEME
        self.widgets = {}  # Almacenar referencias a widgets
    
    def build(self, parent):
        """Construir panel izquierdo completo"""
        left_panel = ttk.Frame(parent, style='Dark.TFrame', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Secciones del panel
        self._add_title(left_panel)
        self._add_file_section(left_panel)
        self._add_variables_section(left_panel)
        self._add_parameters_section(left_panel)
        self._add_train_button(left_panel)
        self._add_status_area(left_panel)
        
        return left_panel
    
    def _add_title(self, parent):
        """Agregar título"""
        model_names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple'
        }
        title_text = f"ML Visualizer - {model_names.get(self.model_type, 'Desconocido')}"
        
        title_label = tk.Label(parent, text=title_text, 
                              font=('Arial', 18, 'bold'),
                              bg=self.theme['bg'], fg=self.theme['highlight'])
        title_label.pack(pady=10)
    
    def _add_file_section(self, parent):
        """Agregar sección de carga de archivos"""
        file_frame = tk.LabelFrame(parent, text="1. Cargar Datos",
                                  bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                  font=('Arial', 11, 'bold'))
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.widgets['file_label'] = tk.Label(file_frame, text="No hay archivo seleccionado",
                                             bg=self.theme['frame_bg'], fg=self.theme['fg'])
        self.widgets['file_label'].pack(padx=10, pady=5)
        
        self.widgets['load_btn'] = tk.Button(file_frame, text="📁 Cargar Archivo (Excel/CSV)",
                                            bg=self.theme['button_bg'], fg=self.theme['button_fg'],
                                            font=('Arial', 10), cursor='hand2')
        self.widgets['load_btn'].pack(padx=10, pady=5)
    
    def _add_variables_section(self, parent):
        """Agregar sección de selección de variables"""
        vars_frame = tk.LabelFrame(parent, text="2. Seleccionar Variables",
                                  bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                  font=('Arial', 11, 'bold'))
        vars_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Variables independientes
        tk.Label(vars_frame, text="Variables Independientes (Features):",
                bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        # Canvas con scrollbar para checkbuttons
        features_container = tk.Frame(vars_frame, bg=self.theme['frame_bg'])
        features_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        features_canvas = tk.Canvas(features_container, bg=self.theme['frame_bg'],
                                   height=120, highlightthickness=0)
        scrollbar = tk.Scrollbar(features_container, orient="vertical",
                                command=features_canvas.yview)
        self.widgets['features_scroll_frame'] = tk.Frame(features_canvas, bg=self.theme['frame_bg'])
        
        features_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        canvas_window = features_canvas.create_window((0, 0), 
                                                     window=self.widgets['features_scroll_frame'],
                                                     anchor="nw", width=features_canvas.winfo_width())
        
        def configure_scroll_region(event):
            features_canvas.configure(scrollregion=features_canvas.bbox("all"))
        
        def configure_canvas_width(event):
            features_canvas.itemconfig(canvas_window, width=event.width)
        
        self.widgets['features_scroll_frame'].bind("<Configure>", configure_scroll_region)
        features_canvas.bind("<Configure>", configure_canvas_width)
        features_canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            features_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        features_canvas.bind("<MouseWheel>", _on_mousewheel)
        self.widgets['features_scroll_frame'].bind("<MouseWheel>", _on_mousewheel)
        
        self.widgets['features_vars'] = {}
        
        # Botones select/deselect
        select_all_frame = tk.Frame(vars_frame, bg=self.theme['frame_bg'])
        select_all_frame.pack(pady=5)
        
        self.widgets['select_all_btn'] = tk.Button(select_all_frame, text="✓ Seleccionar Todas",
                                                  bg=self.theme['button_bg'], 
                                                  fg=self.theme['button_fg'],
                                                  font=('Arial', 8), cursor='hand2')
        self.widgets['select_all_btn'].pack(side=tk.LEFT, padx=5)
        
        self.widgets['deselect_all_btn'] = tk.Button(select_all_frame, 
                                                    text="✗ Deseleccionar Todas",
                                                    bg=self.theme['button_bg'],
                                                    fg=self.theme['button_fg'],
                                                    font=('Arial', 8), cursor='hand2')
        self.widgets['deselect_all_btn'].pack(side=tk.LEFT, padx=5)
        
        # Variable dependiente
        tk.Label(vars_frame, text="Variable Dependiente (Target):",
                bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.widgets['target_var'] = tk.StringVar()
        self.widgets['target_entry'] = tk.Entry(vars_frame, 
                                               textvariable=self.widgets['target_var'],
                                               width=35, bg=self.theme['entry_bg'],
                                               fg=self.theme['entry_fg'],
                                               insertbackground=self.theme['fg'])
        self.widgets['target_entry'].pack(padx=10, pady=5)
        
        # Botón confirmar
        self.widgets['confirm_vars_btn'] = tk.Button(vars_frame, text="✓ Confirmar Variables",
                                                    bg=self.theme['success'], fg='white',
                                                    font=('Arial', 10, 'bold'),
                                                    cursor='hand2',
                                                    state=tk.DISABLED)
        self.widgets['confirm_vars_btn'].pack(padx=10, pady=10)
    
    def _add_parameters_section(self, parent):
        """Agregar sección de parámetros de entrenamiento"""
        params_frame = tk.LabelFrame(parent, text="3. Parámetros de Entrenamiento",
                                    bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                    font=('Arial', 11, 'bold'))
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Test size
        tk.Label(params_frame, text="Tamaño de Prueba (0-1):",
                bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.widgets['test_size_var'] = tk.StringVar(value="0.3")
        tk.Entry(params_frame, textvariable=self.widgets['test_size_var'], width=10,
                bg=self.theme['entry_bg'], fg=self.theme['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        # Epochs
        tk.Label(params_frame, text="Número de Epochs:",
                bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.widgets['epochs_var'] = tk.StringVar(value="100")
        tk.Entry(params_frame, textvariable=self.widgets['epochs_var'], width=10,
                bg=self.theme['entry_bg'], fg=self.theme['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        # Learning rate (solo para regresión)
        if self.model_type in ['simple', 'multiple']:
            tk.Label(params_frame, text="Tasa de Aprendizaje (0.001-0.1):",
                    bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(anchor=tk.W, padx=10, pady=(10, 0))
            self.widgets['learning_rate_var'] = tk.StringVar(value="0.01")
            tk.Entry(params_frame, textvariable=self.widgets['learning_rate_var'], width=10,
                    bg=self.theme['entry_bg'], fg=self.theme['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
        
        # Null handling
        tk.Label(params_frame, text="Manejo de Nulos:\n1=Eliminar, 2=Mediana/Moda, 3=0",
                bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(anchor=tk.W, padx=10, pady=(10, 0))
        self.widgets['nulls_var'] = tk.StringVar(value="2")
        tk.Entry(params_frame, textvariable=self.widgets['nulls_var'], width=10,
                bg=self.theme['entry_bg'], fg=self.theme['entry_fg']).pack(anchor=tk.W, padx=10, pady=5)
    
    def _add_train_button(self, parent):
        """Agregar botón de entrenamiento"""
        self.widgets['train_btn'] = tk.Button(parent, text="🚀 INICIAR ENTRENAMIENTO",
                                             bg=self.theme['highlight'], fg='white',
                                             font=('Arial', 12, 'bold'), cursor='hand2',
                                             state=tk.DISABLED)
        self.widgets['train_btn'].pack(padx=10, pady=20, fill=tk.X)
    
    def _add_status_area(self, parent):
        """Agregar área de estado"""
        status_frame = tk.LabelFrame(parent, text="Estado",
                                    bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                    font=('Arial', 11, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.widgets['status_text'] = tk.Text(status_frame, height=8, width=40,
                                             bg=self.theme['entry_bg'],
                                             fg=self.theme['entry_fg'],
                                             wrap=tk.WORD, font=('Consolas', 9))
        self.widgets['status_text'].pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar_status = tk.Scrollbar(self.widgets['status_text'])
        scrollbar_status.pack(side=tk.RIGHT, fill=tk.Y)
        self.widgets['status_text'].config(yscrollcommand=scrollbar_status.set)
        scrollbar_status.config(command=self.widgets['status_text'].yview)


class RightPanelBuilder:
    """Constructor del panel derecho con gráficas"""
    
    def __init__(self, dark_theme=None):
        self.theme = dark_theme or DARK_THEME
        self.widgets = {}
    
    def build(self, parent):
        """Construir panel derecho completo"""
        right_panel = ttk.Frame(parent, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook para pestañas
        self.widgets['notebook'] = ttk.Notebook(right_panel)
        self.widgets['notebook'].pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de entrenamiento en vivo
        self.widgets['training_frame'] = ttk.Frame(self.widgets['notebook'], style='Dark.TFrame')
        self.widgets['notebook'].add(self.widgets['training_frame'], text="📈 Entrenamiento en Vivo")
        
        self.widgets['plots_frame'] = tk.Frame(self.widgets['training_frame'], 
                                              bg=self.theme['bg'])
        self.widgets['plots_frame'].pack(fill=tk.BOTH, expand=True)
        
        # Pestaña de resultados
        self.widgets['results_frame'] = ttk.Frame(self.widgets['notebook'], style='Dark.TFrame')
        self.widgets['notebook'].add(self.widgets['results_frame'], text="📊 Resultados Finales")
        
        self.widgets['results_display'] = tk.Frame(self.widgets['results_frame'],
                                                  bg=self.theme['bg'])
        self.widgets['results_display'].pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        return right_panel


def setup_ttk_styles(dark_theme=None):
    """Configurar estilos para widgets ttk"""
    theme = dark_theme or DARK_THEME
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configurar colores
    style.configure('Dark.TFrame', background=theme['bg'])
    style.configure('Dark.TLabel', background=theme['bg'], foreground=theme['fg'])
    style.configure('Dark.TButton', background=theme['button_bg'],
                   foreground=theme['button_fg'], borderwidth=0, focuscolor='none')
    style.map('Dark.TButton', background=[('active', theme['button_active'])])
    
    style.configure('Success.TButton', background=theme['success'])
    style.configure('Info.TButton', background=theme['info'])
    
    style.configure('Dark.TEntry', fieldbackground=theme['entry_bg'],
                   foreground=theme['entry_fg'])
