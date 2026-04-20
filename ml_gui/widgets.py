import tkinter as tk
from tkinter import ttk
from config.theme import DARK_THEME
from ml_gui.plots import TrainingPlotsManager

def setup_ttk_styles():
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Dark.TFrame', background=DARK_THEME['bg'])
    style.configure('Dark.TLabel', background=DARK_THEME['bg'], foreground=DARK_THEME['fg'])
    style.configure('Dark.TButton', background=DARK_THEME['button_bg'], foreground=DARK_THEME['button_fg'])
    style.map('Dark.TButton', background=[('active', DARK_THEME['button_active'])])
    style.configure('Success.TButton', background=DARK_THEME['success'])
    style.configure('Info.TButton', background=DARK_THEME['info'])
    style.configure('Dark.TEntry', fieldbackground=DARK_THEME['entry_bg'], foreground=DARK_THEME['entry_fg'])

class LeftPanelBuilder:
    def __init__(self, app):
        self.app = app
        self.file_label = None
        self.features_listbox = None
        self.target_var = None
        self.target_entry = None
        self.confirm_vars_btn = None
        self.train_btn = None
        self.status_text = None
        self.test_size_var = None
        self.epochs_var = None
        self.nulls_var = None
        self.learning_rate_var = None
    
    def build(self):
        left_panel = ttk.Frame(self.app.root, style='Dark.TFrame', width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        title = tk.Label(left_panel, text=f"ML Visualizer - {self.app.get_model_name()}",
                         font=('Arial', 18, 'bold'), bg=DARK_THEME['bg'], fg=DARK_THEME['highlight'])
        title.pack(pady=10)
        
        self._add_file_section(left_panel)
        self._add_variables_section(left_panel)
        self._add_parameters_section(left_panel)
        self._add_train_button(left_panel)
        self._add_status_area(left_panel)
        return left_panel
    
    def _add_file_section(self, parent):
        frame = tk.LabelFrame(parent, text="1. Cargar Datos", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 11, 'bold'))
        frame.pack(fill=tk.X, padx=10, pady=10)
        self.file_label = tk.Label(frame, text="No hay archivo seleccionado",
                                   bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'])
        self.file_label.pack(padx=10, pady=5)
        self.app.load_btn = tk.Button(frame, text="📁 Cargar Archivo (Excel/CSV)",
                                      command=self.app.load_file,
                                      bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                                      font=('Arial', 10), cursor='hand2')
        self.app.load_btn.pack(padx=10, pady=5)
    
    def _add_variables_section(self, parent):
        vars_frame = tk.LabelFrame(parent, text="2. Seleccionar Variables",
                                   bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                   font=('Arial', 11, 'bold'))
        vars_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Etiqueta de instrucciones según el modelo
        if self.app.model_type == 'simple':
            instruccion = "Variables Independientes (seleccione SOLO UNA):"
        else:
            instruccion = "Variables Independientes (seleccione múltiples con Ctrl+Click):"
        
        tk.Label(vars_frame, text=instruccion,
                 bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        
        # Frame para listbox con scroll
        listbox_frame = tk.Frame(vars_frame, bg=DARK_THEME['frame_bg'])
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Listbox con selección múltiple (para todos, luego validamos)
        self.features_listbox = tk.Listbox(
            listbox_frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
            bg=DARK_THEME['entry_bg'],
            fg=DARK_THEME['fg'],
            selectbackground=DARK_THEME['highlight'],
            selectforeground='white',
            height=8,
            font=('Arial', 9)
        )
        self.features_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.features_listbox.yview)
        
        # Botones de selección
        btn_frame = tk.Frame(vars_frame, bg=DARK_THEME['frame_bg'])
        btn_frame.pack(pady=5)
        
        def select_all():
            self.features_listbox.selection_set(0, tk.END)
            # Para regresión simple, corregir después
            if self.app.model_type == 'simple' and self.features_listbox.size() > 0:
                # Dejar solo el primer elemento seleccionado
                self.features_listbox.selection_clear(1, tk.END)
        
        def deselect_all():
            self.features_listbox.selection_clear(0, tk.END)
        
        tk.Button(btn_frame, text="✓ Seleccionar Todas", command=select_all,
                  bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                  font=('Arial', 8), cursor='hand2').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="✗ Deseleccionar Todas", command=deselect_all,
                  bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                  font=('Arial', 8), cursor='hand2').pack(side=tk.LEFT, padx=5)
        
        # Variable dependiente
        tk.Label(vars_frame, text="Variable Dependiente (Target):",
                 bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=10, pady=(10,0))
        
        self.target_var = tk.StringVar()
        self.target_entry = tk.Entry(vars_frame, textvariable=self.target_var, width=35,
                                     bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg'],
                                     insertbackground=DARK_THEME['fg'])
        self.target_entry.pack(padx=10, pady=5)
        
        # Botón confirmar
        self.confirm_vars_btn = tk.Button(vars_frame, text="✓ Confirmar Variables",
                                          command=self.app.confirm_variables,
                                          bg=DARK_THEME['success'], fg='white',
                                          font=('Arial', 10, 'bold'), cursor='hand2',
                                          state=tk.DISABLED)
        self.confirm_vars_btn.pack(padx=10, pady=10)
    
    def _add_parameters_section(self, parent):
        params_frame = tk.LabelFrame(parent, text="3. Parámetros de Entrenamiento",
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
        
        if self.app.model_type in ['simple', 'multiple']:
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
    
    def _add_train_button(self, parent):
        self.train_btn = tk.Button(parent, text="🚀 INICIAR ENTRENAMIENTO",
                                   command=self.app.start_training,
                                   bg=DARK_THEME['highlight'], fg='white',
                                   font=('Arial', 12, 'bold'), cursor='hand2',
                                   state=tk.DISABLED)
        self.train_btn.pack(padx=10, pady=20, fill=tk.X)
    
    def _add_status_area(self, parent):
        status_frame = tk.LabelFrame(parent, text="Estado", bg=DARK_THEME['frame_bg'],
                                     fg=DARK_THEME['fg'], font=('Arial', 11, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.status_text = tk.Text(status_frame, height=8, width=40,
                                   bg=DARK_THEME['entry_bg'], fg=DARK_THEME['entry_fg'],
                                   wrap=tk.WORD, font=('Consolas', 9))
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll = tk.Scrollbar(self.status_text)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scroll.set)
        scroll.config(command=self.status_text.yview)

class RightPanelBuilder:
    def __init__(self, app):
        self.app = app
        self.results_display = None
        self.plots_manager = None
        self.plots_frame = None
    
    def build(self):
        right_panel = ttk.Frame(self.app.root, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pestaña entrenamiento
        training_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(training_frame, text="📈 Entrenamiento en Vivo")
        self.plots_frame = tk.Frame(training_frame, bg=DARK_THEME['bg'])
        self.plots_frame.pack(fill=tk.BOTH, expand=True)
        self.plots_manager = TrainingPlotsManager(self.plots_frame)
        
        # Pestaña resultados
        results_frame = ttk.Frame(notebook, style='Dark.TFrame')
        notebook.add(results_frame, text="📊 Resultados Finales")
        self.results_display = tk.Frame(results_frame, bg=DARK_THEME['bg'])
        self.results_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        return right_panel
