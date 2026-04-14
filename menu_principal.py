import tkinter as tk
from tkinter import ttk
from config.theme import DARK_THEME

class MenuPrincipal:
    def __init__(self, root, on_model_selected):
        self.root = root
        self.on_model_selected = on_model_selected
        self.root.title("ML Visualizer - Selección de Modelo")
        self.root.geometry("650x580")
        self.root.configure(bg=DARK_THEME['bg'])
        self.center_window()
        self.create_widgets()
    
    def center_window(self):
        self.root.update_idletasks()
        width = 1500
        height = 720
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg=DARK_THEME['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        title_label = tk.Label(main_frame, text="🤖 ML Visualizer", font=('Arial', 28, 'bold'),
                               bg=DARK_THEME['bg'], fg=DARK_THEME['highlight'])
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(main_frame, text="Seleccione el tipo de modelo a entrenar",
                                  font=('Arial', 12), bg=DARK_THEME['bg'], fg=DARK_THEME['fg'])
        subtitle_label.pack(pady=(0, 30))
        
        models_frame = tk.Frame(main_frame, bg=DARK_THEME['bg'])
        models_frame.pack(fill=tk.BOTH, expand=True)
        
        self.selected_model = tk.StringVar(value="logistic")
        
        radio_style = {
            'bg': DARK_THEME['bg'], 'fg': DARK_THEME['fg'],
            'selectcolor': DARK_THEME['select_bg'], 'activebackground': DARK_THEME['bg'],
            'activeforeground': DARK_THEME['fg'], 'font': ('Arial', 11)
        }
        
        # Regresión Logística
        logistic_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        logistic_frame.pack(fill=tk.X, pady=8, padx=20)
        tk.Radiobutton(logistic_frame, text="📊 Regresión Logística", variable=self.selected_model,
                       value="logistic", **radio_style).pack(anchor=tk.W, padx=15, pady=(8,2))
        tk.Label(logistic_frame, text="  Clasificación binaria o multiclase. Predice probabilidades.",
                 font=('Arial', 9), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                 justify=tk.LEFT).pack(anchor=tk.W, padx=35, pady=(0,8))
        
        # Regresión Lineal Simple
        simple_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        simple_frame.pack(fill=tk.X, pady=8, padx=20)
        tk.Radiobutton(simple_frame, text="📈 Regresión Lineal Simple", variable=self.selected_model,
                       value="simple", **radio_style).pack(anchor=tk.W, padx=15, pady=(8,2))
        tk.Label(simple_frame, text="  Una variable independiente, una dependiente continua.",
                 font=('Arial', 9), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                 justify=tk.LEFT).pack(anchor=tk.W, padx=35, pady=(0,8))
        
        # Regresión Lineal Múltiple
        multiple_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        multiple_frame.pack(fill=tk.X, pady=8, padx=20)
        tk.Radiobutton(multiple_frame, text="📉 Regresión Lineal Múltiple", variable=self.selected_model,
                       value="multiple", **radio_style).pack(anchor=tk.W, padx=15, pady=(8,2))
        tk.Label(multiple_frame, text="  Múltiples variables independientes, una dependiente continua.",
                 font=('Arial', 9), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                 justify=tk.LEFT).pack(anchor=tk.W, padx=35, pady=(0,8))
        
        # Random Forest (NUEVO)
        rf_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        rf_frame.pack(fill=tk.X, pady=8, padx=20)
        tk.Radiobutton(rf_frame, text="🌲 Random Forest", variable=self.selected_model,
                       value="random_forest", **radio_style).pack(anchor=tk.W, padx=15, pady=(8,2))
        tk.Label(rf_frame, text="  Clasificación por ensamblado de árboles de decisión. Muy robusto.",
                 font=('Arial', 9), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                 justify=tk.LEFT).pack(anchor=tk.W, padx=35, pady=(0,8))
        
        confirm_btn = tk.Button(main_frame, text="✓ CONFIRMAR Y CONTINUAR", command=self.confirm_selection,
                                bg=DARK_THEME['success'], fg='white', font=('Arial', 12, 'bold'),
                                cursor='hand2', pady=10)
        confirm_btn.pack(fill=tk.X, pady=30)
        
        info_label = tk.Label(main_frame, text="💡 Los datos deben estar en formato Excel (.xlsx, .xls) o CSV",
                              font=('Arial', 9), bg=DARK_THEME['bg'], fg=DARK_THEME['info'])
        info_label.pack()
    
    def confirm_selection(self):
        model_type = self.selected_model.get()
        self.root.destroy()
        self.on_model_selected(model_type)
