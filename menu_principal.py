"""
Menú Principal - Selección de tipo de modelo
"""
import tkinter as tk
from tkinter import ttk
from styles import DARK_THEME

class MenuPrincipal:
    def __init__(self, root, on_model_selected):
        self.root = root
        self.on_model_selected = on_model_selected
        
        self.root.title("ML Visualizer - Selección de Modelo")
        self.root.geometry("600x500")
        self.root.configure(bg=DARK_THEME['bg'])
        
        # Centrar ventana
        self.center_window()
        
        self.create_widgets()
    
    def center_window(self):
        """Centrar la ventana en la pantalla"""
        self.root.update_idletasks()
        width = 600
        height = 500
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Crear widgets del menú principal"""
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg=DARK_THEME['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Título
        title_label = tk.Label(
            main_frame, 
            text="🤖 ML Visualizer", 
            font=('Arial', 28, 'bold'),
            bg=DARK_THEME['bg'], 
            fg=DARK_THEME['highlight']
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(
            main_frame,
            text="Seleccione el tipo de modelo a entrenar",
            font=('Arial', 12),
            bg=DARK_THEME['bg'],
            fg=DARK_THEME['fg']
        )
        subtitle_label.pack(pady=(0, 30))
        
        # Frame para los botones de modelo
        models_frame = tk.Frame(main_frame, bg=DARK_THEME['bg'])
        models_frame.pack(fill=tk.BOTH, expand=True)
        
        # Variable para almacenar selección
        self.selected_model = tk.StringVar(value="logistic")
        
        # Estilo para los botones de radio
        radio_style = {
            'bg': DARK_THEME['bg'],
            'fg': DARK_THEME['fg'],
            'selectcolor': DARK_THEME['select_bg'],
            'activebackground': DARK_THEME['bg'],
            'activeforeground': DARK_THEME['fg'],
            'font': ('Arial', 11)
        }
        
        # Opción 1: Regresión Logística
        logistic_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        logistic_frame.pack(fill=tk.X, pady=10, padx=20)
        
        tk.Radiobutton(
            logistic_frame, 
            text="Regresión Logística", 
            variable=self.selected_model, 
            value="logistic",
            **radio_style
        ).pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        tk.Label(
            logistic_frame,
            text="  Clasificación binaria o multiclase. Predice probabilidades de categorías.\n  Ejemplo: Predecir si un estudiante aprueba (Pass/Fail)",
            font=('Arial', 9),
            bg=DARK_THEME['frame_bg'],
            fg=DARK_THEME['fg'],
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=35, pady=(0, 10))
        
        # Opción 2: Regresión Lineal Simple
        simple_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        simple_frame.pack(fill=tk.X, pady=10, padx=20)
        
        tk.Radiobutton(
            simple_frame, 
            text="Regresión Lineal Simple", 
            variable=self.selected_model, 
            value="simple",
            **radio_style
        ).pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        tk.Label(
            simple_frame,
            text="  Una variable independiente, una dependiente. Relación lineal.\n  Ejemplo: Predecir 'math_score' usando 'study_hours_per_day'",
            font=('Arial', 9),
            bg=DARK_THEME['frame_bg'],
            fg=DARK_THEME['fg'],
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=35, pady=(0, 10))
        
        # Opción 3: Regresión Lineal Múltiple
        multiple_frame = tk.Frame(models_frame, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        multiple_frame.pack(fill=tk.X, pady=10, padx=20)
        
        tk.Radiobutton(
            multiple_frame, 
            text="Regresión Lineal Múltiple", 
            variable=self.selected_model, 
            value="multiple",
            **radio_style
        ).pack(anchor=tk.W, padx=15, pady=(10, 5))
        
        tk.Label(
            multiple_frame,
            text="  Múltiples variables independientes, una dependiente continua.\n  Ejemplo: Predecir 'final_exam_score' usando horas estudio, asistencia, etc.",
            font=('Arial', 9),
            bg=DARK_THEME['frame_bg'],
            fg=DARK_THEME['fg'],
            justify=tk.LEFT
        ).pack(anchor=tk.W, padx=35, pady=(0, 10))
        
        # Botón Confirmar
        confirm_btn = tk.Button(
            main_frame,
            text="✓ CONFIRMAR Y CONTINUAR",
            command=self.confirm_selection,
            bg=DARK_THEME['success'],
            fg='white',
            font=('Arial', 12, 'bold'),
            cursor='hand2',
            pady=10
        )
        confirm_btn.pack(fill=tk.X, pady=30)
        
        # Información adicional
        info_label = tk.Label(
            main_frame,
            text="💡 Los datos deben estar en formato Excel (.xlsx, .xls) o CSV",
            font=('Arial', 9),
            bg=DARK_THEME['bg'],
            fg=DARK_THEME['info']
        )
        info_label.pack()
    
    def confirm_selection(self):
        """Confirmar selección y cerrar menú"""
        model_type = self.selected_model.get()
        self.root.destroy()  # Cerrar menú
        self.on_model_selected(model_type)