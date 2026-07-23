import customtkinter as ctk

class MenuPrincipal:
    def __init__(self, root, on_model_selected):
        self.root = root
        self.on_model_selected = on_model_selected
        self.root.title("ML Visualizer - Selección de Modelo")
        self.center_window()
        self.create_widgets()

    def center_window(self):
        self.root.update_idletasks()
        width = 700
        height = 620
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=40, pady=40)

        title_label = ctk.CTkLabel(
            main_frame, text="ML Visualizer",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=("#1a73e8", "#00bcd4")
        )
        title_label.pack(pady=(0, 5))

        subtitle_label = ctk.CTkLabel(
            main_frame, text="Seleccione el tipo de modelo a entrenar",
            font=ctk.CTkFont(size=14)
        )
        subtitle_label.pack(pady=(0, 30))

        self.selected_model = ctk.StringVar(value="logistic")

        models = [
            ("logistic", "Regresion Logistica",
             "Clasificacion binaria o multiclase. Predice probabilidades."),
            ("simple", "Regresion Lineal Simple",
             "Una variable independiente, una dependiente continua."),
            ("multiple", "Regresion Lineal Multiple",
             "Multiples variables independientes, una dependiente continua."),
            ("random_forest", "Random Forest",
             "Clasificacion por ensamblado de arboles de decision. Muy robusto."),
        ]

        for value, title, desc in models:
            card = ctk.CTkFrame(main_frame, corner_radius=10)
            card.pack(fill="x", pady=6, padx=10)

            radio = ctk.CTkRadioButton(
                card, text=title, variable=self.selected_model,
                value=value, font=ctk.CTkFont(size=13, weight="bold")
            )
            radio.pack(anchor="w", padx=15, pady=(10, 0))

            desc_label = ctk.CTkLabel(
                card, text=f"  {desc}",
                font=ctk.CTkFont(size=11),
                text_color=("gray30", "gray70")
            )
            desc_label.pack(anchor="w", padx=35, pady=(0, 10))

        confirm_btn = ctk.CTkButton(
            main_frame, text="Confirmar y Continuar",
            command=self.confirm_selection,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45, corner_radius=8
        )
        confirm_btn.pack(fill="x", pady=30, padx=10)

        info_label = ctk.CTkLabel(
            main_frame, text="Los datos deben estar en formato Excel (.xlsx, .xls) o CSV",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        info_label.pack()

    def confirm_selection(self):
        model_type = self.selected_model.get()
        self.root.destroy()
        self.on_model_selected(model_type)
