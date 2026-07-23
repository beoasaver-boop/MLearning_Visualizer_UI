import customtkinter as ctk
from ml_gui.plots import TrainingPlotsManager

class LeftPanelBuilder:
    def __init__(self, app):
        self.app = app
        self.file_label = None
        self.features_frame = None
        self.feature_vars = {}
        self.target_var = None
        self.target_menu = None
        self.confirm_vars_btn = None
        self.train_btn = None
        self.status_text = None
        self.test_size_var = None
        self.test_size_entry = None
        self.epochs_var = None
        self.epochs_entry = None
        self.nulls_var = None
        self.nulls_menu = None
        self.learning_rate_var = None
        self.learning_rate_entry = None
        self.progress_bar = None

    def build(self):
        left_panel = ctk.CTkFrame(self.app.root, width=380, corner_radius=0)
        left_panel.pack(side="left", fill="y", padx=(0, 0))
        left_panel.pack_propagate(False)

        title = ctk.CTkLabel(
            left_panel,
            text=f"ML Visualizer\n{self.app.get_model_name()}",
            font=ctk.CTkFont(size=18, weight="bold"),
            justify="center"
        )
        title.pack(pady=(15, 10))

        self._add_file_section(left_panel)
        self._add_variables_section(left_panel)
        self._add_parameters_section(left_panel)
        self._add_train_button(left_panel)
        self._add_progress_bar(left_panel)
        self._add_status_area(left_panel)
        return left_panel

    def _add_file_section(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", padx=10, pady=(5, 5))

        header = ctk.CTkLabel(
            frame, text="1. Cargar Datos",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        header.pack(padx=12, pady=(8, 2), anchor="w")

        self.file_label = ctk.CTkLabel(
            frame, text="No hay archivo seleccionado",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        self.file_label.pack(padx=12, pady=(0, 5))

        self.app.load_btn = ctk.CTkButton(
            frame, text="Cargar Archivo (Excel/CSV)",
            command=self.app.load_file,
            font=ctk.CTkFont(size=11),
            height=32, corner_radius=6
        )
        self.app.load_btn.pack(padx=12, pady=(0, 8), fill="x")

    def _add_variables_section(self, parent):
        vars_frame = ctk.CTkFrame(parent, corner_radius=8)
        vars_frame.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        header = ctk.CTkLabel(
            vars_frame, text="2. Seleccionar Variables",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        header.pack(padx=12, pady=(8, 2), anchor="w")

        instr_text = (
            "Variables Independientes:" if self.app.model_type != 'simple'
            else "Variable Independiente (solo UNA):"
        )
        instr = ctk.CTkLabel(
            vars_frame, text=instr_text,
            font=ctk.CTkFont(size=11)
        )
        instr.pack(anchor="w", padx=12, pady=(5, 0))

        self.features_frame = ctk.CTkScrollableFrame(
            vars_frame, height=120, corner_radius=6
        )
        self.features_frame.pack(fill="both", expand=True, padx=10, pady=(3, 5))

        btn_row = ctk.CTkFrame(vars_frame, fg_color="transparent")
        btn_row.pack(pady=(0, 5))

        ctk.CTkButton(
            btn_row, text="Seleccionar Todas",
            command=self.app.select_all,
            font=ctk.CTkFont(size=10),
            width=100, height=26, corner_radius=5
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row, text="Deseleccionar Todas",
            command=self.app.deselect_all,
            font=ctk.CTkFont(size=10),
            width=110, height=26, corner_radius=5
        ).pack(side="left", padx=4)

        target_label = ctk.CTkLabel(
            vars_frame, text="Variable Dependiente (Target):",
            font=ctk.CTkFont(size=11)
        )
        target_label.pack(anchor="w", padx=12, pady=(3, 0))

        self.target_var = ctk.StringVar(value="")
        self.target_menu = ctk.CTkOptionMenu(
            vars_frame, variable=self.target_var, values=[""],
            font=ctk.CTkFont(size=11), dropdown_font=ctk.CTkFont(size=11)
        )
        self.target_menu.pack(padx=10, pady=(3, 5), fill="x")

        self.confirm_vars_btn = ctk.CTkButton(
            vars_frame, text="Confirmar Variables",
            command=self.app.confirm_variables,
            font=ctk.CTkFont(size=11, weight="bold"),
            state="disabled", height=32, corner_radius=6
        )
        self.confirm_vars_btn.pack(padx=10, pady=(0, 8), fill="x")

    def _add_parameters_section(self, parent):
        params_frame = ctk.CTkFrame(parent, corner_radius=8)
        params_frame.pack(fill="x", padx=10, pady=(5, 5))

        header = ctk.CTkLabel(
            params_frame, text="3. Parametros de Entrenamiento",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        header.pack(padx=12, pady=(8, 5), anchor="w")

        grid = ctk.CTkFrame(params_frame, fg_color="transparent")
        grid.pack(padx=12, pady=(0, 5), fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            grid, text="Test size (0-1):",
            font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.test_size_var = ctk.StringVar(value="0.3")
        self.test_size_entry = ctk.CTkEntry(
            grid, textvariable=self.test_size_var, width=70,
            font=ctk.CTkFont(size=11), corner_radius=5
        )
        self.test_size_entry.grid(row=0, column=1, sticky="e", pady=(0, 2), padx=(5, 0))

        ctk.CTkLabel(
            grid, text="Epochs:", font=ctk.CTkFont(size=11)
        ).grid(row=1, column=0, sticky="w", pady=(0, 2))
        self.epochs_var = ctk.StringVar(value="100")
        self.epochs_entry = ctk.CTkEntry(
            grid, textvariable=self.epochs_var, width=70,
            font=ctk.CTkFont(size=11), corner_radius=5
        )
        self.epochs_entry.grid(row=1, column=1, sticky="e", pady=(0, 2), padx=(5, 0))

        row_offset = 2
        if self.app.model_type in ['simple', 'multiple']:
            ctk.CTkLabel(
                grid, text="Learning rate:", font=ctk.CTkFont(size=11)
            ).grid(row=2, column=0, sticky="w", pady=(0, 2))
            self.learning_rate_var = ctk.StringVar(value="0.01")
            self.learning_rate_entry = ctk.CTkEntry(
                grid, textvariable=self.learning_rate_var, width=70,
                font=ctk.CTkFont(size=11), corner_radius=5
            )
            self.learning_rate_entry.grid(row=2, column=1, sticky="e", pady=(0, 2), padx=(5, 0))
            row_offset = 3

        ctk.CTkLabel(
            grid, text="Manejo de nulos:", font=ctk.CTkFont(size=11)
        ).grid(row=row_offset, column=0, sticky="w", pady=(0, 2))
        self.nulls_var = ctk.StringVar(value="2 - Mediana/Moda")
        self.nulls_menu = ctk.CTkOptionMenu(
            grid, variable=self.nulls_var,
            values=["1 - Eliminar filas", "2 - Mediana/Moda", "3 - Rellenar con 0"],
            font=ctk.CTkFont(size=10), dropdown_font=ctk.CTkFont(size=10),
            width=100
        )
        self.nulls_menu.grid(row=row_offset, column=1, sticky="e", pady=(0, 2), padx=(5, 0))

        grid._grid_line = grid.grid_size()[1]

    def _add_train_button(self, parent):
        self.train_btn = ctk.CTkButton(
            parent, text="Iniciar Entrenamiento",
            command=self.app.start_training,
            font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled", height=38, corner_radius=8
        )
        self.train_btn.pack(padx=10, pady=(10, 5), fill="x")

    def _add_progress_bar(self, parent):
        self.progress_bar = ctk.CTkProgressBar(parent, corner_radius=4)
        self.progress_bar.pack(padx=10, pady=(0, 5), fill="x")
        self.progress_bar.set(0)

    def _add_status_area(self, parent):
        status_frame = ctk.CTkFrame(parent, corner_radius=8)
        status_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        header = ctk.CTkLabel(
            status_frame, text="Estado",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        header.pack(padx=12, pady=(8, 2), anchor="w")

        self.status_text = ctk.CTkTextbox(
            status_frame, font=ctk.CTkFont(family="Consolas", size=10),
            corner_radius=6, wrap="word"
        )
        self.status_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))

class RightPanelBuilder:
    def __init__(self, app):
        self.app = app
        self.results_display = None
        self.plots_manager = None
        self.plots_frame = None

    def build(self):
        right_panel = ctk.CTkFrame(self.app.root, corner_radius=0)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0), pady=0)

        tabview = ctk.CTkTabview(right_panel, corner_radius=8)
        tabview.pack(fill="both", expand=True, padx=8, pady=8)

        training_tab = tabview.add("Entrenamiento en Vivo")
        results_tab = tabview.add("Resultados Finales")

        self.plots_frame = ctk.CTkFrame(training_tab, fg_color="transparent", corner_radius=0)
        self.plots_frame.pack(fill="both", expand=True)
        self.plots_manager = TrainingPlotsManager(self.plots_frame)

        self.results_display = ctk.CTkScrollableFrame(
            results_tab, fg_color="transparent", corner_radius=0
        )
        self.results_display.pack(fill="both", expand=True, padx=5, pady=5)

        return right_panel
