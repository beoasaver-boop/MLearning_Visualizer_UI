import customtkinter as ctk

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.show_tip)
        widget.bind('<Leave>', self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        except:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + 25

        self.tip_window = ctk.CTkToplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        self.tip_window.attributes('-topmost', True)

        label = ctk.CTkLabel(
            self.tip_window, text=self.text,
            justify="left",
            corner_radius=4,
            font=ctk.CTkFont(size=10),
            wraplength=280
        )
        label.pack(padx=6, pady=4)

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

TOOLTIPS = {
    "load_file": "Carga archivos Excel (.xlsx, .xls) o CSV para analisis",
    "features_frame": "Seleccione las variables independientes (features) marcando las casillas",
    "target_menu": "Seleccione la variable que desea predecir (target)",
    "confirm_vars_btn": "Confirma la seleccion de variables y prepara el modelo",
    "test_size": "Proporcion de datos para prueba (ej: 0.2 = 20% prueba, 80% entrenamiento)",
    "epochs": "Numero de iteraciones de entrenamiento. Mas epochs = mas preciso pero mas lento",
    "nulls_handling": "Como manejar valores faltantes: 1=Eliminar filas, 2=Mediana/Moda, 3=Rellenar con 0",
    "train_btn": "Inicia el entrenamiento del modelo con los parametros seleccionados",
    "learning_rate": "Tasa de aprendizaje (solo regresion). Valores tipicos: 0.01, 0.001, 0.0001"
}

def add_tooltips_to_widgets(app):
    if hasattr(app, 'load_btn') and app.load_btn:
        ToolTip(app.load_btn, TOOLTIPS["load_file"])

    if hasattr(app, 'features_frame') and app.features_frame:
        ToolTip(app.features_frame, TOOLTIPS["features_frame"])

    if hasattr(app, 'target_menu') and app.target_menu:
        ToolTip(app.target_menu, TOOLTIPS["target_menu"])

    if hasattr(app, 'confirm_vars_btn') and app.confirm_vars_btn:
        ToolTip(app.confirm_vars_btn, TOOLTIPS["confirm_vars_btn"])

    if hasattr(app, 'train_btn') and app.train_btn:
        ToolTip(app.train_btn, TOOLTIPS["train_btn"])

    if hasattr(app, 'test_size_entry') and app.test_size_entry:
        ToolTip(app.test_size_entry, TOOLTIPS["test_size"])

    if hasattr(app, 'epochs_entry') and app.epochs_entry:
        ToolTip(app.epochs_entry, TOOLTIPS["epochs"])

    if hasattr(app, 'nulls_menu') and app.nulls_menu:
        ToolTip(app.nulls_menu, TOOLTIPS["nulls_handling"])

    if hasattr(app, 'learning_rate_entry') and app.learning_rate_entry:
        ToolTip(app.learning_rate_entry, TOOLTIPS["learning_rate"])
