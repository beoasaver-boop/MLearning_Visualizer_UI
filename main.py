import customtkinter as ctk
import analytics
import config
import ml_gui
import utils
from menu_principal import MenuPrincipal

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def start_ml_app(model_type):
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1400) // 2
    y = (screen_height - 900) // 2
    root.geometry(f"1400x900+{x}+{y}")
    root.title(f"MLearning Visualizer - {model_type}")

    app = ml_gui.MLVisualizerApp(root, model_type=model_type)
    root.mainloop()

def main():
    menu_root = ctk.CTk()

    def on_model_selected(model_type):
        start_ml_app(model_type)

    app = MenuPrincipal(menu_root, on_model_selected)
    menu_root.mainloop()

if __name__ == "__main__":
    main()
