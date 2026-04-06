#!/usr/bin/env python3
"""
ML Visualizer - Aplicación de Machine Learning con Interfaz Gráfica
Ahora con soporte para Regresión Logística, Lineal Simple y Lineal Múltiple
"""

import sys
import tkinter as tk
from menu_principal import MenuPrincipal
from ml_gui import MLVisualizerApp

def start_ml_app(model_type):
    """Inicia la aplicación principal con el tipo de modelo seleccionado"""
    root = tk.Tk()
    
    # Configurar ventana
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # Centrar ventana
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1400) // 2
    y = (screen_height - 900) // 2
    root.geometry(f"1400x900+{x}+{y}")
    
    # Iniciar aplicación con el modelo seleccionado
    app = MLVisualizerApp(root, model_type=model_type)
    
    # Ejecutar
    root.mainloop()

def main():
    """Función principal - Muestra menú primero"""
    # Crear ventana del menú
    menu_root = tk.Tk()
    
    def on_model_selected(model_type):
        start_ml_app(model_type)
    
    app = MenuPrincipal(menu_root, on_model_selected)
    menu_root.mainloop()

if __name__ == "__main__":
    main()