#!/usr/bin/env python3
"""
ML Visualizer - Aplicación de Machine Learning con Interfaz Gráfica
Permite cargar datos, seleccionar variables y entrenar modelos con visualización en tiempo real
"""

import sys
import tkinter as tk
from ml_gui import MLVisualizerApp

def main():
    """Función principal"""
    root = tk.Tk()
    
    # Configurar icono (opcional)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    # Centrar ventana en pantalla
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1400) // 2
    y = (screen_height - 900) // 2
    root.geometry(f"1400x900+{x}+{y}")
    
    # Iniciar aplicación
    app = MLVisualizerApp(root)
    
    # Ejecutar
    root.mainloop()

if __name__ == "__main__":
    main()