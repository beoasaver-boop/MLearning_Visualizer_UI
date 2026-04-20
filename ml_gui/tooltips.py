# -*- coding: utf-8 -*-
"""
Módulo para mostrar tooltips informativos al pasar el mouse
Versión corregida para manejar diferentes tipos de widgets
"""

import tkinter as tk
from config.theme import DARK_THEME

class ToolTip:
    """Crea tooltips para widgets"""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind('<Enter>', self.show_tip)
        widget.bind('<Leave>', self.hide_tip)
    
    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        
        # Obtener posición del widget de manera segura
        try:
            # Para widgets que soportan bbox
            if hasattr(self.widget, 'bbox') and self.widget.winfo_class() != 'Listbox':
                x, y, _, _ = self.widget.bbox("insert")
                x += self.widget.winfo_rootx() + 25
                y += self.widget.winfo_rooty() + 25
            else:
                # Para Listbox y otros widgets sin bbox("insert")
                x = self.widget.winfo_rootx() + 25
                y = self.widget.winfo_rooty() + 25
        except:
            # Fallback: usar posición del widget
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 25
        
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background=DARK_THEME['frame_bg'],
                         foreground=DARK_THEME['fg'],
                         relief=tk.SOLID, borderwidth=1,
                         font=('Arial', 9), wraplength=300)
        label.pack()
    
    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# Tooltips para widgets comunes
TOOLTIPS = {
    "load_file": "📁 Carga archivos Excel (.xlsx, .xls) o CSV para análisis",
    "features_listbox": "📊 Seleccione variables independientes.\n• Ctrl+Click: selección múltiple\n• Click: selecciona/deselecciona",
    "target_entry": "🎯 Variable que desea predecir (columna objetivo)",
    "confirm_vars_btn": "✓ Confirma la selección de variables y prepara el modelo",
    "test_size": "📐 Proporción de datos para prueba\nEjemplo: 0.2 = 20% prueba, 80% entrenamiento",
    "epochs": "🔄 Número de iteraciones de entrenamiento\nMás épocas = más preciso pero más lento",
    "nulls_handling": "⚠️ Cómo manejar valores faltantes:\n1=Eliminar filas\n2=Mediana/Moda\n3=Rellenar con 0",
    "train_btn": "🚀 Inicia el entrenamiento del modelo con los parámetros seleccionados",
    "learning_rate": "📉 Tasa de aprendizaje (solo regresión)\nValores típicos: 0.01, 0.001, 0.0001"
}

def add_tooltips_to_widgets(app):
    """Añade tooltips a los widgets principales de la aplicación"""
    
    # Botón cargar archivo
    if hasattr(app, 'load_btn') and app.load_btn:
        ToolTip(app.load_btn, TOOLTIPS["load_file"])
    
    # Listbox de features
    if hasattr(app, 'features_listbox') and app.features_listbox:
        ToolTip(app.features_listbox, TOOLTIPS["features_listbox"])
    
    # Entry de target
    if hasattr(app, 'target_entry') and app.target_entry:
        ToolTip(app.target_entry, TOOLTIPS["target_entry"])
    
    # Botón confirmar variables
    if hasattr(app, 'confirm_vars_btn') and app.confirm_vars_btn:
        ToolTip(app.confirm_vars_btn, TOOLTIPS["confirm_vars_btn"])
    
    # Botón entrenar
    if hasattr(app, 'train_btn') and app.train_btn:
        ToolTip(app.train_btn, TOOLTIPS["train_btn"])
    
    # Buscar entries de parámetros (test_size, epochs, nulls, learning_rate)
    def find_entry_by_var(var_name):
        """Busca un Entry widget asociado a una StringVar"""
        for child in app.root.winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, tk.Entry):
                    if hasattr(subchild, 'textvariable') and subchild.textvariable:
                        if hasattr(subchild.textvariable, '_name') and var_name in str(subchild.textvariable):
                            return subchild
                        elif hasattr(subchild, 'get') and hasattr(app, var_name):
                            try:
                                if subchild.get() == getattr(app, var_name).get():
                                    return subchild
                            except:
                                pass
        return None
    
    # Test size tooltip
    if hasattr(app, 'test_size_var') and app.test_size_var:
        entry = find_entry_by_var('test_size_var')
        if entry:
            ToolTip(entry, TOOLTIPS["test_size"])
    
    # Epochs tooltip
    if hasattr(app, 'epochs_var') and app.epochs_var:
        entry = find_entry_by_var('epochs_var')
        if entry:
            ToolTip(entry, TOOLTIPS["epochs"])
    
    # Nulls handling tooltip
    if hasattr(app, 'nulls_var') and app.nulls_var:
        entry = find_entry_by_var('nulls_var')
        if entry:
            ToolTip(entry, TOOLTIPS["nulls_handling"])
    
    # Learning rate tooltip (solo si existe)
    if hasattr(app, 'learning_rate_var') and app.learning_rate_var:
        entry = find_entry_by_var('learning_rate_var')
        if entry:
            ToolTip(entry, TOOLTIPS["learning_rate"])
