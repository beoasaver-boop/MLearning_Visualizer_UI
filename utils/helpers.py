"""
Funciones auxiliares y utilidades
"""

def center_window(root, width=1400, height=900):
    """
    Centra una ventana en la pantalla
    
    Args:
        root: ventana de tkinter
        width: ancho de la ventana
        height: alto de la ventana
    """
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")


def limit_simple_selection(features_vars, selected_col):
    """
    Limita a una sola selección en regresión simple
    
    Args:
        features_vars: diccionario de variables booleanas
        selected_col: columna actualmente seleccionada
    """
    for col, var in features_vars.items():
        if col != selected_col and var.get():
            var.set(False)


def validate_parameters(test_size, n_epochs, nulls_handling):
    """
    Valida los parámetros de entrenamiento
    
    Args:
        test_size: tamaño del conjunto de prueba
        n_epochs: número de epochs
        nulls_handling: forma de manejar nulos
        
    Returns:
        tuple: (es_valido, mensaje_error)
    """
    try:
        test_size = float(test_size)
        n_epochs = int(n_epochs)
        
        if not (0 < test_size < 1):
            return False, "El tamaño de prueba debe estar entre 0 y 1"
        
        if n_epochs <= 0:
            return False, "El número de epochs debe ser positivo"
        
        return True, ""
    except ValueError:
        return False, "Parámetros inválidos"


def validate_variables(features, target, available_cols):
    """
    Valida que las variables seleccionadas sean válidas
    
    Args:
        features: lista de características
        target: variable objetivo
        available_cols: columnas disponibles en el dataset
        
    Returns:
        tuple: (es_valido, mensaje_error)
    """
    if not features:
        return False, "Por favor seleccione al menos una variable independiente"
    
    if not target:
        return False, "Por favor ingrese la variable dependiente"
    
    if target not in available_cols:
        return False, f"Variable objetivo '{target}' no encontrada"
    
    return True, ""
