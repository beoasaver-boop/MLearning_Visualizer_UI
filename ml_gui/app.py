# -*- coding: utf-8 -*-
"""
Aplicación principal MLVisualizerApp - Coordinadora de UI y lógica
"""

import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np

from config.theme import DARK_THEME
from ml_gui.widgets import LeftPanelBuilder, RightPanelBuilder, setup_ttk_styles
from ml_gui.callbacks import DataLoadingCallbacks, VariableSelectionCallbacks, TrainingCallbacks
from ml_gui.plots import TrainingPlotsManager
from ml_gui.rf_plots import RandomForestPlotsManager
from ml_gui.results import ResultsDisplayer

from analytics import AutoMLVisualizer, LinearRegressionVisualizer, RandomForestVisualizer

class MLVisualizerApp:
    def __init__(self, root, model_type="logistic"):
        self.root = root
        self.model_type = model_type
        self.root.title(f"ML Visualizer - {self.get_model_name()}")
        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_THEME['bg'])
        
        # Variables de estado
        self.file_path = None
        self.automl = None
        self.is_training = False
        self.training_thread = None
        
        # Referencias a widgets
        self.file_label = None
        self.features_listbox = None
        self.target_var = None
        self.target_entry = None
        self.confirm_vars_btn = None
        self.train_btn = None
        self.status_text = None
        self.test_size_var = None
        self.epochs_var = None
        self.nulls_var = None
        self.learning_rate_var = None
        self.results_display = None
        
        # Managers de gráficas
        self.plots_manager = None
        self.rf_plots_manager = None
        self.plots_frame = None  # referencia al frame donde van las gráficas
        
        # Configurar estilos y crear UI
        setup_ttk_styles()
        self._create_ui()
        self._setup_callbacks()
        
        # Inicializar resultados displayer
        self.results_displayer = ResultsDisplayer(self)
    
    def get_model_name(self):
        names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple',
            'random_forest': 'Random Forest'
        }
        return names.get(self.model_type, 'Desconocido')
    
    def _create_ui(self):
        # Construir paneles
        left_builder = LeftPanelBuilder(self)
        right_builder = RightPanelBuilder(self)
        left_panel = left_builder.build()
        right_panel = right_builder.build()
        
        # Guardar referencias
        self.file_label = left_builder.file_label
        self.features_listbox = left_builder.features_listbox
        self.target_var = left_builder.target_var
        self.target_entry = left_builder.target_entry
        self.confirm_vars_btn = left_builder.confirm_vars_btn
        self.train_btn = left_builder.train_btn
        self.status_text = left_builder.status_text
        self.test_size_var = left_builder.test_size_var
        self.epochs_var = left_builder.epochs_var
        self.nulls_var = left_builder.nulls_var
        self.learning_rate_var = getattr(left_builder, 'learning_rate_var', None)
        self.results_display = right_builder.results_display
        self.plots_frame = right_builder.plots_frame
        
        # Crear el manager de gráficas adecuado según el modelo
        if self.model_type == 'random_forest':
            # Si ya existe algún widget en plots_frame, lo eliminamos (por si acaso)
            for widget in self.plots_frame.winfo_children():
                widget.destroy()
            # Creamos el manager específico de Random Forest
            self.rf_plots_manager = RandomForestPlotsManager(self.plots_frame)
            self.plots_manager = None
        else:
            # El manager genérico ya fue creado por RightPanelBuilder
            self.plots_manager = right_builder.plots_manager
            self.rf_plots_manager = None
    
    def _setup_callbacks(self):
        self.data_callbacks = DataLoadingCallbacks(self)
        self.var_callbacks = VariableSelectionCallbacks(self)
        self.train_callbacks = TrainingCallbacks(self)
    
    def populate_feature_checkbuttons(self):
        if self.features_listbox:
            self.features_listbox.delete(0, tk.END)
            columns = self.automl.get_columns()
            for col in columns:
                self.features_listbox.insert(tk.END, col)
            self.update_status(f"✅ {len(columns)} columnas cargadas")
    
    def update_status(self, message):
        if self.status_text:
            self.status_text.insert(tk.END, f"{message}\n")
            self.status_text.see(tk.END)
            self.root.update_idletasks()
    
    def update_training_plots(self, epoch, n_epochs, train_losses, test_losses,
                              train_accuracies, test_accuracies, coef_history=None,
                              is_regression=False, is_simple=False, X_test=None,
                              y_test=None, model=None, extra_data=None):
        if self.model_type == 'random_forest' and self.rf_plots_manager:
            n_estimators = extra_data.get('n_estimators', 0) if extra_data else 0
            feature_imp = extra_data.get('feature_importance', None)
            oob = extra_data.get('oob_score', None)
            cm = extra_data.get('confusion_matrix', None)
            current_train_acc = train_accuracies[-1] if train_accuracies else 0
            current_test_acc = test_accuracies[-1] if test_accuracies else 0
            
            self.rf_plots_manager.update_plots(
                epoch, n_epochs, n_estimators,
                current_train_acc, current_test_acc,
                feature_importance=feature_imp,
                oob_score=oob,
                confusion_matrix=cm
            )
        elif self.plots_manager:
            self.plots_manager.update_loss_and_accuracy(epoch, n_epochs, train_losses, test_losses,
                                                        train_accuracies, test_accuracies, is_regression)
            self.plots_manager.update_coefficients(coef_history, self.automl.feature_names if self.automl else [])
            self.plots_manager.update_feature_importance(self.automl if self.automl else None, is_regression)
            self.plots_manager.update_residuals(is_regression, model, X_test, y_test, epoch)
            self.plots_manager.update_overfitting(train_accuracies, test_accuracies)
            self.plots_manager.draw()
    
    def show_final_results(self, results):
        if self.results_displayer:
            self.results_displayer.display(results)
    
    # Métodos delegados a callbacks
    def load_file(self):
        self.data_callbacks.load_file()
    
    def confirm_variables(self):
        self.var_callbacks.confirm_variables()
    
    def start_training(self):
        self.train_callbacks.start_training()
    
    def _training_worker(self, test_size, n_epochs, nulls_handling, learning_rate=None):
        self.train_callbacks._training_worker(test_size, n_epochs, nulls_handling, learning_rate)
    
    def _enable_buttons(self):
        self.train_callbacks._enable_buttons()
    
    def select_all(self):
        if self.features_listbox:
            self.features_listbox.selection_set(0, tk.END)
    
    def deselect_all(self):
        if self.features_listbox:
            self.features_listbox.selection_clear(0, tk.END)
