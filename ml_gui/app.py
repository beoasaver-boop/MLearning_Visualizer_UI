import customtkinter as ctk
from tkinter import messagebox
import threading
import numpy as np

from ml_gui.widgets import LeftPanelBuilder, RightPanelBuilder
from ml_gui.callbacks import DataLoadingCallbacks, VariableSelectionCallbacks, TrainingCallbacks
from ml_gui.plots import TrainingPlotsManager
from ml_gui.rf_plots import RandomForestPlotsManager
from ml_gui.results import ResultsDisplayer
from ml_gui.tooltips import add_tooltips_to_widgets

from analytics import AutoMLVisualizer, LinearRegressionVisualizer, RandomForestVisualizer

class MLVisualizerApp:
    def __init__(self, root, model_type="logistic"):
        self.root = root
        self.model_type = model_type
        self.root.title(f"MLearning Visualizer - {self.get_model_name()}")

        self.file_path = None
        self.automl = None
        self.is_training = False
        self.training_thread = None

        self.file_label = None
        self.features_frame = None
        self.feature_vars = {}
        self.target_var = None
        self.target_menu = None
        self.confirm_vars_btn = None
        self.train_btn = None
        self.status_text = None
        self.test_size_var = None
        self.epochs_var = None
        self.nulls_var = None
        self.learning_rate_var = None
        self.results_display = None
        self.progress_bar = None

        self.plots_manager = None
        self.rf_plots_manager = None
        self.plots_frame = None

        self._create_ui()
        self._setup_callbacks()
        self.results_displayer = ResultsDisplayer(self)

    def get_model_name(self):
        names = {
            'logistic': 'Regresion Logistica',
            'simple': 'Regresion Lineal Simple',
            'multiple': 'Regresion Lineal Multiple',
            'random_forest': 'Random Forest'
        }
        return names.get(self.model_type, 'Desconocido')

    def _create_ui(self):
        left_builder = LeftPanelBuilder(self)
        right_builder = RightPanelBuilder(self)
        left_panel = left_builder.build()
        right_panel = right_builder.build()

        self.file_label = left_builder.file_label
        self.features_frame = left_builder.features_frame
        self.target_var = left_builder.target_var
        self.target_menu = left_builder.target_menu
        self.confirm_vars_btn = left_builder.confirm_vars_btn
        self.train_btn = left_builder.train_btn
        self.status_text = left_builder.status_text
        self.test_size_var = left_builder.test_size_var
        self.test_size_entry = left_builder.test_size_entry
        self.epochs_var = left_builder.epochs_var
        self.epochs_entry = left_builder.epochs_entry
        self.nulls_var = left_builder.nulls_var
        self.nulls_menu = left_builder.nulls_menu
        self.learning_rate_var = getattr(left_builder, 'learning_rate_var', None)
        self.learning_rate_entry = getattr(left_builder, 'learning_rate_entry', None)
        self.progress_bar = getattr(left_builder, 'progress_bar', None)
        self.results_display = right_builder.results_display

        add_tooltips_to_widgets(self)
        self.plots_frame = right_builder.plots_frame

        if self.model_type == 'random_forest':
            for widget in self.plots_frame.winfo_children():
                widget.destroy()
            self.rf_plots_manager = RandomForestPlotsManager(self.plots_frame)
            self.plots_manager = None
        else:
            self.plots_manager = right_builder.plots_manager
            self.rf_plots_manager = None

    def _setup_callbacks(self):
        self.data_callbacks = DataLoadingCallbacks(self)
        self.var_callbacks = VariableSelectionCallbacks(self)
        self.train_callbacks = TrainingCallbacks(self)

    def populate_feature_checkbuttons(self):
        if self.features_frame:
            for widget in self.features_frame.winfo_children():
                widget.destroy()
            self.feature_vars = {}
            columns = self.automl.get_columns()
            for col in columns:
                var = ctk.StringVar(value="off")
                cb = ctk.CTkCheckBox(
                    self.features_frame, text=col, variable=var,
                    onvalue="on", offvalue="off",
                    font=ctk.CTkFont(size=11)
                )
                cb.pack(anchor="w", padx=8, pady=1)
                self.feature_vars[col] = var
            self.target_menu.configure(values=columns)
            self.target_var.set("")
            self.update_status(f"{len(columns)} columnas cargadas")

    def update_status(self, message):
        if self.status_text:
            self.status_text.insert("end", f"{message}\n")
            self.status_text.see("end")
            self.root.update_idletasks()

    def update_training_plots(self, epoch, n_epochs, train_losses, test_losses,
                              train_accuracies, test_accuracies, coef_history=None,
                              is_regression=False, is_simple=False, X_test=None,
                              y_test=None, model=None, extra_data=None):
        if self.progress_bar:
            self.progress_bar.set((epoch + 1) / n_epochs)
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
            self.plots_manager.update_loss_and_accuracy(train_losses, test_losses,
                                                        train_accuracies, test_accuracies, is_regression)
            self.plots_manager.update_coefficients(coef_history, self.automl.feature_names if self.automl else [])
            importance = self.automl.get_feature_importance() if self.automl else None
            self.plots_manager.update_feature_importance(importance, is_simple=is_simple)
            self.plots_manager.update_residuals(X_test, y_test, model, epoch, is_regression)
            self.plots_manager.update_overfitting(train_accuracies, test_accuracies)
            self.plots_manager.draw()

    def show_final_results(self, results):
        if self.results_displayer:
            self.results_displayer.display(results)

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
        for var in self.feature_vars.values():
            var.set("on")
        if self.model_type == 'simple' and self.feature_vars:
            first = list(self.feature_vars.keys())[0]
            for col, var in self.feature_vars.items():
                var.set("on" if col == first else "off")

    def deselect_all(self):
        for var in self.feature_vars.values():
            var.set("off")
