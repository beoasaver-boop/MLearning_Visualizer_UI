"""
Manejo de gráficas y visualizaciones en tiempo real
"""
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from config.theme import DARK_THEME


class TrainingPlotsManager:
    """Gestiona la creación y actualización de gráficas de entrenamiento"""
    
    def __init__(self, plots_frame, dark_theme=None):
        self.plots_frame = plots_frame
        self.theme = dark_theme or DARK_THEME
        self.fig = None
        self.canvas = None
        self.axes = {}
        self.lines = {}
        self.create_plots()
    
    def create_plots(self):
        """Crear las 6 gráficas de entrenamiento"""
        # Crear figura con 2x3 subplots
        self.fig = Figure(figsize=(10, 8), facecolor=self.theme['bg'])
        
        # Crear subplots
        ax_keys = ['loss', 'accuracy', 'coef', 'feature_importance', 'residuals', 'overfitting']
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
        
        for key, (row, col) in zip(ax_keys, positions):
            ax = self.fig.add_subplot(2, 3, (row - 1) * 3 + col)
            self._configure_axis(ax)
            self.axes[key] = ax
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plots_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar líneas vacías para loss y accuracy
        self.lines['train_loss'], = self.axes['loss'].plot([], [], 'b-', label='Train Loss', linewidth=2)
        self.lines['test_loss'], = self.axes['loss'].plot([], [], 'r-', label='Test Loss', linewidth=2)
        self.lines['train_acc'], = self.axes['accuracy'].plot([], [], 'b-', label='Train Acc', linewidth=2)
        self.lines['test_acc'], = self.axes['accuracy'].plot([], [], 'r-', label='Test Acc', linewidth=2)
        
        # Configurar límites iniciales
        self.axes['loss'].set_xlim(0, 100)
        self.axes['loss'].set_ylim(0, 1.1)
        self.axes['accuracy'].set_xlim(0, 100)
        self.axes['accuracy'].set_ylim(0, 1.05)
        self.axes['coef'].set_xlim(0, 100)
        self.axes['coef'].set_ylim(-1, 1)
        self.axes['feature_importance'].set_xlim(0, 1)
        self.axes['residuals'].set_xlim(-0.5, 1.5)
        self.axes['residuals'].set_ylim(-0.5, 1.5)
        self.axes['overfitting'].set_xlim(0, 100)
        self.axes['overfitting'].set_ylim(-0.5, 0.5)
        self.axes['overfitting'].axhline(y=0, color='white', linestyle='--', alpha=0.5)
        
        # Leyendas
        self.axes['loss'].legend(facecolor=self.theme['frame_bg'], labelcolor=self.theme['fg'])
        self.axes['accuracy'].legend(facecolor=self.theme['frame_bg'], labelcolor=self.theme['fg'])
        self.axes['overfitting'].legend(facecolor=self.theme['frame_bg'], labelcolor=self.theme['fg'])
        
        # Títulos y etiquetas
        titles = {
            'loss': 'Pérdida (Loss)',
            'accuracy': 'Precisión (Accuracy)',
            'coef': 'Evolución de Coeficientes',
            'feature_importance': 'Importancia de Características',
            'residuals': 'Distribución de Residuos',
            'overfitting': 'Sobreajuste'
        }
        
        for key, title in titles.items():
            self.axes[key].set_title(title)
        
        labels = {
            'loss': ('Epoch', 'Loss'),
            'accuracy': ('Epoch', 'Accuracy'),
            'coef': ('Epoch', 'Coeficiente'),
            'feature_importance': ('Importancia', ''),
            'residuals': ('', 'Frecuencia'),
            'overfitting': ('Epoch', 'Gap (Train - Test)')
        }
        
        for key, (xlabel, ylabel) in labels.items():
            if xlabel:
                self.axes[key].set_xlabel(xlabel)
            if ylabel:
                self.axes[key].set_ylabel(ylabel)
        
        self.fig.tight_layout()
    
    def _configure_axis(self, ax):
        """Configurar colores de ejes"""
        ax.set_facecolor(self.theme['frame_bg'])
        ax.tick_params(colors=self.theme['fg'])
        ax.xaxis.label.set_color(self.theme['fg'])
        ax.yaxis.label.set_color(self.theme['fg'])
        ax.title.set_color(self.theme['fg'])
        ax.grid(True, alpha=0.3, color=self.theme.get('grid_color', '#404040'))
    
    def update_loss_and_accuracy(self, train_losses, test_losses, 
                                  train_accuracies, test_accuracies, 
                                  is_regression=False):
        """Actualizar gráficas de pérdida y precisión"""
        # Pérdida
        self.lines['train_loss'].set_data(range(len(train_losses)), train_losses)
        self.lines['test_loss'].set_data(range(len(test_losses)), test_losses)
        
        if len(train_losses) > 1:
            max_loss = max(max(train_losses[-10:]), max(test_losses[-10:])) if train_losses else 1
            self.axes['loss'].set_xlim(0, max(10, len(train_losses)))
            self.axes['loss'].set_ylim(0, min(max_loss * 1.1, 10))
        
        loss_label = "MSE" if is_regression else "Log Loss"
        self.axes['loss'].set_ylabel(loss_label)
        self.axes['loss'].set_title(f'Pérdida ({loss_label})')
        
        # Métrica principal
        self.lines['train_acc'].set_data(range(len(train_accuracies)), train_accuracies)
        self.lines['test_acc'].set_data(range(len(test_accuracies)), test_accuracies)
        
        if len(train_accuracies) > 1:
            self.axes['accuracy'].set_xlim(0, max(10, len(train_accuracies)))
            if is_regression:
                self.axes['accuracy'].set_ylim(-1, 1)
            else:
                self.axes['accuracy'].set_ylim(0, 1.05)
        
        metric_label = "R² Score" if is_regression else "Accuracy"
        current_metric = test_accuracies[-1] if test_accuracies else 0
        self.axes['accuracy'].set_title(
            f'{metric_label} (Epoch {len(train_accuracies)} - {metric_label}: {current_metric:.3f})'
        )
        self.axes['accuracy'].set_ylabel(metric_label)
    
    def update_coefficients(self, coef_history, feature_names=None):
        """Actualizar gráfica de evolución de coeficientes"""
        self.axes['coef'].clear()
        self._configure_axis(self.axes['coef'])
        
        if coef_history and len(coef_history) > 0:
            coef_array = np.array(coef_history)
            n_features = min(5, coef_array.shape[1])
            for i in range(n_features):
                label = (feature_names[i] if feature_names and i < len(feature_names) 
                        else f'Coef {i}')
                self.axes['coef'].plot(range(len(coef_array[:, i])), 
                                      coef_array[:, i], linewidth=1.5, label=label[:15])
            self.axes['coef'].set_xlim(0, max(10, len(coef_array)))
            self.axes['coef'].legend(facecolor=self.theme['frame_bg'], 
                                     labelcolor=self.theme['fg'], fontsize=8)
        else:
            self.axes['coef'].text(0.5, 0.5, 'Actualizando en\ntiempo real...', 
                                   transform=self.axes['coef'].transAxes, 
                                   ha='center', va='center',
                                   color=self.theme['fg'], alpha=0.7)
        
        self.axes['coef'].set_xlabel('Epoch')
        self.axes['coef'].set_ylabel('Coeficiente')
        self.axes['coef'].set_title('Evolución de Coeficientes')
    
    def update_feature_importance(self, importance=None, X_test=None, y_test=None, 
                                   model=None, is_simple=False):
        """Actualizar gráfica de importancia de características"""
        self.axes['feature_importance'].clear()
        self._configure_axis(self.axes['feature_importance'])
        
        if is_simple and X_test is not None and y_test is not None and model is not None:
            # Scatter plot para regresión simple
            y_pred = model.predict(X_test)
            self.axes['feature_importance'].scatter(y_test, y_pred, alpha=0.5, 
                                                    c=self.theme['highlight'], s=10)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            self.axes['feature_importance'].plot([min_val, max_val], [min_val, max_val], 
                                                 'r--', alpha=0.7, label='Ideal')
            self.axes['feature_importance'].set_xlabel('Valores Reales')
            self.axes['feature_importance'].set_ylabel('Predicciones')
            self.axes['feature_importance'].set_title('Predicciones vs Reales')
            self.axes['feature_importance'].legend(facecolor=self.theme['frame_bg'], 
                                                   labelcolor=self.theme['fg'])
        elif importance:
            features = list(importance.keys())
            values = list(importance.values())
            indices = np.argsort(values)
            
            bars = self.axes['feature_importance'].barh(range(len(features)), 
                                                        [values[i] for i in indices])
            self.axes['feature_importance'].set_yticks(range(len(features)))
            self.axes['feature_importance'].set_yticklabels([features[i] for i in indices], fontsize=8)
            self.axes['feature_importance'].set_xlabel('Importancia')
            
            for bar in bars:
                bar.set_color(self.theme['highlight'])
        else:
            self.axes['feature_importance'].text(0.5, 0.5, 'Se mostrará al\nfinal',
                                                 transform=self.axes['feature_importance'].transAxes,
                                                 ha='center', va='center',
                                                 color=self.theme['fg'], alpha=0.7)
    
    def update_residuals(self, X_test=None, y_test=None, model=None, epoch=0, is_regression=False):
        """Actualizar gráfica de residuos o matriz de confusión"""
        self.axes['residuals'].clear()
        self._configure_axis(self.axes['residuals'])
        
        if is_regression and X_test is not None and model is not None and epoch % 20 == 0:
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            self.axes['residuals'].hist(residuals, bins=30, color=self.theme['highlight'], 
                                       alpha=0.7, edgecolor='white')
            self.axes['residuals'].set_xlabel('Residuos')
            self.axes['residuals'].set_ylabel('Frecuencia')
            self.axes['residuals'].set_title('Distribución de Residuos')
            self.axes['residuals'].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        else:
            self.axes['residuals'].text(0.5, 0.5, 'Se actualizará\nperiódicamente',
                                       transform=self.axes['residuals'].transAxes,
                                       ha='center', va='center',
                                       color=self.theme['fg'], alpha=0.7)
    
    def update_overfitting(self, train_accuracies, test_accuracies):
        """Actualizar gráfica de sobreajuste"""
        self.axes['overfitting'].clear()
        if len(train_accuracies) > 1:
            acc_gap = np.array(train_accuracies) - np.array(test_accuracies)
            self.axes['overfitting'].plot(range(len(acc_gap)), acc_gap, 'g-', 
                                         linewidth=2, label='Gap')
            self.axes['overfitting'].set_xlim(0, max(10, len(acc_gap)))
            max_gap = max(abs(max(acc_gap)), abs(min(acc_gap))) if len(acc_gap) > 0 else 0.5
            self.axes['overfitting'].set_ylim(-max_gap * 1.1, max_gap * 1.1)
        else:
            self.axes['overfitting'].set_xlim(0, 100)
            self.axes['overfitting'].set_ylim(-0.5, 0.5)
        
        self._configure_axis(self.axes['overfitting'])
        self.axes['overfitting'].set_xlabel('Epoch')
        self.axes['overfitting'].set_ylabel('Gap (Train - Test)')
        self.axes['overfitting'].set_title('Sobreajuste (gap positivo = sobreajuste)')
        self.axes['overfitting'].axhline(y=0, color='white', linestyle='--', alpha=0.5)
        self.axes['overfitting'].grid(True, alpha=0.3, color=self.theme.get('grid_color', '#404040'))
        self.axes['overfitting'].legend(facecolor=self.theme['frame_bg'], 
                                       labelcolor=self.theme['fg'])
    
    def draw(self):
        """Redibujar canvas"""
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.canvas.flush_events()
