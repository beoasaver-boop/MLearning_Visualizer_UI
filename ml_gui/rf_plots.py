# -*- coding: utf-8 -*-
"""
Módulo de gráficas específicas para Random Forest
Muestra: evolución de árboles, importancia de características,
matriz de confusión, y curvas de aprendizaje.
"""

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config.theme import DARK_THEME, MATPLOTLIB_DARK_STYLE
import matplotlib.pyplot as plt

# Aplicar estilo oscuro de matplotlib
plt.rcParams.update(MATPLOTLIB_DARK_STYLE)

class RandomForestPlotsManager:
    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.fig = Figure(figsize=(10, 8), facecolor=DARK_THEME['bg'])
        self._create_subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Datos para actualización
        self.n_estimators_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.feature_importance_history = []  # lista de dicts
        self.confusion_matrices = []
        
    def _create_subplots(self):
        """Crea 6 subplots específicos para Random Forest"""
        # 2 filas, 3 columnas
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # Evolución de árboles vs accuracy
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Importancia de características (barras)
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # Matriz de confusión (heatmap)
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # Curva de aprendizaje (train/test)
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Error OOB (si disponible)
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # Comparativa de precisión final
        
        # Configuración estética
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
            ax.set_facecolor(DARK_THEME['frame_bg'])
            ax.tick_params(colors=DARK_THEME['fg'])
            ax.xaxis.label.set_color(DARK_THEME['fg'])
            ax.yaxis.label.set_color(DARK_THEME['fg'])
            ax.title.set_color(DARK_THEME['fg'])
            ax.grid(True, alpha=0.3, color='#404040')
        
        # Títulos iniciales
        self.ax1.set_title('Árboles vs Accuracy')
        self.ax1.set_xlabel('Número de árboles')
        self.ax1.set_ylabel('Accuracy')
        
        self.ax2.set_title('Importancia de Características')
        self.ax2.set_xlabel('Importancia')
        
        self.ax3.set_title('Matriz de Confusión (última)')
        
        self.ax4.set_title('Curva de Aprendizaje')
        self.ax4.set_xlabel('Época')
        self.ax4.set_ylabel('Accuracy')
        
        self.ax5.set_title('Error Out-of-Bag (si disponible)')
        self.ax5.set_xlabel('Árboles')
        self.ax5.set_ylabel('Error OOB')
        
        self.ax6.set_title('Precisión Final')
        self.ax6.set_xlabel('Modelo')
        self.ax6.set_ylabel('Accuracy')
        
        self.fig.tight_layout()
    
    def update_plots(self, epoch, n_epochs, n_estimators, train_acc, test_acc,
                     feature_importance=None, oob_score=None, confusion_matrix=None):
        """
        Actualiza las gráficas en tiempo real.
        - epoch: época actual (0-indexed)
        - n_estimators: número de árboles hasta ahora
        - train_acc, test_acc: accuracies actuales
        - feature_importance: dict {feature: importance}
        - oob_score: puntuación OOB (si existe)
        - confusion_matrix: matriz 2x2 o multiclase
        """
        # Guardar historial
        self.n_estimators_history.append(n_estimators)
        self.train_acc_history.append(train_acc)
        self.test_acc_history.append(test_acc)
        
        # Gráfica 1: Árboles vs Accuracy (dispersión)
        self.ax1.clear()
        self.ax1.set_facecolor(DARK_THEME['frame_bg'])
        self.ax1.scatter(self.n_estimators_history, self.train_acc_history,
                         c='blue', alpha=0.6, label='Train', s=20)
        self.ax1.scatter(self.n_estimators_history, self.test_acc_history,
                         c='red', alpha=0.6, label='Test', s=20)
        self.ax1.plot(self.n_estimators_history, self.train_acc_history, 'b--', alpha=0.3)
        self.ax1.plot(self.n_estimators_history, self.test_acc_history, 'r--', alpha=0.3)
        self.ax1.set_xlabel('Número de árboles')
        self.ax1.set_ylabel('Accuracy')
        self.ax1.set_title(f'Árboles vs Accuracy (época {epoch+1}/{n_epochs})')
        self.ax1.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        self.ax1.grid(True, alpha=0.3)
        
        # Gráfica 2: Importancia de características (barras horizontales)
        self.ax2.clear()
        self.ax2.set_facecolor(DARK_THEME['frame_bg'])
        if feature_importance:
            features = list(feature_importance.keys())
            values = list(feature_importance.values())
            # Ordenar descendente
            indices = np.argsort(values)[::-1]
            features = [features[i] for i in indices]
            values = [values[i] for i in indices]
            # Mostrar solo top 10 si hay muchas
            if len(features) > 10:
                features = features[:10]
                values = values[:10]
            y_pos = np.arange(len(features))
            self.ax2.barh(y_pos, values, color=DARK_THEME['highlight'])
            self.ax2.set_yticks(y_pos)
            self.ax2.set_yticklabels(features, fontsize=8)
            self.ax2.set_xlabel('Importancia')
            self.ax2.set_title('Importancia de Características (top 10)')
            self.ax2.invert_yaxis()
        else:
            self.ax2.text(0.5, 0.5, 'Esperando importancia...',
                          transform=self.ax2.transAxes, ha='center', va='center',
                          color=DARK_THEME['fg'])
        
        # Gráfica 3: Matriz de confusión (heatmap)
        self.ax3.clear()
        self.ax3.set_facecolor(DARK_THEME['frame_bg'])
        if confusion_matrix is not None:
            cm = confusion_matrix
            im = self.ax3.imshow(cm, interpolation='nearest', cmap='Blues')
            self.ax3.set_title(f'Matriz de Confusión (época {epoch+1})')
            self.ax3.set_xlabel('Predicción')
            self.ax3.set_ylabel('Real')
            # Etiquetas automáticas
            n_classes = cm.shape[0]
            self.ax3.set_xticks(range(n_classes))
            self.ax3.set_yticks(range(n_classes))
            self.ax3.set_xticklabels([f'Clase {i}' for i in range(n_classes)])
            self.ax3.set_yticklabels([f'Clase {i}' for i in range(n_classes)])
            # Añadir valores numéricos
            for i in range(n_classes):
                for j in range(n_classes):
                    self.ax3.text(j, i, cm[i, j], ha='center', va='center',
                                  color='white' if cm[i, j] > cm.max()/2 else 'black')
            # Colorbar
            if not hasattr(self, '_cbar') or self._cbar is None:
                self._cbar = self.fig.colorbar(im, ax=self.ax3)
            else:
                self._cbar.update_normal(im)
        else:
            self.ax3.text(0.5, 0.5, 'Matriz de confusión\nse actualizará pronto',
                          transform=self.ax3.transAxes, ha='center', va='center',
                          color=DARK_THEME['fg'])
        
        # Gráfica 4: Curva de aprendizaje (train/test vs época)
        self.ax4.clear()
        self.ax4.set_facecolor(DARK_THEME['frame_bg'])
        epochs_range = range(1, len(self.train_acc_history)+1)
        self.ax4.plot(epochs_range, self.train_acc_history, 'b-', label='Train')
        self.ax4.plot(epochs_range, self.test_acc_history, 'r-', label='Test')
        self.ax4.set_xlabel('Época')
        self.ax4.set_ylabel('Accuracy')
        self.ax4.set_title('Curva de Aprendizaje')
        self.ax4.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        self.ax4.grid(True, alpha=0.3)
        
        # Gráfica 5: Error OOB (si disponible)
        self.ax5.clear()
        self.ax5.set_facecolor(DARK_THEME['frame_bg'])
        if oob_score is not None:
            # OOB score es accuracy, el error sería 1 - oob_score
            oob_error = 1 - oob_score
            # Simular histórico si no lo tenemos
            if not hasattr(self, 'oob_history'):
                self.oob_history = []
            self.oob_history.append(oob_error)
            self.ax5.plot(range(1, len(self.oob_history)+1), self.oob_history, 'g-', linewidth=2)
            self.ax5.set_xlabel('Época')
            self.ax5.set_ylabel('Error OOB')
            self.ax5.set_title(f'Error Out-of-Bag (actual: {oob_error:.4f})')
        else:
            self.ax5.text(0.5, 0.5, 'Error OOB no disponible\n(requiere oob_score=True)',
                          transform=self.ax5.transAxes, ha='center', va='center',
                          color=DARK_THEME['fg'])
        self.ax5.grid(True, alpha=0.3)
        
        # Gráfica 6: Comparativa final (se actualiza cada 20 épocas o al final)
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            self.ax6.clear()
            self.ax6.set_facecolor(DARK_THEME['frame_bg'])
            labels = ['Train', 'Test']
            values = [train_acc, test_acc]
            colors = [DARK_THEME['success'], DARK_THEME['info']]
            bars = self.ax6.bar(labels, values, color=colors, alpha=0.7)
            self.ax6.set_ylim(0, 1)
            self.ax6.set_ylabel('Accuracy')
            self.ax6.set_title(f'Precisión actual (época {epoch+1})')
            # Añadir valor sobre barras
            for bar, val in zip(bars, values):
                self.ax6.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                              f'{val:.3f}', ha='center', va='bottom',
                              color=DARK_THEME['fg'])
        
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self.canvas.flush_events()
