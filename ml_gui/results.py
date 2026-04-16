# -*- coding: utf-8 -*-
"""
Visualización de resultados finales adaptada a cada modelo
"""

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config.theme import DARK_THEME

class ResultsDisplayer:
    def __init__(self, app):
        self.app = app
    
    def display(self, results):
        # Limpiar frame de resultados
        for widget in self.app.results_display.winfo_children():
            widget.destroy()
        
        # Crear scrollable frame
        canvas = tk.Canvas(self.app.results_display, bg=DARK_THEME['bg'], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.app.results_display, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=DARK_THEME['bg'])
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Título
        title = tk.Label(scrollable_frame, text=f"📊 RESULTADOS DEL MODELO - {self.app.get_model_name()}",
                         font=('Arial', 16, 'bold'), bg=DARK_THEME['bg'], fg=DARK_THEME['highlight'])
        title.pack(pady=10)
        
        # Métricas principales
        self._show_metrics(scrollable_frame, results)
        
        # Ecuación (solo regresión)
        if self.app.model_type in ['simple', 'multiple'] and 'coefficients' in results:
            self._show_equation(scrollable_frame, results)
        
        # Coeficientes o importancia
        if hasattr(self.app.automl, 'get_feature_importance'):
            importance = self.app.automl.get_feature_importance()
            if importance:
                self._show_coefficients(scrollable_frame, importance)
        
        # Matriz de confusión (clasificación)
        if self.app.model_type in ['logistic', 'random_forest'] and hasattr(self.app.automl, 'get_confusion_matrix'):
            cm = self.app.automl.get_confusion_matrix()
            self._show_confusion_matrix(scrollable_frame, cm)
        
        # Reporte de clasificación (logística y random forest)
        if self.app.model_type in ['logistic', 'random_forest'] and hasattr(self.app.automl, 'get_classification_report'):
            report = self.app.automl.get_classification_report()
            self._show_classification_report(scrollable_frame, report)
        
        # Predicciones vs reales (solo regresión simple)
        if self.app.model_type == 'simple' and hasattr(self.app.automl, 'get_predictions'):
            preds = self.app.automl.get_predictions()
            if preds is not None:
                self._show_predictions_plot(scrollable_frame, preds)
        
        # Diagnóstico
        self._show_diagnostics(scrollable_frame, results)
    
    def _show_metrics(self, parent, results):
        frame = tk.Frame(parent, bg=DARK_THEME['frame_bg'], relief=tk.RIDGE, bd=2)
        frame.pack(fill=tk.X, padx=20, pady=10)
        
        if self.app.model_type == 'logistic' or self.app.model_type == 'random_forest':
            tk.Label(frame, text=f"🎯 Mejor Accuracy: {results['best_accuracy']:.4f}",
                     font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack(pady=5)
            tk.Label(frame, text=f"📈 Accuracy Final: {results['final_accuracy']:.4f}",
                     font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
            tk.Label(frame, text=f"🏆 Mejor Epoch: {results['best_epoch']}",
                     font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['info']).pack(pady=5)
            if 'n_estimators_final' in results:
                tk.Label(frame, text=f"🌲 Árboles finales: {results['n_estimators_final']}",
                         font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['warning']).pack(pady=5)
        else:
            tk.Label(frame, text=f"🎯 Mejor R²: {results['best_r2']:.4f}",
                     font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).pack(pady=5)
            tk.Label(frame, text=f"📈 R² Final: {results['final_r2']:.4f}",
                     font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
            tk.Label(frame, text=f"📉 MSE Final: {results['final_mse']:.4f}",
                     font=('Arial', 12), bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(pady=5)
    
    def _show_equation(self, parent, results):
        frame = tk.LabelFrame(parent, text="📐 Ecuación del Modelo", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 12, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)
        intercept = results['intercept']
        coefs = results['coefficients']
        eq = f"y = {intercept:.4f}"
        for feat, coef in coefs.items():
            sign = " + " if coef >= 0 else " - "
            eq += f"{sign}{abs(coef):.4f} * {feat}"
        tk.Label(frame, text=eq, wraplength=800, font=('Consolas', 10),
                 bg=DARK_THEME['frame_bg'], fg=DARK_THEME['highlight']).pack(padx=10, pady=10)
    
    def _show_coefficients(self, parent, importance):
        frame = tk.LabelFrame(parent, text="📊 Importancia de Características", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 12, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)
        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, val in sorted_items:
            bar_frame = tk.Frame(frame, bg=DARK_THEME['frame_bg'])
            bar_frame.pack(fill=tk.X, padx=10, pady=2)
            tk.Label(bar_frame, text=f"{feature}:", width=25, anchor=tk.W,
                     bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg']).pack(side=tk.LEFT)
            color = DARK_THEME['success'] if val > 0 else DARK_THEME['error']
            tk.Label(bar_frame, text=f"{val:.4f}", width=10, anchor=tk.W,
                     bg=DARK_THEME['frame_bg'], fg=color).pack(side=tk.LEFT)
            max_val = max(abs(v) for v in importance.values())
            bar_width = int((abs(val) / max_val) * 200)
            bar = tk.Canvas(bar_frame, width=200, height=18, bg=DARK_THEME['entry_bg'], highlightthickness=0)
            bar.pack(side=tk.LEFT, padx=5)
            bar.create_rectangle(0, 0, bar_width, 18, fill=DARK_THEME['highlight'], outline='')
    
    def _show_confusion_matrix(self, parent, cm):
        frame = tk.LabelFrame(parent, text="📊 Matriz de Confusión", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 12, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)
        table = tk.Frame(frame, bg=DARK_THEME['frame_bg'])
        table.pack(pady=10)
        tk.Label(table, text="", width=12, bg=DARK_THEME['frame_bg']).grid(row=0, column=0)
        tk.Label(table, text="Predicho Neg", width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['highlight'], fg='white').grid(row=0, column=1, padx=1)
        tk.Label(table, text="Predicho Pos", width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['highlight'], fg='white').grid(row=0, column=2, padx=1)
        tk.Label(table, text="Real Neg", width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=1, column=0, padx=1)
        tk.Label(table, text=str(cm[0,0]), width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=1, column=1, padx=1)
        tk.Label(table, text=str(cm[0,1]), width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=1, column=2, padx=1)
        tk.Label(table, text="Real Pos", width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=2, column=0, padx=1)
        tk.Label(table, text=str(cm[1,0]), width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=2, column=1, padx=1)
        tk.Label(table, text=str(cm[1,1]), width=12, relief=tk.RIDGE,
                 bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=2, column=2, padx=1)
    
    def _show_classification_report(self, parent, report):
        frame = tk.LabelFrame(parent, text="📋 Reporte de Clasificación", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 12, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)
        headers = ["Clase", "Precisión", "Recall", "F1", "Soporte"]
        for i, h in enumerate(headers):
            tk.Label(frame, text=h, width=12, relief=tk.RIDGE, bg=DARK_THEME['highlight'],
                     fg='white').grid(row=0, column=i, padx=1, pady=1)
        row = 1
        for cls, metrics in report.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                tk.Label(frame, text=str(cls)[:12], width=12, relief=tk.RIDGE,
                         bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=row, column=0, padx=1)
                tk.Label(frame, text=f"{metrics['precision']:.3f}", width=12, relief=tk.RIDGE,
                         bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=row, column=1, padx=1)
                tk.Label(frame, text=f"{metrics['recall']:.3f}", width=12, relief=tk.RIDGE,
                         bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=row, column=2, padx=1)
                tk.Label(frame, text=f"{metrics['f1-score']:.3f}", width=12, relief=tk.RIDGE,
                         bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=row, column=3, padx=1)
                tk.Label(frame, text=f"{metrics['support']:.0f}", width=12, relief=tk.RIDGE,
                         bg=DARK_THEME['entry_bg'], fg=DARK_THEME['fg']).grid(row=row, column=4, padx=1)
                row += 1
        if 'accuracy' in report:
            tk.Label(frame, text=f"Accuracy Global: {report['accuracy']:.4f}", font=('Arial', 10, 'bold'),
                     bg=DARK_THEME['frame_bg'], fg=DARK_THEME['success']).grid(row=row, column=0, columnspan=5, pady=5)
    
    def _show_predictions_plot(self, parent, preds):
        frame = tk.LabelFrame(parent, text="📈 Predicciones vs Valores Reales", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 12, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)
        fig = Figure(figsize=(6,4), facecolor=DARK_THEME['frame_bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_THEME['entry_bg'])
        ax.scatter(self.app.automl.y_test, preds, alpha=0.5, c=DARK_THEME['highlight'], s=20)
        minv = min(self.app.automl.y_test.min(), preds.min())
        maxv = max(self.app.automl.y_test.max(), preds.max())
        ax.plot([minv, maxv], [minv, maxv], 'r--', alpha=0.7, label='Ideal')
        ax.set_xlabel('Reales', color=DARK_THEME['fg'])
        ax.set_ylabel('Predicciones', color=DARK_THEME['fg'])
        ax.tick_params(colors=DARK_THEME['fg'])
        ax.legend(facecolor=DARK_THEME['frame_bg'], labelcolor=DARK_THEME['fg'])
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=10, pady=10)
    
    def _show_diagnostics(self, parent, results):
        frame = tk.LabelFrame(parent, text="💡 Diagnóstico del Modelo", bg=DARK_THEME['frame_bg'],
                              fg=DARK_THEME['fg'], font=('Arial', 12, 'bold'))
        frame.pack(fill=tk.X, padx=20, pady=10)
        if self.app.model_type in ['logistic', 'random_forest']:
            acc = results['final_accuracy']
            if acc >= 0.9: msg = "✅ Excelente modelo! Muy alta precisión."
            elif acc >= 0.8: msg = "👍 Buen modelo. Precisión aceptable."
            elif acc >= 0.7: msg = "⚠️ Modelo aceptable. Se puede mejorar."
            else: msg = "❌ Modelo mejorable. Considere más datos o ajustar hiperparámetros."
            if self.app.model_type == 'random_forest':
                msg += " (Random Forest es robusto, prueba aumentando árboles)"
        else:
            r2 = results['final_r2']
            if r2 >= 0.9: msg = "✅ Excelente modelo! Muy alto poder predictivo."
            elif r2 >= 0.7: msg = "👍 Buen modelo. Explica bien la varianza."
            elif r2 >= 0.5: msg = "⚠️ Modelo moderado. Podría mejorar con más features."
            else: msg = "❌ Modelo débil. Considere transformaciones o más datos."
        tk.Label(frame, text=msg, wraplength=800, bg=DARK_THEME['frame_bg'],
                 fg=DARK_THEME['fg'], font=('Arial', 10)).pack(padx=10, pady=10)
