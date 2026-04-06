"""
Visualización de resultados finales del modelo
"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from config.theme import DARK_THEME


class ResultsDisplayer:
    """Gestor de visualización de resultados finales"""
    
    def __init__(self, results_frame, model_type, automl, dark_theme=None):
        self.results_frame = results_frame
        self.model_type = model_type
        self.automl = automl
        self.theme = dark_theme or DARK_THEME
    
    def display(self, results):
        """Mostrar resultados finales adaptados al tipo de modelo"""
        
        # Limpiar frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Canvas scrollable
        canvas = tk.Canvas(self.results_frame, bg=self.theme['bg'], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.results_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.theme['bg'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Mostrar secciones
        self._show_title(scrollable_frame)
        self._show_metrics(scrollable_frame, results)
        
        if self.model_type != 'logistic':
            self._show_equation(scrollable_frame, results)
        
        self._show_coefficients(scrollable_frame)
        
        if self.model_type == 'logistic':
            self._show_confusion_matrix(scrollable_frame)
            self._show_classification_report(scrollable_frame)
        else:
            self._show_predictions(scrollable_frame)
        
        self._show_diagnostics(scrollable_frame, results)
    
    def _show_title(self, parent):
        """Mostrar título de resultados"""
        model_names = {
            'logistic': 'Regresión Logística',
            'simple': 'Regresión Lineal Simple',
            'multiple': 'Regresión Lineal Múltiple'
        }
        
        title = tk.Label(parent, text=f"📊 RESULTADOS DEL MODELO - {model_names.get(self.model_type, 'Modelo')}",
                        font=('Arial', 16, 'bold'),
                        bg=self.theme['bg'], fg=self.theme['highlight'])
        title.pack(pady=10)
    
    def _show_metrics(self, parent, results):
        """Mostrar métricas principales"""
        metrics_frame = tk.Frame(parent, bg=self.theme['frame_bg'], relief=tk.RIDGE, bd=2)
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if self.model_type == 'logistic':
            tk.Label(metrics_frame, text=f"🎯 Mejor Accuracy: {results['best_accuracy']:.4f}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['success']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📈 Accuracy Final: {results['final_accuracy']:.4f}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"🏆 Mejor Epoch: {results['best_epoch']}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['info']).pack(pady=5)
        else:
            tk.Label(metrics_frame, text=f"🎯 Mejor R² Score: {results['best_r2']:.4f}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['success']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📈 R² Score Final: {results['final_r2']:.4f}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📉 MSE Final: {results['final_mse']:.4f}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"📊 MAE Final: {results['final_mae']:.4f}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['fg']).pack(pady=5)
            tk.Label(metrics_frame, text=f"🏆 Mejor Epoch: {results['best_epoch']}",
                    font=('Arial', 12), bg=self.theme['frame_bg'], fg=self.theme['info']).pack(pady=5)
    
    def _show_equation(self, parent, results):
        """Mostrar ecuación del modelo (solo regresión)"""
        if 'coefficients' not in results:
            return
        
        eq_frame = tk.LabelFrame(parent, text="📐 Ecuación del Modelo",
                                bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                font=('Arial', 12, 'bold'))
        eq_frame.pack(fill=tk.X, padx=20, pady=10)
        
        intercept = results['intercept']
        coefs = results['coefficients']
        
        equation = f"y = {intercept[0]:.4f}"
        for feature, coef in coefs.items():
            sign = " + " if coef >= 0 else " - "
            equation += f"{sign}{abs(coef):.4f} * {feature}"
        
        tk.Label(eq_frame, text=equation, wraplength=800,
                font=('Consolas', 10), bg=self.theme['frame_bg'], 
                fg=self.theme['highlight'], justify=tk.LEFT).pack(padx=10, pady=10)
    
    def _show_coefficients(self, parent):
        """Mostrar coeficientes del modelo"""
        if not hasattr(self.automl, 'get_feature_importance'):
            return
        
        importance = self.automl.get_feature_importance()
        if not importance:
            return
        
        imp_frame = tk.LabelFrame(parent, text="📊 Coeficientes del Modelo",
                                bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                font=('Arial', 12, 'bold'))
        imp_frame.pack(fill=tk.X, padx=20, pady=10)
        
        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in sorted_items:
            bar_frame = tk.Frame(imp_frame, bg=self.theme['frame_bg'])
            bar_frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(bar_frame, text=f"{feature}:", width=25, anchor=tk.W,
                    bg=self.theme['frame_bg'], fg=self.theme['fg'],
                    font=('Arial', 9)).pack(side=tk.LEFT)
            
            coef_color = self.theme['success'] if coef > 0 else self.theme['error']
            tk.Label(bar_frame, text=f"{coef:+.4f}", width=12, anchor=tk.W,
                    bg=self.theme['frame_bg'], fg=coef_color,
                    font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
            
            max_abs = max(abs(v) for v in importance.values()) if importance else 1
            bar_width = int((abs(coef) / max_abs) * 200)
            
            bar = tk.Canvas(bar_frame, width=200, height=18,
                           bg=self.theme['entry_bg'], highlightthickness=0)
            bar.pack(side=tk.LEFT, padx=5)
            bar_color = self.theme['highlight'] if coef > 0 else self.theme['warning']
            bar.create_rectangle(0, 0, bar_width, 18, fill=bar_color, outline='')
            
            tk.Label(bar_frame, text=f"|{abs(coef):.4f}|", width=10, anchor=tk.W,
                    bg=self.theme['frame_bg'], fg=self.theme['fg'],
                    font=('Arial', 9)).pack(side=tk.LEFT)
    
    def _show_confusion_matrix(self, parent):
        """Mostrar matriz de confusión (solo clasificación)"""
        if not hasattr(self.automl, 'get_confusion_matrix'):
            return
        
        cm = self.automl.get_confusion_matrix()
        
        cm_frame = tk.LabelFrame(parent, text="📊 Matriz de Confusión",
                                bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                font=('Arial', 12, 'bold'))
        cm_frame.pack(fill=tk.X, padx=20, pady=10)
        
        table_frame = tk.Frame(cm_frame, bg=self.theme['frame_bg'])
        table_frame.pack(pady=10)
        
        # Headers
        tk.Label(table_frame, text="", width=12, bg=self.theme['frame_bg']).grid(row=0, column=0)
        tk.Label(table_frame, text="Predicho Negativo", width=15, relief=tk.RIDGE,
                bg=self.theme['highlight'], fg='white').grid(row=0, column=1, padx=1, pady=1)
        tk.Label(table_frame, text="Predicho Positivo", width=15, relief=tk.RIDGE,
                bg=self.theme['highlight'], fg='white').grid(row=0, column=2, padx=1, pady=1)
        
        # Rows
        tk.Label(table_frame, text="Real Negativo", width=12, relief=tk.RIDGE,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).grid(row=1, column=0, padx=1, pady=1)
        tk.Label(table_frame, text=str(cm[0, 0]), width=15, relief=tk.RIDGE,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).grid(row=1, column=1, padx=1, pady=1)
        tk.Label(table_frame, text=str(cm[0, 1]), width=15, relief=tk.RIDGE,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).grid(row=1, column=2, padx=1, pady=1)
        
        tk.Label(table_frame, text="Real Positivo", width=12, relief=tk.RIDGE,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).grid(row=2, column=0, padx=1, pady=1)
        tk.Label(table_frame, text=str(cm[1, 0]), width=15, relief=tk.RIDGE,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).grid(row=2, column=1, padx=1, pady=1)
        tk.Label(table_frame, text=str(cm[1, 1]), width=15, relief=tk.RIDGE,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).grid(row=2, column=2, padx=1, pady=1)
        
        # Métricas
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_table = tk.Frame(cm_frame, bg=self.theme['frame_bg'])
        metrics_table.pack(pady=10)
        
        tk.Label(metrics_table, text=f"Accuracy: {accuracy:.4f}", width=15,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).pack(side=tk.LEFT, padx=5)
        tk.Label(metrics_table, text=f"Precision: {precision:.4f}", width=15,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).pack(side=tk.LEFT, padx=5)
        tk.Label(metrics_table, text=f"Recall: {recall:.4f}", width=15,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).pack(side=tk.LEFT, padx=5)
        tk.Label(metrics_table, text=f"F1-Score: {f1:.4f}", width=15,
                bg=self.theme['entry_bg'], fg=self.theme['fg']).pack(side=tk.LEFT, padx=5)
    
    def _show_classification_report(self, parent):
        """Mostrar reporte de clasificación"""
        if not hasattr(self.automl, 'get_classification_report'):
            return
        
        report = self.automl.get_classification_report()
        
        report_frame = tk.LabelFrame(parent, text="📋 Reporte de Clasificación",
                                    bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                    font=('Arial', 12, 'bold'))
        report_frame.pack(fill=tk.X, padx=20, pady=10)
        
        headers_frame = tk.Frame(report_frame, bg=self.theme['frame_bg'])
        headers_frame.pack(pady=5)
        
        headers = ["Clase", "Precisión", "Recall", "F1-Score", "Soporte"]
        for i, header in enumerate(headers):
            tk.Label(headers_frame, text=header, width=12, relief=tk.RIDGE,
                    bg=self.theme['highlight'], fg='white',
                    font=('Arial', 9, 'bold')).grid(row=0, column=i, padx=1, pady=1)
        
        row = 1
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                tk.Label(headers_frame, text=str(class_name)[:12], width=12, relief=tk.RIDGE,
                        bg=self.theme['entry_bg'], fg=self.theme['fg'],
                        font=('Arial', 9)).grid(row=row, column=0, padx=1, pady=1)
                tk.Label(headers_frame, text=f"{metrics['precision']:.3f}", width=12, relief=tk.RIDGE,
                        bg=self.theme['entry_bg'], fg=self.theme['fg'],
                        font=('Arial', 9)).grid(row=row, column=1, padx=1, pady=1)
                tk.Label(headers_frame, text=f"{metrics['recall']:.3f}", width=12, relief=tk.RIDGE,
                        bg=self.theme['entry_bg'], fg=self.theme['fg'],
                        font=('Arial', 9)).grid(row=row, column=2, padx=1, pady=1)
                tk.Label(headers_frame, text=f"{metrics['f1-score']:.3f}", width=12, relief=tk.RIDGE,
                        bg=self.theme['entry_bg'], fg=self.theme['fg'],
                        font=('Arial', 9)).grid(row=row, column=3, padx=1, pady=1)
                tk.Label(headers_frame, text=f"{metrics['support']:.0f}", width=12, relief=tk.RIDGE,
                        bg=self.theme['entry_bg'], fg=self.theme['fg'],
                        font=('Arial', 9)).grid(row=row, column=4, padx=1, pady=1)
                row += 1
        
        if 'accuracy' in report:
            acc_frame = tk.Frame(report_frame, bg=self.theme['frame_bg'])
            acc_frame.pack(pady=5)
            tk.Label(acc_frame, text=f"Accuracy Global: {report['accuracy']:.4f}",
                    font=('Arial', 10, 'bold'),
                    bg=self.theme['frame_bg'], fg=self.theme['success']).pack()
    
    def _show_predictions(self, parent):
        """Mostrar predicciones vs reales (regresión simple)"""
        if self.model_type != 'simple' or not hasattr(self.automl, 'get_predictions'):
            return
        
        preds = self.automl.get_predictions()
        if preds is None:
            return
        
        pred_frame = tk.LabelFrame(parent, text="📈 Predicciones vs Valores Reales",
                                  bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                  font=('Arial', 12, 'bold'))
        pred_frame.pack(fill=tk.X, padx=20, pady=10)
        
        fig = Figure(figsize=(6, 4), facecolor=self.theme['bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.theme['entry_bg'])
        
        ax.scatter(self.automl.y_test, preds, alpha=0.5, c=self.theme['highlight'], s=20)
        
        min_val = min(self.automl.y_test.min(), preds.min())
        max_val = max(self.automl.y_test.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal')
        
        ax.set_xlabel('Valores Reales', color=self.theme['fg'])
        ax.set_ylabel('Predicciones', color=self.theme['fg'])
        ax.set_title('Predicciones vs Reales', color=self.theme['fg'])
        ax.tick_params(colors=self.theme['fg'])
        ax.legend(facecolor=self.theme['frame_bg'], labelcolor=self.theme['fg'])
        ax.grid(True, alpha=0.3)
        
        canvas_plot = FigureCanvasTkAgg(fig, master=pred_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(padx=10, pady=10)
    
    def _show_diagnostics(self, parent, results):
        """Mostrar diagnóstico del modelo"""
        diag_frame = tk.LabelFrame(parent, text="💡 Diagnóstico del Modelo",
                                  bg=self.theme['frame_bg'], fg=self.theme['fg'],
                                  font=('Arial', 12, 'bold'))
        diag_frame.pack(fill=tk.X, padx=20, pady=10)
        
        if self.model_type == 'logistic':
            final_acc = results['final_accuracy']
            if final_acc >= 0.9:
                diag_text = "✅ Excelente modelo! Muy alta precisión."
            elif final_acc >= 0.8:
                diag_text = "👍 Buen modelo. Precisión aceptable."
            elif final_acc >= 0.7:
                diag_text = "⚠️ Modelo aceptable. Se puede mejorar."
            else:
                diag_text = "❌ Modelo mejorable. Considere más datos o diferentes features."
        else:
            final_r2 = results['final_r2']
            if final_r2 >= 0.9:
                diag_text = "✅ Excelente modelo! Muy alto poder predictivo."
            elif final_r2 >= 0.7:
                diag_text = "👍 Buen modelo. El modelo explica bien la varianza."
            elif final_r2 >= 0.5:
                diag_text = "⚠️ Modelo moderado. Podría mejorar con más features."
            else:
                diag_text = "❌ Modelo débil. El R² es bajo, considere transformaciones o más datos."
        
        tk.Label(diag_frame, text=diag_text, wraplength=800,
                bg=self.theme['frame_bg'], fg=self.theme['fg'],
                font=('Arial', 10)).pack(padx=10, pady=10)
