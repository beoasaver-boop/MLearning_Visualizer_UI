import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config.theme import DARK_THEME

class ResultsDisplayer:
    def __init__(self, app):
        self.app = app

    def display(self, results):
        for widget in self.app.results_display.winfo_children():
            widget.destroy()

        container = ctk.CTkFrame(self.app.results_display, fg_color="transparent")
        container.pack(fill="both", expand=True)

        title = ctk.CTkLabel(
            container,
            text=f"RESULTADOS DEL MODELO - {self.app.get_model_name()}",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.pack(pady=(10, 15))

        self._show_metrics(container, results)

        if self.app.model_type in ['simple', 'multiple'] and 'coefficients' in results:
            self._show_equation(container, results)

        if hasattr(self.app.automl, 'get_feature_importance'):
            importance = self.app.automl.get_feature_importance()
            if importance:
                self._show_coefficients(container, importance)

        if self.app.model_type in ['logistic', 'random_forest'] and hasattr(self.app.automl, 'get_confusion_matrix'):
            cm = self.app.automl.get_confusion_matrix()
            self._show_confusion_matrix(container, cm)

        if self.app.model_type in ['logistic', 'random_forest'] and hasattr(self.app.automl, 'get_classification_report'):
            report = self.app.automl.get_classification_report()
            self._show_classification_report(container, report)

        if self.app.model_type == 'simple' and hasattr(self.app.automl, 'get_predictions'):
            preds = self.app.automl.get_predictions()
            if preds is not None:
                self._show_predictions_plot(container, preds)

        self._show_diagnostics(container, results)

    def _section_frame(self, parent, title):
        frame = ctk.CTkFrame(parent, corner_radius=8)
        frame.pack(fill="x", padx=15, pady=(0, 10))
        label = ctk.CTkLabel(
            frame, text=title,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        label.pack(anchor="w", padx=12, pady=(8, 5))
        return frame

    def _show_metrics(self, parent, results):
        frame = self._section_frame(parent, "Metricas Principales")
        if self.app.model_type in ['logistic', 'random_forest']:
            best = results['best_accuracy']
            final = results['final_accuracy']
            epoch = results['best_epoch']
            ctk.CTkLabel(
                frame,
                text=f"Mejor Accuracy: {best:.4f}",
                font=ctk.CTkFont(size=12),
                text_color=("#1a7a1a", "#4caf50")
            ).pack(anchor="w", padx=12, pady=2)
            ctk.CTkLabel(
                frame,
                text=f"Accuracy Final: {final:.4f}",
                font=ctk.CTkFont(size=12)
            ).pack(anchor="w", padx=12, pady=2)
            ctk.CTkLabel(
                frame,
                text=f"Mejor Epoch: {epoch}",
                font=ctk.CTkFont(size=12),
                text_color=("#1a5a8a", "#2196f3")
            ).pack(anchor="w", padx=12, pady=2)
            if 'n_estimators_final' in results:
                ctk.CTkLabel(
                    frame,
                    text=f"Arboles finales: {results['n_estimators_final']}",
                    font=ctk.CTkFont(size=12),
                    text_color=("#cc7a00", "#ff9800")
                ).pack(anchor="w", padx=12, pady=2)
        else:
            best = results['best_r2']
            final = results['final_r2']
            mse = results['final_mse']
            ctk.CTkLabel(
                frame,
                text=f"Mejor R²: {best:.4f}",
                font=ctk.CTkFont(size=12),
                text_color=("#1a7a1a", "#4caf50")
            ).pack(anchor="w", padx=12, pady=2)
            ctk.CTkLabel(
                frame,
                text=f"R² Final: {final:.4f}",
                font=ctk.CTkFont(size=12)
            ).pack(anchor="w", padx=12, pady=2)
            ctk.CTkLabel(
                frame,
                text=f"MSE Final: {mse:.4f}",
                font=ctk.CTkFont(size=12)
            ).pack(anchor="w", padx=12, pady=2)

    def _show_equation(self, parent, results):
        frame = self._section_frame(parent, "Ecuacion del Modelo")
        intercept = results['intercept']
        coefs = results['coefficients']
        eq = f"y = {intercept:.4f}"
        for feat, coef in coefs.items():
            sign = " + " if coef >= 0 else " - "
            eq += f"{sign}{abs(coef):.4f} * {feat}"
        ctk.CTkLabel(
            frame, text=eq,
            font=ctk.CTkFont(family="Consolas", size=11),
            wraplength=800
        ).pack(padx=12, pady=5)

    def _show_coefficients(self, parent, importance):
        frame = self._section_frame(parent, "Importancia de Caracteristicas")
        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        max_val = max(abs(v) for v in importance.values()) if importance else 1

        for feature, val in sorted_items:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=1)
            ctk.CTkLabel(
                row, text=f"{feature}:", width=25,
                font=ctk.CTkFont(size=11),
                anchor="w"
            ).pack(side="left")
            color = ("#1a7a1a", "#4caf50") if val > 0 else ("#aa2222", "#f44336")
            ctk.CTkLabel(
                row, text=f"{val:.4f}", width=10,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=color, anchor="w"
            ).pack(side="left")
            bar_frame = ctk.CTkFrame(row, fg_color="#2d2d2d",
                                      corner_radius=3, height=14, width=200)
            bar_frame.pack(side="left", padx=5)
            bar_frame.pack_propagate(False)
            bar_width = max(1, int((abs(val) / max_val) * 196))
            fill = ctk.CTkFrame(bar_frame, fg_color=("#1a73e8", "#00bcd4"),
                                 corner_radius=2, height=12, width=bar_width)
            fill.pack(side="left", padx=1, pady=1)

    def _show_confusion_matrix(self, parent, cm):
        frame = self._section_frame(parent, "Matriz de Confusion")
        table = ctk.CTkFrame(frame, fg_color="transparent")
        table.pack(pady=10)

        headers_cfg = {"font": ctk.CTkFont(size=11, weight="bold"),
                       "fg_color": ("#1a73e8", "#1a5a8a"),
                       "text_color": "white", "corner_radius": 3,
                       "width": 100, "height": 28}
        data_cfg = {"font": ctk.CTkFont(size=11),
                    "fg_color": "#2d2d2d",
                    "text_color": "white", "corner_radius": 3,
                    "width": 100, "height": 28}

        ctk.CTkLabel(table, text="", width=100, fg_color="transparent").grid(row=0, column=0)
        ctk.CTkLabel(table, text="Predicho Neg", **headers_cfg).grid(row=0, column=1, padx=1)
        ctk.CTkLabel(table, text="Predicho Pos", **headers_cfg).grid(row=0, column=2, padx=1)

        labels_data = [
            ("Real Neg", cm[0, 0], cm[0, 1]),
            ("Real Pos", cm[1, 0], cm[1, 1])
        ]
        for i, (label, v1, v2) in enumerate(labels_data, 1):
            ctk.CTkLabel(table, text=label, **data_cfg).grid(row=i, column=0, padx=1)
            ctk.CTkLabel(table, text=str(v1), **data_cfg).grid(row=i, column=1, padx=1)
            ctk.CTkLabel(table, text=str(v2), **data_cfg).grid(row=i, column=2, padx=1)

    def _show_classification_report(self, parent, report):
        frame = self._section_frame(parent, "Reporte de Clasificacion")
        headers = ["Clase", "Precision", "Recall", "F1", "Soporte"]
        hdr_cfg = {"font": ctk.CTkFont(size=11, weight="bold"),
                    "fg_color": ("#1a73e8", "#1a5a8a"),
                    "text_color": "white", "corner_radius": 3,
                    "width": 100, "height": 28}
        cell_cfg = {"font": ctk.CTkFont(size=11),
                     "fg_color": "#2d2d2d",
                     "text_color": "white", "corner_radius": 3,
                     "width": 100, "height": 28}

        table = ctk.CTkFrame(frame, fg_color="transparent")
        table.pack(pady=8)

        for i, h in enumerate(headers):
            ctk.CTkLabel(table, text=h, **hdr_cfg).grid(row=0, column=i, padx=1, pady=1)

        row = 1
        for cls, metrics in report.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                ctk.CTkLabel(table, text=str(cls)[:12], **cell_cfg).grid(row=row, column=0, padx=1)
                ctk.CTkLabel(table, text=f"{metrics['precision']:.3f}", **cell_cfg).grid(row=row, column=1, padx=1)
                ctk.CTkLabel(table, text=f"{metrics['recall']:.3f}", **cell_cfg).grid(row=row, column=2, padx=1)
                ctk.CTkLabel(table, text=f"{metrics['f1-score']:.3f}", **cell_cfg).grid(row=row, column=3, padx=1)
                ctk.CTkLabel(table, text=f"{metrics['support']:.0f}", **cell_cfg).grid(row=row, column=4, padx=1)
                row += 1

        if 'accuracy' in report:
            ctk.CTkLabel(
                frame,
                text=f"Accuracy Global: {report['accuracy']:.4f}",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=("#1a7a1a", "#4caf50")
            ).pack(pady=(5, 8))

    def _show_predictions_plot(self, parent, preds):
        frame = self._section_frame(parent, "Predicciones vs Valores Reales")
        fig = Figure(figsize=(6, 4), facecolor=DARK_THEME['frame_bg'])
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_THEME['entry_bg'])
        ax.scatter(self.app.automl.y_test, preds, alpha=0.5,
                   c=DARK_THEME['highlight'], s=20)
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
        frame = self._section_frame(parent, "Diagnostico del Modelo")
        if self.app.model_type in ['logistic', 'random_forest']:
            acc = results['final_accuracy']
            if acc >= 0.9:
                msg = "Excelente modelo! Muy alta precision."
                color = ("#1a7a1a", "#4caf50")
            elif acc >= 0.8:
                msg = "Buen modelo. Precision aceptable."
                color = ("#1a5a8a", "#2196f3")
            elif acc >= 0.7:
                msg = "Modelo aceptable. Se puede mejorar."
                color = ("#cc7a00", "#ff9800")
            else:
                msg = "Modelo mejorable. Considere mas datos o ajustar hiperparametros."
                color = ("#aa2222", "#f44336")
            if self.app.model_type == 'random_forest':
                msg += " (Random Forest es robusto, prueba aumentando arboles)"
        else:
            r2 = results['final_r2']
            if r2 >= 0.9:
                msg = "Excelente modelo! Muy alto poder predictivo."
                color = ("#1a7a1a", "#4caf50")
            elif r2 >= 0.7:
                msg = "Buen modelo. Explica bien la varianza."
                color = ("#1a5a8a", "#2196f3")
            elif r2 >= 0.5:
                msg = "Modelo moderado. Podria mejorar con mas features."
                color = ("#cc7a00", "#ff9800")
            else:
                msg = "Modelo debil. Considere transformaciones o mas datos."
                color = ("#aa2222", "#f44336")

        ctk.CTkLabel(
            frame, text=msg,
            font=ctk.CTkFont(size=11),
            text_color=color, wraplength=800
        ).pack(padx=12, pady=8)
