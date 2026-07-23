import customtkinter as ctk
from tkinter import ttk
import pandas as pd
import numpy as np

class EDAViewer:
    def __init__(self, parent, df, filename):
        self.df = df
        self.filename = filename
        self.window = ctk.CTkToplevel(parent)
        self.window.title(f"Analisis Exploratorio - {filename}")
        self.window.geometry("900x700")
        self.window.transient(parent)
        self.window.grab_set()
        self.center_window()
        self._style_treeview()
        self._create_widgets()

    def center_window(self):
        self.window.update_idletasks()
        width = 900
        height = 700
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')

    def _style_treeview(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                        background="#2d2d2d",
                        foreground="#ffffff",
                        fieldbackground="#2d2d2d",
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        background="#1e1e1e",
                        foreground="#ffffff",
                        relief="flat")
        style.map("Treeview",
                  background=[("selected", "#144870")],
                  foreground=[("selected", "#ffffff")])

    def _create_widgets(self):
        tabview = ctk.CTkTabview(self.window, corner_radius=8)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self._add_preview_tab(tabview)
        self._add_columns_info_tab(tabview)
        self._add_statistics_tab(tabview)
        self._add_null_tab(tabview)

        close_btn = ctk.CTkButton(
            self.window, text="Cerrar",
            command=self.window.destroy,
            font=ctk.CTkFont(size=11)
        )
        close_btn.pack(pady=(0, 10))

    def _add_preview_tab(self, tabview):
        tab = tabview.add("Vista Previa")

        info_label = ctk.CTkLabel(
            tab,
            text=f"Dimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        info_label.pack(anchor="w", padx=10, pady=(10, 5))

        frame = ctk.CTkFrame(tab, fg_color="#2d2d2d", corner_radius=6)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        scroll_y = ttk.Scrollbar(frame, orient="vertical")
        scroll_x = ttk.Scrollbar(frame, orient="horizontal")

        tree = ttk.Treeview(
            frame,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
            style="Treeview"
        )
        scroll_y.configure(command=tree.yview)
        scroll_x.configure(command=tree.xview)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)

        columns = list(self.df.columns)
        tree["columns"] = columns
        tree["show"] = "headings"

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        for _, row in self.df.head(10).iterrows():
            tree.insert("", "end", values=list(row))

    def _add_columns_info_tab(self, tabview):
        tab = tabview.add("Informacion de Columnas")

        frame = ctk.CTkFrame(tab, fg_color="#2d2d2d", corner_radius=6)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        scroll_y = ttk.Scrollbar(frame, orient="vertical")
        scroll_x = ttk.Scrollbar(frame, orient="horizontal")

        tree = ttk.Treeview(
            frame,
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set,
            style="Treeview"
        )
        scroll_y.configure(command=tree.yview)
        scroll_x.configure(command=tree.xview)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)

        tree["columns"] = ("tipo", "nulos", "unicos", "ejemplo")
        tree["show"] = "headings"
        tree.heading("tipo", text="Tipo de Dato")
        tree.heading("nulos", text="Valores Nulos")
        tree.heading("unicos", text="Valores Unicos")
        tree.heading("ejemplo", text="Ejemplo")
        tree.column("tipo", width=120)
        tree.column("nulos", width=100)
        tree.column("unicos", width=100)
        tree.column("ejemplo", width=300)

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            nulos = self.df[col].isnull().sum()
            unicos = self.df[col].nunique()
            ejemplo = str(self.df[col].iloc[0]) if len(self.df) > 0 else "N/A"
            tree.insert("", "end", values=(dtype, nulos, unicos, ejemplo[:50]))

    def _add_statistics_tab(self, tabview):
        tab = tabview.add("Estadisticas")

        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) > 0:
            num_section = ctk.CTkFrame(scroll_frame, corner_radius=6)
            num_section.pack(fill="x", pady=(0, 10))

            ctk.CTkLabel(
                num_section, text="Variables Numericas",
                font=ctk.CTkFont(size=13, weight="bold")
            ).pack(anchor="w", padx=12, pady=(8, 5))

            for col in numeric_cols:
                stats = self.df[col].describe()
                info = (f"{col}: media={stats['mean']:.2f}, "
                        f"mediana={stats['50%']:.2f}, "
                        f"min={stats['min']:.2f}, max={stats['max']:.2f}")
                ctk.CTkLabel(
                    num_section, text=info,
                    font=ctk.CTkFont(family="Consolas", size=10),
                    justify="left"
                ).pack(anchor="w", padx=12, pady=1)

            if len(categorical_cols) > 0:
                cat_section = ctk.CTkFrame(scroll_frame, corner_radius=6)
                cat_section.pack(fill="x")

                ctk.CTkLabel(
                    cat_section, text="Variables Categoricas",
                    font=ctk.CTkFont(size=13, weight="bold")
                ).pack(anchor="w", padx=12, pady=(8, 5))

                for col in categorical_cols:
                    top_values = self.df[col].value_counts().head(3)
                    top_str = ", ".join([f"{v}: {c}" for v, c in top_values.items()])
                    info = f"{col}: {self.df[col].nunique()} categorias. Top: {top_str}"
                    ctk.CTkLabel(
                        cat_section, text=info,
                        font=ctk.CTkFont(family="Consolas", size=10),
                        justify="left", wraplength=700
                    ).pack(anchor="w", padx=12, pady=1)

    def _add_null_tab(self, tabview):
        tab = tabview.add("Valores Nulos")

        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        null_counts = self.df.isnull().sum()
        null_cols = null_counts[null_counts > 0]

        if len(null_cols) > 0:
            ctk.CTkLabel(
                frame,
                text=f"Se encontraron {null_counts.sum()} valores nulos en total:",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=("#cc7a00", "#ff9800")
            ).pack(pady=(10, 5))

            for col, count in null_cols.items():
                pct = (count / len(self.df)) * 100
                ctk.CTkLabel(
                    frame, text=f"  - {col}: {count} nulos ({pct:.1f}%)",
                    font=ctk.CTkFont(size=11)
                ).pack(anchor="w", padx=20, pady=1)
        else:
            ctk.CTkLabel(
                frame,
                text="No hay valores nulos en el dataset",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=("#1a7a1a", "#4caf50")
            ).pack(pady=40)
