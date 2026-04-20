# -*- coding: utf-8 -*-
"""
Módulo para Análisis Exploratorio de Datos (EDA)
Muestra información detallada del dataset cargado
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from config.theme import DARK_THEME

class EDAViewer:
    def __init__(self, parent, df, filename):
        self.df = df
        self.filename = filename
        self.window = tk.Toplevel(parent)
        self.window.title(f"Análisis Exploratorio - {filename}")
        self.window.geometry("900x700")
        self.window.configure(bg=DARK_THEME['bg'])
        self.window.transient(parent)
        self.window.grab_set()
        
        # Centrar ventana
        self.center_window()
        
        self._create_widgets()
    
    def center_window(self):
        self.window.update_idletasks()
        width = 900
        height = 700
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_widgets(self):
        # Notebook para pestañas
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Pestaña 1: Vista previa
        self._add_preview_tab(notebook)
        
        # Pestaña 2: Información de columnas
        self._add_columns_info_tab(notebook)
        
        # Pestaña 3: Estadísticas
        self._add_statistics_tab(notebook)
        
        # Pestaña 4: Valores nulos
        self._add_null_tab(notebook)
        
        # Botón cerrar
        close_btn = tk.Button(self.window, text="Cerrar", command=self.window.destroy,
                              bg=DARK_THEME['button_bg'], fg=DARK_THEME['button_fg'],
                              font=('Arial', 10), cursor='hand2')
        close_btn.pack(pady=10)
    
    def _add_preview_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="📋 Vista Previa")
        
        # Info de dimensiones
        info_frame = tk.Frame(tab, bg=DARK_THEME['bg'])
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(info_frame, text=f"Dimensiones: {self.df.shape[0]} filas × {self.df.shape[1]} columnas",
                 bg=DARK_THEME['bg'], fg=DARK_THEME['fg'], font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        # Treeview para mostrar datos
        frame = tk.Frame(tab, bg=DARK_THEME['bg'])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scroll_y = tk.Scrollbar(frame, orient=tk.VERTICAL)
        scroll_x = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        
        tree = ttk.Treeview(frame, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configurar columnas
        columns = list(self.df.columns)
        tree["columns"] = columns
        tree["show"] = "headings"
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Insertar primeras 10 filas
        for idx, row in self.df.head(10).iterrows():
            tree.insert("", tk.END, values=list(row))
    
    def _add_columns_info_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="📊 Información de Columnas")
        
        frame = tk.Frame(tab, bg=DARK_THEME['bg'])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview para info de columnas
        scroll_y = tk.Scrollbar(frame, orient=tk.VERTICAL)
        scroll_x = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        
        tree = ttk.Treeview(frame, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)
        
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tree["columns"] = ("tipo", "nulos", "unicos", "ejemplo")
        tree["show"] = "headings"
        tree.heading("tipo", text="Tipo de Dato")
        tree.heading("nulos", text="Valores Nulos")
        tree.heading("unicos", text="Valores Únicos")
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
            
            # Color según tipo
            if 'int' in dtype or 'float' in dtype:
                tipo_display = f"🔢 {dtype}"
            else:
                tipo_display = f"📝 {dtype}"
            
            tree.insert("", tk.END, values=(tipo_display, nulos, unicos, ejemplo[:50]))
    
    def _add_statistics_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="📈 Estadísticas")
        
        frame = tk.Frame(tab, bg=DARK_THEME['bg'])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Separar numéricas y categóricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        # Frame para scroll
        canvas = tk.Canvas(frame, bg=DARK_THEME['bg'], highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=DARK_THEME['bg'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Estadísticas numéricas
        if len(numeric_cols) > 0:
            num_frame = tk.LabelFrame(scrollable_frame, text="Variables Numéricas",
                                      bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                      font=('Arial', 11, 'bold'))
            num_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for col in numeric_cols:
                stats = self.df[col].describe()
                info = f"{col}: media={stats['mean']:.2f}, mediana={stats['50%']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}"
                tk.Label(num_frame, text=info, bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                         font=('Consolas', 9), anchor=tk.W).pack(fill=tk.X, padx=10, pady=2)
        
        # Estadísticas categóricas
        if len(categorical_cols) > 0:
            cat_frame = tk.LabelFrame(scrollable_frame, text="Variables Categóricas",
                                      bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                                      font=('Arial', 11, 'bold'))
            cat_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for col in categorical_cols:
                top_values = self.df[col].value_counts().head(3)
                top_str = ", ".join([f"{v}: {c}" for v, c in top_values.items()])
                info = f"{col}: {self.df[col].nunique()} categorías distintas. Top: {top_str}"
                tk.Label(cat_frame, text=info, bg=DARK_THEME['frame_bg'], fg=DARK_THEME['fg'],
                         font=('Consolas', 9), anchor=tk.W, wraplength=700).pack(fill=tk.X, padx=10, pady=2)
    
    def _add_null_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="⚠️ Valores Nulos")
        
        frame = tk.Frame(tab, bg=DARK_THEME['bg'])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        null_counts = self.df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
        
        if len(null_cols) > 0:
            tk.Label(frame, text=f"Se encontraron {null_counts.sum()} valores nulos en total:",
                     bg=DARK_THEME['bg'], fg=DARK_THEME['warning'], font=('Arial', 10, 'bold')).pack(pady=5)
            
            for col, count in null_cols.items():
                pct = (count / len(self.df)) * 100
                tk.Label(frame, text=f"  • {col}: {count} nulos ({pct:.1f}%)",
                         bg=DARK_THEME['bg'], fg=DARK_THEME['fg']).pack(anchor=tk.W, padx=20, pady=2)
        else:
            tk.Label(frame, text="✅ No hay valores nulos en el dataset",
                     bg=DARK_THEME['bg'], fg=DARK_THEME['success'], font=('Arial', 12, 'bold')).pack(pady=20)
