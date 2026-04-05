"""
Clase para Regresión Lineal (Simple y Múltiple)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionVisualizer:
    """
    Clase para regresión lineal con visualización en tiempo real
    Soporta: Simple (1 feature) y Múltiple (múltiples features)
    """
    
    def __init__(self, status_callback=None, plot_callback=None):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_names = []
        self.is_simple = False  # True = regresión simple, False = múltiple
        self.status_callback = status_callback
        self.plot_callback = plot_callback
        
    def log_status(self, message):
        """Envía mensaje de estado a la GUI"""
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_data(self, file_path):
        """Carga datos desde Excel o CSV"""
        self.log_status(f"📂 Cargando archivo: {file_path}")
        
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Formato no soportado. Use .xlsx, .xls o .csv")
        
        self.log_status(f"✅ Datos cargados: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")
        return self.df
    
    def get_columns(self):
        """Retorna lista de columnas disponibles"""
        if self.df is not None:
            return list(self.df.columns)
        return []
    
    def set_variables(self, feature_cols, target_col):
        """Establece variables independientes y dependiente"""
        self.feature_names = feature_cols
        self.is_simple = len(feature_cols) == 1
        
        self.X = self.df[feature_cols].copy()
        self.y = self.df[target_col].copy()
        
        model_type = "Simple" if self.is_simple else "Múltiple"
        self.log_status(f"✅ Regresión Lineal {model_type}")
        self.log_status(f"   Features ({len(feature_cols)}): {feature_cols}")
        self.log_status(f"   Target: {target_col}")
        
    def clean_data(self, handle_nulls='2'):
        """Limpieza y preprocesamiento automático de datos"""
        self.log_status("🔍 Limpiando y preprocesando datos...")
        
        # Manejo de valores nulos en X
        null_counts = self.X.isnull().sum()
        if null_counts.sum() > 0:
            self.log_status(f"⚠️ Encontrados {null_counts.sum()} valores nulos en features")
            
            if handle_nulls == '1':
                combined = pd.concat([self.X, self.y], axis=1)
                combined = combined.dropna()
                self.X = combined[self.feature_names]
                self.y = combined[self.y.name]
                self.log_status(f"✅ Eliminadas filas con nulos. Nuevo tamaño: {len(self.X)}")
            elif handle_nulls == '2':
                for col in self.X.columns:
                    if self.X[col].dtype in ['object', 'category']:
                        self.X[col].fillna(self.X[col].mode()[0] if not self.X[col].mode().empty else 0, inplace=True)
                    else:
                        self.X[col].fillna(self.X[col].median(), inplace=True)
                self.log_status("✅ Nulos rellenados con mediana/moda")
            elif handle_nulls == '3':
                self.X.fillna(0, inplace=True)
                self.log_status("✅ Nulos rellenados con 0")
        else:
            self.log_status("✅ No hay valores nulos en features")
        
        # Manejo de nulos en y (target)
        if self.y.isnull().sum() > 0:
            self.log_status(f"⚠️ {self.y.isnull().sum()} valores nulos en target, eliminando...")
            valid_idx = ~self.y.isnull()
            self.X = self.X[valid_idx]
            self.y = self.y[valid_idx]
        
        # Codificación de variables categóricas en X
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            self.log_status(f"🏷️ Codificando {len(categorical_cols)} variables categóricas...")
            for col in categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le
        
        # Verificar que y es numérica (para regresión)
        if self.y.dtype == 'object' or pd.api.types.is_string_dtype(self.y):
            self.log_status("⚠️ Variable objetivo categórica detectada. Convertiendo a numérica...")
            le = LabelEncoder()
            self.y = le.fit_transform(self.y.astype(str))
            self.log_status(f"   Valores mapeados: {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            self.y = self.y.astype(float)
        
        # Escalado de características (importante para SGD)
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.log_status("📏 Escalado completado")
        
    def split_data(self, test_size=0.3):
        """Divide datos en entrenamiento y prueba"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42
        )
        
        # Convertir a arrays de numpy
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        
        self.log_status(f"✅ Datos divididos: Train={len(self.X_train)}, Test={len(self.X_test)}")
    
    def train_and_visualize(self, n_epochs=10, learning_rate=0.01):
        """Entrena el modelo con SGD para visualización en tiempo real"""
        self.log_status("🚀 Iniciando entrenamiento con SGD...")
        
        # Usar SGDRegressor para visualización paso a paso
        self.model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            learning_rate='constant',
            eta0=learning_rate,
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42
        )
        
        # Métricas
        train_losses = []  # MSE
        test_losses = []
        train_r2s = []
        test_r2s = []
        coef_history = []
        
        for epoch in range(n_epochs):
            # Entrenar un epoch
            self.model.partial_fit(self.X_train, self.y_train)
            
            # Predicciones
            y_train_pred = self.model.predict(self.X_train)
            y_test_pred = self.model.predict(self.X_test)
            
            # Calcular métricas
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            train_losses.append(train_mse)
            test_losses.append(test_mse)
            train_r2s.append(train_r2)
            test_r2s.append(test_r2)
            
            # Guardar coeficientes
            coef_history.append(self.model.coef_.copy())
            
            # Enviar a GUI para actualizar gráficas
            if self.plot_callback:
                self.plot_callback(
                    epoch, n_epochs, 
                    train_losses, test_losses,
                    train_r2s, test_r2s,
                    coef_history,
                    is_regression=True,
                    is_simple=self.is_simple,
                    X_test=self.X_test if self.is_simple else None,
                    y_test=self.y_test if self.is_simple else None,
                    model=self.model
                )
            
            # Actualizar estado cada 10 epochs
            if (epoch + 1) % 10 == 0:
                self.log_status(f"Epoch {epoch+1}/{n_epochs} - Test R²: {test_r2:.4f} | Test MSE: {test_mse:.4f}")
        
        self.log_status("✅ Entrenamiento completado!")
        
        # Resultados finales
        y_test_pred = self.model.predict(self.X_test)
        
        return {
            'best_epoch': np.argmax(test_r2s) + 1,
            'best_r2': max(test_r2s),
            'final_r2': test_r2s[-1],
            'final_mse': test_losses[-1],
            'final_mae': mean_absolute_error(self.y_test, y_test_pred),
            'coefficients': dict(zip(self.feature_names, self.model.coef_)),
            'intercept': self.model.intercept_
        }
    
    def get_feature_importance(self):
        """Retorna coeficientes como importancia"""
        if self.model is not None:
            return dict(zip(self.feature_names, np.abs(self.model.coef_)))
        return {}
    
    def get_predictions(self):
        """Retorna predicciones para gráficas"""
        if self.model is not None:
            return self.model.predict(self.X_test)
        return None