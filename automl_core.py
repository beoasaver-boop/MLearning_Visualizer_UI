import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class AutoMLVisualizer:
    """
    Clase para análisis automático de machine learning con visualización en tiempo real
    Modificada para integrarse con interfaz gráfica
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
        self.is_multiclass = False
        self.n_classes = 2
        self.status_callback = status_callback  # Para actualizar estado en GUI
        self.plot_callback = plot_callback      # Para enviar gráficas a GUI
        
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
        self.X = self.df[feature_cols].copy()
        self.y = self.df[target_col].copy()
        
        self.log_status(f"✅ Features ({len(feature_cols)}): {feature_cols}")
        self.log_status(f"✅ Target: {target_col}")
        
    def clean_data(self, handle_nulls='2'):
        """Limpieza y preprocesamiento automático de datos"""
        self.log_status("🔍 Limpiando y preprocesando datos...")
        
        # Manejo de valores nulos
        null_counts = self.X.isnull().sum()
        if null_counts.sum() > 0:
            self.log_status(f"⚠️ Encontrados {null_counts.sum()} valores nulos")
            
            if handle_nulls == '1':
                combined = pd.concat([self.X, self.y], axis=1)
                combined = combined.dropna()
                self.X = combined[self.feature_names]
                self.y = combined[self.y.name]
                self.log_status(f"✅ Eliminadas filas con nulos. Nuevo tamaño: {len(self.X)}")
                
            elif handle_nulls == '2':
                for col in self.X.columns:
                    if self.X[col].dtype in ['object', 'category']:
                        self.X[col].fillna(self.X[col].mode()[0] if not self.X[col].mode().empty else 'Unknown', inplace=True)
                    else:
                        self.X[col].fillna(self.X[col].median(), inplace=True)
                self.log_status("✅ Nulos rellenados con mediana/moda")
                
            elif handle_nulls == '3':
                self.X.fillna(0, inplace=True)
                self.log_status("✅ Nulos rellenados con 0")
        else:
            self.log_status("✅ No hay valores nulos")
        
        # Codificación de variables categóricas
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            self.log_status(f"🏷️ Codificando {len(categorical_cols)} variables categóricas...")
            for col in categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
                self.label_encoders[col] = le
        
        # Procesar variable objetivo
        if self.y.dtype == 'object' or self.y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)
            self.n_classes = len(self.target_encoder.classes_)
            self.is_multiclass = self.n_classes > 2
            self.log_status(f"🎯 Target codificado: {self.n_classes} clases")
        else:
            self.n_classes = len(np.unique(self.y))
            self.is_multiclass = self.n_classes > 2
        
        # Escalado
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.log_status("📏 Escalado completado")
        
    def split_data(self, test_size=0.3):
        """Divide datos en entrenamiento y prueba"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        self.log_status(f"✅ Datos divididos: Train={len(self.X_train)}, Test={len(self.X_test)}")
        
    def train_and_visualize(self, n_epochs=100):
        """Entrena el modelo y envía visualizaciones a la GUI"""
        self.log_status("🚀 Iniciando entrenamiento...")
        
        # Configurar modelo
        self.model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            learning_rate='optimal',
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42
        )
        
        # Métricas
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(n_epochs):
            # Entrenar
            self.model.partial_fit(self.X_train, self.y_train, classes=np.unique(self.y))
            
            # Calcular métricas
            train_probs = self.model.predict_proba(self.X_train)
            test_probs = self.model.predict_proba(self.X_test)
            
            if self.is_multiclass:
                train_loss = -np.mean([np.log(train_probs[i, self.y_train[i]] + 1e-10) for i in range(len(self.y_train))])
                test_loss = -np.mean([np.log(test_probs[i, self.y_test[i]] + 1e-10) for i in range(len(self.y_test))])
            else:
                train_loss = -np.mean(self.y_train * np.log(train_probs[:, 1] + 1e-10) + 
                                     (1 - self.y_train) * np.log(1 - train_probs[:, 1] + 1e-10))
                test_loss = -np.mean(self.y_test * np.log(test_probs[:, 1] + 1e-10) + 
                                    (1 - self.y_test) * np.log(1 - test_probs[:, 1] + 1e-10))
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            train_acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
            test_acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            # Enviar a GUI para actualizar gráficas
            if self.plot_callback:
                self.plot_callback(epoch, n_epochs, train_losses, test_losses, 
                                  train_accuracies, test_accuracies)
            
            # Actualizar estado cada 10 epochs
            if (epoch + 1) % 10 == 0:
                self.log_status(f"Epoch {epoch+1}/{n_epochs} - Test Acc: {test_acc:.4f}")
        
        # Resultados finales
        self.log_status("✅ Entrenamiento completado!")
        
        return {
            'best_epoch': np.argmax(test_accuracies) + 1,
            'best_accuracy': max(test_accuracies),
            'final_accuracy': test_accuracies[-1],
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    
    def get_feature_importance(self):
        """Retorna importancia de características"""
        if hasattr(self.model, 'coef_'):
            if self.is_multiclass:
                importances = np.abs(self.model.coef_).mean(axis=0)
            else:
                importances = np.abs(self.model.coef_[0])
            return dict(zip(self.feature_names, importances))
        return {}
    
    def get_confusion_matrix(self):
        """Retorna matriz de confusión"""
        y_pred = self.model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self):
        """Retorna reporte de clasificación como diccionario"""
        y_pred = self.model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        return report