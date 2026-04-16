"""
Módulo para Random Forest (Clasificación)
Integración con la interfaz gráfica y visualización en tiempo real
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class RandomForestVisualizer:
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
        self.status_callback = status_callback
        self.plot_callback = plot_callback
        self.n_estimators_per_epoch = 5
        
    def log_status(self, message):
        if self.status_callback:
            self.status_callback(message)
        print(message)
    
    def load_data(self, file_path):
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
        return list(self.df.columns) if self.df is not None else []
    
    def set_variables(self, feature_cols, target_col):
        self.feature_names = feature_cols
        self.X = self.df[feature_cols].copy()
        self.y = self.df[target_col].copy()
        self.log_status(f"✅ Random Forest configurado con {len(feature_cols)} features")
    
    def clean_data(self, handle_nulls='2'):
        self.log_status("🔍 Limpiando y preprocesando datos...")
        null_counts = self.X.isnull().sum()
        if null_counts.sum() > 0:
            self.log_status(f"⚠️ Encontrados {null_counts.sum()} valores nulos")
            if handle_nulls == '1':
                combined = pd.concat([self.X, self.y], axis=1).dropna()
                self.X = combined[self.feature_names]
                self.y = combined[self.y.name]
            elif handle_nulls == '2':
                for col in self.X.columns:
                    if self.X[col].dtype in ['object', 'category']:
                        self.X[col].fillna(self.X[col].mode()[0] if not self.X[col].mode().empty else 'Unknown', inplace=True)
                    else:
                        self.X[col].fillna(self.X[col].median(), inplace=True)
            elif handle_nulls == '3':
                self.X.fillna(0, inplace=True)
            self.log_status("✅ Nulos manejados")
        
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.X[col] = le.fit_transform(self.X[col].astype(str))
            self.label_encoders[col] = le
        
        if self.y.dtype == 'object' or pd.api.types.is_string_dtype(self.y):
            self.target_encoder = LabelEncoder()
            self.y = self.target_encoder.fit_transform(self.y)
            self.n_classes = len(self.target_encoder.classes_)
        else:
            self.y = self.y.astype(int)
            self.n_classes = len(np.unique(self.y))
        
        self.is_multiclass = self.n_classes > 2
        self.log_status(f"🎯 Target: {self.n_classes} clases")
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.log_status("📏 Escalado completado")
    
    def split_data(self, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        self.log_status(f"✅ Datos divididos: Train={len(self.X_train)}, Test={len(self.X_test)}")
    
    def train_and_visualize(self, n_epochs=100):
        self.log_status("🚀 Iniciando entrenamiento incremental de Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=0,
            warm_start=True,
            random_state=42,
            n_jobs=-1,
            oob_score=True  # Para obtener error OOB
        )
        self.model.n_estimators = 0
        self.model.classes_ = np.unique(self.y_train)
        
        train_accuracies = []
        test_accuracies = []
        train_losses = []
        test_losses = []
        
        for epoch in range(n_epochs):
            current_trees = self.model.n_estimators
            self.model.set_params(n_estimators=current_trees + self.n_estimators_per_epoch)
            self.model.fit(self.X_train, self.y_train)
            
            train_acc = accuracy_score(self.y_train, self.model.predict(self.X_train))
            test_acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
            train_loss = 1 - train_acc
            test_loss = 1 - test_acc
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            # Obtener importancia de características
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Obtener matriz de confusión actual
            cm = confusion_matrix(self.y_test, self.model.predict(self.X_test))
            
            # Obtener OOB score si está disponible
            oob = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
            
            # Extra data para gráficas personalizadas
            extra_data = {
                'n_estimators': self.model.n_estimators,
                'feature_importance': feature_importance,
                'oob_score': oob,
                'confusion_matrix': cm
            }
            
            if self.plot_callback:
                self.plot_callback(
                    epoch, n_epochs,
                    train_losses, test_losses,
                    train_accuracies, test_accuracies,
                    coef_history=None,
                    is_regression=False,
                    is_simple=False,
                    X_test=None, y_test=None, model=None,
                    extra_data=extra_data
                )
            
            if (epoch + 1) % 10 == 0:
                self.log_status(f"Epoch {epoch+1}/{n_epochs} - Árboles: {self.model.n_estimators} - Test Acc: {test_acc:.4f}")
        
        self.log_status("✅ Entrenamiento completado!")
        return {
            'best_epoch': np.argmax(test_accuracies) + 1,
            'best_accuracy': max(test_accuracies),
            'final_accuracy': test_accuracies[-1],
            'n_estimators_final': self.model.n_estimators
        }
    
    def get_feature_importance(self):
        if self.model and hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}
    
    def get_confusion_matrix(self):
        y_pred = self.model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self):
        y_pred = self.model.predict(self.X_test)
        return classification_report(self.y_test, y_pred, output_dict=True)
