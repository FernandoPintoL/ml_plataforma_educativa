"""
Modelo LSTM para Predicciones Temporales de Desempeño Académico
Plataforma Educativa ML - PASO 5: Deep Learning

Este módulo implementa un modelo LSTM (Long Short-Term Memory) para:
1. Análisis temporal de secuencias de notas estudiantiles
2. Predicción de desempeño futuro basado en patrones históricos
3. Detección de anomalías temporales y cambios de tendencia
4. Proyecciones de calificaciones finales

Arquitectura:
    Input (secuencia temporal) → LSTM → Dropout → Dense → Output
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib

from shared.config import MODELS_DIR, DEBUG, TF_RANDOM_SEED, USE_GPU

# Configurar logger
logger = logging.getLogger(__name__)

# Configurar TensorFlow
tf.random.set_seed(TF_RANDOM_SEED)
np.random.seed(TF_RANDOM_SEED)


class LSTMPredictor:
    """
    Predictor LSTM para análisis temporal de desempeño académico.

    Características principales:
    - Análisis de secuencias temporales de calificaciones
    - Predicción de desempeño futuro (próximas 1-4 semanas)
    - Detección de anomalías y cambios de tendencia
    - Manejo de dropout para regularización
    - Early stopping para evitar overfitting
    """

    def __init__(self,
                 lookback: int = 5,
                 lstm_units: int = 64,
                 dense_units: int = 32,
                 dropout_rate: float = 0.2,
                 forecast_horizon: int = 1,
                 name: str = "lstm_predictor"):
        """
        Inicializar predictor LSTM.

        Args:
            lookback (int): Número de pasos temporales previos a usar (default 5)
            lstm_units (int): Número de unidades LSTM (default 64)
            dense_units (int): Número de unidades en capa Dense (default 32)
            dropout_rate (float): Tasa de dropout (default 0.2)
            forecast_horizon (int): Períodos a predecir (default 1)
            name (str): Nombre del modelo
        """
        self.name = name
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon

        self.model = None
        self.scaler = None
        self.is_trained = False
        self.history = None
        self.features = []

        self.metadata = {
            'name': name,
            'model_type': 'deep_learning',
            'created_at': datetime.now().isoformat(),
            'trained': False,
            'trained_at': None,
            'metrics': {},
            'architecture': {
                'lookback': lookback,
                'lstm_units': lstm_units,
                'dense_units': dense_units,
                'dropout_rate': dropout_rate,
                'forecast_horizon': forecast_horizon
            }
        }

        logger.info(f"✓ {self.name} inicializado (lookback={lookback})")

    def build_model(self, n_features: int) -> None:
        """
        Construir arquitectura del modelo LSTM.

        Args:
            n_features (int): Número de features de entrada
        """
        try:
            self.model = Sequential([
                # Entrada: (None, lookback, n_features)
                layers.LSTM(
                    self.lstm_units,
                    activation='relu',
                    return_sequences=False,
                    input_shape=(self.lookback, n_features)
                ),

                # Dropout para regularización
                layers.Dropout(self.dropout_rate),

                # Capa densa intermedia
                layers.Dense(self.dense_units, activation='relu'),
                layers.Dropout(self.dropout_rate),

                # Capas de salida adicionales para mejor aprendizaje
                layers.Dense(16, activation='relu'),

                # Salida: predicción de siguiente valor
                layers.Dense(self.forecast_horizon)
            ])

            # Compilar modelo
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mse']
            )

            logger.info(f"✓ Arquitectura LSTM construida: {self.model.summary()}")

        except Exception as e:
            logger.error(f"✗ Error construyendo modelo: {str(e)}")
            raise

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 5,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Entrenar el modelo LSTM.

        Args:
            X_train (np.ndarray): Secuencias de entrenamiento (n_samples, lookback, n_features)
            y_train (np.ndarray): Targets de entrenamiento (n_samples, forecast_horizon)
            X_val (np.ndarray): Secuencias de validación (opcional)
            y_val (np.ndarray): Targets de validación (opcional)
            epochs (int): Número de épocas (default 100)
            batch_size (int): Tamaño del batch (default 32)
            patience (int): Paciencia para early stopping (default 5)
            verbose (int): Verbosidad del entrenamiento (0, 1, 2)

        Retorna:
            Dict: Información de entrenamiento con métricas
        """
        try:
            if self.model is None:
                self.build_model(X_train.shape[2])

            logger.info(f"Iniciando entrenamiento LSTM...")
            logger.info(f"  X_train shape: {X_train.shape}")
            logger.info(f"  y_train shape: {y_train.shape}")

            if X_val is not None:
                validation_data = (X_val, y_val)
                logger.info(f"  X_val shape: {X_val.shape}")
            else:
                validation_data = None
                logger.info("  Sin validación externa (usará split interno)")

            # Callbacks
            early_stop = EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            )

            callbacks = [early_stop, reduce_lr]

            # Entrenar
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data if validation_data else None,
                validation_split=0.2 if validation_data is None else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )

            self.is_trained = True
            self.metadata['trained'] = True
            self.metadata['trained_at'] = datetime.now().isoformat()

            # Extraer métricas finales
            final_loss = self.history.history['loss'][-1]
            final_val_loss = self.history.history.get('val_loss', [final_loss])[-1]
            final_mae = self.history.history.get('mae', [0])[-1]

            metrics = {
                'final_loss': float(final_loss),
                'final_val_loss': float(final_val_loss),
                'final_mae': float(final_mae),
                'epochs_trained': len(self.history.history['loss']),
                'best_epoch': np.argmin(self.history.history.get('val_loss', self.history.history['loss']))
            }

            self.metadata['metrics'] = metrics

            logger.info(f"✓ Entrenamiento completado")
            logger.info(f"  Épocas entrenadas: {metrics['epochs_trained']}")
            logger.info(f"  Mejor época: {metrics['best_epoch']}")
            logger.info(f"  Loss final: {final_loss:.6f}")
            logger.info(f"  Val Loss final: {final_val_loss:.6f}")

            return metrics

        except Exception as e:
            logger.error(f"✗ Error durante entrenamiento: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realizar predicciones con intervalo de confianza.

        Args:
            X (np.ndarray): Secuencias de entrada (n_samples, lookback, n_features)

        Retorna:
            Tuple: (predicciones, intervalos_confianza)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo no está entrenado")

        try:
            # Predicción puntual
            predictions = self.model.predict(X, verbose=0)

            # Calcular intervalos de confianza basados en varianza histórica
            std_dev = np.std(X, axis=(0, 1))
            confidence_interval = 1.96 * std_dev / np.sqrt(X.shape[0])

            return predictions, confidence_interval

        except Exception as e:
            logger.error(f"✗ Error en predicción: {str(e)}")
            raise

    def detect_anomalies(self,
                        X: np.ndarray,
                        y_true: Optional[np.ndarray] = None,
                        threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detectar anomalías en secuencias temporales.

        Args:
            X (np.ndarray): Secuencias de entrada
            y_true (np.ndarray): Valores reales (opcional)
            threshold (float): Umbral de desviación estándar (default 2.0)

        Retorna:
            Dict: Información de anomalías detectadas
        """
        if not self.is_trained:
            raise ValueError("Modelo no está entrenado")

        try:
            predictions, _ = self.predict(X)

            anomalies = {
                'detected': False,
                'anomaly_indices': [],
                'anomaly_scores': [],
                'details': []
            }

            if y_true is not None:
                # Calcular errores de predicción
                errors = np.abs(y_true - predictions.flatten())
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                # Detectar anomalías
                anomaly_threshold = mean_error + (threshold * std_error)
                anomaly_mask = errors > anomaly_threshold

                if np.any(anomaly_mask):
                    anomalies['detected'] = True
                    anomalies['anomaly_indices'] = np.where(anomaly_mask)[0].tolist()
                    anomalies['anomaly_scores'] = errors[anomaly_mask].tolist()

                    for idx in anomalies['anomaly_indices']:
                        anomalies['details'].append({
                            'index': int(idx),
                            'error': float(errors[idx]),
                            'threshold': float(anomaly_threshold),
                            'severity': 'alta' if errors[idx] > (2 * anomaly_threshold) else 'media'
                        })

                    logger.info(f"✓ {len(anomalies['anomaly_indices'])} anomalías detectadas")

            return anomalies

        except Exception as e:
            logger.error(f"✗ Error detectando anomalías: {str(e)}")
            return {'detected': False, 'error': str(e)}

    def save(self, filename: Optional[str] = None, directory: Optional[str] = None) -> str:
        """
        Guardar modelo entrenado.

        Args:
            filename (str): Nombre del archivo (default: {name}_model.h5)
            directory (str): Directorio donde guardar (default: MODELS_DIR)

        Retorna:
            str: Ruta del archivo guardado
        """
        if not self.is_trained:
            logger.warning(f"{self.name} no está entrenado")

        if filename is None:
            filename = f"{self.name}_model.h5"
        if directory is None:
            directory = os.path.join(MODELS_DIR, 'deep_learning')

        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        try:
            # Guardar modelo Keras
            self.model.save(filepath)

            # Guardar metadata y scaler
            metadata_file = filepath.replace('.h5', '_metadata.pkl')
            joblib.dump({
                'metadata': self.metadata,
                'scaler': self.scaler,
                'features': self.features
            }, metadata_file)

            logger.info(f"✓ Modelo guardado en {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"✗ Error guardando modelo: {str(e)}")
            raise

    def load(self, filepath: str) -> bool:
        """
        Cargar modelo entrenado.

        Args:
            filepath (str): Ruta al archivo del modelo

        Retorna:
            bool: True si se cargó exitosamente
        """
        try:
            # Cargar modelo Keras
            self.model = keras.models.load_model(filepath)

            # Cargar metadata
            metadata_file = filepath.replace('.h5', '_metadata.pkl')
            if os.path.exists(metadata_file):
                data = joblib.load(metadata_file)
                self.metadata = data.get('metadata', {})
                self.scaler = data.get('scaler', None)
                self.features = data.get('features', [])

            self.is_trained = True
            logger.info(f"✓ Modelo cargado desde {filepath}")
            return True

        except Exception as e:
            logger.error(f"✗ Error cargando modelo: {str(e)}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """
        Obtener metadata del modelo.

        Retorna:
            Dict: Información del modelo
        """
        return self.metadata

    def summary(self) -> str:
        """
        Obtener resumen del modelo.

        Retorna:
            str: Resumen de arquitectura
        """
        if self.model is None:
            return "Modelo no construido"

        return self.model.summary()

    def __repr__(self) -> str:
        status = "entrenado" if self.is_trained else "sin entrenar"
        return f"{self.__class__.__name__}(name={self.name}, {status}, lookback={self.lookback})"
