"""
Data Loader para Secuencias Temporales
Plataforma Educativa ML - PASO 5: Deep Learning

Este módulo convierte datos planos de estudiantes en secuencias temporales
para el entrenamiento de modelos LSTM. Incluye:

1. Carga de datos desde la base de datos
2. Creación de ventanas deslizantes (sliding windows)
3. Normalización de secuencias
4. Validación y limpieza de datos
5. División entrenamiento/validación/test
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SequenceLoader:
    """
    Cargador de secuencias temporales para modelos LSTM.

    Funcionalidad:
    - Convierte datos de estudiantes en secuencias temporales
    - Aplica normalización y validación
    - Maneja missing data
    - Genera train/val/test splits
    """

    def __init__(self, lookback: int = 5, lookahead: int = 1):
        """
        Inicializar cargador de secuencias.

        Args:
            lookback (int): Número de pasos previos a usar (default 5)
            lookahead (int): Número de pasos futuros a predecir (default 1)
        """
        self.lookback = lookback
        self.lookahead = lookahead
        self.scaler = StandardScaler()
        self.features = []
        self.n_features = 0
        self.is_fitted = False

        logger.info(f"✓ SequenceLoader inicializado (lookback={lookback}, lookahead={lookahead})")

    def create_sequences(self,
                        data: np.ndarray,
                        lookback: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crear secuencias con ventana deslizante.

        Args:
            data (np.ndarray): Array de datos (n_timesteps, n_features)
            lookback (int): Tamaño de ventana (usa default si None)

        Retorna:
            Tuple: (X, y) secuencias de entrada y targets
        """
        if lookback is None:
            lookback = self.lookback

        if len(data) < lookback + self.lookahead:
            logger.warning(f"Datos insuficientes para crear secuencias (necesario: {lookback + self.lookahead}, tenido: {len(data)})")
            return None, None

        X, y = [], []

        for i in range(len(data) - lookback - self.lookahead + 1):
            # Ventana de entrada
            X.append(data[i:i + lookback])
            # Target
            y.append(data[i + lookback:i + lookback + self.lookahead, 0])  # Usar primer feature como target

        return np.array(X), np.array(y)

    def load_from_dataframe(self,
                           df: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           groupby_column: str = 'estudiante_id',
                           sort_column: str = 'fecha') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Cargar secuencias desde un DataFrame.

        Args:
            df (pd.DataFrame): DataFrame con datos de estudiantes
            feature_columns (List[str]): Columnas a usar como features
            groupby_column (str): Columna para agrupar por estudiante
            sort_column (str): Columna para ordenar temporal

        Retorna:
            Tuple: (X_sequences, y, metadata)
        """
        try:
            # Usar todas las columnas numéricas si no se especifican features
            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if groupby_column in feature_columns:
                    feature_columns.remove(groupby_column)

            # Filtrar features que realmente existen en el DataFrame
            feature_columns = [col for col in feature_columns if col in df.columns]

            if not feature_columns:
                logger.warning("No hay features especificadas, usando todas las numéricas")
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if groupby_column in feature_columns:
                    feature_columns.remove(groupby_column)

            self.features = feature_columns
            self.n_features = len(feature_columns)

            logger.info(f"Usando {self.n_features} features: {feature_columns}")

            X_all = []
            y_all = []
            student_ids = []
            valid_students = 0
            skipped_students = 0

            # Procesar cada estudiante
            for student_id, group in df.groupby(groupby_column):
                try:
                    # Ordenar por tiempo
                    group = group.sort_values(sort_column)

                    # Necesita mínimo lookback + lookahead datos
                    if len(group) < self.lookback + self.lookahead:
                        skipped_students += 1
                        continue

                    # Extraer features
                    student_data = group[feature_columns].values

                    # Normalizar si es la primera vez
                    if not self.is_fitted:
                        student_data = self.scaler.fit_transform(student_data)
                    else:
                        student_data = self.scaler.transform(student_data)

                    # Crear secuencias
                    X_seq, y_seq = self.create_sequences(student_data)

                    if X_seq is not None and len(X_seq) > 0:
                        X_all.extend(X_seq)
                        y_all.extend(y_seq)
                        student_ids.extend([student_id] * len(X_seq))
                        valid_students += 1

                except Exception as e:
                    logger.warning(f"Error procesando estudiante {student_id}: {str(e)}")
                    skipped_students += 1
                    continue

            if not X_all:
                raise ValueError("No hay datos suficientes para crear secuencias")

            self.is_fitted = True

            X = np.array(X_all)
            y = np.array(y_all)

            metadata = {
                'n_samples': len(X),
                'n_features': self.n_features,
                'lookback': self.lookback,
                'lookahead': self.lookahead,
                'valid_students': valid_students,
                'skipped_students': skipped_students,
                'feature_names': self.features,
                'X_shape': X.shape,
                'y_shape': y.shape
            }

            logger.info(f"✓ Secuencias creadas exitosamente")
            logger.info(f"  Estudiantes válidos: {valid_students}")
            logger.info(f"  Estudiantes omitidos: {skipped_students}")
            logger.info(f"  Secuencias totales: {len(X)}")
            logger.info(f"  Shape X: {X.shape}")
            logger.info(f"  Shape y: {y.shape}")

            return X, y, metadata

        except Exception as e:
            logger.error(f"✗ Error cargando datos: {str(e)}")
            raise

    def split_data(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  test_size: float = 0.2,
                  val_size: float = 0.1,
                  random_state: int = 42) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Dividir datos en train/validation/test.

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            test_size (float): Proporción de test
            val_size (float): Proporción de validación (del training)
            random_state (int): Seed para reproducibilidad

        Retorna:
            Tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        try:
            # Primero separar test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

            # Luego separar validación del training
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state
            )

            metadata = {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test),
                'train_ratio': len(X_train) / len(X),
                'val_ratio': len(X_val) / len(X),
                'test_ratio': len(X_test) / len(X)
            }

            logger.info(f"✓ Datos divididos")
            logger.info(f"  Train: {len(X_train)} ({metadata['train_ratio']:.1%})")
            logger.info(f"  Val: {len(X_val)} ({metadata['val_ratio']:.1%})")
            logger.info(f"  Test: {len(X_test)} ({metadata['test_ratio']:.1%})")

            return (X_train, y_train), (X_val, y_val), (X_test, y_test)

        except Exception as e:
            logger.error(f"✗ Error dividiendo datos: {str(e)}")
            raise

    def create_evaluation_sequences(self,
                                   df: pd.DataFrame,
                                   feature_columns: Optional[List[str]] = None,
                                   groupby_column: str = 'estudiante_id',
                                   sort_column: str = 'fecha') -> Dict[str, Tuple]:
        """
        Crear secuencias para evaluación (una por estudiante).

        Args:
            df (pd.DataFrame): DataFrame con datos
            feature_columns (List[str]): Columnas a usar
            groupby_column (str): Columna de agrupación
            sort_column (str): Columna temporal

        Retorna:
            Dict: {student_id: (X_seq, metadata)}
        """
        try:
            if feature_columns is None:
                feature_columns = self.features

            sequences = {}

            for student_id, group in df.groupby(groupby_column):
                group = group.sort_values(sort_column)

                if len(group) < self.lookback:
                    continue

                # Últimas lookback observaciones
                student_data = group[feature_columns].tail(self.lookback).values
                student_data = self.scaler.transform(student_data)

                sequences[student_id] = {
                    'sequence': np.expand_dims(student_data, axis=0),
                    'last_date': group[sort_column].iloc[-1],
                    'n_points': len(group)
                }

            logger.info(f"✓ {len(sequences)} secuencias de evaluación creadas")
            return sequences

        except Exception as e:
            logger.error(f"✗ Error creando secuencias de evaluación: {str(e)}")
            raise

    def get_scaler(self) -> StandardScaler:
        """
        Obtener scaler usado.

        Retorna:
            StandardScaler: Scaler entrenado
        """
        return self.scaler

    def get_features(self) -> List[str]:
        """
        Obtener nombres de features.

        Retorna:
            List: Lista de nombres de features
        """
        return self.features

    def __repr__(self) -> str:
        return f"SequenceLoader(lookback={self.lookback}, lookahead={self.lookahead}, fitted={self.is_fitted})"
