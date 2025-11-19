"""
Script de Entrenamiento para Modelo LSTM
Plataforma Educativa ML - PASO 5: Deep Learning

Este script proporciona una interfaz CLI para:
1. Cargar datos de estudiantes
2. Crear secuencias temporales
3. Entrenar modelo LSTM
4. Evaluar desempeño
5. Guardar modelo entrenado
6. Generar reportes

Uso:
    python train_lstm.py --limit 100 --epochs 50 --batch-size 16
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_learning.models.lstm_predictor import LSTMPredictor
from deep_learning.data.sequence_loader import SequenceLoader
from shared.config import MODELS_DIR, DEBUG, TF_RANDOM_SEED
from shared.database.connection import get_db_connection

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    Gestor de entrenamiento para modelos LSTM.

    Flujo:
    1. Cargar datos de BD
    2. Crear secuencias
    3. Dividir datos
    4. Construir y entrenar LSTM
    5. Evaluar modelo
    6. Guardar resultados
    """

    def __init__(self,
                 lookback: int = 5,
                 lstm_units: int = 64,
                 dense_units: int = 32,
                 batch_size: int = 32,
                 epochs: int = 100):
        """
        Inicializar entrenador.

        Args:
            lookback (int): Número de pasos previos
            lstm_units (int): Unidades LSTM
            dense_units (int): Unidades Dense
            batch_size (int): Tamaño de batch
            epochs (int): Número de épocas
        """
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.batch_size = batch_size
        self.epochs = epochs

        self.sequence_loader = None
        self.lstm_model = None
        self.history = None

        logger.info("✓ LSTMTrainer inicializado")

    def load_data(self, limit: int = None) -> pd.DataFrame:
        """
        Cargar datos de estudiantes desde BD.

        Carga datos de rendimiento académico combinando información de
        rendimiento_academico y calificaciones individuales.

        Args:
            limit (int): Máximo número de registros (None = todos)

        Retorna:
            pd.DataFrame: Datos de estudiantes con calificaciones
        """
        try:
            logger.info("Cargando datos de BD...")

            db = get_db_connection()
            if db is None:
                raise ConnectionError("No se pudo conectar a la base de datos")

            # Query para obtener datos de estudiantes
            # Combinamos rendimiento_academico con calificaciones
            query = """
                SELECT
                    u.id as estudiante_id,
                    u.name as nombre,
                    c.fecha_calificacion as fecha,
                    c.puntaje as calificacion,
                    ra.promedio as promedio_histor,
                    ra.tendencia_temporal as tendencia,
                    u.created_at as fecha_registro
                FROM users u
                LEFT JOIN calificaciones c ON u.id = c.evaluador_id OR c.id > 0
                LEFT JOIN rendimiento_academico ra ON u.id = ra.estudiante_id
                WHERE c.puntaje IS NOT NULL
                ORDER BY u.id, c.fecha_calificacion
            """

            # Alternativa más simple si lo anterior no funciona bien
            simple_query = """
                SELECT
                    ra.estudiante_id,
                    u.name as nombre,
                    ra.promedio as calificacion,
                    ra.created_at as fecha,
                    ra.tendencia_temporal as tendencia
                FROM rendimiento_academico ra
                LEFT JOIN users u ON ra.estudiante_id = u.id
                WHERE ra.promedio IS NOT NULL
                ORDER BY ra.estudiante_id, ra.created_at
            """

            # Usar SQLAlchemy para ejecutar la query
            from sqlalchemy import text

            try:
                result = db.execute(text(query))
                rows = result.fetchall()
                if not rows:
                    logger.info("Intentando con query simplificada...")
                    result = db.execute(text(simple_query))
                    rows = result.fetchall()
            except Exception as e:
                logger.warning(f"Query principal falló, usando simplificada: {e}")
                result = db.execute(text(simple_query))
                rows = result.fetchall()

            # Convertir a DataFrame
            if rows:
                df = pd.DataFrame(rows, columns=result.keys())
                # Asegurar que tenemos al menos las columnas necesarias
                if 'estudiante_id' not in df.columns:
                    logger.error("No se encontró columna 'estudiante_id'")
                    df = pd.DataFrame()
            else:
                logger.warning("No hay datos disponibles en la BD")
                df = pd.DataFrame()

            db.close()

            if not df.empty:
                logger.info(f"✓ {len(df)} registros cargados")
                logger.info(f"  Estudiantes únicos: {df['estudiante_id'].nunique()}")
                if 'fecha' in df.columns:
                    logger.info(f"  Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
            else:
                logger.warning("No se pudieron cargar datos de la BD")

            return df

        except Exception as e:
            logger.error(f"✗ Error cargando datos: {str(e)}")
            logger.error(f"  Tipo: {type(e).__name__}")
            raise

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preparar secuencias desde datos.

        Args:
            df (pd.DataFrame): Datos de estudiantes

        Retorna:
            Tuple: (X, y, metadata)
        """
        try:
            logger.info("Preparando secuencias...")

            if df.empty:
                raise ValueError("DataFrame vacío")

            self.sequence_loader = SequenceLoader(lookback=self.lookback)

            # Usar solo las columnas que realmente existen en los datos
            # Verificar qué columnas tenemos
            logger.info(f"Columnas disponibles: {df.columns.tolist()}")

            # Features a usar - solo las que existan
            possible_features = [
                'calificacion',
                'asistencia',
                'participacion',
                'tareas_completadas',
                'promedio',
                'promedio_histor',
                'puntaje'
            ]

            feature_columns = [col for col in possible_features if col in df.columns]

            if not feature_columns:
                # Si no hay ningún feature numérico, usar la calificación
                if 'calificacion' in df.columns:
                    feature_columns = ['calificacion']
                else:
                    raise ValueError(f"No se encontraron columnas numéricas. Disponibles: {df.columns.tolist()}")

            logger.info(f"Usando features: {feature_columns}")

            # Crear secuencias
            X, y, metadata = self.sequence_loader.load_from_dataframe(
                df,
                feature_columns=feature_columns,
                groupby_column='estudiante_id',
                sort_column='fecha'
            )

            logger.info(f"✓ Secuencias preparadas")
            logger.info(f"  X shape: {X.shape}")
            logger.info(f"  y shape: {y.shape}")

            return X, y, metadata

        except Exception as e:
            logger.error(f"✗ Error preparando secuencias: {str(e)}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray, verbose: int = 1) -> Dict[str, Any]:
        """
        Entrenar modelo LSTM.

        Args:
            X (np.ndarray): Secuencias de entrada
            y (np.ndarray): Targets
            verbose (int): Verbosidad del entrenamiento

        Retorna:
            Dict: Resultados del entrenamiento
        """
        try:
            logger.info("Entrenando modelo LSTM...")

            # Dividir datos
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
                self.sequence_loader.split_data(X, y, test_size=0.2, val_size=0.1)

            # Crear modelo
            self.lstm_model = LSTMPredictor(
                lookback=self.lookback,
                lstm_units=self.lstm_units,
                dense_units=self.dense_units
            )

            # Entrenar
            metrics = self.lstm_model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=verbose
            )

            # Evaluar en test
            logger.info("\nEvaluando en test set...")
            test_predictions, _ = self.lstm_model.predict(X_test)
            test_mae = np.mean(np.abs(y_test - test_predictions.flatten()))
            test_rmse = np.sqrt(np.mean((y_test - test_predictions.flatten()) ** 2))

            metrics['test_mae'] = float(test_mae)
            metrics['test_rmse'] = float(test_rmse)

            logger.info(f"  Test MAE: {test_mae:.6f}")
            logger.info(f"  Test RMSE: {test_rmse:.6f}")

            # Detectar anomalías
            logger.info("\nDetectando anomalías...")
            anomalies = self.lstm_model.detect_anomalies(X_test, y_test)
            metrics['anomalies_detected'] = len(anomalies.get('anomaly_indices', []))

            if anomalies['detected']:
                logger.info(f"  {metrics['anomalies_detected']} anomalías detectadas")

            return metrics

        except Exception as e:
            logger.error(f"✗ Error durante entrenamiento: {str(e)}")
            raise

    def save_model(self, directory: str = None) -> str:
        """
        Guardar modelo entrenado.

        Args:
            directory (str): Directorio de guardado

        Retorna:
            str: Ruta del modelo guardado
        """
        try:
            if self.lstm_model is None:
                raise ValueError("No hay modelo para guardar")

            if directory is None:
                directory = os.path.join(MODELS_DIR, 'deep_learning')

            filepath = self.lstm_model.save(directory=directory)
            logger.info(f"✓ Modelo guardado en {filepath}")

            # Guardar también el scaler
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            import joblib
            joblib.dump(self.sequence_loader.get_scaler(), scaler_path)
            logger.info(f"✓ Scaler guardado en {scaler_path}")

            return filepath

        except Exception as e:
            logger.error(f"✗ Error guardando modelo: {str(e)}")
            raise

    def generate_report(self, metrics: Dict[str, Any], output_file: str = None) -> str:
        """
        Generar reporte de entrenamiento.

        Args:
            metrics (Dict): Métricas del entrenamiento
            output_file (str): Archivo de salida (opcional)

        Retorna:
            str: Reporte formateado
        """
        report = f"""
================================================================================
REPORTE DE ENTRENAMIENTO LSTM - PASO 5: DEEP LEARNING
================================================================================

Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURACIÓN DEL MODELO
------------------------
Lookback:           {self.lookback} pasos temporales
LSTM Units:         {self.lstm_units}
Dense Units:        {self.dense_units}
Batch Size:         {self.batch_size}
Épocas:             {self.epochs}

RESULTADOS DEL ENTRENAMIENTO
-----------------------------
Épocas entrenadas:  {metrics.get('epochs_trained', 'N/A')}
Mejor época:        {metrics.get('best_epoch', 'N/A')}
Loss final:         {metrics.get('final_loss', 'N/A'):.6f}
Val Loss final:     {metrics.get('final_val_loss', 'N/A'):.6f}
MAE final:          {metrics.get('final_mae', 'N/A'):.6f}

EVALUACIÓN EN TEST SET
---------------------
Test MAE:           {metrics.get('test_mae', 'N/A'):.6f}
Test RMSE:          {metrics.get('test_rmse', 'N/A'):.6f}

DETECCIÓN DE ANOMALÍAS
---------------------
Anomalías detectadas: {metrics.get('anomalies_detected', 0)}

RESUMEN
-------
Estado:             ENTRENAMIENTO COMPLETADO EXITOSAMENTE
Modelo:             LSTM (Long Short-Term Memory)
Aplicación:         Predicción de Desempeño Académico Temporal
Próximos pasos:     - Evaluar modelo en producción
                    - Integrar en pipeline ML
                    - Monitorear desempeño en tiempo real

================================================================================
"""

        logger.info(report)

        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write(report)
                logger.info(f"✓ Reporte guardado en {output_file}")
            except Exception as e:
                logger.warning(f"No se pudo guardar reporte: {str(e)}")

        return report

    def run_pipeline(self, limit: int = None, verbose: int = 1) -> Dict[str, Any]:
        """
        Ejecutar pipeline completo de entrenamiento.

        Args:
            limit (int): Máximo de registros a cargar
            verbose (int): Verbosidad

        Retorna:
            Dict: Resultados del pipeline
        """
        try:
            logger.info("INICIANDO PIPELINE DE ENTRENAMIENTO LSTM\n")

            # Paso 1: Cargar datos
            logger.info("PASO 1: Cargando datos...")
            df = self.load_data(limit=limit)

            # Paso 2: Preparar secuencias
            logger.info("\nPASO 2: Preparando secuencias...")
            X, y, metadata = self.prepare_sequences(df)

            # Paso 3: Entrenar
            logger.info("\nPASO 3: Entrenando modelo...")
            metrics = self.train(X, y, verbose=verbose)

            # Paso 4: Guardar modelo
            logger.info("\nPASO 4: Guardando modelo...")
            model_path = self.save_model()

            # Paso 5: Generar reporte
            logger.info("\nPASO 5: Generando reporte...")
            report_path = os.path.join(MODELS_DIR, 'deep_learning', 'training_report.txt')
            report = self.generate_report(metrics, report_path)

            results = {
                'success': True,
                'model_path': model_path,
                'metrics': metrics,
                'metadata': metadata,
                'report_path': report_path
            }

            logger.info("\n✓ PIPELINE COMPLETADO EXITOSAMENTE\n")
            return results

        except Exception as e:
            logger.error(f"\n✗ ERROR EN PIPELINE: {str(e)}\n")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    """Función principal con argumentos de CLI."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo LSTM para predicción temporal de desempeño'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Máximo número de registros a cargar (default: todos)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número de épocas de entrenamiento (default: 100)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamaño de batch (default: 32)'
    )

    parser.add_argument(
        '--lookback',
        type=int,
        default=5,
        help='Número de pasos temporales previos (default: 5)'
    )

    parser.add_argument(
        '--lstm-units',
        type=int,
        default=64,
        help='Número de unidades LSTM (default: 64)'
    )

    parser.add_argument(
        '--dense-units',
        type=int,
        default=32,
        help='Número de unidades Dense (default: 32)'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosidad del entrenamiento (0, 1, 2) (default: 1)'
    )

    args = parser.parse_args()

    # Crear entrenador
    trainer = LSTMTrainer(
        lookback=args.lookback,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Ejecutar pipeline
    results = trainer.run_pipeline(limit=args.limit, verbose=args.verbose)

    # Retornar código de salida
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()
