# Sistema de Machine Learning - Plataforma Educativa
# Versión 2.0

"""
Sistema completo de Machine Learning para Plataforma Educativa

Organizado en 3 fases:
1. SUPERVISADO (Ahora - 70% esfuerzo)
   - Predictor de Desempeño
   - Recomendador de Carreras
   - Predicción de Tendencia
   - Análisis de Progreso

2. NO SUPERVISADO (Mes 3 - 20% esfuerzo)
   - K-Means Clustering
   - Isolation Forest (Anomalías)
   - Hierarchical Clustering
   - Collaborative Filtering

3. DEEP LEARNING (Mes 6+ - 10% esfuerzo)
   - LSTM (Análisis Temporal)
   - BERT/Transformer (NLP)
   - Autoencoder (Anomalías Avanzadas)

Uso:
    from ml_educativas.supervisado.models import PerformancePredictor
    from ml_educativas.no_supervisado.models import KMeansClustering
    from ml_educativas.deep_learning.models import LSTMModel

Configuración:
    Editar shared/config.py con variables de entorno

Documentación:
    Ver ML_CLASIFICACION_ALGORITMOS.md
    Ver STRUCTURE.md
"""

__version__ = "2.0.0"
__author__ = "Plataforma Educativa"
__license__ = "MIT"

# Configuración global
try:
    from shared.config import (
        PROJECT_NAME,
        PROJECT_VERSION,
        DATABASE_URL,
        DEBUG,
        LOG_LEVEL
    )
except ImportError:
    # Fallback si no se puede importar
    PROJECT_NAME = "Plataforma Educativa ML"
    PROJECT_VERSION = "2.0.0"
    DATABASE_URL = "postgresql://localhost/educativa_db"
    DEBUG = True
    LOG_LEVEL = "INFO"
