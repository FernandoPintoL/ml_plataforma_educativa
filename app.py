"""
API FastAPI para ML Educativa
Expone todos los modelos (supervisados, no supervisados, deep learning)
"""

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar configuración
try:
    from shared.config import (
        API_HOST,
        API_PORT,
        API_WORKERS,
        PROJECT_NAME,
        PROJECT_VERSION,
        DEBUG
    )
except ImportError as e:
    logger.error(f"Error cargando configuración: {e}")
    API_HOST = "0.0.0.0"
    API_PORT = 8001
    API_WORKERS = 4
    PROJECT_NAME = "Plataforma Educativa ML"
    PROJECT_VERSION = "2.0.0"
    DEBUG = False

# =====================================================
# CREAR APP
# =====================================================

app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description="API de Machine Learning para Plataforma Educativa",
    docs_url="/docs",
    redoc_url="/redoc",
)

# =====================================================
# MIDDLEWARE CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODELOS PYDANTIC
# =====================================================

class PredictionRequest(BaseModel):
    """Request para predicción de desempeño"""
    student_id: int
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "student_id": 1,
                "features": [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]
            }
        }

class RiskPredictionResponse(BaseModel):
    """Response para predicción de riesgo"""
    student_id: int
    risk_level: str
    risk_score: float
    status: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    debug: bool

class ClusterRequest(BaseModel):
    """Request para clustering"""
    students_data: List[List[float]]
    n_clusters: Optional[int] = 3

class AnomalyRequest(BaseModel):
    """Request para detección de anomalías"""
    student_data: List[float]

# =====================================================
# MODELOS ENTRENADOS (Lazy loading)
# =====================================================

models = {
    'performance_predictor': None,
    'career_recommender': None,
    'progress_analyzer': None,
    'trend_predictor': None,
    'kmeans_segmenter': None,
    'isolation_forest': None,
    'lstm_predictor': None,
}

def load_model(model_name: str):
    """Cargar modelo de forma lazy"""
    global models

    if models[model_name] is not None:
        return models[model_name]

    try:
        if model_name == 'performance_predictor':
            from supervisado.models.performance_predictor import PerformancePredictor
            models[model_name] = PerformancePredictor()
            logger.info(f"✓ {model_name} cargado")

        elif model_name == 'career_recommender':
            from supervisado.models.career_recommender import CareerRecommender
            models[model_name] = CareerRecommender()
            logger.info(f"✓ {model_name} cargado")

        elif model_name == 'progress_analyzer':
            from supervisado.models.progress_analyzer import ProgressAnalyzer
            models[model_name] = ProgressAnalyzer()
            logger.info(f"✓ {model_name} cargado")

        elif model_name == 'trend_predictor':
            from supervisado.models.trend_predictor import TrendPredictor
            models[model_name] = TrendPredictor()
            logger.info(f"✓ {model_name} cargado")

        elif model_name == 'kmeans_segmenter':
            from no_supervisado.models.kmeans_segmenter import KMeansSegmenter
            models[model_name] = KMeansSegmenter()
            logger.info(f"✓ {model_name} cargado")

        elif model_name == 'isolation_forest':
            from no_supervisado.models.isolation_forest_anomaly import IsolationForestAnomaly
            models[model_name] = IsolationForestAnomaly()
            logger.info(f"✓ {model_name} cargado")

        elif model_name == 'lstm_predictor':
            from deep_learning.models.lstm_predictor import LSTMPredictor
            models[model_name] = LSTMPredictor()
            logger.info(f"✓ {model_name} cargado")

        return models[model_name]

    except Exception as e:
        logger.error(f"Error cargando modelo {model_name}: {str(e)}")
        return None

# =====================================================
# RUTAS HEALTH CHECK
# =====================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check de la API"""
    return HealthResponse(
        status="healthy",
        service=PROJECT_NAME,
        version=PROJECT_VERSION,
        debug=DEBUG
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"API {PROJECT_NAME}",
        "version": PROJECT_VERSION,
        "docs": "/docs",
        "endpoints": {
            "supervisado": "/supervisado/",
            "no_supervisado": "/no-supervisado/",
            "deep_learning": "/deep-learning/",
        }
    }

# =====================================================
# RUTAS SUPERVISADO
# =====================================================

@app.post("/supervisado/performance/predict", response_model=RiskPredictionResponse)
async def predict_performance(request: PredictionRequest):
    """
    Predecir riesgo de bajo desempeño

    **Parámetros:**
    - student_id: ID del estudiante
    - features: Lista de 10 características

    **Retorna:**
    - risk_level: Alto, Medio o Bajo
    - risk_score: Probabilidad de riesgo (0-1)
    - status: critical, warning, ok
    """
    try:
        model = load_model('performance_predictor')
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo Performance Predictor no disponible"
            )

        if not model.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Modelo no entrenado"
            )

        X = np.array(request.features).reshape(1, -1)
        predictions = model.predict_risk_level(X)

        if not predictions:
            raise HTTPException(
                status_code=400,
                detail="Error en predicción"
            )

        pred = predictions[0]
        return RiskPredictionResponse(
            student_id=request.student_id,
            risk_level=pred['risk_level'],
            risk_score=pred['risk_score'],
            status=pred['status']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predict_performance: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno: {str(e)}"
        )

@app.post("/supervisado/performance/predict-batch")
async def predict_performance_batch(requests: List[PredictionRequest]):
    """
    Predicción en batch para múltiples estudiantes
    """
    try:
        model = load_model('performance_predictor')
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo no disponible"
            )

        if not model.is_trained:
            raise HTTPException(
                status_code=503,
                detail="Modelo no entrenado"
            )

        results = []
        for req in requests:
            X = np.array(req.features).reshape(1, -1)
            predictions = model.predict_risk_level(X)
            if predictions:
                pred = predictions[0]
                results.append({
                    "student_id": req.student_id,
                    "risk_level": pred['risk_level'],
                    "risk_score": pred['risk_score'],
                    "status": pred['status']
                })

        return {"predictions": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predict_performance_batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/supervisado/performance/model-info")
async def get_performance_model_info():
    """Obtener información del modelo"""
    try:
        model = load_model('performance_predictor')
        if model is None:
            raise HTTPException(status_code=503)

        return {
            "model_name": model.name,
            "model_type": model.model_type,
            "is_trained": model.is_trained,
            "metadata": model.metadata,
            "feature_importance": model.get_top_features(n=10) if model.is_trained else {}
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# RUTAS NO SUPERVISADO
# =====================================================

@app.post("/no-supervisado/clustering/predict")
async def predict_clustering(request: ClusterRequest):
    """
    Clustering de estudiantes con K-Means
    """
    try:
        model = load_model('kmeans_segmenter')
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo de clustering no disponible"
            )

        X = np.array(request.students_data)
        n_clusters = request.n_clusters or 3

        predictions = model.predict(X, n_clusters=n_clusters)

        return {
            "clusters": predictions.tolist(),
            "n_clusters": n_clusters,
            "n_samples": len(predictions)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en clustering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/no-supervisado/anomaly/detect")
async def detect_anomaly(request: AnomalyRequest):
    """
    Detectar anomalías en datos de estudiantes
    """
    try:
        model = load_model('isolation_forest')
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo de anomalía no disponible"
            )

        X = np.array(request.student_data).reshape(1, -1)
        anomaly_score = model.predict(X)

        return {
            "is_anomaly": int(anomaly_score[0]) == -1,
            "anomaly_score": float(anomaly_score[0]),
            "interpretation": "Anomalía detectada" if anomaly_score[0] == -1 else "Normal"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en detect_anomaly: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# RUTAS DEEP LEARNING
# =====================================================

@app.post("/deep-learning/lstm/predict")
async def predict_lstm(request: Dict):
    """
    Predicción con LSTM (series temporales)

    Requiere secuencia temporal de datos
    """
    try:
        model = load_model('lstm_predictor')
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo LSTM no disponible"
            )

        sequence = np.array(request.get('sequence', [])).reshape(1, -1, 1)
        prediction = model.predict(sequence)

        return {
            "prediction": float(prediction[0][0]) if prediction is not None else None,
            "model": "LSTM",
            "input_length": len(request.get('sequence', []))
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predict_lstm: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# MANEJO DE ERRORES
# =====================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador de excepciones HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error": True
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador general de excepciones"""
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno del servidor",
            "error": True
        }
    )

# =====================================================
# STARTUP/SHUTDOWN
# =====================================================

@app.on_event("startup")
async def startup():
    """Eventos al iniciar"""
    logger.info(f"Starting {PROJECT_NAME} v{PROJECT_VERSION}")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info("API pronta para recibir solicitudes")

@app.on_event("shutdown")
async def shutdown():
    """Eventos al cerrar"""
    logger.info("Shutting down API")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=int(API_PORT),
        workers=int(API_WORKERS),
        reload=DEBUG,
        log_level="info"
    )
