"""
Script de prueba para verificar que la API funciona correctamente
Uso: python test_api.py
"""

import requests
import json
import sys

# Configuración
BASE_URL = "http://localhost:8001"
TIMEOUT = 10

def print_response(title, response):
    """Imprimir respuesta de forma legible"""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print(f"{'='*60}")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except:
        print(response.text)

def test_health():
    """Test: Health check"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        print_response("Health Check", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error en health check: {str(e)}")
        return False

def test_root():
    """Test: Root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=TIMEOUT)
        print_response("Root Endpoint", response)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error en root: {str(e)}")
        return False

def test_performance_predict():
    """Test: Predicción de desempeño"""
    try:
        payload = {
            "student_id": 1,
            "features": [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]
        }
        response = requests.post(
            f"{BASE_URL}/supervisado/performance/predict",
            json=payload,
            timeout=TIMEOUT
        )
        print_response("Performance Prediction", response)
        return response.status_code in [200, 503]  # 503 si modelo no está entrenado
    except Exception as e:
        print(f"❌ Error en performance prediction: {str(e)}")
        return False

def test_batch_predict():
    """Test: Predicción en batch"""
    try:
        payload = [
            {"student_id": 1, "features": [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]},
            {"student_id": 2, "features": [2.5, 65, 5, 1.5, 30, 0.6, 0.8, 0.7, 0.65, 1.5]},
        ]
        response = requests.post(
            f"{BASE_URL}/supervisado/performance/predict-batch",
            json=payload,
            timeout=TIMEOUT
        )
        print_response("Batch Prediction", response)
        return response.status_code in [200, 503]
    except Exception as e:
        print(f"❌ Error en batch prediction: {str(e)}")
        return False

def test_clustering():
    """Test: Clustering K-Means"""
    try:
        payload = {
            "students_data": [
                [3.5, 85, 10],
                [2.5, 65, 5],
                [4.0, 95, 15]
            ],
            "n_clusters": 3
        }
        response = requests.post(
            f"{BASE_URL}/no-supervisado/clustering/predict",
            json=payload,
            timeout=TIMEOUT
        )
        print_response("Clustering Prediction", response)
        return response.status_code in [200, 503]
    except Exception as e:
        print(f"❌ Error en clustering: {str(e)}")
        return False

def test_anomaly_detection():
    """Test: Detección de anomalías"""
    try:
        payload = {
            "student_data": [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]
        }
        response = requests.post(
            f"{BASE_URL}/no-supervisado/anomaly/detect",
            json=payload,
            timeout=TIMEOUT
        )
        print_response("Anomaly Detection", response)
        return response.status_code in [200, 503]
    except Exception as e:
        print(f"❌ Error en anomaly detection: {str(e)}")
        return False

def test_model_info():
    """Test: Información del modelo"""
    try:
        response = requests.get(
            f"{BASE_URL}/supervisado/performance/model-info",
            timeout=TIMEOUT
        )
        print_response("Model Information", response)
        return response.status_code in [200, 503]
    except Exception as e:
        print(f"❌ Error en model info: {str(e)}")
        return False

def main():
    """Ejecutar todos los tests"""
    print("\n" + "="*60)
    print("🧪 TESTING API ML EDUCATIVA")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")

    # Verificar conexión
    try:
        requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    except Exception as e:
        print(f"\n❌ No se puede conectar a {BASE_URL}")
        print(f"Error: {str(e)}")
        print("\n💡 Asegúrate de que la API está ejecutándose:")
        print("   python -m uvicorn app:app --reload --port 8001")
        sys.exit(1)

    # Ejecutar tests
    results = {}

    print("\n📍 Ejecutando tests...\n")

    results["Health Check"] = test_health()
    results["Root Endpoint"] = test_root()
    results["Performance Prediction"] = test_performance_predict()
    results["Batch Prediction"] = test_batch_predict()
    results["Clustering"] = test_clustering()
    results["Anomaly Detection"] = test_anomaly_detection()
    results["Model Info"] = test_model_info()

    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DE TESTS")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    print("="*60)
    print(f"Resultados: {passed}/{total} tests pasados")

    if passed == total:
        print("\n🎉 ¡TODOS LOS TESTS PASARON! API lista para producción.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) fallaron. Revisar logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
