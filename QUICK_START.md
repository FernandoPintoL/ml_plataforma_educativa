# ⚡ Quick Start - ML Educativa a Railway

> 5 minutos para ir de 0 a producción

## 🎯 Tu Situación

✅ Ya tienes `plataforma-educativa` corriendo en Railway
❌ Necesitas desplegar `ml_educativas` de forma separada
🎯 Objetivo: API de ML en Railway comunicándose con Laravel

## 📦 Lo Que Hemos Preparado

Todo está listo para Railway. Solo necesitas:

```
✅ app.py              - API completamente funcional
✅ Dockerfile          - Container listo para producción
✅ docker-compose.yml  - Para testing local
✅ Variables de entorno - Configuradas correctamente
✅ Health checks       - Incluidos
✅ CORS               - Habilitado para tu Laravel
```

## 🚀 Pasos Rápidos

### 1️⃣ GitHub (2 minutos)

```bash
cd ml_educativas
git init
git add .
git commit -m "ML API ready for Railway"
git remote add origin https://github.com/tu-usuario/ml_educativas.git
git push -u origin main
```

### 2️⃣ Railway (3 minutos)

1. Ir a https://railway.app → "Create New Project"
2. Seleccionar "Deploy from GitHub repo"
3. Conectar a tu repositorio `ml_educativas`
4. Railway detectará el Dockerfile automáticamente
5. Esperar a que termine el build

### 3️⃣ Configurar Variables (1 minuto)

En Railway Dashboard, en "Variables" del servicio, añadir:

```
ENVIRONMENT=production
DEBUG=False
API_RELOAD=False
API_WORKERS=4
LOG_LEVEL=INFO
SECRET_KEY=tu-clave-super-secreta-aqui
```

### 4️⃣ Verificar (1 minuto)

```bash
# Reemplazar con tu URL de Railway
curl https://tu-ml-api.railway.app/health

# Respuesta esperada:
# {"status":"healthy","service":"Plataforma Educativa ML","version":"2.0.0","debug":false}
```

## 📍 Conectar desde Laravel

En tu código Laravel:

```php
// config/services.php
'ml_api' => [
    'url' => env('ML_API_URL', 'http://ml-api:8001'),
],

// Controller
use Illuminate\Support\Facades\Http;

$response = Http::post(config('services.ml_api.url') . '/supervisado/performance/predict', [
    'student_id' => $student->id,
    'features' => [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]
]);

$prediction = $response->json();
```

En `.env` de Laravel:
```
ML_API_URL=https://tu-ml-api.railway.app
```

## 🧪 Testing Local Antes de Railway (OPCIONAL)

Si quieres probar antes de subir:

### Sin Docker:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8001

# Ir a http://localhost:8001/docs para ver la API
```

### Con Docker:
```bash
docker-compose up -d
curl http://localhost:8001/health
docker-compose down
```

## 📊 Endpoints Disponibles

```
GET  /health                           - Health check
GET  /                                 - Info API

POST /supervisado/performance/predict          - Predicción individual
POST /supervisado/performance/predict-batch    - Batch de predicciones
GET  /supervisado/performance/model-info       - Info del modelo

POST /no-supervisado/clustering/predict        - Clustering K-Means
POST /no-supervisado/anomaly/detect            - Anomalías

POST /deep-learning/lstm/predict               - LSTM predictions
```

Documentación interactiva: `https://tu-ml-api.railway.app/docs`

## ⚠️ Cosas Importantes

### NO Commitear
```
❌ .env (solo .env.example)
❌ venv/ (está en .gitignore)
❌ __pycache__/ (está en .gitignore)
❌ *.log (está en .gitignore)
```

### Cambiar Antes de Producción
```
🔑 SECRET_KEY          (variable aleatoria/segura)
🔓 DEBUG=False         (siempre en producción)
🔄 API_RELOAD=False    (siempre en producción)
```

### Credenciales BD
Railway proporciona automáticamente `DATABASE_URL` cuando añades PostgreSQL.
No necesitas configurarla manualmente.

## 🆘 Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| "Modelo no disponible" | Revisar que `trained_models/` tenga los archivos |
| "Database connection failed" | Asegurar que DATABASE_URL está en variables |
| "Port already in use" | Cambiar puerto en `uvicorn` |
| "Out of memory" | Reducir `API_WORKERS` |
| "Build timeout" | Usar `requirements-prod.txt` sin desarrollo |

## 📚 Más Información

- **DEPLOYMENT.md** - Guía completa (25 minutos de lectura)
- **SETUP_SUMMARY.md** - Resumen técnico detallado
- **/docs** - Documentación automática de API (Swagger)

## ✅ Checklist Final

- [ ] Código en GitHub
- [ ] Railway proyecto creado
- [ ] Variables de entorno configuradas
- [ ] Health check responde
- [ ] API accesible desde Internet
- [ ] Laravel conecta correctamente

## 🎉 ¡Listo!

Tu ML API está en producción y comunicándose con tu plataforma educativa.

---

**Preguntas frecuentes:**

Q: ¿Puedo usar la misma BD que Laravel?
R: Sí, pero separadas es mejor. Railway permite múltiples servicios.

Q: ¿Cuánto cuesta?
R: Railway tiene tier gratuito. Luego $5/mes aproximadamente.

Q: ¿Cómo actualizar modelos?
R: Push a GitHub → Railway redeploy automático.

Q: ¿Cómo escalar?
R: Railway permite aumentar CPU/RAM del servicio desde Dashboard.
