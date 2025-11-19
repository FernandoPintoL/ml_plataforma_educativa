# ❓ Preguntas Frecuentes (FAQ)

## General

### ¿Realmente necesito tener dos servicios separados?

**Sí**, por estas razones:

| Aspecto | Monolítico | Separado |
|---------|-----------|----------|
| Si ML falla | ❌ Derriba todo | ✅ Laravel sigue activo |
| Escalabilidad | ❌ Escalar todo | ✅ Escalar solo ML |
| Actualizaciones | ❌ Todo juntos | ✅ Independiente |
| Recursos | ❌ Más caros | ✅ Optimizados |
| Debugging | ❌ Más complejo | ✅ Aislado |

En producción, si los modelos consumen mucha memoria, puedes tener un servicio de ML escalado sin afectar Laravel.

### ¿Cuándo usar monolítico?

Solo si:
- Proyecto muy pequeño (< 50 estudiantes)
- Modelos muy simples
- Sin requisitos de escalabilidad
- Equipo de 1-2 personas

Mejor: Hacer separados desde el inicio.

---

## Deployment en Railway

### ¿Cuánto cuesta en Railway?

| Recurso | Costo |
|---------|-------|
| Primer mes | $5 crédito gratis |
| Desarrollo | $0 (tier gratuito limitado) |
| Producción | ~$5-20/mes por servicio |
| Cada 500GB/mes | $0.50 |

**Nuestro caso típico:**
- 1 vCPU + 512MB RAM = $5/mes
- PostgreSQL = $15/mes (opcional)
- Total = ~$20/mes para ambos servicios

Mucho más barato que tener servidor propio.

### ¿Railway redeploy automáticamente en git push?

**Sí**, es automático:

```
Tu push a GitHub → Railway detecta → Build → Deploy
```

No necesitas hacer nada manualmente. Tarda ~3-5 minutos.

### ¿Puedo probar antes de subir a Railway?

**Sí**, dos opciones:

**Opción 1: Local rápido (recomendado)**
```bash
python -m venv venv
pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8001
# Ir a http://localhost:8001/docs
```

**Opción 2: Local con Docker**
```bash
docker-compose up -d
curl http://localhost:8001/health
docker-compose down
```

Opción 1 es más rápida (1 minuto vs 5 minutos).

### ¿Qué pasa si mi app crashea en Railway?

Railway:
1. Detecta el crash
2. Intenta reiniciar automáticamente
3. Si sigue fallando, muestra error
4. Puedes ver logs en Dashboard

**Rollback manual:**
```
Dashboard → Deployments → Click en previous → Redeploy
```

---

## Integración con Laravel

### ¿Cómo conectar Laravel a la API de ML?

```php
// config/services.php
'ml_api' => [
    'url' => env('ML_API_URL', 'http://localhost:8001'),
],

// En tu controller
use Illuminate\Support\Facades\Http;

$response = Http::post(
    config('services.ml_api.url') . '/supervisado/performance/predict',
    [
        'student_id' => $student->id,
        'features' => [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]
    ]
);

if ($response->successful()) {
    $data = $response->json();
    // $data['risk_level'], $data['risk_score'], etc
}
```

En `.env`:
```
ML_API_URL=https://tu-ml-api.railway.app
```

### ¿Necesito autenticación entre servicios?

**Para desarrollo:** No necesitas

**Para producción:** Recomendado
- Token JWT simple
- API Key
- mTLS (complicado, no lo hagas)

Implementar:
```python
# En app.py
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/supervisado/performance/predict")
async def predict_performance(
    request: PredictionRequest,
    credentials: HTTPAuthCredentials = Depends(security)
):
    # Validar token
    # ...
```

En Laravel:
```php
$response = Http::withToken($token)->post(...)
```

---

## Modelos ML

### ¿Dónde van los modelos entrenados?

En la carpeta `trained_models/`:
```
trained_models/
├── performance_predictor.pkl    (Modelo Random Forest)
├── xgboost_model.pkl            (Modelo XGBoost)
├── kmeans_model.pkl             (Modelo K-Means)
├── lstm_model.h5                (Modelo LSTM)
└── ...
```

Los modelos deben estar entrenados **antes** de desplegar.

### ¿Puedo actualizar modelos sin redeploy?

**Opción 1: Con redeploy (recomendado)**
```bash
# Reentrenar localmente
python supervisado/training/train_performance.py

# Guardar en trained_models/
# Push a GitHub
git add trained_models/
git commit -m "Updated models"
git push

# Railway redeploy automático
```

**Opción 2: Sin redeploy (avanzado)**
- Guardar modelos en base de datos
- API carga desde BD
- Requiere cambios en `app.py`

Opción 1 es más simple.

### ¿Qué pasa si el modelo es muy grande?

Si el modelo > 200MB:

1. **Comprimirlo:**
   ```bash
   # Guardar sin compresión
   joblib.dump(model, 'model.pkl', compress=0)

   # Guardar con compresión
   joblib.dump(model, 'model.pkl', compress=3)
   ```

2. **Usar S3 en lugar de git:**
   - Guardar en AWS S3
   - App descarga en startup
   - Más complejo

3. **Segmentar el modelo:**
   - Splits en múltiples archivos
   - Cargar en paralelo

Para la mayoría de casos, compresión level 3 es suficiente.

---

## Problemas Comunes

### Error: "Modelo no está entrenado"

**Causa:** El archivo `.pkl` o `.h5` no existe o está corrupto

**Solución:**
```bash
# Reentrenar modelo
cd supervisado/training
python train_performance.py

# Verificar archivo
ls -lh ../models/trained_models/
```

### Error: "Database connection failed"

**Causa:** DATABASE_URL inválida o BD no accesible

**Solución en Railway:**
1. Ir a Dashboard
2. Verificar "Variables" tiene DATABASE_URL
3. Probar conexión:
   ```bash
   psql $DATABASE_URL -c "SELECT 1"
   ```

### Error: "Port 8001 already in use"

**En desarrollo local:**
```bash
# Usar otro puerto
python -m uvicorn app:app --port 8002

# O matar proceso
lsof -i :8001
kill -9 <PID>
```

**En Railway:** Automático, no hay problema

### Error: "Out of memory"

**Causas:**
- Modelos muy grandes (TensorFlow/PyTorch)
- Batch size muy grande
- Memory leak en código

**Soluciones:**
1. Aumentar RAM del servicio en Railway
2. Reducir batch size
3. Usar cuantización en modelos
4. Usar `requirements-prod.txt` sin heavy deps

---

## Performance y Optimización

### ¿Qué tan rápido es la API?

Tiempos típicos (desde cliente):

```
Health check:           ~50ms
Predicción simple:      100-500ms (depende modelo)
Batch 10 predicciones:  200-1000ms
Clustering:             500-2000ms (depende datos)
```

Railway ≈ 50-100ms de latencia adicional por ubicación.

### ¿Cómo cachear predicciones?

```python
# En app.py con Redis
from redis import Redis

redis_client = Redis.from_url(REDIS_URL)

@app.post("/supervisado/performance/predict")
async def predict_performance(request: PredictionRequest):
    # Generar cache key
    cache_key = f"pred:{request.student_id}"

    # Intentar obtener del cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Sino, calcular
    prediction = model.predict(...)

    # Guardar en cache por 1 hora
    redis_client.setex(cache_key, 3600, json.dumps(prediction))

    return prediction
```

### ¿Cómo monitorear performance?

Railway proporciona:
- 📊 CPU/RAM/Network en Dashboard
- 📝 Logs en tiempo real
- 🔔 Alertas configurables

Para más detalle:
```python
# Usar time tracking
import time

@app.post("/supervisado/performance/predict")
async def predict_performance(request: PredictionRequest):
    start = time.time()
    result = model.predict(...)
    duration = time.time() - start

    logger.info(f"Prediction took {duration:.3f}s")

    return result
```

---

## Seguridad

### ¿Debo exponer todos los endpoints?

**En desarrollo:** Sí, es útil para testing

**En producción:** Considera:
- `/docs` - Documenta toda tu API (útil, pero puede leakear info)
- `/health` - Debe estar público (monitoreo)
- Endpoints de modelos - Proteger con auth si es sensible

Para deshabilitar docs:
```python
app = FastAPI(docs_url=None, redoc_url=None)
```

### ¿Cómo proteger con JWT?

```python
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from jose import JWTError, jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/supervisado/performance/predict")
async def predict_performance(
    request: PredictionRequest,
    token_data = Depends(verify_token)
):
    # Token validado, continuar
    ...
```

---

## Maintenance

### ¿Cómo actualizar dependencias?

```bash
# Ver actualizaciones disponibles
pip list --outdated

# Actualizar específico
pip install --upgrade scikit-learn

# Actualizar todo (cuidado!)
pip install --upgrade -r requirements.txt

# Verificar cambios
python -m uvicorn app:app --reload

# Commit y push
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### ¿Cómo hacer backup de modelos?

```bash
# Opción 1: GitHub (si < 100MB)
git add trained_models/
git commit -m "Backup models"
git push

# Opción 2: AWS S3
aws s3 cp trained_models/ s3://mi-bucket/ --recursive

# Opción 3: Manual desde Railway
# Dashboard → archivos → descargar
```

### ¿Cada cuánto actualizar modelos?

Depende:
- **Semanal**: Si hay datos nuevos y cambios importantes
- **Mensual**: Reentrenamiento estándar
- **Trimestral**: Performance review

Recomendación: **Semanal o mensual** para educación (datos cambian frecuentemente).

---

## Testing

### ¿Cómo probar endpoints?

**Opción 1: test_api.py (rápido)**
```bash
python test_api.py
```

**Opción 2: curl**
```bash
curl -X POST "http://localhost:8001/supervisado/performance/predict" \
  -H "Content-Type: application/json" \
  -d '{"student_id":1,"features":[3.5,85,10,2.1,45,0.8,1.2,0.9,0.85,2.0]}'
```

**Opción 3: /docs (interfaz)**
```
http://localhost:8001/docs
```

**Opción 4: Postman**
- Importar endpoint
- Guardar en Postman Cloud
- Compartir con equipo

---

## Soporte

### ¿Dónde reporto bugs?

1. **En desarrollo:**
   - Ver logs: `docker-compose logs ml-api`
   - Ejecutar `test_api.py`
   - Revisar `DEPLOYMENT.md` troubleshooting

2. **En Railway:**
   - Ir a Dashboard → Logs
   - Ver histórico de deployments
   - Revisar variables de entorno

3. **Comunidad:**
   - Railway Docs: https://docs.railway.app
   - FastAPI Docs: https://fastapi.tiangolo.com
   - Stack Overflow

---

## Referencia Rápida

| Comando | Uso |
|---------|-----|
| `python -m uvicorn app:app --reload` | Desarrollo local |
| `docker-compose up -d` | Testing con Docker |
| `python test_api.py` | Probar endpoints |
| `curl http://localhost:8001/health` | Health check |
| `http://localhost:8001/docs` | API docs interactivos |
| `git push` | Deploy automático en Railway |

---

**¿Tu pregunta no está aquí?**

Revisar:
1. `QUICK_START.md` - Guía rápida
2. `DEPLOYMENT.md` - Guía completa
3. `SETUP_SUMMARY.md` - Referencia técnica
4. `/docs` en tu API - Documentación automática
