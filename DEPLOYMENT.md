# Guía de Deployment para Railway

## Descripción General

Este documento describe cómo preparar y desplegar `ml_educativas` en Railway de forma separada de la plataforma educativa (que ya está en Railway).

## Arquitectura

```
┌─────────────────────────────────────────────┐
│          Railway - Plataforma Educativa     │
│        (Laravel - puerto 80/443)            │
└─────────────────────────────────────────────┘
                      ↓
        Llamadas HTTP a http://ml-api:8001
                      ↓
┌─────────────────────────────────────────────┐
│         Railway - ML Educativa               │
│   (FastAPI/Python - puerto 8001)            │
│                                             │
│  ├─ /supervisado/performance/*             │
│  ├─ /no-supervisado/clustering/*           │
│  ├─ /no-supervisado/anomaly/*              │
│  └─ /deep-learning/lstm/*                  │
└─────────────────────────────────────────────┘
```

## Archivos Creados

### 1. **app.py**
- API FastAPI con todos los endpoints
- Lazy loading de modelos
- Health check endpoint
- CORS habilitado
- Manejo de errores

### 2. **Dockerfile**
- Build en 2 stages (builder + runtime)
- Python 3.11-slim
- Usuario no-root (mluser)
- Health check integrado
- Optimizado para tamaño

### 3. **.dockerignore**
- Excluye archivos innecesarios
- Reduce tamaño de imagen

### 4. **docker-compose.yml**
- Servicios: ML API + PostgreSQL + Redis
- Para pruebas locales antes de Railway
- Redes configuradas
- Volúmenes para modelos entrenados

### 5. **requirements-prod.txt**
- Versión optimizada para producción
- Sin herramientas de desarrollo (pytest, black, etc)
- Dependencias críticas solamente

### 6. **railway.json**
- Configuración específica de Railway
- Build settings
- Deploy settings
- Variables de entorno

### 7. **.env.railway**
- Template de variables para Railway
- Instrucciones de configuración

## Pasos para Desplegar en Railway

### Paso 1: Preparar Repositorio Local

```bash
cd ml_educativas
git init
git add .
git commit -m "Initial commit: ML API ready for Railway"
```

### Paso 2: Subir a GitHub

```bash
git remote add origin https://github.com/tu-usuario/ml_educativas.git
git branch -M main
git push -u origin main
```

### Paso 3: Crear Proyecto en Railway

1. Ir a https://railway.app
2. Hacer login
3. Click en "Create New Project"
4. Seleccionar "Deploy from GitHub repo"
5. Conectar tu repositorio GitHub

### Paso 4: Configurar Servicio en Railway

1. **Crear servicio de ML API:**
   - Click en "Add Service" → "GitHub Repo"
   - Seleccionar `ml_educativas`
   - Railway detectará el Dockerfile automáticamente

2. **Configurar variables de entorno:**
   - En el dashboard, ir a "Variables" del servicio
   - Añadir variables del archivo `.env.railway`:
     ```
     ENVIRONMENT=production
     DEBUG=False
     API_RELOAD=False
     API_HOST=0.0.0.0
     API_WORKERS=4
     LOG_LEVEL=INFO
     SECRET_KEY=tu-clave-secreta-generada
     ```

3. **Conectar PostgreSQL (OPCIONAL):**
   - Si deseas BD separada:
     - Click "Add" → "Add Plugin" → "PostgreSQL"
   - Si compartes BD con Laravel:
     - Usar mismo DATABASE_URL que Laravel

### Paso 5: Configurar Puertos

- Railway asignará automáticamente puerto
- El servicio detectará variable `PORT`
- API escuchará en `http://tu-servicio.railway.app`

### Paso 6: Verificar Health Check

```bash
curl https://tu-servicio.railway.app/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "service": "Plataforma Educativa ML",
  "version": "2.0.0",
  "debug": false
}
```

## Comunicación entre Servicios

### Desde Laravel hacia ML API

En tu código Laravel, llamar a:

```php
// config/services.php
'ml_api' => [
    'url' => env('ML_API_URL', 'http://ml-api:8001'),
],

// Uso en controller
$response = Http::post(config('services.ml_api.url') . '/supervisado/performance/predict', [
    'student_id' => 123,
    'features' => [3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0]
]);
```

### Variable de Entorno en Railway

Añadir en el servicio de Laravel:
```
ML_API_URL=https://tu-ml-api.railway.app
```

## Testing Local Antes de Railway

### Opción 1: Con Docker Compose

```bash
# Construir imagen
docker-compose build

# Ejecutar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f ml-api

# Probar API
curl http://localhost:8001/health
curl http://localhost:8001/docs  # Swagger UI

# Parar servicios
docker-compose down
```

### Opción 2: Sin Docker (desarrollo)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar API
python -m uvicorn app:app --reload --port 8001

# Acceder a documentación
# http://localhost:8001/docs
```

## Endpoints Disponibles

### Health Check
```
GET /health
GET /
```

### Modelos Supervisados
```
POST /supervisado/performance/predict
POST /supervisado/performance/predict-batch
GET /supervisado/performance/model-info
```

### Modelos No Supervisados
```
POST /no-supervisado/clustering/predict
POST /no-supervisado/anomaly/detect
```

### Deep Learning
```
POST /deep-learning/lstm/predict
```

## Monitoreo en Railway

1. **Logs:**
   - Ir a "Deployments" → Ver logs
   - Buscar errores o advertencias

2. **Métricas:**
   - CPU, RAM, requests
   - Railway muestra en tiempo real

3. **Redeploy:**
   - Push a GitHub → Railway redeploy automático
   - O manual desde Dashboard

## Troubleshooting

### Error: "Modelo no disponible"
- Verificar que modelos entrenados existen en `trained_models/`
- Revisar permisos de lectura

### Error: "Database connection failed"
- Verificar DATABASE_URL en variables de entorno
- Asegurar que base de datos existe y es accesible

### Error: "Out of memory"
- Modelos grandes (TensorFlow, torch)
- En Railway: aumentar RAM del servicio
- O usar `requirements-prod.txt` para versiones lite

### API lenta
- Aumentar workers en API_WORKERS
- Revisar logs para detectar bottlenecks

## Optimizaciones Futuras

1. **Caché de modelos:**
   - Usar Redis para cachear predicciones
   - Evitar recargar modelos

2. **Modelos versionados:**
   - Sistema de versionado de modelos
   - A/B testing de predicciones

3. **Monitoreo:**
   - Integración con Sentry para errores
   - Logging estructurado

4. **CDN:**
   - Cachear responses estáticas
   - Comprimir payloads grandes

## Referencias

- [Railway Docs](https://docs.railway.app)
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Docker Compose](https://docs.docker.com/compose)
- [Uvicorn](https://www.uvicorn.org)
