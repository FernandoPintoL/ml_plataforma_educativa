# 🐳 Testing con Docker Desktop

Este documento te guía paso a paso para probar `ml_educativas` localmente con Docker Desktop.

## 🏗️ Arquitectura Optimizada

```
┌─────────────────────────────────────────┐
│   ML API Container (Docker)             │
│   http://localhost:8001                 │
│                                         │
│   • FastAPI                             │
│   • Modelos ML                          │
│   • Documentación (/docs)               │
└────────────────┬────────────────────────┘
                 │
                 ├──→ PostgreSQL Externa
                 │    127.0.0.1:5432
                 │    (Tu BD existente)
                 │
                 └──→ Redis (Opcional)
                      127.0.0.1:6379
```

**Ventajas:**
- ✅ Sin duplicidad de BD
- ✅ Usa tu DB existente
- ✅ Menor consumo de RAM
- ✅ Startup más rápido (~20 seg)
- ✅ Misma BD para Laravel y ML

## Requisitos Previos

✅ Docker Desktop instalado (Ya lo tienes)
✅ Docker Compose disponible (Incluido en Docker Desktop)
✅ PostgreSQL corriendo en `127.0.0.1:5432`
✅ ~500MB de espacio libre (solo imagen ML)
✅ 1GB RAM mínimo para Docker (es suficiente)

## 🚀 Opción 1: Script Automático (Recomendado)

### En Windows (PowerShell)

1. **Abre PowerShell en la carpeta `ml_educativas`:**
   ```powershell
   cd "D:\PLATAFORMA EDUCATIVA\ml_educativas"
   ```

2. **Ejecuta el script:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File docker-test.ps1
   ```

   El script automáticamente:
   - Verifica Docker
   - Construye la imagen
   - Inicia docker-compose
   - Espera a que servicios estén listos
   - Prueba health check
   - Muestra instrucciones

### En Linux/Mac

```bash
cd ml_educativas
bash docker-test.sh
```

---

## 🔧 Opción 2: Paso a Paso Manual

Si prefieres hacerlo manualmente o el script no funciona:

### Paso 1: Verificar Docker

```bash
docker --version
docker-compose --version
```

Debes ver algo como:
```
Docker version 28.5.1
Docker Compose version v2.40.2
```

### Paso 2: Construir la imagen

```bash
cd "D:\PLATAFORMA EDUCATIVA\ml_educativas"
docker build -t ml-educativa:latest .
```

**Tiempo:** 2-5 minutos (la primera vez)

**Salida esperada:**
```
[+] Building 240.5s (15/15) FINISHED
...
=> => naming to docker.io/library/ml-educativa:latest
```

### Paso 3: Iniciar docker-compose

```bash
docker-compose up -d
```

**Salida:**
```
[+] Running 3/3
 ✓ Network educativa-network Created
 ✓ Container educativa-postgres Started
 ✓ Container educativa-redis Started
 ✓ Container ml-educativa-api Started
```

### Paso 4: Verificar servicios

```bash
docker-compose ps
```

Debes ver:
```
NAME                  COMMAND                  SERVICE     STATUS
educativa-postgres    "docker-entrypoint..."   postgres    Up 30s
educativa-redis       "redis-server..."        redis       Up 30s
ml-educativa-api      "python -m uvicorn..."   ml-api      Up 30s
```

### Paso 5: Ver logs

```bash
# Últimos 20 líneas
docker-compose logs ml-api

# En tiempo real
docker-compose logs -f ml-api

# Parar logs: Ctrl+C
```

---

## ✅ Verificar que Funciona

### Opción A: Navegador (Más fácil)

1. Abre: **http://localhost:8001/docs**

Deberías ver la documentación interactiva de Swagger con todos los endpoints.

### Opción B: Health Check (Terminal)

```bash
curl http://localhost:8001/health
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

### Opción C: Script Python

```bash
python test_api.py
```

Ejecuta 7 tests de endpoints y muestra resultados:
```
✅ Health Check
✅ Root Endpoint
❌ Performance Prediction (modelo no entrenado - normal)
❌ Batch Prediction (modelo no entrenado - normal)
...
```

---

## 📊 Testing Manual de Endpoints

### Desde el navegador (http://localhost:8001/docs)

Haz click en cada endpoint y presiona "Try it out":

1. **GET /health** - Presiona "Execute"
   ```
   Status: 200 OK
   Response: {"status": "healthy", ...}
   ```

2. **GET /** - Info de API
   ```
   Devuelve versión y enlaces a documentación
   ```

3. **POST /supervisado/performance/predict**
   - Presiona "Try it out"
   - Mantén los valores de ejemplo
   - Presiona "Execute"
   - Respuesta: Puede ser:
     - 503 si modelo no está entrenado (NORMAL)
     - 200 con predicción si lo está

### Desde Terminal (PowerShell/cmd)

```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:8001/health

# Predicción
$body = @{
    student_id = 1
    features = @(3.5, 85, 10, 2.1, 45, 0.8, 1.2, 0.9, 0.85, 2.0)
} | ConvertTo-Json

Invoke-WebRequest -Uri http://localhost:8001/supervisado/performance/predict `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
```

---

## 📁 Puertos Disponibles

| Servicio | Puerto | Tipo | Ubicación |
|----------|--------|------|-----------|
| API | 8001 | Docker | http://localhost:8001 |
| API Docs | 8001 | Docker | http://localhost:8001/docs |
| PostgreSQL | 5432 | **Externa** | localhost:5432 (tu BD existente) |
| Redis | 6379 | Opcional | localhost:6379 (si descomenta) |

**Nota:** PostgreSQL no se crea en Docker, usa tu BD existente en `127.0.0.1:5432`

---

## 🔍 Troubleshooting

### Error: "Docker daemon is not running"

**Solución:**
- Abre Docker Desktop (aplicación)
- Espera ~30 segundos a que inicie
- El ícono en la bandeja debe estar azul
- Intenta de nuevo

### Error: "Port 8001 already in use"

Algo más está usando puerto 8001:

```bash
# Encontrar qué usa puerto 8001
netstat -ano | findstr :8001

# O matar servicio en ese puerto
taskkill /PID <PID> /F
```

O usa otro puerto:
```bash
docker-compose down  # Para todo
# Editar docker-compose.yml: cambiar "8001:8001" a "8002:8001"
docker-compose up -d
```

### Error: "Image build failed"

Probablemente dependencias no se instalaron correctamente:

```bash
# Limpiar y reintentar
docker-compose down -v   # Elimina volúmenes
docker system prune -a   # Limpia imágenes viejas
docker build -t ml-educativa:latest . --no-cache
docker-compose up -d
```

### Error: "Connection refused" al probar API

Servicios aún están iniciando. Espera 30-60 segundos:

```bash
# Ver logs
docker-compose logs ml-api

# Esperar a que muestre: "Uvicorn running on http://0.0.0.0:8001"
```

### Error: "Out of memory"

Docker necesita más RAM:

1. Abre Docker Desktop
2. Settings → Resources
3. Aumenta "Memory" a 4GB o más
4. Aplica y reinicia Docker
5. Intenta de nuevo

---

## 🧪 Testing Completo

### 1. Verificar Health Check

```bash
curl http://localhost:8001/health
```

### 2. Ver documentación

```
Navegador: http://localhost:8001/docs
```

### 3. Ejecutar test script

```bash
python test_api.py
```

### 4. Ver logs en tiempo real

```bash
docker-compose logs -f ml-api
```

### 5. Verificar base de datos

```bash
# Conectar a PostgreSQL
docker exec -it educativa-postgres psql -U postgres -d educativa

# Ver tablas
\dt

# Salir
\q
```

---

## 📊 Esperado vs Realidad

### ✅ Funcionará correctamente

- Health check responde
- Documentación es accesible (/docs)
- Logs muestran servidor corriendo
- Navegador muestra Swagger UI

### ⚠️ Normal que falle

- Predicciones devuelven 503 (modelos no entrenados aún)
- Clustering devuelve 503 (modelo no entrenado)
- Anomaly detection devuelve 503 (modelo no entrenado)

**¿Por qué?** Los modelos `.pkl` deben estar en `trained_models/` y entrenados. Eso es paso aparte.

### ❌ No debería pasar

- Error de conexión (Docker no corriendo)
- Puerto en uso (cambiar puerto o matar proceso)
- Image build fail (problemas con dependencias)

---

## 🛑 Detener Todo

### Solo pausar (mantiene datos)

```bash
docker-compose stop
```

### Detener y limpiar (elimina todo)

```bash
docker-compose down
```

### Eliminar todo incluyendo volúmenes

```bash
docker-compose down -v
```

### Limpiar Docker completamente

```bash
docker system prune -a --volumes
```

---

## 📈 Performance Esperado

| Acción | Tiempo |
|--------|--------|
| Build primera vez | 2-5 minutos |
| Build después | 30 seg (cached) |
| Iniciar servicios | 10-20 segundos |
| Health check | < 50ms |
| Predicción simple | 100-500ms |
| Documentación (/docs) | < 100ms |

---

## 🎯 Próximo Paso

Una vez que todo funcione localmente:

1. ✅ Verifica que `/docs` es accesible
2. ✅ Verifica que health check funciona
3. ✅ Verifica que logs muestran servidor activo
4. Procede a **DEPLOYMENT.md** para subir a Railway

---

## 💡 Tips Útiles

**Ver todo en Docker Desktop:**
1. Abre aplicación "Docker Desktop"
2. Pestaña "Containers"
3. Verás `educativa-network` con 3 servicios
4. Click en `ml-educativa-api` para ver logs interactivos

**Reiniciar servicios sin rebuild:**
```bash
docker-compose restart ml-api
```

**Reconstruir sin caché:**
```bash
docker build -t ml-educativa:latest . --no-cache
```

**Ver uso de recursos:**
```bash
docker stats
```

---

## 🎉 Success!

Cuando veas esto, estás listo:

```
✅ docker-compose ps - muestra 3 servicios "Up"
✅ curl http://localhost:8001/health - responde 200
✅ http://localhost:8001/docs - es accesible
✅ docker-compose logs ml-api - muestra "Application startup complete"
```

Ahora puedes:
1. Probar endpoints manualmente
2. Integrar con Laravel localmente
3. Hacer cambios y probar
4. Luego pushear a GitHub para Railway
