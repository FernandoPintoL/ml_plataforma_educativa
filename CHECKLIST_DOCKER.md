# ✅ CHECKLIST INTERACTIVO PARA DOCKERIZACIÓN

> Sigue este checklist paso a paso. Marca cada item cuando lo completes.

---

## 📋 PASO 1: VERIFICAR POSTGRESQL

### ☐ Paso 1.1: Abre PowerShell

```
Presiona: Win + R
Escribe: powershell
Presiona: Enter
```

**Deberías ver:** Una ventana negra con `PS C:\>`

### ☐ Paso 1.2: Verifica que PostgreSQL está corriendo

```powershell
psql -U postgres -d educativa -c "SELECT 1"
```

**Esperado:**
```
 ?column?
──────────
        1
(1 fila)
```

**Si ves esto: ✅ Marca esta casilla**

---

## 📋 PASO 2: DOCKER DESKTOP

### ☐ Paso 2.1: Abre Docker Desktop

```
Win + Buscar "Docker Desktop"
Click para abrir
Espera 30 segundos
```

### ☐ Paso 2.2: Verifica que está listo

Busca ícono en bandeja de tareas (abajo a la derecha):
- 🔵 Azul = Listo ✅
- ⚫ Negro = Iniciando (espera más)

**Si ícono es AZUL: ✅ Marca esta casilla**

### ☐ Paso 2.3: Verifica Docker en terminal

```powershell
docker --version
docker-compose --version
```

**Deberías ver versiones (ej: Docker version 28.5.1)**

**Si ves versiones: ✅ Marca esta casilla**

---

## 📋 PASO 3: NAVEGAR A CARPETA

### ☐ Paso 3.1: Abre File Explorer

```
Win + E
```

### ☐ Paso 3.2: Navega a carpeta

```
D:\PLATAFORMA EDUCATIVA\ml_educativas
```

### ☐ Paso 3.3: Abre PowerShell aquí

```
Shift + Click derecho en carpeta vacía
"Open PowerShell window here"
```

**Deberías ver:**
```
PS D:\PLATAFORMA EDUCATIVA\ml_educativas>
```

**Si ves esta ruta: ✅ Marca esta casilla**

### ☐ Paso 3.4: Verifica archivos

```powershell
ls
```

**Deberías ver:**
```
    Directory: D:\PLATAFORMA EDUCATIVA\ml_educativas

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a---          11/18/2025 12:40 PM          14 KB app.py
-a---          11/18/2025 12:40 PM         1.6 KB Dockerfile
-a---          11/18/2025 12:40 PM         2.4 KB docker-compose.yml
...
```

**Si ves archivos del proyecto: ✅ Marca esta casilla**

---

## 📋 PASO 4: BUILD IMAGEN DOCKER

### ☐ Paso 4.1: Inicia build

```powershell
docker build -t ml-educativa:latest .
```

### ☐ Paso 4.2: Espera a que termine

Verás:
```
[+] Building ...
[+] Running 15/15 FINISHED
 => => naming to docker.io/library/ml-educativa:latest
```

**Tiempo: 2-5 minutos (solo la primera vez)**

### ☐ Paso 4.3: Verifica que build fue exitoso

Deberías ver al final:
```
=> => naming to docker.io/library/ml-educativa:latest
```

**Si ves "FINISHED": ✅ Marca esta casilla**

### ☐ Paso 4.4: Verifica imagen creada

```powershell
docker image ls ml-educativa
```

**Esperado:**
```
REPOSITORY      TAG       IMAGE ID      SIZE
ml-educativa    latest    abc123def456  750MB
```

**Si ves imagen listada: ✅ Marca esta casilla**

---

## 📋 PASO 5: DOCKER-COMPOSE UP

### ☐ Paso 5.1: Inicia servicios

```powershell
docker-compose up -d
```

**Verás:**
```
[+] Running 1/1
 ✓ Container ml-educativa-api  Started
```

### ☐ Paso 5.2: Verifica que container está corriendo

```powershell
docker-compose ps
```

**Esperado:**
```
NAME                   COMMAND                 SERVICE   STATUS
ml-educativa-api       "python -m uvicorn..."  ml-api    Up 10 seconds
```

**Si ves "Up": ✅ Marca esta casilla**

### ☐ Paso 5.3: Ver logs

```powershell
docker-compose logs ml-api | tail -20
```

**Deberías ver:**
```
... Application startup complete
```

**Si ves "startup complete": ✅ Marca esta casilla**

---

## 📋 PASO 6: PROBAR HEALTH CHECK

### ☐ Paso 6.1: Health check básico

```powershell
curl http://localhost:8001/health
```

O en PowerShell:
```powershell
Invoke-WebRequest http://localhost:8001/health
```

**Esperado:**
```json
{
  "status": "healthy",
  "service": "Plataforma Educativa ML",
  "version": "2.0.0",
  "debug": false
}
```

**Si ves status "healthy": ✅ Marca esta casilla**

### ☐ Paso 6.2: Repite si no responde

Si no responde:
```powershell
# Espera 10 segundos más
Start-Sleep -Seconds 10

# Reintenta
Invoke-WebRequest http://localhost:8001/health
```

**Si ahora sí responde: ✅ Marca esta casilla**

---

## 📋 PASO 7: SWAGGER UI (Visual)

### ☐ Paso 7.1: Abre navegador

```
Abre tu navegador favorito (Chrome, Edge, Firefox)
```

### ☐ Paso 7.2: Ve a URL

```
http://localhost:8001/docs
```

### ☐ Paso 7.3: Deberías ver

Una página con:
- Título "Plataforma Educativa ML"
- Lista de endpoints (GET, POST)
- Interfaz Swagger interactiva

**Si ves Swagger: ✅ Marca esta casilla**

### ☐ Paso 7.4: Prueba un endpoint

1. Haz click en cualquier endpoint (ej: `/health`)
2. Click en **"Try it out"**
3. Click en **"Execute"**
4. Deberías ver respuesta en JSON

**Si ves respuesta: ✅ Marca esta casilla**

---

## 📋 PASO 8: SCRIPT DE TESTING

### ☐ Paso 8.1: Ejecuta tests

```powershell
python test_api.py
```

### ☐ Paso 8.2: Verifica resultados

Deberías ver algo como:
```
✅ Health Check
✅ Root Endpoint
❌ Performance Prediction (503 - normal, modelo no entrenado)
...
═══════════════════
Resultados: 5/7 tests pasados
```

**Nota:** Es NORMAL que algunos fallen con 503 si modelos no están entrenados.

**Si ves "tests pasados": ✅ Marca esta casilla**

---

## 🎉 RESUMEN

Si marcaste todos los ☑️ anteriores:

```
┌─────────────────────────────────────────┐
│  ✅ DOCKERIZACIÓN COMPLETADA EXITOSAMENTE│
│                                         │
│  Tu ML API está corriendo en:           │
│  http://localhost:8001                  │
│                                         │
│  ✅ Conectada a tu BD existente         │
│  ✅ Documentación en /docs              │
│  ✅ Tests ejecutándose                  │
│  ✅ Lista para desarrollo                │
└─────────────────────────────────────────┘
```

---

## 🐛 SOLUCIÓN RÁPIDA DE PROBLEMAS

### ❌ "docker: command not found"
```powershell
# Docker Desktop no está instalado
# Descarga en: https://www.docker.com/products/docker-desktop
# Reinstala e reinicia PowerShell
```

### ❌ "error during connect: docker daemon is not running"
```powershell
# Abre Docker Desktop (aplicación)
# Espera 30 segundos
# Reintenta
```

### ❌ "couldn't connect to database"
```powershell
# PostgreSQL no está corriendo
# Verifica: psql -U postgres -d educativa -c "SELECT 1"
# Si falla, reinicia PostgreSQL
# Luego: docker-compose restart
```

### ❌ "Port 8001 already in use"
```powershell
# Otro proceso usa puerto 8001
# Opción 1:
netstat -ano | findstr :8001
taskkill /PID <PID> /F

# Opción 2: Cambia puerto en docker-compose.yml
# "8002:8001" en lugar de "8001:8001"
```

### ❌ "Connection refused"
```powershell
# API aún se está inicializando
# Espera 30 segundos más
# Ver logs: docker-compose logs ml-api
```

---

## 📊 DURACIÓN POR PASO

| Paso | Duración |
|------|----------|
| 1. PostgreSQL | 1 min |
| 2. Docker Desktop | 1 min |
| 3. Navegar | 30 seg |
| 4. Build imagen | **2-5 min** |
| 5. docker-compose | 1 min |
| 6. Health check | 1 min |
| 7. Swagger | 2 min |
| 8. Tests | 1 min |
| **TOTAL PRIMERO** | **~10 minutos** |
| Siguientes veces | **~2 minutos** |

---

## ✨ DESPUÉS DE DOCKERIZAR

```
✅ Desarrollar:
   - Edita código (app.py)
   - docker-compose restart
   - Cambios inmediatos

✅ Probar:
   - http://localhost:8001/docs
   - python test_api.py
   - Ver logs: docker-compose logs -f

✅ Prepararse para Railway:
   - git push origin main
   - Railway detecta cambios
   - Deploy automático

✅ Monitoreo:
   - docker-compose ps
   - docker-compose logs -f ml-api
   - Railway Dashboard
```

---

## 🚀 PRÓXIMOS PASOS

Una vez completado este checklist:

1. Lee: **DEPLOYMENT.md**
2. Sube a GitHub
3. Crea proyecto en Railway
4. Deploy automático

**¡Tu API estará en PRODUCCIÓN!** 🎉

---

## 📞 AYUDA RÁPIDA

Cualquier error:
1. Copia el mensaje de error
2. Abre **DOCKER_TESTING.md** sección "Troubleshooting"
3. O consulta **ARCHITECTURE.md** para entender la arquitectura

Estás casi ahí. **¡Continúa con el próximo paso!** 💪
