# 📋 Pasos para Dockerizar ML Educativa

> Guía paso a paso para levantar todo con Docker Desktop

---

## ✅ PASO 1: Verificar PostgreSQL Corriendo

PostgreSQL **DEBE estar activo** antes de iniciar Docker.

### Opción A: Desde PowerShell/CMD

```powershell
# Abre PowerShell y ejecuta:
psql -U postgres -d educativa -c "SELECT 1"
```

**Resultado esperado:**
```
 ?column?
──────────
        1
(1 fila)
```

### Opción B: Desde pgAdmin (Visual)

1. Abre pgAdmin (si lo tienes instalado)
2. Conéctate a `localhost:5432`
3. Usuario: `postgres`
4. Contraseña: `1234`
5. Debería conectarse sin problemas

### ❌ Si PostgreSQL NO está corriendo:

```powershell
# Windows - Busca PostgreSQL
# Win + Buscar "Services"
# Busca "PostgreSQL" → Click derecho → Start

# O ejecuta en PowerShell (como Admin):
# net start postgresql-x64-XX
```

**Continúa solo cuando veas respuesta del `SELECT 1`**

---

## ✅ PASO 2: Abrir Docker Desktop

Docker Desktop debe estar corriendo para todo.

### En Windows:

1. **Presiona:** `Win + R`
2. **Escribe:** `Docker Desktop`
3. **Presiona:** `Enter`

O busca en Inicio:
1. **Win** (presiona tecla Windows)
2. **Busca:** `Docker Desktop`
3. **Click** para abrir

### Espera ~30 segundos

Verás ícono en bandeja de tareas (esquina inferior derecha):
- ⚫ Negro = Iniciando
- 🔵 Azul = Listo ✅

**No continúes hasta que el ícono sea AZUL**

---

## ✅ PASO 3: Abrir Terminal/PowerShell

Necesitas una terminal en la carpeta del proyecto.

### Opción A: Rápida (Recomendada)

1. Abre File Explorer
2. Navega a: `D:\PLATAFORMA EDUCATIVA\ml_educativas`
3. **Shift + Click derecho** en carpeta vacía
4. **"Open PowerShell window here"**

### Opción B: Manual

```powershell
# Abre PowerShell y navega manualmente
cd "D:\PLATAFORMA EDUCATIVA\ml_educativas"
```

### Verificar ubicación correcta

```powershell
# Deberías ver estos archivos:
ls

# Output esperado:
Dockerfile
docker-compose.yml
app.py
requirements.txt
# ... más archivos
```

**Continúa solo si ves estos archivos**

---

## ✅ PASO 4: Construir Imagen Docker

Esto crea la imagen que Docker usará.

### Comando:

```powershell
docker build -t ml-educativa:latest .
```

### Qué verás:

```
[+] Building 0.5s (8/8) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [builder] FROM python:3.11-slim
 => [builder] RUN apt-get update && apt-get install -y
 ...
 => => naming to docker.io/library/ml-educativa:latest
```

### Tiempo esperado:

- **Primera vez:** 2-5 minutos (descarga dependencias)
- **Siguientes:** 30 segundos (usa caché)

**Espera hasta ver "FINISHED"**

### ❌ Si falla:

```powershell
# Limpia y reintenta
docker system prune -a
docker build -t ml-educativa:latest . --no-cache
```

---

## ✅ PASO 5: Iniciar docker-compose

Esto levanta el contenedor ML API.

### Comando:

```powershell
docker-compose up -d
```

### Qué verás:

```
[+] Running 1/1
 ✓ Container ml-educativa-api  Started
```

### Verificar que está corriendo:

```powershell
docker-compose ps
```

**Deberías ver:**
```
NAME                   STATUS
ml-educativa-api       Up 5 seconds
```

### ❌ Si no levanta:

```powershell
# Ver qué pasó
docker-compose logs ml-api

# Si hay problema, parar e intentar de nuevo
docker-compose down
docker-compose up -d
```

---

## ✅ PASO 6: Esperar a que Esté Listo

La API tarda 10-20 segundos en inicializar.

### Comando:

```powershell
# Ejecuta repetidamente hasta ver respuesta
curl http://localhost:8001/health
```

O desde PowerShell:

```powershell
Invoke-WebRequest http://localhost:8001/health
```

### Resultado esperado:

```json
{
  "status": "healthy",
  "service": "Plataforma Educativa ML",
  "version": "2.0.0",
  "debug": false
}
```

### Si no responde:

```powershell
# Ver logs para entender qué está pasando
docker-compose logs ml-api

# Busca: "Application startup complete"
# Si ves errores de conexión BD:
# 1. Verifica PostgreSQL está corriendo (Paso 1)
# 2. Verifica DATABASE_URL en docker-compose.yml

# Si está todo bien, espera 30 segundos más
```

---

## ✅ PASO 7: Verificar que Funciona

### Opción A: Swagger UI (Visual) ⭐ Recomendado

1. Abre navegador
2. Ve a: **http://localhost:8001/docs**
3. Deberías ver interfaz con todos los endpoints
4. Haz click en cualquier endpoint
5. Click en **"Try it out"**
6. Click en **"Execute"**
7. Ves respuesta en JSON

**Esto prueba que la API funciona completamente**

### Opción B: Health Check (Terminal)

```powershell
curl http://localhost:8001/health

# Deberías ver JSON con status "healthy"
```

### Opción C: Script de Testing

```powershell
python test_api.py

# Ejecuta 7 tests automáticos
# Muestra cuáles pasaron (✅) y cuáles fallaron (❌)
```

---

## 📊 RESUMEN DE PASOS

```
┌─────────────────────────────────────────────────────────┐
│ 1. PostgreSQL corriendo                                 │
│    psql -U postgres -d educativa -c "SELECT 1"          │
│                                                         │
│ 2. Docker Desktop abierto                               │
│    (ícono azul en bandeja)                              │
│                                                         │
│ 3. PowerShell en ml_educativas/                          │
│    cd "D:\PLATAFORMA EDUCATIVA\ml_educativas"           │
│                                                         │
│ 4. Build imagen                                         │
│    docker build -t ml-educativa:latest .                │
│    (Espera 2-5 minutos)                                 │
│                                                         │
│ 5. Iniciar compose                                      │
│    docker-compose up -d                                 │
│                                                         │
│ 6. Esperar a que esté listo                             │
│    curl http://localhost:8001/health                    │
│    (Espera 10-20 segundos)                              │
│                                                         │
│ 7. Probar                                               │
│    http://localhost:8001/docs                           │
│                                                         │
│ ✅ ¡LISTO! API corriendo en http://localhost:8001      │
└─────────────────────────────────────────────────────────┘
```

---

## 🐳 COMANDOS ÚTILES DESPUÉS

```powershell
# Ver estado de servicios
docker-compose ps

# Ver logs en tiempo real
docker-compose logs -f ml-api

# Parar servicios (sin eliminar)
docker-compose stop

# Reinicar servicios
docker-compose restart

# Parar y eliminar todo
docker-compose down

# Limpiar espacio Docker
docker system prune -a
```

---

## ❓ TROUBLESHOOTING

### Error: "docker: command not found"
```
→ Docker Desktop no está instalado o no está en PATH
→ Reinstala Docker Desktop
→ Reinicia PowerShell después de instalar
```

### Error: "error during connect: docker daemon is not running"
```
→ Docker Desktop no está corriendo
→ Abre Docker Desktop (aplicación)
→ Espera 30 segundos a que inicie
→ Reintenta
```

### Error: "Port 8001 already in use"
```
→ Algo más usa puerto 8001
→ Opción 1: Mata el proceso
   netstat -ano | findstr :8001
   taskkill /PID <PID> /F
→ Opción 2: Usa otro puerto
   Edita docker-compose.yml: "8002:8001"
```

### Error: "couldn't connect to database"
```
→ PostgreSQL no está corriendo
→ Verifica Paso 1: psql -U postgres -d educativa -c "SELECT 1"
→ Si falla, reinicia PostgreSQL
→ Reintenta: docker-compose restart
```

### Error: "Application not ready, still loading"
```
→ La API se está iniciando, espera 30 segundos más
→ Ver logs: docker-compose logs ml-api
→ Busca: "Application startup complete"
```

---

## 📈 FLUJO VISUAL COMPLETO

```
INICIO
  │
  ├─→ ¿PostgreSQL corriendo?
  │   NO → Iniciar PostgreSQL
  │   SÍ → Continuar
  │
  ├─→ ¿Docker Desktop abierto?
  │   NO → Abrir Docker Desktop (esperar 30 seg)
  │   SÍ → Continuar
  │
  ├─→ Abre PowerShell en D:\...\ml_educativas
  │
  ├─→ docker build -t ml-educativa:latest .
  │   (Espera 2-5 minutos)
  │   ✅ FINISHED → Continuar
  │
  ├─→ docker-compose up -d
  │   ✅ Container started → Continuar
  │
  ├─→ curl http://localhost:8001/health
  │   (Repite cada 5 segundos)
  │   ✅ Response 200 → Continuar
  │
  ├─→ http://localhost:8001/docs
  │   ✅ Swagger UI visible → ¡LISTO!
  │
  └─→ 🎉 DOCKERIZACIÓN COMPLETADA
```

---

## ✨ AHORA PUEDES:

```
✅ Ejecutar: http://localhost:8001/docs
   └─ Probar todos los endpoints visualmente

✅ Ejecutar: python test_api.py
   └─ Tests automáticos

✅ Ver logs: docker-compose logs -f
   └─ Monitoreo en tiempo real

✅ Cambiar código y reiniciar:
   docker-compose restart

✅ Cuando esté perfecto:
   git push → Deploy en Railway
```

---

## 🎯 DURACIÓN TOTAL

| Paso | Tiempo |
|------|--------|
| 1. PostgreSQL | 1 min |
| 2. Docker Desktop | 1 min |
| 3. Terminal | 30 seg |
| 4. Build imagen | 2-5 min |
| 5. docker-compose | 1 min |
| 6. Esperar listo | 1 min |
| 7. Pruebas | 2 min |
| **TOTAL** | **~8-12 minutos** |

**La primera vez tarda más. Siguientes veces: ~2 minutos**

---

## ✅ CHECKLIST FINAL

- [ ] PostgreSQL responde a `SELECT 1`
- [ ] Docker Desktop icono es azul
- [ ] `docker build` terminó con FINISHED
- [ ] `docker-compose ps` muestra container UP
- [ ] `curl http://localhost:8001/health` responde
- [ ] http://localhost:8001/docs es accesible
- [ ] Hice click en un endpoint y funcionó

Si todos los ☑️ están marcados: **¡Dockerización completada!** 🎉

---

## 🚀 PRÓXIMOS PASOS

Una vez que todo funciona localmente:

1. ✅ Prueba endpoints en `/docs`
2. ✅ Ejecuta `python test_api.py`
3. ✅ Lee `DEPLOYMENT.md` para subir a Railway
4. ✅ Sube a GitHub
5. ✅ Crea proyecto en Railway
6. ✅ Deploy automático en Railway

**Luego tu API estará en PRODUCCIÓN con todo configurado.** 🚀
