# 🏗️ Arquitectura Optimizada - Sin Duplicidad de BD

## Problema Resuelto

❌ **Antes (Incorrecto):**
```
Docker:
├── ML API (8001)
├── PostgreSQL (5432)  ← Duplicada
└── Redis (6379)

Laravel (127.0.0.1):
└── PostgreSQL (5432)  ← Otra instancia

Resultado: 2 bases de datos, datos inconsistentes
```

✅ **Ahora (Optimizado):**
```
Docker:
└── ML API (8001)  ────┐
                       ├─→ PostgreSQL Externa (127.0.0.1:5432)
                       └─→ Redis Opcional

Laravel (127.0.0.1):
└── Comparte misma BD  ✅

Resultado: 1 BD, datos sincronizados
```

---

## 🎯 Configuración

### Variable de Conexión

```env
DATABASE_URL=postgresql://postgres:1234@127.0.0.1:5432/educativa
```

**Componentes:**
- `postgresql://` - Protocolo
- `postgres` - Usuario
- `1234` - Contraseña
- `127.0.0.1:5432` - Host:Puerto (tu BD local)
- `educativa` - Nombre de BD

### En Docker

El contenedor ML API se conecta automáticamente:
1. Lee `DATABASE_URL` del environment
2. Se conecta a `127.0.0.1:5432` (tu PostgreSQL local)
3. Comparte misma BD que Laravel

### En Railway (Producción)

Railway proporciona `DATABASE_URL` automáticamente:
```
DATABASE_URL=postgresql://user:pass@shortline.proxy.rlwy.net:10870/railway
```

El mismo contenedor Docker funciona en ambos lugares.

---

## 📊 Ventajas

| Aspecto | Monolítico | Optimizado |
|---------|-----------|-----------|
| **BD creadas** | 2+ | 1 ✅ |
| **RAM usado** | 3-4GB | 1-2GB ✅ |
| **Startup** | 60+ seg | 20 seg ✅ |
| **Consistencia** | ⚠️ Problemas | ✅ Garantizada |
| **Mantenimiento** | ❌ Complejo | ✅ Simple |
| **Escalabilidad** | ❌ Difícil | ✅ Fácil |

---

## 🔧 Cómo Funciona Localmente

### 1. PostgreSQL Corriendo (Local)

Tu BD ya está en: `127.0.0.1:5432`

```bash
# Verificar que PostgreSQL está activo
psql -U postgres -d educativa -c "SELECT 1"
```

### 2. Docker Levanta ML API

```bash
docker-compose up -d
```

Esto:
- Construye imagen ML (solo app)
- Inicia contenedor con ml-api en 8001
- Se conecta a tu PostgreSQL local

### 3. Ambas Aplicaciones Ven Misma BD

```
Laravel (8000) ────┐
                   ├─→ PostgreSQL (5432)
ML API (8001) ─────┘

✅ Mismos datos
✅ Transacciones consistentes
✅ No hay sincronización manual
```

---

## 📈 En Producción (Railway)

### Arquitectura

```
┌─────────────────────────────────────────┐
│         Railway Project                 │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   plataforma-educativa (Laravel) │  │
│  │   https://tu-dominio.com         │  │
│  └──────────────────┬───────────────┘  │
│                     │                   │
│                     ├─────────┐         │
│                     │         │         │
│  ┌──────────────────▼──┐  ┌──▼───────┐ │
│  │  ml-educativa (ML)  │  │ PostgreSQL│ │
│  │ https://ml.app.com  │  │ (Shared)  │ │
│  └─────────────────────┘  └───────────┘ │
│                                         │
└─────────────────────────────────────────┘

1 BD compartida por ambos servicios
```

### Variables Automáticas

Railway proporciona automáticamente:
```
DATABASE_URL=postgresql://user:pwd@host:port/db
```

Mismo contenedor Docker funciona en local y producción.

---

## ✅ Verificación

### Local

```bash
# 1. PostgreSQL está activo
psql -U postgres -d educativa -c "SELECT version();"

# 2. Levantar docker
docker-compose up -d

# 3. Verificar que ML API se conecta
curl http://localhost:8001/health

# 4. Ver logs
docker-compose logs ml-api | grep -i "database\|connected"
```

### En Railway

```bash
# Revisar logs
railway logs --service ml-api

# Buscar errores de conexión
railway logs --service ml-api | grep -i "database\|error"
```

---

## 🔒 Seguridad

### Credenciales

```
⚠️ NUNCA comitear .env a Git
✅ Variables en .env.example (sin valores reales)
✅ Variables en Railway Dashboard
✅ Credenciales en variables de entorno
```

### Local Development

```env
DATABASE_URL=postgresql://postgres:1234@127.0.0.1:5432/educativa
DEBUG=False
SECRET_KEY=development-only-key
```

### Production (Railway)

```env
DATABASE_URL=<proporcionado por Railway>
DEBUG=False
SECRET_KEY=<generar con secrets seguros>
```

---

## 🧪 Testing

### Verificar Conexión

```bash
# Desde container
docker exec ml-educativa-api python -c \
  "import psycopg2; \
   conn = psycopg2.connect('postgresql://postgres:1234@127.0.0.1:5432/educativa'); \
   print('✅ Conectado'); \
   conn.close()"
```

### Ver Logs de Conexión

```bash
docker-compose logs ml-api | tail -50
```

Busca: "Application startup complete"

---

## 📚 Archivos Relevantes

| Archivo | Propósito |
|---------|-----------|
| `docker-compose.yml` | Stack (solo ML API + BD externa) |
| `.env.example` | Variables template |
| `.env.railway` | Variables para Railway |
| `app.py` | Inicialización de BD |
| `shared/config.py` | Configuración de conexión |

---

## 🚀 Flujo Completo

### Desarrollo Local

1. PostgreSQL corre en `127.0.0.1:5432`
2. Laravel corre en `localhost:8000`
3. `docker-compose up -d` levanta ML API en 8001
4. Ambas ven misma BD

### Cambios en Código

```bash
# Editar código
vim app.py

# Reconstruir imagen
docker build -t ml-educativa:latest .

# Reiniciar contenedor
docker-compose up -d
```

### Deploy en Railway

```bash
git push origin main
# Railway:
# 1. Detecta cambios
# 2. Construye imagen (misma que local)
# 3. Deploy en production
# 4. Conecta a DATABASE_URL de Railway
# 5. ✅ Listo
```

---

## ❓ FAQ

**P: ¿Puedo cambiar contraseña de BD sin afectar ML?**
A: No. Actualiza DATABASE_URL en ambos lugares.

**P: ¿Qué pasa si BD desaparece?**
A: ML API fallará con error conexión. Reinicia BD y contenedor.

**P: ¿Redis es obligatorio?**
A: No. Está comentado en docker-compose.yml. Solo descomentar si lo usas.

**P: ¿Cómo migro datos?**
A: Misma BD = no hay que migrar. Todo está sincronizado.

**P: ¿En Railway puedo cambiar BD?**
A: No. Railway usa su PostgreSQL. No es recomendable compartir con otra aplicación.

---

## 🎉 Resumen

✅ **1 BD** para Laravel + ML (desarrollo)
✅ **Sincronización automática** de datos
✅ **Menor uso de recursos** (RAM, disco)
✅ **Mismo contenedor** funciona local + production
✅ **Escalabilidad sencilla** en ambos servicios

**Arquitectura limpia y eficiente.** 🚀
