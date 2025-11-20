# 🔧 Configuración de Puertos: Railway vs Docker Desktop

## 📋 Resumen

Railway asigna automáticamente el puerto **8080** mediante la variable de entorno `PORT`. Sin esta configuración correcta, se genera un error **502 Bad Gateway**.

## ⚙️ Cómo Funciona

### En Railway (Producción)
```
Railway asigna: PORT=8080
Uvicorn recibe: ${PORT} → 8080
Resultado: Aplicación escucha en 0.0.0.0:8080 ✅
```

### En Docker Desktop (Local)
```
docker-compose: No establece PORT
Uvicorn recibe: ${PORT:-8001} → 8001 (default)
Resultado: Aplicación escucha en 0.0.0.0:8001 ✅
```

## 🔍 Archivos Configurados

### 1. Dockerfile
```dockerfile
# Expone ambos puertos
EXPOSE 8001 8080

# Health check dinámico
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# CMD usa variable PORT con default 8001
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1"]
```

### 2. docker-compose.yml
```yaml
environment:
  # PORT: Para local usa 8001, Railway la asigna automáticamente como 8080
  - PORT=${PORT:-8001}
```

## ✅ Verificación

### Local (Docker Desktop)
```bash
docker-compose up -d
curl http://localhost:8001/health
# Esperado: {"status": "ok", ...}
```

### Production (Railway)
1. Railway Dashboard → ML Service
2. View Logs - buscar: `INFO:     Uvicorn running on http://0.0.0.0:8080`
3. Health Check: `curl https://your-domain.railway.app/health`

## 🚨 Troubleshooting: Error 502 en Railway

Si ves error **502 Bad Gateway** en Railway:

1. **Verificar logs:**
   ```
   Railway Dashboard → Deployments → View Logs
   ```

2. **Buscar puertos:**
   - ✅ Correcto: `Uvicorn running on http://0.0.0.0:8080`
   - ❌ Incorrecto: `Uvicorn running on http://0.0.0.0:8000` o `8001`

3. **Verificar variables:**
   ```
   Railway Dashboard → Variables
   - Debe estar: PORT=8080 (asignado automáticamente)
   - No debe estar: Valores hardcodeados de puerto
   ```

4. **Si sigue fallando:**
   - Redeploy desde Railway Dashboard
   - Limpiar cache: `docker system prune -a`

## 📊 Tabla de Puertos

| Entorno | Puerto | Variable | Default | Nota |
|---------|--------|----------|---------|------|
| **Railway** | 8080 | `PORT=8080` | N/A | Asignado automáticamente |
| **Docker Desktop** | 8001 | `${PORT:-8001}` | 8001 | Configurable localmente |
| **Dockerfile EXPOSE** | 8001, 8080 | N/A | N/A | Ambos expuestos |
| **Health Check** | Dinámico | `${PORT:-8001}` | 8001 | Respeta variable PORT |

## 🔗 Referencias Importantes

- **Railway docs:** Usa `PORT` automáticamente
- **Uvicorn:** Respeta `--port` en CLI
- **Docker:** `EXPOSE` es solo documentación, no afecta network
- **docker-compose:** `ports` mapea entre host:container

---

**¡Commit:** Cambios enviados a `main` para ambos repos (ml_educativas y plataforma-educativa)
