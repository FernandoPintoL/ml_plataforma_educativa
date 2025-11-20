# =====================================================
# Stage 1: Builder
# =====================================================

FROM python:3.11-slim AS builder

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# =====================================================
# Stage 2: Runtime
# =====================================================

FROM python:3.11-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/mluser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Instalar dependencias de runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Crear usuario no-root PRIMERO (antes de copiar archivos)
RUN useradd -m -u 1000 mluser

# Copiar Python packages del builder y asignar permisos
COPY --from=builder /root/.local /home/mluser/.local
RUN chown -R mluser:mluser /home/mluser/.local

# Copiar código de la aplicación
COPY --chown=mluser:mluser . .

# Health check (usa puerto dinámico)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Cambiar a usuario no-root
USER mluser

# Exponer puertos (8001 para local, 8080 para Railway)
EXPOSE 8001 8080

# Command
# En Railway: Usa variable PORT (asignada automáticamente como 8080)
# En Docker Desktop: Usa 8001 como default
# El puerto dinámico se respeta con ${PORT:-8001}
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8001} --workers 1"]
