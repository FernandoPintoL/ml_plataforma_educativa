#!/bin/bash

# =====================================================
# Script para probar ml_educativas con Docker Desktop
# =====================================================

set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ML EDUCATIVA - DOCKER DESKTOP TESTING                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Verificar Docker
echo "🔍 Verificando Docker Desktop..."
if ! docker --version &> /dev/null; then
    echo "❌ Docker no está instalado o no está corriendo"
    echo "   Inicia Docker Desktop y vuelve a intentar"
    exit 1
fi

docker_version=$(docker --version)
compose_version=$(docker-compose --version 2>/dev/null || echo "No encontrado")

echo "   ✅ Docker: $docker_version"
echo "   ✅ Compose: $compose_version"
echo ""

# 2. Build de imagen
echo "🏗️  Construyendo imagen Docker..."
echo "   (Este paso tarda ~2-5 minutos la primera vez)"
echo ""

if docker build -t ml-educativa:latest .; then
    echo ""
    echo "   ✅ Imagen construida correctamente"
else
    echo "   ❌ Error construyendo imagen"
    exit 1
fi
echo ""

# 3. Verificar imagen
echo "📦 Imágenes disponibles:"
docker image ls | grep ml-educativa
echo ""

# 4. Iniciar stack
echo "🚀 Iniciando servicios con docker-compose..."
echo "   • ml-api        (Puerto 8001)"
echo "   • postgresql    (Puerto 5432)"
echo "   • redis         (Puerto 6379)"
echo ""

if docker-compose up -d; then
    echo "   ✅ Servicios iniciados"
else
    echo "   ❌ Error iniciando servicios"
    exit 1
fi
echo ""

# 5. Esperar a que servicios estén listos
echo "⏳ Esperando a que servicios estén listos..."
echo "   (Esto puede tomar 30-60 segundos)"

for i in {1..30}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "   ✅ API está lista"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo ""

# 6. Probar health check
echo "🏥 Probando Health Check..."
curl -s http://localhost:8001/health | python -m json.tool
echo ""

# 7. Mostrar logs recientes
echo ""
echo "📝 Últimos logs de la API:"
echo "   (Primeros 15 líneas)"
docker-compose logs ml-api | tail -15
echo ""

# 8. Información de servicios
echo "🔗 Servicios disponibles:"
echo ""
echo "   API Documentation:"
echo "   └─ http://localhost:8001/docs"
echo ""
echo "   Health Check:"
echo "   └─ http://localhost:8001/health"
echo ""
echo "   PostgreSQL:"
echo "   └─ localhost:5432"
echo "   └─ user: postgres"
echo "   └─ password: password"
echo ""
echo "   Redis:"
echo "   └─ localhost:6379"
echo ""

# 9. Instrucciones de testing
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              SIGUIENTE: PRUEBA LOS ENDPOINTS                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "OPCIÓN 1: Interfaz web (Recomendado)"
echo "   Abre en navegador: http://localhost:8001/docs"
echo ""
echo "OPCIÓN 2: Script de testing"
echo "   Ejecuta: python test_api.py"
echo ""
echo "OPCIÓN 3: curl desde terminal"
echo "   curl http://localhost:8001/health"
echo ""
echo "OPCIÓN 4: Ver logs en tiempo real"
echo "   docker-compose logs -f ml-api"
echo ""

# 10. Limpiar (Opcional)
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              CUANDO TERMINES DE PROBAR                       ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Para parar servicios sin eliminar datos:"
echo "   docker-compose stop"
echo ""
echo "Para parar y eliminar todo:"
echo "   docker-compose down"
echo ""
echo "Para ver logs:"
echo "   docker-compose logs -f"
echo ""

echo "✅ Setup completado. API lista en http://localhost:8001"
echo ""
