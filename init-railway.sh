#!/bin/bash

# =====================================================
# Script de Inicialización para Railway
# =====================================================
# Ejecutar con: bash init-railway.sh

set -e

echo "📦 Inicializando ml_educativas para Railway..."
echo ""

# 1. Verificar si git existe
if ! command -v git &> /dev/null; then
    echo "❌ Git no está instalado. Por favor, instálalo primero."
    exit 1
fi

# 2. Inicializar git si no existe
if [ ! -d ".git" ]; then
    echo "🔧 Inicializando repositorio Git..."
    git init
    git config user.name "ML Educativa"
    git config user.email "ml@educativa.local"
else
    echo "✓ Repositorio Git ya existe"
fi

# 3. Verificar .env
if [ ! -f ".env.railway" ]; then
    echo "⚠️  Archivo .env.railway no encontrado"
else
    echo "✓ Archivo .env.railway presente"
fi

# 4. Crear .gitignore si no existe
if [ ! -f ".gitignore" ]; then
    echo "🔧 Creando .gitignore..."
    cat > .gitignore << EOF
.env
.env.local
venv/
env/
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
logs/
*.log
.DS_Store
.vscode/
.idea/
.swp
nul
EOF
fi

# 5. Añadir archivos
echo "📝 Añadiendo archivos a Git..."
git add -A

# 6. Crear commit inicial
if git diff --cached --quiet; then
    echo "✓ No hay cambios para hacer commit"
else
    git commit -m "🚀 Initial commit: ML API ready for Railway deployment"
fi

# 7. Información de siguientes pasos
echo ""
echo "✅ Inicialización completada"
echo ""
echo "📋 Siguientes pasos:"
echo "1. Crear repositorio en GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Conectar remoto local:"
echo "   git remote add origin https://github.com/tu-usuario/ml_educativas.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. En Railway Dashboard:"
echo "   - Create New Project"
echo "   - Deploy from GitHub repo"
echo "   - Seleccionar ml_educativas"
echo ""
echo "4. Configurar variables de entorno en Railway:"
echo "   - Copiar contenido de .env.railway"
echo "   - Cambiar SECRET_KEY por valor seguro"
echo ""
echo "5. Verificar health check:"
echo "   curl https://tu-servicio.railway.app/health"
echo ""
echo "📖 Para más información, ver DEPLOYMENT.md"
