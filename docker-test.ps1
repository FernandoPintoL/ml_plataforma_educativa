# =====================================================
# Script para probar ml_educativas con Docker Desktop
# PowerShell para Windows
# =====================================================
# Ejecutar con: powershell -ExecutionPolicy Bypass -File docker-test.ps1

Write-Host "`n" -ForegroundColor White
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║      ML EDUCATIVA - DOCKER DESKTOP TESTING                ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# 1. Verificar Docker
Write-Host "🔍 Verificando Docker Desktop..." -ForegroundColor Cyan

try {
    $dockerVersion = docker --version
    $composeVersion = docker-compose --version
    Write-Host "   ✅ Docker: $dockerVersion" -ForegroundColor Green
    Write-Host "   ✅ Compose: $composeVersion" -ForegroundColor Green
}
catch {
    Write-Host "   ❌ Docker no está instalado o no está corriendo" -ForegroundColor Red
    Write-Host "   Inicia Docker Desktop y vuelve a intentar" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 2. Build de imagen
Write-Host "🏗️  Construyendo imagen Docker..." -ForegroundColor Cyan
Write-Host "   (Este paso tarda ~2-5 minutos la primera vez)" -ForegroundColor Yellow
Write-Host ""

if (docker build -t ml-educativa:latest .) {
    Write-Host "   ✅ Imagen construida correctamente" -ForegroundColor Green
}
else {
    Write-Host "   ❌ Error construyendo imagen" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 3. Mostrar imagen
Write-Host "📦 Imágenes disponibles:" -ForegroundColor Cyan
docker image ls | Select-String "ml-educativa"
Write-Host ""

# 4. Iniciar stack
Write-Host "🚀 Iniciando servicios con docker-compose..." -ForegroundColor Cyan
Write-Host "   • ml-api        (Puerto 8001)" -ForegroundColor White
Write-Host "   • PostgreSQL    (EXTERNA - 127.0.0.1:5432)" -ForegroundColor Yellow
Write-Host "   • Redis         (OPCIONAL - comentado en compose)" -ForegroundColor Yellow
Write-Host ""

if (docker-compose up -d) {
    Write-Host "   ✅ Servicios iniciados" -ForegroundColor Green
}
else {
    Write-Host "   ❌ Error iniciando servicios" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 5. Esperar a que servicios estén listos
Write-Host "⏳ Esperando a que servicios estén listos..." -ForegroundColor Cyan
Write-Host "   (Esto puede tomar 30-60 segundos)" -ForegroundColor Yellow

$maxWait = 60
$waited = 0
$ready = $false

while ($waited -lt $maxWait -and -not $ready) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
            Write-Host "   ✅ API está lista" -ForegroundColor Green
        }
    }
    catch {
        Write-Host -NoNewline "." -ForegroundColor Yellow
        Start-Sleep -Seconds 2
        $waited += 2
    }
}

if (-not $ready) {
    Write-Host "`n   ⚠️  Timeout esperando API (puede que esté cargando modelos...)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host ""

# 6. Probar health check
Write-Host "🏥 Probando Health Check..." -ForegroundColor Cyan
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing | ConvertFrom-Json
    Write-Host ($health | ConvertTo-Json -Depth 2) -ForegroundColor Green
}
catch {
    Write-Host "   ⚠️  No se pudo obtener health check (puede estar cargando)" -ForegroundColor Yellow
}

Write-Host ""

# 7. Información de servicios
Write-Host "🔗 Servicios disponibles:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   API Documentation (SWAGGER):" -ForegroundColor White
Write-Host "   └─ http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "   Health Check:" -ForegroundColor White
Write-Host "   └─ http://localhost:8001/health" -ForegroundColor Cyan
Write-Host ""
Write-Host "   API Endpoints:" -ForegroundColor White
Write-Host "   └─ http://localhost:8001" -ForegroundColor Cyan
Write-Host ""
Write-Host "   PostgreSQL (EXTERNA):" -ForegroundColor Yellow
Write-Host "   └─ localhost:5432 (tu BD existente)" -ForegroundColor Cyan
Write-Host "   └─ usuario: postgres" -ForegroundColor Cyan
Write-Host "   └─ contraseña: 1234" -ForegroundColor Cyan
Write-Host ""

# 8. Instrucciones de testing
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           SIGUIENTE: PRUEBA LOS ENDPOINTS                 ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "OPCIÓN 1: Interfaz web (Recomendado)" -ForegroundColor Yellow
Write-Host "   Abre en navegador: http://localhost:8001/docs" -ForegroundColor White
Write-Host ""
Write-Host "OPCIÓN 2: Script de testing" -ForegroundColor Yellow
Write-Host "   Ejecuta: python test_api.py" -ForegroundColor White
Write-Host ""
Write-Host "OPCIÓN 3: PowerShell/cmd" -ForegroundColor Yellow
Write-Host "   Invoke-WebRequest -Uri http://localhost:8001/health" -ForegroundColor White
Write-Host ""
Write-Host "OPCIÓN 4: Ver logs en tiempo real" -ForegroundColor Yellow
Write-Host "   docker-compose logs -f ml-api" -ForegroundColor White
Write-Host ""

# 9. Limpiar (Opcional)
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           CUANDO TERMINES DE PROBAR                       ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Para parar servicios sin eliminar datos:" -ForegroundColor Yellow
Write-Host "   docker-compose stop" -ForegroundColor White
Write-Host ""
Write-Host "Para parar y eliminar todo:" -ForegroundColor Yellow
Write-Host "   docker-compose down" -ForegroundColor White
Write-Host ""
Write-Host "Para ver logs:" -ForegroundColor Yellow
Write-Host "   docker-compose logs -f" -ForegroundColor White
Write-Host ""

Write-Host "✅ Setup completado. API lista en http://localhost:8001" -ForegroundColor Green
Write-Host ""
