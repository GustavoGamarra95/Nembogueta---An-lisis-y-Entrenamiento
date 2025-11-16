# Script helper para ejecutar contenedores Docker con o sin GPU (Windows PowerShell)

param(
    [Parameter(Position=0)]
    [ValidateSet('dev', 'test', 'ci')]
    [string]$Command,

    [switch]$Gpu,
    [switch]$Cpu,
    [switch]$Help
)

function Print-Help {
    Write-Host "Uso: .\scripts\docker-run.ps1 [COMANDO] [OPCIONES]" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Comandos disponibles:"
    Write-Host "  dev          Inicia contenedor de desarrollo (interactivo)"
    Write-Host "  test         Ejecuta tests con cobertura"
    Write-Host "  ci           Ejecuta pipeline CI (linting + tests)"
    Write-Host ""
    Write-Host "Opciones:"
    Write-Host "  -Gpu         Usa GPU (requiere configuración previa)"
    Write-Host "  -Cpu         Usa CPU (por defecto)"
    Write-Host "  -Help        Muestra esta ayuda"
    Write-Host ""
    Write-Host "Ejemplos:"
    Write-Host "  .\scripts\docker-run.ps1 dev              # Desarrollo con CPU" -ForegroundColor Green
    Write-Host "  .\scripts\docker-run.ps1 dev -Gpu         # Desarrollo con GPU" -ForegroundColor Green
    Write-Host "  .\scripts\docker-run.ps1 test             # Tests con CPU" -ForegroundColor Green
    Write-Host "  .\scripts\docker-run.ps1 ci -Gpu          # CI con GPU" -ForegroundColor Green
}

function Test-GpuAvailable {
    # En WSL2, verificar si los dispositivos GPU están disponibles
    $wslGpu = wsl.exe -e test -e /dev/nvidia0
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ GPU detectada" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠ GPU no detectada. Ver docs/GPU_DOCKER_SETUP.md para configuración" -ForegroundColor Yellow
        return $false
    }
}

# Mostrar ayuda si se solicita
if ($Help -or -not $Command) {
    Print-Help
    exit 0
}

# Determinar perfil (CPU o GPU)
$Profile = "cpu"
if ($Gpu) {
    $Profile = "gpu"
    if (-not (Test-GpuAvailable)) {
        Write-Host "Continuando con CPU..." -ForegroundColor Yellow
        $Profile = "cpu"
    }
}

# Determinar el servicio a ejecutar
$Service = switch ($Command) {
    'dev' {
        if ($Profile -eq "gpu") { "nembogueta-gpu" } else { "nembogueta" }
    }
    'test' {
        if ($Profile -eq "gpu") { "nembogueta-test-gpu" } else { "nembogueta-test" }
    }
    'ci' {
        if ($Profile -eq "gpu") { "nembogueta-ci-gpu" } else { "nembogueta-ci" }
    }
}

# Ejecutar el comando
Write-Host "Iniciando $Command ($Profile)..." -ForegroundColor Blue
docker compose --profile $Profile up $Service

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Completado" -ForegroundColor Green
} else {
    Write-Host "✗ Error al ejecutar" -ForegroundColor Red
    exit $LASTEXITCODE
}
