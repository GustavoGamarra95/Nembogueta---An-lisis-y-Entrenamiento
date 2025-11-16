# Script para ejecutar preprocesamiento de videos en Docker con GPU
# Uso: .\scripts\docker-preprocess.ps1 [DATASET] [OPCIONES]

param(
    [Parameter(Position=0)]
    [ValidateSet('vlibrasil', 'lspy', 'custom')]
    [string]$Dataset = 'vlibrasil',

    [Parameter(Position=1)]
    [int]$MaxVideos,

    [ValidateSet('hands', 'upper_body', 'holistic')]
    [string]$Preset = 'hands',

    [int]$TargetLength = 300,

    [switch]$Gpu,
    [switch]$Stats,
    [switch]$NoSkip,
    [switch]$Help
)

function Print-Help {
    Write-Host "Uso: .\scripts\docker-preprocess.ps1 [DATASET] [OPCIONES]" -ForegroundColor Blue
    Write-Host ""
    Write-Host "Datasets disponibles:"
    Write-Host "  vlibrasil    Procesa V-LIBRASIL (4,086 videos)"
    Write-Host "  lspy         Procesa LSPy"
    Write-Host "  custom       Configuración personalizada"
    Write-Host ""
    Write-Host "Opciones:"
    Write-Host "  -MaxVideos N     Procesar solo N videos (para pruebas)"
    Write-Host "  -Preset TYPE     Tipo de extracción: hands, upper_body, holistic (default: hands)"
    Write-Host "  -TargetLength N  Longitud de secuencia (default: 300)"
    Write-Host "  -Gpu             Usar GPU (recomendado, requiere configuración)"
    Write-Host "  -Stats           Solo mostrar estadísticas"
    Write-Host "  -NoSkip          Reprocesar videos existentes"
    Write-Host "  -Help            Muestra esta ayuda"
    Write-Host ""
    Write-Host "Ejemplos:"
    Write-Host "  # Ver estadísticas de V-LIBRASIL" -ForegroundColor Green
    Write-Host "  .\scripts\docker-preprocess.ps1 vlibrasil -Stats" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  # Prueba rápida (10 videos) con GPU" -ForegroundColor Green
    Write-Host "  .\scripts\docker-preprocess.ps1 vlibrasil -MaxVideos 10 -Gpu" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  # Procesar todos los videos de V-LIBRASIL con GPU" -ForegroundColor Green
    Write-Host "  .\scripts\docker-preprocess.ps1 vlibrasil -Gpu" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  # LSPy con preset holistic" -ForegroundColor Green
    Write-Host "  .\scripts\docker-preprocess.ps1 lspy -Preset holistic -Gpu" -ForegroundColor Yellow
}

function Test-DockerRunning {
    try {
        $null = docker ps 2>&1
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    } catch {
        return $false
    }
    return $false
}

function Test-GpuAvailable {
    # Verificar si Docker Desktop tiene acceso a GPU
    Write-Host "Verificando GPU..." -ForegroundColor Cyan

    # Verificar NVIDIA SMI
    try {
        $nvidiaSmi = docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ GPU NVIDIA detectada y accesible desde Docker" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "⚠ No se pudo acceder a la GPU desde Docker" -ForegroundColor Yellow
    }

    # Verificar en WSL2
    try {
        $wslGpu = wsl.exe -e test -e /dev/nvidia0
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ GPU detectada en WSL2" -ForegroundColor Green
            Write-Host "⚠ Asegúrate de que Docker Desktop esté configurado para usar WSL2" -ForegroundColor Yellow
            return $true
        }
    } catch {
        Write-Host "⚠ GPU no detectada en WSL2" -ForegroundColor Yellow
    }

    return $false
}

# Mostrar ayuda
if ($Help) {
    Print-Help
    exit 0
}

# Verificar Docker
Write-Host "`n=== Verificación del Sistema ===" -ForegroundColor Cyan
if (-not (Test-DockerRunning)) {
    Write-Host "✗ Docker no está corriendo. Inicia Docker Desktop" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Docker está corriendo" -ForegroundColor Green

# Verificar GPU si se solicita
$UseGpu = $false
if ($Gpu) {
    if (Test-GpuAvailable) {
        $UseGpu = $true
        Write-Host "✓ Usando GPU para procesamiento" -ForegroundColor Green
    } else {
        Write-Host "⚠ GPU no disponible, continuando con CPU" -ForegroundColor Yellow
        Write-Host "  Ver DOCKER_GPU_GUIDE.md para configuración" -ForegroundColor Yellow
    }
} else {
    Write-Host "ℹ Usando CPU (añade -Gpu para usar GPU)" -ForegroundColor Cyan
}

# Construir argumentos según el dataset
$VideosDir = ""
$Annotations = ""
$OutputDir = ""
$ExtraArgs = ""

switch ($Dataset) {
    'vlibrasil' {
        $VideosDir = "src/data/videos UFPE (V-LIBRASIL)/data"
        $Annotations = "src/data/videos UFPE (V-LIBRASIL)/annotations.csv"
        $OutputDir = "data/processed/v-librasil"
    }
    'lspy' {
        $VideosDir = "data/raw/lspy"
        $OutputDir = "data/processed/lspy"
        $ExtraArgs = "--auto-infer"
    }
    'custom' {
        Write-Host "Para datasets personalizados, usa el comando docker directamente:" -ForegroundColor Yellow
        Write-Host "docker run --rm --gpus all -v `${PWD}:/app nembogueta-dev python scripts/preprocess_sign_language.py ..." -ForegroundColor Yellow
        exit 0
    }
}

# Construir comando Python
$PythonCmd = "python scripts/preprocess_sign_language.py"
$PythonCmd += " --videos-dir `"$VideosDir`""
if ($Annotations) {
    $PythonCmd += " --annotations `"$Annotations`""
}
$PythonCmd += " --output-dir `"$OutputDir`""
$PythonCmd += " --preset $Preset"
$PythonCmd += " --target-length $TargetLength"

if ($MaxVideos) {
    $PythonCmd += " --max-videos $MaxVideos"
}
if ($Stats) {
    $PythonCmd += " --stats"
}
if ($NoSkip) {
    $PythonCmd += " --no-skip"
}
if ($ExtraArgs) {
    $PythonCmd += " $ExtraArgs"
}

# Mostrar información
Write-Host "`n=== Configuración ===" -ForegroundColor Cyan
Write-Host "Dataset: $Dataset" -ForegroundColor White
Write-Host "Preset: $Preset" -ForegroundColor White
Write-Host "Target Length: $TargetLength" -ForegroundColor White
if ($MaxVideos) {
    Write-Host "Max Videos: $MaxVideos" -ForegroundColor White
}
Write-Host "GPU: $UseGpu" -ForegroundColor White
Write-Host "`nComando: $PythonCmd" -ForegroundColor Gray

# Construir imagen si es necesario
Write-Host "`n=== Construyendo imagen Docker ===" -ForegroundColor Cyan
docker build -t nembogueta-dev .
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Error construyendo imagen Docker" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Imagen construida" -ForegroundColor Green

# Ejecutar el comando
Write-Host "`n=== Ejecutando Procesamiento ===" -ForegroundColor Cyan

if ($UseGpu) {
    # Con GPU
    docker run --rm --gpus all `
        -v "${PWD}:/app" `
        -w /app `
        nembogueta-dev `
        bash -c $PythonCmd
} else {
    # Sin GPU
    docker run --rm `
        -v "${PWD}:/app" `
        -w /app `
        nembogueta-dev `
        bash -c $PythonCmd
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Procesamiento completado exitosamente" -ForegroundColor Green

    if (-not $Stats) {
        Write-Host "`nArchivos procesados guardados en: $OutputDir" -ForegroundColor Cyan

        # Mostrar algunos archivos de ejemplo
        Write-Host "`nEjemplos de archivos generados:" -ForegroundColor Cyan
        Get-ChildItem -Path $OutputDir -Recurse -Filter "*.npy" -ErrorAction SilentlyContinue |
            Select-Object -First 5 |
            ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor Gray }
    }
} else {
    Write-Host "`n✗ Error durante el procesamiento" -ForegroundColor Red
    Write-Host "Revisa el log para más detalles: sign_language_processing.log" -ForegroundColor Yellow
    exit $LASTEXITCODE
}
