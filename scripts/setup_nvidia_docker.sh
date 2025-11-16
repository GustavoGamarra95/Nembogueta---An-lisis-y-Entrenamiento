#!/bin/bash
# Script para instalar NVIDIA Container Toolkit en WSL2 (Ubuntu/Debian)

set -e

echo "============================================================"
echo "INSTALACIÓN DE NVIDIA CONTAINER TOOLKIT"
echo "============================================================"
echo ""

# Verificar si estamos en WSL2
if ! grep -qi microsoft /proc/version; then
    echo "⚠️  Advertencia: No parece ser WSL2"
    read -p "¿Deseas continuar de todos modos? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verificar si Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "✗ Docker no está instalado"
    echo "  Instala Docker primero: https://docs.docker.com/engine/install/"
    exit 1
fi

echo "✓ Docker está instalado: $(docker --version)"
echo ""

# Configurar repositorio de NVIDIA
echo "Configurando repositorio de NVIDIA..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "✓ Repositorio configurado"
echo ""

# Actualizar lista de paquetes
echo "Actualizando lista de paquetes..."
sudo apt-get update

# Instalar NVIDIA Container Toolkit
echo "Instalando NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

echo "✓ NVIDIA Container Toolkit instalado"
echo ""

# Configurar Docker runtime
echo "Configurando Docker runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "✓ Docker runtime configurado"
echo ""

# Reiniciar Docker
echo "Reiniciando Docker..."
sudo systemctl restart docker

echo "✓ Docker reiniciado"
echo ""

# Verificar instalación
echo "============================================================"
echo "VERIFICANDO INSTALACIÓN"
echo "============================================================"
echo ""

echo "Ejecutando prueba con nvidia-smi..."
if docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi; then
    echo ""
    echo "============================================================"
    echo "✓ INSTALACIÓN EXITOSA"
    echo "============================================================"
    echo ""
    echo "Próximos pasos:"
    echo "  1. cd /mnt/c/Users/gugar/Nembogueta---An-lisis-y-Entrenamiento"
    echo "  2. docker-compose build"
    echo "  3. docker-compose up -d nembogueta"
    echo "  4. docker exec -it nembogueta-dev bash"
    echo "  5. python scripts/verify_gpu.py"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "✗ ERROR EN LA VERIFICACIÓN"
    echo "============================================================"
    echo ""
    echo "Posibles soluciones:"
    echo "  1. Verifica que los drivers NVIDIA estén instalados en Windows"
    echo "  2. Ejecuta 'nvidia-smi' en PowerShell para verificar"
    echo "  3. Reinicia WSL2: wsl --shutdown (desde PowerShell)"
    echo "  4. Consulta: docs/GPU_DOCKER_SETUP.md"
    echo ""
    exit 1
fi
