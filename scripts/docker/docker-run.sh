#!/bin/bash

# Script helper para ejecutar contenedores Docker con o sin GPU

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    echo -e "${BLUE}Uso: ./scripts/docker-run.sh [COMANDO] [OPCIONES]${NC}"
    echo ""
    echo "Comandos disponibles:"
    echo "  dev          Inicia contenedor de desarrollo (interactivo)"
    echo "  test         Ejecuta tests con cobertura"
    echo "  ci           Ejecuta pipeline CI (linting + tests)"
    echo ""
    echo "Opciones:"
    echo "  --gpu        Usa GPU (requiere configuración previa)"
    echo "  --cpu        Usa CPU (por defecto)"
    echo "  --help       Muestra esta ayuda"
    echo ""
    echo "Ejemplos:"
    echo "  ${GREEN}./scripts/docker-run.sh dev${NC}              # Desarrollo con CPU"
    echo "  ${GREEN}./scripts/docker-run.sh dev --gpu${NC}        # Desarrollo con GPU"
    echo "  ${GREEN}./scripts/docker-run.sh test${NC}             # Tests con CPU"
    echo "  ${GREEN}./scripts/docker-run.sh ci --gpu${NC}         # CI con GPU"
}

# Función para verificar si GPU está disponible
check_gpu() {
    if [ -e /dev/nvidia0 ] && [ -e /dev/nvidiactl ]; then
        echo -e "${GREEN}✓ GPU detectada${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ GPU no detectada. Ver docs/GPU_DOCKER_SETUP.md para configuración${NC}"
        return 1
    fi
}

# Valores por defecto
COMMAND=""
PROFILE="cpu"

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        dev|test|ci)
            COMMAND=$1
            shift
            ;;
        --gpu)
            PROFILE="gpu"
            shift
            ;;
        --cpu)
            PROFILE="cpu"
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Argumento desconocido '$1'${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Validar que se especificó un comando
if [ -z "$COMMAND" ]; then
    echo -e "${RED}Error: Debes especificar un comando${NC}"
    print_help
    exit 1
fi

# Verificar GPU si se solicitó
if [ "$PROFILE" = "gpu" ]; then
    if ! check_gpu; then
        echo -e "${YELLOW}Continuando con CPU...${NC}"
        PROFILE="cpu"
    fi
fi

# Determinar el servicio a ejecutar
case $COMMAND in
    dev)
        if [ "$PROFILE" = "gpu" ]; then
            SERVICE="nembogueta-gpu"
        else
            SERVICE="nembogueta"
        fi
        echo -e "${BLUE}Iniciando contenedor de desarrollo (${PROFILE})...${NC}"
        docker compose --profile "$PROFILE" up "$SERVICE"
        ;;
    test)
        if [ "$PROFILE" = "gpu" ]; then
            SERVICE="nembogueta-test-gpu"
        else
            SERVICE="nembogueta-test"
        fi
        echo -e "${BLUE}Ejecutando tests (${PROFILE})...${NC}"
        docker compose --profile "$PROFILE" up "$SERVICE"
        ;;
    ci)
        if [ "$PROFILE" = "gpu" ]; then
            SERVICE="nembogueta-ci-gpu"
        else
            SERVICE="nembogueta-ci"
        fi
        echo -e "${BLUE}Ejecutando CI pipeline (${PROFILE})...${NC}"
        docker compose --profile "$PROFILE" up "$SERVICE"
        ;;
esac

echo -e "${GREEN}✓ Completado${NC}"
