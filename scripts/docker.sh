#!/bin/bash
# =============================================================================
# KZA Voice Assistant - Docker Helper Script
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# =============================================================================
# Commands
# =============================================================================

cmd_check() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker installed: $(docker --version)"
    else
        print_error "Docker not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        print_success "Docker Compose installed: $(docker compose version --short)"
    else
        print_error "Docker Compose not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker
    if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        print_success "NVIDIA Container Toolkit installed"
    else
        print_warning "NVIDIA Container Toolkit not found (needed for GPU containers)"
        echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
    
    # Check GPUs
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
        print_success "NVIDIA GPUs detected: $gpu_count"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
            echo "    - $line"
        done
    else
        print_warning "nvidia-smi not found"
    fi
    
    # Check .env file
    if [ -f ".env" ]; then
        print_success ".env file exists"
    else
        print_warning ".env file not found"
        echo "  Run: cp .env.docker.example .env"
    fi
    
    # Check models directory
    if [ -d "models" ]; then
        print_success "models/ directory exists"
        model_count=$(find models -name "*.gguf" -o -name "*.onnx" 2>/dev/null | wc -l)
        echo "    - Found $model_count model files"
    else
        print_warning "models/ directory not found"
        echo "  Run: ./scripts/download_models.sh"
    fi
}

cmd_build() {
    print_header "Building Docker Images"
    
    # Build specific service or all
    if [ -n "$1" ]; then
        echo "Building service: $1"
        docker compose build "$1"
    else
        echo "Building all services..."
        docker compose build
    fi
    
    print_success "Build complete"
}

cmd_up() {
    print_header "Starting Services"
    
    # Check .env
    if [ ! -f ".env" ]; then
        print_error ".env file not found"
        echo "Run: cp .env.docker.example .env"
        exit 1
    fi
    
    # Start services
    if [ "$1" == "-d" ] || [ "$1" == "--detach" ]; then
        docker compose up -d
        print_success "Services started in background"
        echo ""
        echo "View logs: ./scripts/docker.sh logs"
        echo "Check status: ./scripts/docker.sh status"
    else
        docker compose up
    fi
}

cmd_down() {
    print_header "Stopping Services"
    docker compose down
    print_success "Services stopped"
}

cmd_restart() {
    print_header "Restarting Services"
    
    if [ -n "$1" ]; then
        docker compose restart "$1"
        print_success "Restarted: $1"
    else
        docker compose restart
        print_success "All services restarted"
    fi
}

cmd_logs() {
    if [ -n "$1" ]; then
        docker compose logs -f "$1"
    else
        docker compose logs -f
    fi
}

cmd_status() {
    print_header "Service Status"
    docker compose ps
    
    echo ""
    echo "Health checks:"
    for service in chromadb stt embeddings router tts reasoner pipeline; do
        container="kza-$service"
        if docker ps --format '{{.Names}}' | grep -q "$container"; then
            health=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no healthcheck")
            if [ "$health" == "healthy" ]; then
                print_success "$service: $health"
            elif [ "$health" == "unhealthy" ]; then
                print_error "$service: $health"
            else
                print_warning "$service: $health"
            fi
        else
            echo "  $service: not running"
        fi
    done
}

cmd_shell() {
    if [ -z "$1" ]; then
        echo "Usage: $0 shell <service>"
        echo "Services: chromadb, stt, embeddings, router, tts, reasoner, pipeline"
        exit 1
    fi
    docker compose exec "$1" /bin/bash
}

cmd_gpu() {
    print_header "GPU Status"
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
    else
        print_error "nvidia-smi not found"
    fi
}

cmd_clean() {
    print_header "Cleaning Docker Resources"
    
    echo "This will remove:"
    echo "  - Stopped containers"
    echo "  - Unused networks"
    echo "  - Dangling images"
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down --remove-orphans
        docker system prune -f
        print_success "Cleanup complete"
    else
        echo "Cancelled"
    fi
}

cmd_test() {
    print_header "Testing Services"
    
    echo "Testing health endpoints..."
    
    services=(
        "chromadb:8000"
        "stt:8001"
        "embeddings:8002"
        "router:8003"
        "tts:8004"
        "reasoner:8005"
        "pipeline:8080"
    )
    
    for svc in "${services[@]}"; do
        name="${svc%%:*}"
        port="${svc##*:}"
        
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            print_success "$name (port $port): healthy"
        else
            print_error "$name (port $port): unreachable"
        fi
    done
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "KZA Voice Assistant - Docker Helper"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  check       Check prerequisites (Docker, GPUs, etc.)"
    echo "  build       Build Docker images"
    echo "  up          Start all services"
    echo "  up -d       Start services in background"
    echo "  down        Stop all services"
    echo "  restart     Restart services"
    echo "  logs        View logs (follow mode)"
    echo "  status      Show service status"
    echo "  shell       Open shell in container"
    echo "  gpu         Show GPU status"
    echo "  test        Test service health endpoints"
    echo "  clean       Remove stopped containers and unused images"
    echo ""
    echo "Examples:"
    echo "  $0 check              # Verify setup"
    echo "  $0 build              # Build all images"
    echo "  $0 up -d              # Start in background"
    echo "  $0 logs pipeline      # View pipeline logs"
    echo "  $0 shell reasoner     # Shell into reasoner container"
}

case "$1" in
    check)   cmd_check ;;
    build)   cmd_build "$2" ;;
    up)      cmd_up "$2" ;;
    down)    cmd_down ;;
    restart) cmd_restart "$2" ;;
    logs)    cmd_logs "$2" ;;
    status)  cmd_status ;;
    shell)   cmd_shell "$2" ;;
    gpu)     cmd_gpu ;;
    test)    cmd_test ;;
    clean)   cmd_clean ;;
    help|-h|--help) show_help ;;
    *)
        show_help
        exit 1
        ;;
esac
