#!/bin/bash
# AI Research Lab - Railway Deployment Script
# One-click deployment to Railway

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[X]${NC} $1"; }
print_info() { echo -e "${BLUE}[*]${NC} $1"; }
print_step() { echo -e "\n${BLUE}>>> $1${NC}"; }

# Check for Railway CLI
check_railway() {
    if ! command -v railway &> /dev/null; then
        print_info "Installing Railway CLI..."
        if command -v npm &> /dev/null; then
            npm install -g @railway/cli
        elif command -v brew &> /dev/null; then
            brew install railway
        else
            print_error "Please install Railway CLI: https://docs.railway.app/develop/cli"
            exit 1
        fi
    fi
}

# Login to Railway
login_railway() {
    print_step "Logging into Railway..."
    railway login
}

# Create new project
create_project() {
    print_step "Creating Railway project..."
    railway init
}

# Deploy frontend
deploy_frontend() {
    print_step "Deploying Frontend..."

    cd frontend

    # Create railway.json for frontend
    cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "npm run build"
  },
  "deploy": {
    "startCommand": "npm start",
    "healthcheckPath": "/api/health",
    "healthcheckTimeout": 300
  }
}
EOF

    railway up

    cd ..
    print_success "Frontend deployed"
}

# Deploy backend
deploy_backend() {
    print_step "Deploying Backend..."

    cd backend

    # Create railway.json for backend
    cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
EOF

    railway up

    cd ..
    print_success "Backend deployed"
}

# Add PostgreSQL
add_postgres() {
    print_step "Adding PostgreSQL..."
    railway add -p PostgreSQL
    print_success "PostgreSQL added"
}

# Set environment variables
set_env_vars() {
    print_step "Setting environment variables..."

    railway variables set NODE_ENV=production
    railway variables set NEXT_PUBLIC_API_URL=\${{BACKEND_URL}}

    print_info "Set your API keys:"
    echo "  railway variables set ANTHROPIC_API_KEY=your-key"
    echo "  railway variables set OPENAI_API_KEY=your-key"
}

# Get deployment URL
get_url() {
    print_step "Deployment Info"
    railway status
    print_info "Your lab is being deployed!"
    print_info "Check Railway dashboard for the URL"
}

# Full deployment
full_deploy() {
    check_railway
    login_railway
    create_project
    add_postgres
    deploy_backend
    deploy_frontend
    set_env_vars
    get_url
}

# Usage
usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  deploy       Full deployment"
    echo "  frontend     Deploy frontend only"
    echo "  backend      Deploy backend only"
    echo "  status       Check deployment status"
    echo "  logs         View logs"
    echo ""
}

# Main
main() {
    case "$1" in
        deploy)
            full_deploy
            ;;
        frontend)
            check_railway
            deploy_frontend
            ;;
        backend)
            check_railway
            deploy_backend
            ;;
        status)
            check_railway
            railway status
            ;;
        logs)
            check_railway
            railway logs
            ;;
        *)
            usage
            ;;
    esac
}

main "$@"
