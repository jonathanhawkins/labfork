# AI Research Lab - Windows Setup Script
# Run in PowerShell as Administrator

param(
    [switch]$SkipDocker,
    [switch]$SkipOllama,
    [switch]$UseWSL
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[X] $msg" -ForegroundColor Red }
function Write-Info { param($msg) Write-Host "[*] $msg" -ForegroundColor Cyan }
function Write-Step { param($msg) Write-Host "`n>>> $msg" -ForegroundColor Blue }

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This script should be run as Administrator for full functionality"
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Blue
Write-Host "  AI Research Lab - Windows Setup" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue
Write-Host ""

# Detect architecture
$arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "x86" }
Write-Info "Architecture: $arch"

# Check for NVIDIA GPU
$hasNvidia = $false
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpu) {
        $hasNvidia = $true
        Write-Info "NVIDIA GPU detected: $($gpu.Name)"
    }
} catch {
    Write-Info "No NVIDIA GPU detected"
}

# Step 1: Check/Install Chocolatey
Write-Step "Checking Chocolatey..."
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    $env:Path += ";$env:ALLUSERSPROFILE\chocolatey\bin"
} else {
    Write-Success "Chocolatey is installed"
}

# Step 2: Install Git
Write-Step "Checking Git..."
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Git..."
    choco install git -y
    $env:Path += ";C:\Program Files\Git\bin"
} else {
    Write-Success "Git is installed"
}

# Step 3: Install Docker Desktop
if (-not $SkipDocker) {
    Write-Step "Checking Docker..."
    if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Info "Installing Docker Desktop..."
        choco install docker-desktop -y

        Write-Warning "Docker Desktop has been installed."
        Write-Warning "Please restart your computer and run this script again to continue."
        Write-Warning "After restart, make sure Docker Desktop is running."
        exit 0
    } else {
        # Check if Docker is running
        try {
            docker info | Out-Null
            Write-Success "Docker is running"
        } catch {
            Write-Warning "Docker is installed but not running"
            Write-Info "Please start Docker Desktop and run this script again"
            exit 1
        }
    }
}

# Step 4: Install Node.js
Write-Step "Checking Node.js..."
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Node.js..."
    choco install nodejs-lts -y
    $env:Path += ";C:\Program Files\nodejs"
} else {
    $nodeVersion = node -v
    Write-Success "Node.js $nodeVersion is installed"
}

# Step 5: Install Python
Write-Step "Checking Python..."
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Python..."
    choco install python311 -y
    $env:Path += ";C:\Python311;C:\Python311\Scripts"
} else {
    $pythonVersion = python --version
    Write-Success "$pythonVersion is installed"
}

# Step 6: Install Ollama
if (-not $SkipOllama) {
    Write-Step "Checking Ollama..."
    if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Info "Installing Ollama..."

        # Download Ollama installer
        $ollamaUrl = "https://ollama.com/download/windows"
        $installerPath = "$env:TEMP\OllamaSetup.exe"

        Write-Info "Downloading Ollama..."
        Invoke-WebRequest -Uri $ollamaUrl -OutFile $installerPath

        Write-Info "Running Ollama installer..."
        Start-Process -FilePath $installerPath -Wait

        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        Write-Success "Ollama is installed"
    }

    # Pull model
    Write-Step "Pulling Ollama model..."
    $model = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen3-coder:30b" }

    $modelList = ollama list 2>$null
    if ($modelList -notmatch $model) {
        Write-Info "Pulling $model (this may take a while)..."
        ollama pull $model
    } else {
        Write-Success "$model is already pulled"
    }
}

# Step 7: Create environment file
Write-Step "Setting up environment..."
if (!(Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Success "Created .env file from template"
    Write-Warning "Please edit .env to add your API keys"
} else {
    Write-Success ".env file already exists"
}

# Step 8: Install frontend dependencies
Write-Step "Installing frontend dependencies..."
Push-Location frontend
npm install
Pop-Location
Write-Success "Frontend dependencies installed"

# Step 9: Install backend dependencies
Write-Step "Installing backend dependencies..."
Push-Location backend
if (!(Test-Path "venv")) {
    python -m venv venv
}
& ".\venv\Scripts\Activate.ps1"
pip install -r requirements.txt
deactivate
Pop-Location
Write-Success "Backend dependencies installed"

# Final summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the lab:"
Write-Host ""
Write-Host "  Option 1: Docker (Recommended)" -ForegroundColor Cyan

if ($hasNvidia) {
    Write-Host "    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d"
} else {
    Write-Host "    docker compose up -d"
}

Write-Host ""
Write-Host "  Option 2: Local Development" -ForegroundColor Cyan
Write-Host "    # PowerShell 1 - Frontend"
Write-Host "    cd frontend; npm run dev"
Write-Host ""
Write-Host "    # PowerShell 2 - Backend"
Write-Host "    cd backend; .\venv\Scripts\Activate.ps1; python main.py"
Write-Host ""
Write-Host "Lab will be available at: http://localhost:3003"
Write-Host ""

if ($hasNvidia) {
    Write-Host "NVIDIA GPU Detected:" -ForegroundColor Blue
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    Write-Host ""
}
