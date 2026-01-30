# AI Research Lab - Deployment Guide

Deploy your own AI Research Lab in minutes. Choose the deployment method that works best for you.

## Quick Start (5 Minutes)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/labfork.git
cd labfork

# Run the setup script
./setup.sh

# Start the lab
docker compose up -d

# Open http://localhost:3003
```

### Option 2: Interactive Wizard

```bash
./deploy/wizard.sh
```

The wizard will guide you through:
- Choosing a deployment method
- Configuring your hardware
- Setting up API keys
- Creating your first lab

## Deployment Methods

### 1. Docker Compose (Local)

Best for: Development, testing, and local research.

**Requirements:**
- Docker Desktop
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

**Without GPU:**
```bash
docker compose up -d
```

**With NVIDIA GPU:**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

**Development Mode (with hot reload):**
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### 2. Vercel + Cloud GPU

Best for: Production deployments with scalable GPU access.

**Deploy Frontend to Vercel:**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-repo/labfork/tree/main/frontend)

**Environment Variables:**
- `NEXT_PUBLIC_API_URL` - Backend API URL
- `NEXT_PUBLIC_OLLAMA_URL` - Ollama server URL

**Deploy GPU Backend to RunPod:**
```bash
export RUNPOD_API_KEY="your-api-key"
./deploy/deploy-runpod.sh create "NVIDIA RTX 4090" 1
```

### 3. Railway

Best for: Simple cloud deployment without GPU.

```bash
./deploy/deploy-railway.sh deploy
```

### 4. Local Development (No Docker)

**macOS:**
```bash
./deploy/setup-mac.sh

# Start services
cd frontend && npm run dev &
cd backend && source venv/bin/activate && python main.py
```

**Linux:**
```bash
./deploy/setup-linux.sh

# Start services
cd frontend && npm run dev &
cd backend && source venv/bin/activate && python main.py
```

**Windows (PowerShell as Admin):**
```powershell
.\deploy\setup-windows.ps1

# Start services (in separate terminals)
cd frontend; npm run dev
cd backend; .\venv\Scripts\Activate.ps1; python main.py
```

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 20GB | 50GB+ |
| CPU | 4 cores | 8+ cores |
| GPU | None (CPU-only) | 8GB+ VRAM |

### GPU Recommendations

| GPU | VRAM | Models | Performance |
|-----|------|--------|-------------|
| None (CPU) | - | qwen3-coder:7b | Slow |
| RTX 3060 | 12GB | qwen3-coder:14b | Good |
| RTX 4070 | 12GB | qwen3-coder:14b | Good |
| RTX 4090 | 24GB | qwen3-coder:30b | Excellent |
| A100 | 40GB+ | qwen3-coder:30b | Best |
| Apple M3 Pro | 18GB | qwen3-coder:14b | Good |
| Apple M4 Max | 64GB+ | qwen3-coder:30b | Excellent |

## Cloud Provider Comparison

| Provider | GPU Available | Cost/Hour | Difficulty | Best For |
|----------|--------------|-----------|------------|----------|
| Local Docker | Depends | Free | Easy | Development |
| Vercel | No | Free tier | Easy | Frontend |
| Railway | No | $5/mo+ | Easy | Small projects |
| RunPod | Yes (RTX 4090) | $0.44/hr | Medium | GPU compute |
| AWS EC2 | Yes (A10G) | $1.00+/hr | Hard | Enterprise |
| GCP | Yes (T4, A100) | $0.80+/hr | Hard | Enterprise |

## Cost Estimation

### Development (Local)
- **Cost:** Free
- **Requirements:** Your own hardware

### Light Usage (Cloud)
- **Vercel Free Tier:** $0/month
- **Railway:** $5-20/month
- **Estimated Total:** $5-20/month

### Regular Research (Cloud GPU)
- **Vercel Pro:** $20/month
- **RunPod (10 hrs/week):** $18/month
- **Estimated Total:** $40-50/month

### Heavy Research (Cloud GPU)
- **Vercel Pro:** $20/month
- **RunPod (40 hrs/week):** $70/month
- **Estimated Total:** $90-100/month

## Environment Variables

### Required

None - the lab works out of the box with sensible defaults.

### Optional (Enhance Functionality)

```bash
# API Keys
ANTHROPIC_API_KEY=       # Claude-powered features
OPENAI_API_KEY=          # Alternative LLM provider
SEMANTIC_SCHOLAR_API_KEY= # Increased paper fetch rate limits

# Ports (customize if needed)
FRONTEND_PORT=3003
BACKEND_PORT=8003
OLLAMA_PORT=11434
DB_PORT=5432

# Database
DB_PASSWORD=your-secure-password

# GPU
USE_GPU=true             # Enable GPU acceleration
```

## Health Checks

Check if all services are running:

```bash
./deploy/health-check.sh
```

Or via API:
```bash
curl http://localhost:3003/api/health
```

## Troubleshooting

### Docker won't start

```bash
# Check Docker is running
docker info

# View logs
docker compose logs -f

# Restart services
docker compose restart
```

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# ... (see setup-linux.sh for full instructions)
```

### Ollama model won't load

```bash
# Check available memory
ollama ps

# Pull model manually
ollama pull qwen3-coder:14b

# Use smaller model
export OLLAMA_MODEL=qwen3-coder:7b
```

### Port already in use

```bash
# Find process using port
lsof -i :3003

# Kill process or change port
export FRONTEND_PORT=3004
docker compose up -d
```

### Out of memory

```bash
# Check memory usage
docker stats

# Reduce model size
export OLLAMA_MODEL=qwen3-coder:7b

# Increase Docker memory limit (Docker Desktop settings)
```

## Security Best Practices

1. **Never commit .env files** - They contain secrets
2. **Use strong DB passwords** - Not the defaults
3. **Limit network exposure** - Only expose necessary ports
4. **Regular updates** - Keep dependencies updated
5. **API key rotation** - Rotate keys periodically

## Backup & Recovery

### Backup Data

```bash
# Backup database
docker compose exec postgres pg_dump -U lab researchlab > backup.sql

# Backup volumes
docker run --rm -v labfork_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/db-backup.tar.gz /data
```

### Restore Data

```bash
# Restore database
docker compose exec -T postgres psql -U lab researchlab < backup.sql

# Restore volumes
docker run --rm -v labfork_postgres-data:/data -v $(pwd):/backup alpine tar xzf /backup/db-backup.tar.gz -C /
```

## Updating

```bash
# Pull latest changes
git pull

# Rebuild containers
docker compose build

# Restart services
docker compose up -d
```

## Getting Help

- **Documentation:** Check the `/docs` directory
- **Issues:** Open a GitHub issue
- **Discord:** Join our community (link TBD)

---

Happy researching! 🔬
