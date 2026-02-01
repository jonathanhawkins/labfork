# Distributed Compute Network

This document describes the distributed compute architecture that replaces Workers AI with our own GPU infrastructure.

## Architecture Overview

```
                    CLOUDFLARE WORKERS (FREE)
                    ========================
                    - Orchestration only
                    - No AI inference
                    - Stores state in D1
                              |
              +---------------+---------------+
              |               |               |
        [D1 Database]   [Cron Trigger]   [API Routes]
              |           (15 min)            |
              |               |               |
              +---------------+---------------+
                              |
                     [Task Dispatcher]
                              |
              +---------------+---------------+
              |               |               |
        [Power Tier]    [Standard]      [Crowd Tier]
        RTX 4090        Contributors    Browser WebGPU
        82.6 TFLOPS     5-40 TFLOPS     0.5-5 TFLOPS
```

## Components

### 1. Cloudflare Workers (Orchestrator)

**Location:** `workers/src/`

The Workers code is now FREE - it only handles:
- Task creation and dispatch
- Device registration
- Result collection
- State management in D1

**No AI inference happens in Workers.**

### 2. Compute API Routes

**Location:** `workers/src/api/compute.ts`

New API endpoints for the compute network:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/compute/devices` | POST | Register a compute device |
| `/api/compute/devices/:id` | PATCH | Heartbeat / status update |
| `/api/compute/devices` | GET | List all devices |
| `/api/compute/tasks` | POST | Create a compute task |
| `/api/compute/tasks/pending` | GET | Get pending tasks for device |
| `/api/compute/tasks/:id/claim` | POST | Claim a task |
| `/api/compute/tasks/:id/complete` | POST | Report completion |
| `/api/compute/stats` | GET | Network statistics |

### 3. 4090 Compute Agent

**Location:** `backend/compute_agent/`

A Python service that runs on the RTX 4090 server:

```bash
# Start the agent (connects to local Workers)
cd backend/compute_agent
./start.sh --local

# Or connect to production
./start.sh --production --workers-url https://labfork-agents.xxx.workers.dev
```

**Features:**
- Registers as a "power" tier device
- Polls for pending tasks
- Executes inference using local models (Llama, Qwen, etc.)
- Reports results back to Workers

### 4. Browser DeviceAgent (WebGPU)

**Location:** `frontend/lib/compute/device-agent.ts`

Browser-based agent that:
- Registers as "crowd" tier device
- Uses WebLLM for inference
- Handles light tasks (draft tokens, embeddings)

## Device Tiers

| Tier | Compute | Memory | Examples | Task Types |
|------|---------|--------|----------|------------|
| **Power** | >=40 TFLOPS | >=16 GB | RTX 4090, A100 | Full inference, planning, verification |
| **Standard** | >=5 TFLOPS | >=8 GB | RTX 3070, M1 Pro | Medium inference, summarization |
| **Crowd** | <5 TFLOPS | <8 GB | Browser WebGPU | Draft tokens, embeddings, classification |

## Database Schema

New tables in D1:

```sql
-- Registered compute devices
CREATE TABLE compute_devices (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  tier TEXT NOT NULL,  -- 'power', 'standard', 'crowd'
  platform TEXT NOT NULL,  -- 'cuda', 'metal', 'webgpu', 'cpu'
  capabilities TEXT NOT NULL,  -- JSON
  status TEXT DEFAULT 'offline',
  current_task_id TEXT,
  last_heartbeat TEXT,
  ...
);

-- Compute tasks (dispatched to network)
CREATE TABLE compute_tasks (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,  -- 'inference', 'embedding', 'assessment', etc.
  input TEXT NOT NULL,  -- JSON: {prompt, systemPrompt, model}
  config TEXT NOT NULL,  -- JSON: {maxTokens, temperature}
  status TEXT DEFAULT 'pending',
  assigned_device_id TEXT,
  result TEXT,  -- JSON
  ...
);
```

## Task Flow

1. **Workflow creates task:** Manager or Worker workflow needs AI inference
2. **Insert compute_task:** Task inserted as "pending" in D1
3. **Device polls:** 4090 agent or browser polls `/tasks/pending`
4. **Device claims:** Sends POST to `/tasks/:id/claim`
5. **Device executes:** Runs inference locally
6. **Device reports:** Sends result to `/tasks/:id/complete`
7. **Workflow continues:** Original workflow polls and receives result

## Running the System

### Local Development

```bash
# Terminal 1: Start Workers locally
cd workers
npm run dev

# Terminal 2: Start 4090 agent (if on GPU server)
cd backend/compute_agent
./start.sh --local
```

### Production

1. Deploy Workers:
```bash
cd workers
npm run deploy
```

2. Update wrangler.toml with production D1 database ID

3. Apply schema:
```bash
wrangler d1 execute labfork-agents-db --file=./schema.sql
```

4. Start 4090 agent on GPU server:
```bash
cd backend/compute_agent
./start.sh --production --workers-url https://labfork-agents.xxx.workers.dev
```

## Cost Savings

| Before | After |
|--------|-------|
| Workers AI: $0.01/1K tokens | Workers AI: $0.00 |
| Limited to Workers AI models | Any local model (Llama, Qwen, etc.) |
| Metered compute | Own GPU = unlimited |

**Result:** Cloudflare costs are now essentially FREE (just D1 storage and minimal Workers invocations).

## Future Enhancements

1. **Speculative Decoding:** Crowd devices generate drafts, power devices verify
2. **Model Sharding:** Split large models across multiple devices
3. **Geographic Routing:** Route tasks to nearest devices
4. **Credit System:** Reward contributors for compute time
