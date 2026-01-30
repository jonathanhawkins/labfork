# Phase 4: Lab Creation Wizard

## Overview

The Lab Creation Wizard provides a guided multi-step experience for setting up new research labs. Users can configure their domain, hardware, research goals, and let AI assist with initial task generation.

## Architecture

```
/lab/new                      Entry point
    │
    ▼
LabWizard                     Main container component
    │
    ├── WizardStepWelcome     Introduction and quick setup
    ├── WizardStepDomain      Domain selection/creation
    ├── WizardStepHardware    Hardware configuration
    ├── WizardStepResearch    Goal input and AI analysis
    └── WizardStepReview      Final review and launch
```

## Components

### LabWizard (`/components/lab-wizard/LabWizard.tsx`)

Main orchestrator component that manages:
- Step navigation (URL-based state)
- Configuration state
- Validation
- Lab creation API calls

**Props:**
- `existingDomains?: DomainConfig[]` - Existing domains to select from
- `initialDomain?: string` - Pre-selected domain slug
- `onClose?: () => void` - Close handler

### WizardStepWelcome

Introduction step with:
- Feature overview (4 cards)
- What gets created list
- "Get Started" and "Quick Setup" buttons
- Estimated completion time

### WizardStepDomain

Domain configuration with:
- Existing domain selection (if any)
- Template selection (Voice Clone, Quant Trading, Robotics, Biotech)
- Custom domain form with name, description, arXiv categories

### WizardStepHardware

Hardware setup with:
- Local machine detection (GPU, Ollama)
- Remote SSH configuration with connection testing
- Cloud provider selection (RunPod, AWS, GCP, Lambda Labs)
- GPU requirement info per domain

### WizardStepResearch

Research goal configuration:
- Free-text goal input (500 char limit)
- Keywords input
- AI analysis button (Claude-powered)
- Task selection from AI suggestions
- Suggested papers display

### WizardStepReview

Final review showing:
- Domain summary with Edit link
- Hardware configuration with Edit link
- Research goal with Edit link
- Tasks to be created (collapsible)
- Files to be created (collapsible)
- Validation errors (if any)
- Launch Lab button

## Utilities

### `/lib/lab-wizard/types.ts`

Type definitions including:
- `LabConfig` - Complete wizard configuration
- `HardwareConfig` - Hardware setup (local/SSH/cloud)
- `ResearchGoal` - Research goal and tasks
- `InitialTask` - Task structure
- `GpuInfo`, `SSHConfig`, `CloudConfig`
- `WIZARD_STEPS` - Step definitions

### `/lib/lab-wizard/gpu-detection.ts`

GPU detection utilities:
- `detectLocalGpu()` - Detect local GPU via API
- `detectLocalSystem()` - Get full system info
- `checkGpuMeetsDomain()` - Check GPU requirements
- `formatGpuInfo()` - Format GPU for display
- `getGpuRecommendation()` - Get domain-specific recommendations
- `DOMAIN_GPU_REQUIREMENTS` - Requirements by domain

### `/lib/lab-wizard/ssh-tester.ts`

SSH connection utilities:
- `testSSHConnection()` - Test SSH connection
- `detectRemoteGpu()` - Detect GPU on remote machine
- `getRemoteSystemInfo()` - Get remote system details
- `validateSSHConfig()` - Validate SSH configuration
- `formatSSHError()` - Format errors for display
- `KNOWN_SSH_HOSTS` - Pre-configured hosts (RTX 4090)

### `/lib/lab-wizard/goal-analyzer.ts`

AI-powered goal analysis:
- `analyzeGoal()` - Call Claude for goal analysis
- `generateInitialTasks()` - Generate tasks from goal
- `parseGoalAnalysisResponse()` - Parse AI response
- `estimateTimeline()` - Estimate research timeline
- `applyAnalysisToGoal()` - Apply analysis to config

### `/lib/lab-wizard/scaffolding.ts`

Lab scaffolding utilities:
- `generateDomainYaml()` - Generate domain.yaml content
- `generateHardwareSection()` - Generate hardware config
- `generateInitialTasksFromConfig()` - Extract initial tasks
- `createLab()` - Create lab via API
- `validateLabConfig()` - Validate complete config
- `getLabDirectoryStructure()` - Get files to create
- `DEFAULT_PROMPTS` - Default prompt templates

## API Endpoints

### `POST /api/lab/hardware`

Detect local hardware.

**Response:**
```json
{
  "success": true,
  "gpu": {
    "name": "Apple M4 Pro",
    "vram": 48,
    "available": true
  },
  "ollama": { "available": true, "models": ["qwen3-coder:30b"] }
}
```

### `POST /api/lab/hardware/ssh`

Test SSH connection and detect remote GPU.

**Request:**
```json
{
  "action": "test" | "detect-gpu" | "system-info",
  "host": "192.168.1.100",
  "port": 22,
  "user": "admin"
}
```

### `POST /api/lab/analyze-goal`

Analyze research goal with AI.

**Request:**
```json
{
  "goalText": "Build TTS system with emotion control",
  "preferredDomain": "voice-clone",
  "hardwareVram": 24,
  "generateTasks": true
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "suggestedDomain": "voice-clone",
    "arxivCategories": ["cs.SD", "eess.AS"],
    "keywords": ["TTS", "prosody"],
    "reasoning": "Your goal focuses on...",
    "suggestedTasks": [...],
    "estimatedTimeline": "About 2 weeks"
  }
}
```

### `POST /api/lab/create`

Create new lab.

**Request:** Complete `LabConfig` object

**Response:**
```json
{
  "success": true,
  "labId": "lab_123",
  "domainSlug": "voice-clone",
  "files": ["domain.yaml", "prompts/research.md", ...],
  "taskIds": ["task_1", "task_2"],
  "redirectUrl": "/lab?domain=voice-clone"
}
```

## File Structure Created

When creating a new domain:

```
.domains/{slug}/
├── domain.yaml           # Domain configuration
└── prompts/
    ├── research.md       # Research agent prompt
    ├── implementation.md # Implementation agent prompt
    └── evaluation.md     # Evaluation agent prompt
```

## Testing

### Unit Tests (103 tests)

```
__tests__/lib/lab-wizard/
├── gpu-detection.test.ts    # 21 tests
├── ssh-tester.test.ts       # 18 tests
├── goal-analyzer.test.ts    # 35 tests
└── scaffolding.test.ts      # 29 tests
```

### Component Tests (180 tests)

```
__tests__/components/lab-wizard/
├── WizardStepWelcome.test.tsx   # 12 tests
├── WizardStepDomain.test.tsx    # 21 tests
├── WizardStepHardware.test.tsx  # 37 tests
├── WizardStepResearch.test.tsx  # 39 tests
├── WizardStepReview.test.tsx    # 45 tests
└── LabWizard.test.tsx           # 26 tests
```

### Integration Tests (15 tests)

```
__tests__/integration/
└── lab-wizard-flow.test.tsx     # 15 tests
```

Run all tests:
```bash
npm test -- __tests__/lib/lab-wizard/ __tests__/components/lab-wizard/ __tests__/integration/lab-wizard-flow.test.tsx
```

## Usage

### Quick Setup

```typescript
// Navigate to wizard
router.push('/lab/new');

// Or with pre-selected domain
router.push('/lab/new?domain=voice-clone');
```

### Programmatic Lab Creation

```typescript
import { createLab, validateLabConfig } from '@/lib/lab-wizard/scaffolding';

const config: LabConfig = {
  createNewDomain: true,
  domain: {
    name: "My Lab",
    slug: "my-lab",
    // ...
  },
  hardware: { type: "local" },
  research: {
    path: "goal",
    goal: { description: "My research goal" }
  }
};

const { valid, errors } = validateLabConfig(config);
if (valid) {
  const result = await createLab(config);
  if (result.success) {
    router.push(result.redirectUrl);
  }
}
```

## Known SSH Hosts

The wizard includes pre-configured quick-connect for known hosts:

| Name | Host | User | Description |
|------|------|------|-------------|
| RTX 4090 (Tailscale) | $REMOTE_GPU_HOST | doc | 24GB VRAM - Fast training |

## Domain Templates

| Template | arXiv Categories | GPU Requirement |
|----------|-----------------|-----------------|
| Voice Cloning | cs.SD, eess.AS, cs.CL | 24GB recommended |
| Quant Trading | q-fin.ST, cs.LG, stat.ML | 8GB minimum |
| Robotics | cs.RO, cs.AI, cs.LG | 24GB recommended |
| Biotech | q-bio.BM, cs.CL, cs.LG | 48GB+ recommended |

## Future Enhancements

1. **Cloud Provider Integration** - Direct API integration with RunPod/AWS
2. **Team Collaboration** - Share lab configurations
3. **Template Marketplace** - Community-contributed templates
4. **Advanced Hardware Detection** - Multi-GPU support
5. **Cost Estimation** - Estimate cloud costs before creation
