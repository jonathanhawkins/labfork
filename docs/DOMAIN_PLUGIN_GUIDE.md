# Domain Plugin Development Guide

This guide explains how to create custom domain plugins for the AI Research Lab Platform.

## Overview

Domain plugins allow you to customize the research lab for any domain (voice cloning, trading, robotics, etc.). Each domain plugin includes:

- **domain.yaml** - Main configuration file
- **prompts/** - Agent prompt templates
- **activities/** - Activity definitions for 3D visualization

## Quick Start

### 1. Copy the Template

```bash
cp -r .domains/_template .domains/my-domain
cd .domains/my-domain
```

### 2. Edit domain.yaml

```yaml
name: "My Research Lab"
slug: "my-domain"
description: "A custom research domain"

branding:
  primaryColor: "#3b82f6"
  accentColor: "#22c55e"
  backgroundStyle: "sky"  # sky | space | grid | gradient | minimal

scene:
  props:
    - id: computer
      type: supercomputer
      position: [-6, 0, -5]
      scale: 1.3

research:
  arxivCategories:
    - cs.LG
  keywords:
    - machine learning
    - deep learning
```

### 3. Test Your Domain

```bash
# Start the dev server
npm run dev

# Access your domain
curl http://localhost:3003/api/domain/my-domain
```

## Configuration Reference

### domain.yaml Schema

```yaml
# Required fields
name: string           # Display name
slug: string           # URL-safe identifier (lowercase, hyphens)
description: string    # Short description

# Branding (required)
branding:
  primaryColor: string      # Hex color for primary elements
  accentColor: string       # Hex color for accents/highlights
  backgroundStyle: string   # sky | space | grid | gradient | minimal
  backgroundColor?: string  # Optional custom background color
  gradientColors?: [string, string]  # For gradient backgroundStyle
  logo?: string            # Path to logo image

# 3D Scene (required)
scene:
  props: []  # Array of prop definitions (see Props section)
  decorations:
    plants: boolean       # Show decorative plants
    floatingCubes: boolean # Show floating cubes
    particles: boolean    # Show ambient particles
  lighting:
    ambientIntensity: number   # 0-1
    mainLightIntensity: number # 0-1
    mainLightColor: string     # Hex color
  fog:
    enabled: boolean
    near: number
    far: number

# Research Configuration (required)
research:
  arxivCategories: string[]  # arXiv category codes
  keywords: string[]         # Search keywords
  additionalSources?: string[] # semantic-scholar | github | papers-with-code
  maxPapersPerSession?: number

# Evaluation Configuration (optional)
evaluation:
  primaryMetric: string  # ID of the main metric
  baselineComparison?: boolean
  metrics:
    - id: string
      name: string
      description?: string
      range?: [number, number]
      higherIsBetter: boolean
      unit?: string

# Hardware Requirements (optional)
hardware:
  gpuRequired?: boolean
  minGpuVram?: number    # GB
  recommendedGpuVram?: number
  minRam?: number        # GB
  platforms?: string[]   # darwin | linux | win32

# Prompt Configuration (optional)
prompts:
  research?: string      # Path to research prompt template
  implementation?: string # Path to implementation prompt template
  evaluation?: string    # Path to evaluation prompt template
  preamble?: string      # Text prepended to all prompts

# Metadata
tags?: string[]
version?: string
difficulty?: string  # beginner | intermediate | advanced
```

### Props

Available prop types:

| Type | Description | Good For |
|------|-------------|----------|
| `supercomputer` | Large computing cluster | Training, heavy compute |
| `microphone` | Audio input device | Voice recording, audio |
| `speaker` | Audio output device | Audio playback, TTS |
| `server` | Server rack | APIs, inference |
| `emotion-verify` | Emotion display panel | Sentiment, emotions |
| `terminal` | Computer terminal | Coding, development |
| `chart-wall` | Data visualization | Trading, analytics |
| `robot-arm` | Robotic arm | Robotics, automation |
| `camera` | Video camera | Computer vision |
| `molecule` | Molecule model | Chemistry, biology |

Prop definition:

```yaml
props:
  - id: my-prop          # Unique identifier
    type: supercomputer  # Prop type from list above
    position: [-6, 0, -5] # [x, y, z] coordinates
    scale: 1.3           # Size multiplier (default: 1)
    rotation: 0          # Y-axis rotation in radians
    accentColor: 0x00ffaa # Hex number for glow color
```

## Activities

Activities define what agents can do and how it's visualized.

### Activity YAML Schema

Create files in `.domains/my-domain/activities/`:

```yaml
# activities/training.yaml
id: training
name: Model Training
description: Training a machine learning model
icon: Brain  # Lucide icon name

visualization:
  prop: supercomputer      # Which prop to use/highlight
  animation: focused       # idle | focused | typing | walking | thinking
  particles: data-flow     # none | sparks | data-flow | audio-waves | code-rain
  color: 0x4ade80         # Accent color (hex number)
  workLocation: supercomputer  # Where agent moves
  showProgress: true       # Show progress bar
  highlightProp: true      # Highlight the prop

agentBehavior:
  defaultStatus: working   # idle | working | thinking
  typingSpeed: 1.0        # Multiplier for typing animation
  faceProp: true          # Agent faces the prop

progressDetection:
  patterns:
    - "Epoch\\s+(\\d+)/(\\d+)"
    - "loss:\\s+([\\d.]+)"
  completionPattern: "Training complete"
  completionMessage: "Training finished"
```

### Available Icons

The following Lucide icons are available:

- Activity icons: `Brain`, `Mic`, `Sparkles`, `BarChart`, `Search`, `Code`, `Pause`, `Play`
- Hardware: `Cpu`, `Server`, `Database`, `HardDrive`, `Cloud`
- Process: `Zap`, `Flame`, `Activity`, `TrendingUp`
- Audio: `AudioWaveform`, `FileAudio`, `Volume2`, `Radio`
- Development: `GitBranch`, `Terminal`, `FileCode`
- Science: `FlaskConical`, `TestTube`, `Microscope`, `Beaker`
- General: `Lightbulb`, `Target`, `Rocket`, `Settings`, `Wrench`, `Bug`
- Status: `CheckCircle`, `XCircle`, `Clock`, `Timer`, `RefreshCw`

## Prompts

Create prompt templates in `.domains/my-domain/prompts/`:

### research.md

Template for web research agents:

```markdown
# Research Agent Prompt

You are a research agent for the {{domainName}} lab.

## Your Research Topic
{{topic}}

## arXiv Categories
{{arxivCategories}}

## Instructions
1. Use WebSearch to find papers on this topic
2. Check for duplicates before creating tasks
3. Create tasks for promising findings
```

### implementation.md

Template for implementation agents:

```markdown
# Implementation Agent Prompt

## Task Details
**Task ID**: {{taskId}}
**Subject**: {{taskSubject}}

## Instructions
1. Read existing code for patterns
2. Implement the technique
3. Write tests
4. Mark task complete
```

### evaluation.md

Template for evaluation agents:

```markdown
# Evaluation Agent Prompt

## Metrics to Compute
{{#each metrics}}
- **{{name}}** ({{id}}): {{description}}
{{/each}}

## Primary Metric: {{primaryMetric}}
```

## Best Practices

### Naming Conventions

- **slug**: lowercase, hyphens only (`my-domain`, not `My_Domain`)
- **Activity IDs**: lowercase, underscores OK (`model_training`)
- **Prop IDs**: lowercase, descriptive (`main-computer`)

### Color Selection

- **primaryColor**: Use for main branding elements
- **accentColor**: Use for highlights and active states
- **Prop colors**: Use hex numbers (0xAARRGGBB format)

### Performance

- Limit props to 5-7 per scene
- Use `showProgress: false` for quick activities
- Set appropriate `particles` (none for idle activities)

### Research Configuration

- Include 2-4 arXiv categories
- Use specific, relevant keywords
- Set `maxPapersPerSession` to avoid API limits

## Example Domains

### Voice Clone (TTS)

```yaml
name: "Voice Clone Research"
slug: "voice-clone"
research:
  arxivCategories: [cs.SD, cs.CL, eess.AS]
  keywords: [prosody, emotion TTS, voice cloning]
```

### Quant Trading

```yaml
name: "Quant Trading Lab"
slug: "quant-trading"
research:
  arxivCategories: [q-fin.TR, cs.LG, stat.ML]
  keywords: [algorithmic trading, market prediction]
scene:
  props:
    - id: chart-wall
      type: chart-wall
      position: [0, 0, -6]
```

### Robotics

```yaml
name: "Robotics Research"
slug: "robotics"
research:
  arxivCategories: [cs.RO, cs.AI, cs.CV]
  keywords: [robot learning, manipulation, control]
scene:
  props:
    - id: robot-arm
      type: robot-arm
      position: [0, 0, 0]
```

## API Reference

### Loading Domain Config

```typescript
import { loadDomainConfig, domainExists } from '@/lib/domain';

// Check if domain exists
if (domainExists('my-domain')) {
  // Load config
  const config = loadDomainConfig('my-domain');
  console.log(config.name);
}
```

### Using Domain Context

```tsx
import { DomainProvider, useDomain } from '@/components/domain';

// Wrap your app
<DomainProvider slug="my-domain">
  <MyApp />
</DomainProvider>

// Use in components
function MyComponent() {
  const { config, isLoading, error } = useDomain();

  if (isLoading) return <Spinner />;
  if (error) return <Error message={error.message} />;

  return <div>{config?.name}</div>;
}
```

### Loading Activities

```typescript
import { loadDomainActivities, getActivityConfig } from '@/lib/activities';

// Load all activities for a domain
const activities = loadDomainActivities('my-domain');

// Get a specific activity
const training = getActivityConfig('training', 'my-domain');
```

## Testing Your Domain

### Unit Tests

```bash
# Run all tests
npm run test

# Run domain tests only
npm run test -- domain
```

### Visual Verification

1. Start the dev server: `npm run dev`
2. Navigate to `/lab?domain=my-domain`
3. Verify:
   - Props render correctly
   - Colors match branding
   - Activities show proper icons
   - Agents move to correct locations

### API Verification

```bash
# Test domain config endpoint
curl http://localhost:3003/api/domain/my-domain

# Should return JSON with full config
# Or 404 if domain not found
```

## Troubleshooting

### Domain Not Loading

1. Check `domain.yaml` syntax (use YAML validator)
2. Verify slug matches directory name
3. Check file permissions

### Props Not Rendering

1. Verify prop type is valid
2. Check position is within scene bounds (-10 to 10)
3. Verify scale is positive number

### Activities Not Working

1. Check activity YAML syntax
2. Verify icon name is valid
3. Check workLocation matches a prop or valid location

### Colors Look Wrong

1. Verify hex format (`#RRGGBB` for CSS, `0xRRGGBB` for props)
2. Check color contrast for accessibility
3. Test in both light and dark modes

## Support

For issues with domain plugins:

1. Check this documentation first
2. Look at the voice-clone domain as reference
3. Open an issue on GitHub with:
   - Your domain.yaml
   - Error messages
   - Expected vs actual behavior
