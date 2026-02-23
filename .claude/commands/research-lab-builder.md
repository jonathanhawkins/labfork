---
description: Create and configure new research labs from papers and proposals
allowed-tools: Bash, Read, Write, Edit, Task, WebSearch, WebFetch, Grep, Glob
---

# Research Lab Builder Command

You are the Research Lab Builder for LabFork. Your job is to take research proposals, papers, or ideas and turn them into fully functional research lab pages with:

1. **Compiled Research** - Find and summarize relevant papers
2. **Technical Design** - Create build specifications
3. **Interactive Tools** - Design configurators, simulators
4. **Community Infrastructure** - Data collection, leaderboards
5. **Open Source Assets** - STL files, code, documentation

## Workflow

### Phase 1: Research Compilation
```
1. WebSearch for relevant academic papers
2. Extract key findings, methods, materials
3. Identify open source tools and datasets
4. Create bibliography with direct links
```

### Phase 2: Technical Specification
```
1. Bill of materials with costs and sources
2. Build instructions (step by step)
3. Safety considerations
4. Expected performance metrics
```

### Phase 3: Page Generation
```
1. Create Next.js page at /labs/[lab-name]
2. Follow LabFork design system (dark theme, Tailwind)
3. Include tabs: Overview, Science, Build, Simulate, Community
4. Add interactive configurators with sliders
5. Connect to compute backend for simulations
```

### Phase 4: Asset Creation
```
1. Generate parametric CAD models (CadQuery/OpenSCAD)
2. Export STL files for 3D printing
3. Create GitHub repository for assets
4. Set up community data collection (Supabase)
```

## Lab Page Template Structure

```tsx
// /app/labs/[lab-name]/page.tsx
"use client";

export default function LabPage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Hero Header */}
      {/* Tab Navigation */}
      {/* Tab Content */}
      {/* Footer CTA */}
    </div>
  );
}
```

## Required Sections

### Overview Tab
- How it works (visual diagrams)
- Key stats (cost, performance, requirements)
- Use cases (when to build / when not to)

### Science Tab
- Research papers with links
- Physics/chemistry explanation
- Performance data from literature

### Build Tab
- Bill of materials table
- 3D printable components
- Assembly instructions
- Open source tools

### Simulate Tab
- Design configurator (sliders, dropdowns)
- Real-time performance estimation
- Full simulation button (4090 backend)
- Export options (STL, share link)

### Community Tab
- Submit results form
- Leaderboard / data table
- Discussion links

## Example Labs

1. **Water Harvester** - Atmospheric water collection
2. **Solar Concentrator** - DIY solar thermal
3. **Biochar Reactor** - Carbon sequestration
4. **Wind Turbine** - Small-scale generation
5. **Aquaponics System** - Food production

## Integration Points

- **Compute**: POST /api/simulations
- **Storage**: Supabase for community data
- **Assets**: GitHub for STL/code
- **Papers**: /api/papers for bibliography
