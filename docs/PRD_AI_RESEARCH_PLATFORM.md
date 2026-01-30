# Product Requirements Document: AI Research Lab Platform

**Version:** 1.0
**Date:** January 28, 2026
**Author:** Generated via EPG Orchestrator

---

## Executive Summary

Transform the Voice Clone Pipeline research lab into a **generalized AI Research Lab Platform** where anyone can create their own research lab for any domain, watch AI agents work in real-time 3D visualization, share results, and collaborate with others.

---

## Vision

A "GitHub for Live AI Research Labs" where:
1. Anyone can choose a research domain (ML, trading, robotics, etc.)
2. Import papers/GitHub repos and have agents implement techniques
3. Watch cute robot agents work in beautiful 3D environments
4. Share labs publicly to inspire others
5. Discover synergies across all labs via meta-research intelligence

---

## Current State Analysis

### Existing Components (to preserve/generalize)

| Component | Current State | Generalization Needed |
|-----------|--------------|----------------------|
| Lab3D.tsx | 3D scene with voice-clone props | Load domain-specific props dynamically |
| Activities system | Voice-clone activities (training, recording) | Plugin-defined activity registry |
| Orchestrator | TTS-specific task prompts | Domain-configurable prompts |
| Props (3D) | Microphone, Speaker, Waveform, GPU | Domain-specific prop templates |
| Lab page | Single voice-clone lab | Multi-lab support with routing |
| Public viewer | Voice-clone specific | Generic with domain theming |

### Already Generalizable
- Agent rendering and animations
- Task prioritization and lifecycle
- Cost tracking and health monitoring
- CSS2D label system
- Particle effects and decorations

---

## Implementation Phases

### Phase 1: Core Generalization (Weeks 1-3)
Extract domain plugin system, generalize activity system, make orchestrator domain-agnostic.

### Phase 2: Domain Selection UI (Weeks 4-5)
Domain selector component, dynamic routing, domain context provider.

### Phase 3: Paper Ingestion (Weeks 6-8)
ArXiv API integration, paper analysis agent, task generation from papers.

### Phase 4: Create Your Lab Wizard (Weeks 9-11)
Multi-step wizard, Claude-assisted setup, lab scaffolding.

### Phase 5: Multi-Lab & Sharing (Weeks 12-14)
Lab registry database, search/filter, lab metrics.

### Phase 6: Social Layer (Weeks 15-17)
Lab portals, result cards, model sharing, star/fork system.

### Phase 7: Multi-Source Input (Weeks 18-19)
GitHub repo analysis, custom research goals, universal paper ingestion.

### Phase 8: Easy Deployment (Weeks 20-22)
One-click deploy templates, Docker configs, setup wizards.

### Phase 9: Meta-Research System (Weeks 23-28)
Knowledge graph, meta-agents, synergy discovery, genetic evolution.

### Phase 10: Example Domains (Weeks 29-32)
Build 5-7 example domains with full 3D props.

---

## Phase 1: Core Generalization - Detailed Breakdown

### Week 1: Domain Plugin Architecture

#### Task 1.1: Create Domain Configuration Schema
- Define `domain.yaml` schema with TypeScript types
- Include: name, slug, description, branding, scene config, research config, prompts, evaluation, hardware requirements
- Location: `frontend/lib/domain/types.ts`
- **Acceptance:** TypeScript interfaces compile, schema validates example configs

#### Task 1.2: Create .domains Directory Structure
- Create `.domains/` at project root
- Create `.domains/voice-clone/` as first domain plugin
- Move voice-clone specific configs to plugin directory
- **Acceptance:** Directory exists, voice-clone domain has domain.yaml

#### Task 1.3: Implement Domain Config Loader
- Create `loadDomainConfig(slug: string)` function
- Support loading from `.domains/[slug]/domain.yaml`
- Validate against schema, provide defaults
- Location: `frontend/lib/domain/loader.ts`
- **Acceptance:** Can load voice-clone config, returns typed object

#### Task 1.4: Create DomainProvider Context
- React context providing current domain config
- Hook: `useDomain()` returns config and helpers
- Support domain switching without full page reload
- Location: `frontend/components/domain/DomainProvider.tsx`
- **Acceptance:** Components can access domain config via hook

#### Task 1.5: Create Domain Branding Component
- Apply domain-specific colors, logo, background style
- Support: primaryColor, accentColor, backgroundStyle (sky, space, grid, etc.)
- Location: `frontend/components/domain/DomainBranding.tsx`
- **Acceptance:** UI reflects domain colors when context changes

### Week 2: Activity System Generalization

#### Task 2.1: Define Activity Type Schema
- Create YAML schema for activity definitions
- Include: id, name, description, visualization (prop, animation, particles, color), agent behavior, progress detection
- Location: `frontend/lib/activities/types.ts`
- **Acceptance:** Schema types defined, example activities validate

#### Task 2.2: Refactor ActivityRegistry to Load from Domain
- Currently hardcoded activity types in activities.ts
- Load activity configs from `.domains/[slug]/activities/*.yaml`
- Fall back to built-in activities if domain doesn't define custom ones
- Location: `frontend/components/lab/activities/registry.ts`
- **Acceptance:** Voice-clone activities load from YAML, system still works

#### Task 2.3: Move Voice-Clone Activities to Plugin
- Create `.domains/voice-clone/activities/` directory
- Move training.yaml, recording.yaml, generation.yaml definitions
- Extract hardcoded strings from current activities.ts
- **Acceptance:** Activities load from domain plugin, Lab3D renders correctly

#### Task 2.4: Create Activity Config Types for 3D
- Map activity configs to 3D scene properties
- agentAnimation: "focused" | "typing" | "walking"
- particleEffect: "sparks" | "data-flow" | "audio-waves"
- color: hex number for accent
- **Acceptance:** Activity config controls agent animations in Lab3D

#### Task 2.5: Add Activity Icon Registry
- Map activity types to Lucide icons dynamically
- Support custom icon names in activity config
- Fall back to sensible defaults
- **Acceptance:** Custom activities show appropriate icons in UI

### Week 3: Orchestrator Generalization

#### Task 3.1: Make Orchestrator Read Domain Config
- Load domain.yaml for current project
- Use domain-specific research keywords for web search agent
- Use domain-specific prompt preambles for task agents
- **Acceptance:** Orchestrator uses domain config, logs show domain name

#### Task 3.2: Create Domain-Specific Prompt Templates
- Move task prompt templates to `.domains/[slug]/prompts/`
- research.md - for web research agent
- implementation.md - for task implementation agent
- evaluation.md - for evaluation agent
- **Acceptance:** Agents receive domain-appropriate prompts

#### Task 3.3: Generalize arXiv Category Configuration
- Currently hardcoded to cs.SD, cs.CL, eess.AS
- Read from domain.yaml research.arxivCategories
- Pass to research agent for web searches
- **Acceptance:** Research agent searches domain-appropriate categories

#### Task 3.4: Abstract Evaluation Metrics
- Currently hardcoded to audio metrics (MOS, F0_RMSE)
- Read metric definitions from domain.yaml evaluation.metrics
- Pass to evaluation prompts
- **Acceptance:** Evaluation tasks use domain-specific metrics

#### Task 3.5: Add Domain Slug to Task Metadata
- Store domain slug in task metadata when creating tasks
- Filter tasks by domain in multi-lab scenarios (future)
- **Acceptance:** New tasks have domain field in JSON

---

## Phase 2: Domain Selection UI - Detailed Breakdown

### Week 4: Domain Selector & Routing

#### Task 4.1: Create DomainSelector Component
- Grid of preset domains with icons and descriptions
- Preset domains: voice-clone, quant-trading, game-ai, computer-vision, nlp, robotics, custom
- Each card shows: icon, name, description, difficulty badge
- Location: `frontend/components/domain/DomainSelector.tsx`
- **Acceptance:** Renders domain grid, emits selection event

#### Task 4.2: Create Domain Icons
- SVG icons for each preset domain
- Consistent style, suitable for 3D rendering
- Location: `frontend/components/domain/icons/`
- **Acceptance:** Each domain has unique, recognizable icon

#### Task 4.3: Implement Dynamic Routing [domain]
- Create `frontend/app/[domain]/page.tsx`
- Load domain config based on URL param
- Wrap with DomainProvider
- **Acceptance:** /voice-clone shows voice-clone lab, /quant-trading shows trading lab

#### Task 4.4: Create Domain Not Found Page
- Show friendly error when domain doesn't exist
- Suggest similar domains or link to domain selector
- **Acceptance:** Invalid domain URL shows helpful error

#### Task 4.5: Add Domain to Navigation
- Show current domain name in nav
- Dropdown to switch domains (if multiple installed)
- Link to domain selector for new domain
- **Acceptance:** Navigation shows domain context

### Week 5: Domain-Specific Lab Views

#### Task 5.1: Create DomainScene Component
- Load domain-specific 3D props based on config
- Support scene.props array from domain.yaml
- Handle prop positions, scales, types
- Location: `frontend/components/lab/DomainScene.tsx`
- **Acceptance:** Voice-clone scene loads voice props, trading scene loads trading props

#### Task 5.2: Implement Lazy Prop Loading
- Dynamic import for domain-specific props
- Load props from `.domains/[slug]/props/` if custom
- Fall back to built-in props library
- **Acceptance:** Custom domain props load without bundling all props

#### Task 5.3: Create Public Watch Route [domain]/watch
- Public viewer for each domain's lab
- Apply domain theming
- Sanitize sensitive data
- **Acceptance:** /voice-clone/watch shows public view with voice theme

#### Task 5.4: Add Domain-Specific Sidebar Content
- Activity types from domain config
- Metrics from domain config
- Research topics from domain config
- **Acceptance:** Sidebar shows domain-appropriate content

#### Task 5.5: Create Domain Onboarding Modal
- First-time user sees brief domain intro
- Links to documentation
- Option to not show again
- **Acceptance:** New visitors get oriented quickly

---

## Phase 3: Paper Ingestion - Detailed Breakdown

### Week 6: Paper Fetching Service

#### Task 6.1: Create arXiv API Client
- Fetch paper metadata by ID or search query
- Parse XML response to structured format
- Handle rate limiting and errors
- Location: `backend/services/arxiv_client.py`
- **Acceptance:** Can fetch paper by arXiv ID, returns structured data

#### Task 6.2: Create Semantic Scholar API Client
- Alternative paper source for citation data
- Search by title or DOI
- Get related papers
- **Acceptance:** Can search papers, returns structured data

#### Task 6.3: Create Paper Database Schema
- Store ingested papers in SQLite/Postgres
- Fields: id, source, arxivId, title, authors, abstract, url, relevanceScore, status, createdAt
- Location: Create Prisma schema or SQLite setup
- **Acceptance:** Papers persist between sessions

#### Task 6.4: Create Paper Ingestion API Endpoint
- POST /api/papers/ingest
- Accept: source, identifier, domain, priority
- Return: paper metadata, analysis, suggested tasks
- **Acceptance:** API ingests paper and returns analysis

#### Task 6.5: Add PDF Text Extraction
- Download paper PDF
- Extract text for analysis
- Store extracted text
- **Acceptance:** Can extract text from arXiv PDFs

### Week 7: Paper Analysis Agent

#### Task 7.1: Create Paper Relevance Scoring Prompt
- Claude analyzes abstract against domain keywords
- Returns 0-100 relevance score with reasoning
- Identifies key techniques
- **Acceptance:** Papers get relevance scores

#### Task 7.2: Create Task Generation From Paper
- Analyze paper and generate research, implementation, evaluation tasks
- Include paper context in task description
- Set appropriate priority based on relevance
- **Acceptance:** Ingesting paper creates 1-3 tasks

#### Task 7.3: Implement Duplicate Detection
- Check if paper already ingested
- Check if similar tasks exist
- Warn user if duplicate
- **Acceptance:** Duplicate papers don't create duplicate tasks

#### Task 7.4: Create Paper Implementation Complexity Estimator
- Estimate "easy" | "medium" | "hard" based on abstract
- Factor in: dependencies, dataset needs, compute needs
- **Acceptance:** Papers have complexity estimate

#### Task 7.5: Add Paper Metadata Enrichment
- Fetch citation count
- Find GitHub repos linked to paper
- Get author affiliations
- **Acceptance:** Papers have enriched metadata

### Week 8: Paper Management UI

#### Task 8.1: Create Paper Ingestion Form
- Input field for arXiv ID or URL
- Domain selector
- Priority selector
- Preview before ingest
- **Acceptance:** Users can ingest papers via UI

#### Task 8.2: Create Paper Queue View
- List of ingested papers
- Status: pending, implementing, completed
- Filter by domain, status
- **Acceptance:** Can view and filter paper queue

#### Task 8.3: Create Paper Detail View
- Show full paper metadata
- List generated tasks
- Show implementation status
- Link to original paper
- **Acceptance:** Detailed view of single paper

#### Task 8.4: Add Batch Paper Import
- Upload text file with arXiv IDs
- Import multiple papers at once
- Show progress and results
- **Acceptance:** Can batch import 10+ papers

#### Task 8.5: Create Paper Search
- Search papers by title, author, keywords
- Filter by domain
- Sort by relevance, date, citations
- **Acceptance:** Papers are searchable

---

## Phase 4: Create Your Lab Wizard - Detailed Breakdown

### Week 9: Wizard Flow

#### Task 9.1: Create Wizard Container Component
- Multi-step wizard with progress indicator
- Steps: Domain, Configure, Hardware, Research, Review
- Navigation: back, next, skip
- Location: `frontend/app/create/page.tsx`
- **Acceptance:** Wizard navigates through steps

#### Task 9.2: Create Domain Selection Step
- Display preset domain cards
- "Custom" option for advanced users
- Import existing domain.yaml option
- **Acceptance:** User can select domain type

#### Task 9.3: Create Lab Configuration Step
- Name your lab
- Description
- Visibility: public, private, unlisted
- Branding: color picker, icon selector, background theme
- **Acceptance:** User can configure basic lab settings

#### Task 9.4: Create Hardware Setup Step
- Detect local GPU (if browser supports)
- Options: local only, remote GPU, cloud GPU
- Remote: SSH connection setup helper
- Cloud: provider selection (RunPod, etc.)
- **Acceptance:** Hardware configuration captured

#### Task 9.5: Create Research Focus Step
- Natural language research goal input
- Suggested arXiv categories based on goal
- Optional: initial papers to implement
- **Acceptance:** Research direction configured

### Week 10: Claude-Assisted Setup

#### Task 10.1: Create Goal Analysis API
- POST /api/labs/analyze-goal
- Accept: natural language research goal
- Return: suggested domain, arXiv categories, starter papers, tasks
- **Acceptance:** Claude provides setup suggestions

#### Task 10.2: Implement Goal-to-Tasks Conversion
- Parse Claude suggestions into task format
- Create initial tasks for new lab
- Set appropriate priorities
- **Acceptance:** Lab starts with suggested tasks

#### Task 10.3: Create Setup Assistant Chat
- Optional chat interface during wizard
- Claude helps refine research direction
- Suggests adjustments based on hardware
- **Acceptance:** Users can chat for setup help

#### Task 10.4: Add Hardware Requirement Warnings
- Check if selected domain needs GPU
- Warn if hardware insufficient
- Suggest alternatives (cloud, simpler tasks)
- **Acceptance:** Users warned about hardware mismatch

#### Task 10.5: Create Wizard Review Step
- Summary of all selections
- Estimated costs (if using cloud/paid APIs)
- Edit any previous step
- Create lab button
- **Acceptance:** User can review and create lab

### Week 11: Lab Scaffolding

#### Task 11.1: Create Lab Scaffold Generator
- Generate domain.yaml from wizard inputs
- Create directory structure
- Set up initial files
- Location: Backend API + file operations
- **Acceptance:** New lab directory created correctly

#### Task 11.2: Create Lab Initialization API
- POST /api/labs/create
- Accept: wizard configuration
- Return: lab ID, URL, initial status
- **Acceptance:** Labs can be created via API

#### Task 11.3: Initialize Git Repository for Lab
- Optional: create .git in lab directory
- Initial commit with generated files
- .gitignore for sensitive files
- **Acceptance:** New labs optionally have git setup

#### Task 11.4: Create Initial Task Population
- Create starter tasks based on wizard input
- Include setup/verification tasks
- Add first research task if papers specified
- **Acceptance:** New labs have useful initial tasks

#### Task 11.5: Redirect to New Lab
- After creation, redirect to /[domain] with new lab active
- Show welcome modal with next steps
- Auto-start orchestrator if configured
- **Acceptance:** User lands in functional new lab

---

## Phase 5: Multi-Lab & Sharing - Detailed Breakdown

### Week 12: Lab Registry Database

#### Task 12.1: Design Lab Database Schema
- Lab model: id, slug, name, description, domain, ownerId, visibility, config (JSON), stats, createdAt
- Use Prisma or direct SQLite
- **Acceptance:** Schema defined, migrations work

#### Task 12.2: Create Lab CRUD API
- GET /api/labs - list user's labs
- POST /api/labs - create lab
- GET /api/labs/[id] - get lab details
- PATCH /api/labs/[id] - update lab
- DELETE /api/labs/[id] - delete lab
- **Acceptance:** Full CRUD operations work

#### Task 12.3: Add Lab Stats Tracking
- Track: tasksCompleted, papersProcessed, agentHours, stars, viewers
- Update stats on relevant events
- **Acceptance:** Lab stats update automatically

#### Task 12.4: Create Public Labs Index
- List public labs with pagination
- Filter by domain
- Sort by activity, stars, recent
- **Acceptance:** Public labs discoverable

#### Task 12.5: Add Lab Search
- Full-text search on name, description
- Filter by domain, visibility, activity level
- **Acceptance:** Labs are searchable

### Week 13: Lab Sharing Features

#### Task 13.1: Create Shareable Lab URL
- Generate unique public URL for lab
- /labs/[username]/[lab-slug] format
- Handle URL collisions
- **Acceptance:** Labs have shareable URLs

#### Task 13.2: Create Lab Embed Widget
- Embeddable iframe version of public view
- Configurable size, features shown
- /labs/[username]/[lab-slug]/embed
- **Acceptance:** Labs can be embedded on external sites

#### Task 13.3: Create Lab Showcase Page
- Featured labs section
- Filter by domain
- Lab cards with preview, stats
- **Acceptance:** Showcase page displays labs attractively

#### Task 13.4: Add Lab Activity Badge
- "Live" badge when agents running
- Agent count
- Recent activity timestamp
- **Acceptance:** Lab cards show real-time status

#### Task 13.5: Create Lab Statistics Dashboard
- Detailed stats for lab owner
- Charts: tasks over time, agent hours, costs
- Export stats as CSV
- **Acceptance:** Owners see detailed analytics

### Week 14: Star & Fork System

#### Task 14.1: Implement Lab Star Feature
- Star/unstar endpoint
- Store user-lab star relationships
- Display star count on lab card
- **Acceptance:** Users can star labs

#### Task 14.2: Create Starred Labs Page
- List user's starred labs
- Quick access from navigation
- **Acceptance:** Users can view starred labs

#### Task 14.3: Implement Lab Fork Feature
- Copy lab config to create new lab
- Fork from: domain.yaml, prompts, initial tasks
- NOT copied: models, user data, results
- **Acceptance:** Labs can be forked

#### Task 14.4: Show Fork Lineage
- Display "forked from [lab]" on forked labs
- Link to parent lab
- Count forks on parent
- **Acceptance:** Fork relationships visible

#### Task 14.5: Add Fork Notification
- Notify lab owner when forked
- Include forker username, new lab name
- **Acceptance:** Owners notified of forks

---

## Phase 6: Social Layer - Detailed Breakdown

### Week 15: Lab Portal System

#### Task 15.1: Create User Profile Page
- /u/[username] route
- Display user's public labs
- Bio, avatar, stats
- Activity heatmap (GitHub-style)
- **Acceptance:** Users have profile pages

#### Task 15.2: Create Lab Portal Page
- /labs/[username]/[lab-slug] enhanced view
- Live 3D viewer
- Results showcase
- Community features
- **Acceptance:** Lab portal pages work

#### Task 15.3: Add Portal Customization
- Custom banner image
- Pinned results
- Featured demos
- **Acceptance:** Lab owners can customize portal

#### Task 15.4: Create Activity Feed for Lab
- Stream of recent events
- Tasks completed, results achieved, models trained
- **Acceptance:** Labs have activity feeds

#### Task 15.5: Add Lab Following
- Follow/unfollow labs
- Get notifications on activity
- Following feed
- **Acceptance:** Users can follow labs

### Week 16: Result Sharing

#### Task 16.1: Create Result Card Component
- Display research result with metrics
- Media: charts, audio samples, videos
- Share buttons
- **Acceptance:** Results display attractively

#### Task 16.2: Implement Result Publishing API
- POST /api/results - create result
- Attach to lab, link to paper/task
- Set visibility
- **Acceptance:** Results can be published

#### Task 16.3: Create Model Sharing Integration
- Link to Hugging Face model card
- Display model metrics, usage
- Download/clone buttons
- **Acceptance:** Models integrated with HF Hub

#### Task 16.4: Create Interactive Demo Component
- Embed runnable demo for result
- TTS: text input, audio output
- Trading: backtest visualization
- **Acceptance:** Results can have interactive demos

#### Task 16.5: Add Result Comments
- Comment thread on results
- Markdown support
- @ mentions
- **Acceptance:** Results can be discussed

### Week 17: Discovery & Community

#### Task 17.1: Create Explore Page
- /explore route
- Filter by domain, activity, difficulty
- View modes: grid, list, map
- **Acceptance:** Users can explore public labs

#### Task 17.2: Create Trending Labs Algorithm
- Track engagement signals
- Compute trending score
- Update periodically
- **Acceptance:** Trending labs surface interesting content

#### Task 17.3: Create Domain Directories
- /domains/[domain] listing page
- Featured labs per domain
- Domain-specific leaderboards
- **Acceptance:** Each domain has directory

#### Task 17.4: Implement Suggestion Voting
- Upvote/downvote community suggestions
- Sort by votes
- Lab owner response status
- **Acceptance:** Suggestions can be voted on

#### Task 17.5: Create Lab Feed
- Aggregated feed from followed labs
- Filter by event type
- Infinite scroll
- **Acceptance:** Users see followed lab activity

---

## Phase 7: Multi-Source Input - Detailed Breakdown

### Week 18: Additional Research Sources

#### Task 18.1: Create GitHub Repo Analyzer
- Parse README for research context
- Find linked papers
- Extract key techniques
- **Acceptance:** GitHub repos can be analyzed

#### Task 18.2: Add Direct Paper URL Support
- Accept URLs from various paper sites
- Extract metadata via scraping
- Handle various formats
- **Acceptance:** Papers ingested from URLs

#### Task 18.3: Add PDF Upload Support
- Upload PDF directly
- Extract text and metadata
- Store PDF for reference
- **Acceptance:** Papers can be uploaded

#### Task 18.4: Create Custom Research Goal Parser
- Natural language goal input
- Claude extracts: domain, techniques, metrics
- Generates tasks without paper source
- **Acceptance:** Goals create tasks directly

#### Task 18.5: Add Source Verification
- Verify paper/repo exists
- Check for retractions
- Warn on outdated sources
- **Acceptance:** Sources are validated

### Week 19: Universal Paper Ingestion

#### Task 19.1: Create Unified Source Interface
- Abstract interface for all sources
- normalize() method returns common format
- Error handling for each source type
- **Acceptance:** All sources use same interface

#### Task 19.2: Add Multi-Source Search
- Search across arXiv, Semantic Scholar, GitHub
- Deduplicate results
- Rank by relevance
- **Acceptance:** Single search across sources

#### Task 19.3: Create Source-Specific Task Templates
- Different task structures per source
- GitHub: reproduction, comparison
- Paper: implementation, evaluation
- **Acceptance:** Tasks appropriate to source type

#### Task 19.4: Add Source Citation Tracking
- Track which sources led to which tasks
- Show source lineage in task detail
- **Acceptance:** Task-source relationship tracked

#### Task 19.5: Create Paper Import History
- Log all imports
- Show success/failure rates
- Export import history
- **Acceptance:** Import history available

---

## Phase 8: Easy Deployment - Detailed Breakdown

### Week 20: One-Click Deploy Templates

#### Task 20.1: Create Vercel Deploy Template
- vercel.json configuration
- Environment variable setup
- Build configuration
- **Acceptance:** Deploy to Vercel works

#### Task 20.2: Create Docker Compose Configuration
- docker-compose.yml for full stack
- docker-compose.gpu.yml for GPU support
- Volumes for persistence
- **Acceptance:** docker-compose up works

#### Task 20.3: Create Cloud Deploy Scripts
- ./deploy.sh --cloud=runpod
- ./deploy.sh --cloud=aws
- Provider-specific setup
- **Acceptance:** One-command cloud deploy

#### Task 20.4: Create Install Script
- curl | sh one-liner
- Interactive prompts
- Automatic dependency installation
- **Acceptance:** Fresh machine setup works

#### Task 20.5: Add Deploy Status Page
- Show deployment progress
- Handle errors gracefully
- Provide recovery options
- **Acceptance:** Users see deploy progress

### Week 21: Setup Wizards

#### Task 21.1: Create Hardware Detection
- Detect local GPU via browser API
- Detect CPU cores, memory
- Suggest appropriate settings
- **Acceptance:** Hardware auto-detected

#### Task 21.2: Create SSH Connection Helper
- Guide user through SSH setup
- Test connection
- Store credentials securely
- **Acceptance:** Remote GPU setup guided

#### Task 21.3: Create Dependency Installer
- Detect missing dependencies
- Provide install commands
- Verify installation
- **Acceptance:** Dependencies installed automatically

#### Task 21.4: Add Ollama Setup Wizard
- Download and install Ollama
- Pull required models
- Configure context size
- **Acceptance:** Ollama ready to use

#### Task 21.5: Create Post-Deploy Verification
- Test all endpoints
- Verify GPU access
- Check Ollama models
- **Acceptance:** Deployment verified working

### Week 22: Deployment Modes

#### Task 22.1: Document Local Deployment
- Step-by-step local setup
- Troubleshooting guide
- Performance tips
- **Acceptance:** Local docs complete

#### Task 22.2: Document Remote Deployment
- SSH setup guide
- Tailscale configuration
- Security best practices
- **Acceptance:** Remote docs complete

#### Task 22.3: Document Cloud Deployment
- Provider-specific guides
- Cost estimates
- Scaling options
- **Acceptance:** Cloud docs complete

#### Task 22.4: Create Deployment Mode Selector
- Choose: local, remote, cloud
- Configure based on selection
- Save deployment config
- **Acceptance:** Users can switch modes

#### Task 22.5: Add Cost Estimator
- Estimate cloud costs
- Track actual vs estimated
- Budget alerts
- **Acceptance:** Users understand costs

---

## Phase 9: Meta-Research System - Detailed Breakdown

### Week 23-24: Knowledge Graph

#### Task 23.1: Set Up Graph Database
- Neo4j or similar
- Define connection in backend
- **Acceptance:** Graph DB running

#### Task 23.2: Create Knowledge Graph Schema
- TechniqueNode, DomainNode, ConceptNode, LabNode, ResultNode
- Edge types: derived_from, similar_to, combines_with, etc.
- **Acceptance:** Schema implemented

#### Task 23.3: Create Knowledge Graph API
- GraphQL or REST endpoints
- Query techniques, domains, concepts
- Traverse relationships
- **Acceptance:** API queries work

#### Task 23.4: Implement Auto-Indexing
- Index new techniques from lab activity
- Index papers when ingested
- Index results when published
- **Acceptance:** Graph populates automatically

#### Task 23.5: Create Knowledge Graph Visualization
- Interactive graph view
- Click nodes to explore
- Filter by domain
- **Acceptance:** Graph visualized in UI

### Week 25-26: Meta-Agents

#### Task 25.1: Implement Pattern Recognition Agent
- Analyze trends across all labs
- Detect rising/declining techniques
- Generate trend reports
- **Acceptance:** Weekly trend reports generated

#### Task 25.2: Implement Synergy Discovery Agent
- Find technique combinations
- Score synergy potential
- Suggest collaborations
- **Acceptance:** Synergy reports generated

#### Task 25.3: Implement Gap Analysis Agent
- Find unexplored areas
- Identify missing combinations
- Prioritize opportunities
- **Acceptance:** Gap reports generated

#### Task 25.4: Implement Genetic Research Agent
- Technique genes, crossover, mutation
- Fitness evaluation
- Evolution proposals
- **Acceptance:** Evolution proposals generated

#### Task 25.5: Implement Cross-Domain Transfer Agent
- Map concepts between domains
- Suggest transfers
- Assess feasibility
- **Acceptance:** Transfer proposals generated

### Week 27-28: Collaboration Features

#### Task 27.1: Create Meta-Task System
- Multi-lab task coordination
- Shared objectives
- Progress tracking
- **Acceptance:** Meta-tasks can be created

#### Task 27.2: Create Collaboration Opportunity Matcher
- Detect synergies requiring collaboration
- Notify relevant labs
- Track acceptance
- **Acceptance:** Collaborations suggested automatically

#### Task 27.3: Create Weekly Research Digest
- Aggregate activity across platform
- Highlight breakthroughs
- Send to subscribed users
- **Acceptance:** Weekly digest sent

#### Task 27.4: Create Opportunity Board
- List all identified opportunities
- Filter by effort, impact, domain
- Claim opportunities
- **Acceptance:** Opportunity board functional

#### Task 27.5: Add Collaboration Dashboard
- Track active collaborations
- Show progress
- Facilitate communication
- **Acceptance:** Collaborations have dashboard

---

## Phase 10: Example Domains - Detailed Breakdown

### Week 29-30: Core Example Domains

#### Task 29.1: Create Quant Trading Domain
- domain.yaml with trading config
- Props: chart-wall, orderbook, trading-terminal
- Activities: backtesting, optimization, paper-trading
- **Acceptance:** Trading lab functional

#### Task 29.2: Create Game AI Domain
- domain.yaml with game config
- Props: game-screen, controller, leaderboard
- Activities: training, evaluation, tournament
- **Acceptance:** Game AI lab functional

#### Task 29.3: Create Computer Vision Domain
- domain.yaml with CV config
- Props: camera, image-grid, detection-display
- Activities: training, inference, annotation
- **Acceptance:** CV lab functional

#### Task 29.4: Create NLP Domain
- domain.yaml with NLP config
- Props: text-display, embedding-viz, chat-interface
- Activities: training, inference, evaluation
- **Acceptance:** NLP lab functional

### Week 31-32: Additional Domains & Polish

#### Task 31.1: Create Robotics Domain
- domain.yaml with robotics config
- Props: robot-arm, simulation-screen, sensor-display
- Activities: training, simulation, deployment
- **Acceptance:** Robotics lab functional

#### Task 31.2: Create Biology/Chemistry ML Domain
- domain.yaml with bio config
- Props: molecule-viewer, protein-structure, experiment-log
- Activities: prediction, docking, analysis
- **Acceptance:** Bio/Chem lab functional

#### Task 31.3: Domain Documentation
- README for each domain
- Setup instructions
- Example research goals
- **Acceptance:** All domains documented

#### Task 31.4: Domain Showcase
- Landing page showing all domains
- Comparison table
- Quick start for each
- **Acceptance:** Domain showcase page live

#### Task 31.5: Domain Testing
- End-to-end test for each domain
- Verify props render
- Verify activities work
- **Acceptance:** All domains pass tests

---

## Success Metrics

### Platform Metrics
- Labs created: Target 100+ in first month
- Active labs (weekly): Target 50% of created
- Papers ingested: Target 500+ in first month
- Tasks completed: Target 1000+ in first month

### Quality Metrics
- Page load time: < 3s
- 3D frame rate: 60fps
- API response time: < 500ms
- Uptime: 99.5%

### Community Metrics
- Stars given: Target 10x labs created
- Forks: Target 20% of public labs forked
- Comments: Target 5+ per result
- Collaborations: Target 10+ active

---

## Technical Stack

### Frontend
- Next.js 14 (App Router)
- React 18
- Three.js (3D)
- Tailwind CSS
- Vitest for testing

### Backend
- FastAPI (Python)
- Node.js (Orchestrator)
- SQLite/Postgres (database)
- Neo4j (knowledge graph, Phase 9)

### Infrastructure
- Vercel (frontend hosting)
- Docker (containerization)
- Ollama (local LLM)
- Tailscale (networking)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 3D performance on low-end devices | Progressive enhancement, fallback to 2D |
| API costs for paper analysis | Batch processing, caching, Ollama fallback |
| Complex domain plugin system | Start simple, iterate based on usage |
| Multi-lab database scaling | Start with SQLite, migrate to Postgres |
| Knowledge graph complexity | Phase 9 is optional, validate value first |

---

## Appendix: File Locations

```
/Users/light/dev/web-apps/labfork/
├── .domains/                          # NEW: Domain plugins
│   ├── voice-clone/                   # First domain (migrate existing)
│   │   ├── domain.yaml
│   │   ├── prompts/
│   │   ├── activities/
│   │   └── props/
│   └── quant-trading/                 # Example domain
├── frontend/
│   ├── app/
│   │   ├── [domain]/                  # NEW: Dynamic domain routes
│   │   ├── create/                    # NEW: Lab wizard
│   │   ├── explore/                   # NEW: Discovery
│   │   └── labs/[username]/[slug]/    # NEW: Lab portals
│   ├── components/
│   │   ├── domain/                    # NEW: Domain components
│   │   └── lab/                       # Existing (generalize)
│   └── lib/
│       ├── domain/                    # NEW: Domain utilities
│       └── activities/                # NEW: Activity registry
├── backend/
│   ├── services/
│   │   ├── arxiv_client.py           # NEW: Paper fetching
│   │   └── paper_analysis.py         # NEW: Paper analysis
│   └── api/
│       └── papers.py                  # NEW: Paper endpoints
└── docs/
    ├── PRD_AI_RESEARCH_PLATFORM.md   # This document
    ├── GENERALIZED_AI_LAB_PLATFORM.md # Original design
    └── META_RESEARCH_SYSTEM.md        # Meta-research design
```

---

**End of PRD**
