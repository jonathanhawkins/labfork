# Paper Ingestion System (Phase 3)

The paper ingestion system allows users to add research papers from arXiv, Semantic Scholar, DOIs, GitHub repositories, and other sources, analyze them for relevance using AI, and automatically generate implementation tasks.

## Architecture Overview

```
User Input (arXiv ID, DOI, URL)
         │
         ▼
   ┌─────────────┐
   │   Parser    │ ← Detects input type (arxiv, doi, github, etc.)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │ API Client  │ ← Fetches metadata from source (arXiv, S2, GitHub)
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │  Analysis   │ ← AI analyzes paper for relevance, techniques, complexity
   └─────────────┘
         │
         ▼
   ┌─────────────┐
   │Task Generator│ ← Creates research → implementation → evaluation tasks
   └─────────────┘
```

## Core Components

### Library (`/lib/papers/`)

| File | Purpose |
|------|---------|
| `types.ts` | TypeScript types and interfaces for papers, analysis, and tasks |
| `arxiv.ts` | arXiv API client - fetch papers by ID or search |
| `semantic-scholar.ts` | Semantic Scholar API client - DOI lookup and search |
| `github.ts` | GitHub repository analyzer - extracts paper references from repos |
| `parser.ts` | Input detection and routing to appropriate API client |
| `prompts.ts` | AI prompts for paper analysis |
| `task-generator.ts` | Converts paper analysis into linked tasks |
| `index.ts` | Module exports |

### API Endpoints (`/app/api/papers/`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/papers` | GET | List papers with filters |
| `/api/papers` | POST | Add new paper by input |
| `/api/papers` | PATCH | Update paper status |
| `/api/papers` | DELETE | Remove paper |
| `/api/papers/fetch` | GET | Detect input type |
| `/api/papers/fetch` | POST | Fetch paper metadata |
| `/api/papers/analyze` | POST | Analyze paper with AI |

### UI Components (`/components/papers/`)

| Component | Purpose |
|-----------|---------|
| `AddPaperDialog` | Multi-step dialog for adding papers |
| `PaperCard` | Displays paper with status, badges, actions |
| `PaperQueue` | List of papers with search/filter/sort |
| `PaperDetailView` | Full paper details with analysis and task preview |

## Supported Input Formats

The system auto-detects and handles various input formats:

- **arXiv ID**: `2401.12345`, `2401.12345v2`
- **arXiv URL**: `https://arxiv.org/abs/2401.12345`
- **arXiv PDF URL**: `https://arxiv.org/pdf/2401.12345.pdf`
- **DOI**: `10.1234/example.paper`
- **DOI URL**: `https://doi.org/10.1234/example.paper`
- **Semantic Scholar URL**: `https://semanticscholar.org/paper/...`
- **GitHub URL**: `https://github.com/owner/repo`
- **Papers With Code URL**: `https://paperswithcode.com/paper/...`
- **Hugging Face Papers**: `https://huggingface.co/papers/2401.12345`
- **PDF URL**: Any URL ending in `.pdf`

## Paper Analysis

When a paper is analyzed, the AI evaluates:

1. **Relevance Score (0-100)**: How relevant to the current domain
2. **Relevance Reason**: Explanation of relevance assessment
3. **Complexity**: `simple`, `moderate`, `complex`, or `research`
4. **Novelty**: What's new/different about this paper
5. **Techniques**: List of techniques with descriptions
6. **Resources**: Required datasets, compute, models
7. **Task Breakdown**: Research, implementation, and evaluation tasks

### Analysis Prompt Structure

The analysis uses a structured prompt that includes:
- Paper title, abstract, and authors
- Domain context (if available)
- Expected JSON output format

See `/lib/papers/prompts.ts` for the full prompt template.

## Task Generation

From each analyzed paper, three tasks are generated:

1. **Research Task** (`[Research]`)
   - Study the paper methodology
   - Blocked by: nothing

2. **Implementation Task** (`[Implement]`)
   - Build the technique
   - Blocked by: Research task

3. **Evaluation Task** (`[Evaluate]`)
   - Test and measure results
   - Blocked by: Implementation task

Tasks include metadata linking back to the paper:
- `paperId`, `paperTitle`, `paperUrl`
- `phase`: `research`, `implementation`, or `evaluation`
- `domainSlug`: Current domain context
- `complexity`, `relevanceScore`

## Paper Status Flow

```
pending → fetching → fetched → analyzing → analyzed → accepted → implemented
                         ↓                     ↓
                       error                rejected
```

| Status | Description |
|--------|-------------|
| `pending` | Paper queued but not yet fetched |
| `fetching` | Currently fetching metadata |
| `fetched` | Metadata retrieved, ready for analysis |
| `analyzing` | AI is analyzing the paper |
| `analyzed` | Analysis complete, awaiting decision |
| `accepted` | User approved, tasks will be created |
| `rejected` | User rejected, no tasks created |
| `implemented` | Tasks created and linked |
| `error` | An error occurred |

## Storage

Papers are stored in JSON format at `/data/papers/papers.json`:

```json
{
  "papers": [
    {
      "id": "paper_abc123",
      "metadata": { ... },
      "status": "analyzed",
      "analysis": { ... },
      "addedAt": "2024-01-15T10:00:00Z"
    }
  ],
  "lastUpdated": "2024-01-15T10:00:00Z"
}
```

## Usage

### Adding a Paper

1. Navigate to `/papers`
2. Click "Add Paper"
3. Paste arXiv ID, DOI, or URL
4. System auto-detects source and fetches metadata
5. Click "Analyze" to run AI analysis
6. Review analysis results
7. Click "Accept" to create tasks or "Reject" to skip

### Via API

```typescript
// Add a paper
const response = await fetch('/api/papers', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    input: '2401.12345',
    domainSlug: 'voice-cloning'
  })
});

// Analyze a paper
await fetch('/api/papers/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    paperId: 'paper_abc123',
    domainSlug: 'voice-cloning'
  })
});
```

## Rate Limiting

API clients implement rate limiting to avoid being blocked:

- **arXiv**: 350ms between requests (max 3/second)
- **Semantic Scholar**: 100ms between requests
- **GitHub**: 100ms between requests

## Testing

```bash
# Run all paper tests
npx vitest run __tests__/lib/papers/
npx vitest run __tests__/components/papers/
npx vitest run __tests__/integration/paper-flow.test.ts

# Test counts
# - Unit tests for API clients: 68 tests
# - Component tests: 40 tests
# - Integration tests: 16 tests
# - Total: 124 tests
```

## Configuration

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for AI analysis (falls back to mock if not set) |
| `GITHUB_TOKEN` | Optional for higher GitHub API limits |
| `SEMANTIC_SCHOLAR_API_KEY` | Optional for higher S2 API limits |

### Domain Context

Papers can be analyzed with domain context for more relevant scoring:

```typescript
generateAnalysisPrompt(paper, domainConfig);
```

The domain's `focusAreas` and research goals are included in the prompt.

## Future Enhancements

- [ ] PDF text extraction for direct uploads
- [ ] Bulk paper import from CSV/BibTeX
- [ ] Citation graph visualization
- [ ] Automatic paper recommendations
- [ ] Integration with reference managers (Zotero, Mendeley)
