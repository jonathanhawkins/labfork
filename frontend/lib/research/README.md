# Multi-Source Research Input (Phase 7)

This module provides a unified interface for ingesting research papers and resources from multiple sources.

## Overview

The research input pipeline supports:
- **GitHub Repository Analysis** - Analyze ML repositories for frameworks, architectures, and techniques
- **Custom Research Goals** - Natural language goal analysis with paper suggestions and research plans
- **PDF Upload** - Extract metadata from uploaded PDF papers
- **Enhanced Semantic Scholar** - Citation networks, related papers, and influence scores
- **Universal Ingestion Pipeline** - Unified interface for all sources with deduplication

## Architecture

```
                    +-----------------------+
                    |   Universal Ingest    |
                    |   /api/research/ingest|
                    +-----------+-----------+
                                |
        +-----------------------+-----------------------+
        |           |           |           |           |
   +----+----+ +----+----+ +----+----+ +----+----+ +----+----+
   | arXiv   | | GitHub  | | Goal    | | PDF     | | DOI     |
   | Parser  | | Analyzer| | Analyzer| | Parser  | | Lookup  |
   +---------+ +---------+ +---------+ +---------+ +---------+
```

## Services

### GitHub Repository Analyzer (`github-analyzer.ts`)

Analyzes GitHub repositories to detect ML patterns:

```typescript
import { analyzeRepository, RepoAnalysis } from "@/lib/research/github-analyzer";

const analysis: RepoAnalysis = analyzeRepository(
  "https://github.com/owner/repo",
  readmeContent,
  codeContent,
  fileList,
  requirementsContent
);

// Analysis includes:
// - framework: "pytorch" | "tensorflow" | "jax" | etc.
// - architectures: ["transformer", "cnn", ...]
// - techniques: [{ name, confidence }]
// - linkedPapers: [{ type, id }]
// - suggestedTasks: [{ description, priority }]
```

### Custom Research Goal Analyzer (`goal-analyzer.ts`)

Analyzes natural language research goals:

```typescript
import { analyzeGoal, GoalAnalysis } from "@/lib/research/goal-analyzer";

const analysis: GoalAnalysis = analyzeGoal(
  "Develop a transformer-based speech synthesis model with prosody control"
);

// Analysis includes:
// - concepts: extracted key concepts
// - techniques: identified techniques
// - recommendedDomain: best matching research domain
// - paperSuggestions: relevant papers to read
// - plan: research plan with milestones
// - resources: time and hardware estimates
// - complexity: "low" | "medium" | "high" | "very-high"
```

### PDF Parser (`pdf-parser.ts`)

Parses PDF papers to extract metadata:

```typescript
import { parsePDF, parsePDFFromText, fromManualEntry } from "@/lib/research/pdf-parser";

// From file
const result = await parsePDF(file);

// From text content
const result = parsePDFFromText(textContent, filename);

// Manual entry
const metadata = fromManualEntry({
  title: "Paper Title",
  authors: "Author 1, Author 2",
  abstract: "Abstract text",
}, "paper.pdf");
```

### Enhanced Semantic Scholar (`semantic-scholar-enhanced.ts`)

Extended Semantic Scholar API with citation networks:

```typescript
import {
  fetchCitations,
  fetchReferences,
  getCitationNetwork,
  findRelatedPapers,
  getInfluenceScore,
} from "@/lib/research/semantic-scholar-enhanced";

// Get citation network (depth 2)
const network = await getCitationNetwork(paperId, 2);

// Find related papers
const related = await findRelatedPapers(paperId, { limit: 10 });

// Calculate influence score
const influence = await getInfluenceScore(paperId);
```

### Universal Ingestion Pipeline (`ingestion-pipeline.ts`)

Unified interface for all sources:

```typescript
import {
  ingest,
  ingestBatch,
  detectSourceType,
  searchAcrossSources,
} from "@/lib/research/ingestion-pipeline";

// Detect source type
const type = detectSourceType("2106.09685"); // "arxiv"

// Ingest single paper
const result = await ingest("2106.09685", { enrichWithCitations: true });

// Batch ingestion with deduplication
const batch = await ingestBatch([
  "2106.09685",
  "https://github.com/owner/repo",
  { type: "goal", value: "Research prosody control" },
]);

// Search across sources
const results = await searchAcrossSources("voice cloning", {
  sources: ["semantic-scholar", "arxiv"],
  limit: 10,
});
```

## API Endpoints

### POST `/api/research/github`
Analyze a GitHub repository.

```json
{
  "repoUrl": "owner/repo"
}
```

### POST `/api/research/goal`
Analyze a research goal.

```json
{
  "goal": "Develop a voice cloning system with prosody control"
}
```

### POST `/api/research/pdf`
Upload and parse a PDF.

FormData with `file` field, or JSON:
```json
{
  "text": "PDF text content",
  "filename": "paper.pdf"
}
```

Or manual entry:
```json
{
  "manual": {
    "title": "Paper Title",
    "authors": "Author 1, Author 2"
  }
}
```

### POST `/api/research/ingest`
Universal ingestion endpoint.

```json
{
  "input": "2106.09685",
  "options": {
    "enrichWithCitations": true,
    "enrichWithRelated": true
  }
}
```

Or batch:
```json
{
  "inputs": ["2106.09685", "10.1234/test"],
  "options": {}
}
```

### GET `/api/research/ingest`
Detect source type or search.

- `?query=2106.09685&detect=true` - Detect source type
- `?query=voice+cloning&sources=arxiv,semantic-scholar&limit=10` - Search

## UI Components

### AddFromGitHub
Dialog for analyzing GitHub repositories.

```tsx
import { AddFromGitHub } from "@/components/research";

<AddFromGitHub
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  onAnalysisComplete={(analysis, paper) => {
    console.log("Analyzed:", analysis);
  }}
/>
```

### AddCustomGoal
Dialog for defining research goals.

```tsx
import { AddCustomGoal } from "@/components/research";

<AddCustomGoal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  onAnalysisComplete={(analysis, papers) => {
    console.log("Goal analysis:", analysis);
  }}
/>
```

### AddPDF
Dialog for PDF upload with drag-and-drop.

```tsx
import { AddPDF } from "@/components/research";

<AddPDF
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  onUploadComplete={(parseResult, paper) => {
    console.log("Parsed PDF:", parseResult);
  }}
/>
```

### SourceComparison
Side-by-side paper comparison from different sources.

```tsx
import { SourceComparison } from "@/components/research";

<SourceComparison
  papers={papers}
  onSelect={(paper) => console.log("Selected:", paper)}
  onDismiss={(paper) => console.log("Dismissed:", paper)}
/>
```

## Source Type Detection

The pipeline automatically detects source types:

| Pattern | Detected Type |
|---------|---------------|
| `2106.09685` | arxiv |
| `https://arxiv.org/abs/...` | arxiv |
| `10.1234/...` | doi |
| `https://doi.org/...` | doi |
| `https://github.com/...` | github |
| `owner/repo` | github |
| Semantic Scholar ID | semantic-scholar |

## Testing

Run all Phase 7 tests:

```bash
npm test -- --run __tests__/lib/research/ __tests__/api/research/ __tests__/components/research/ __tests__/integration/research-pipeline.test.ts
```

Test counts:
- Library tests: 249 tests
- API tests: 52 tests
- Component tests: 72 tests
- Integration tests: 20 tests
- **Total: 393 tests**
