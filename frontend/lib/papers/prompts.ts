/**
 * Paper Analysis Prompt Templates
 *
 * Prompts for AI-powered paper analysis using Claude/Codex.
 */

import type { DomainConfig } from "../domain/types";

/**
 * System prompt for paper analysis
 */
export const PAPER_ANALYSIS_SYSTEM_PROMPT = `You are an expert research analyst specializing in machine learning and AI research. Your task is to analyze academic papers and assess their relevance, novelty, and implementation feasibility.

You will receive:
1. Paper metadata (title, authors, abstract)
2. Domain context (research focus, keywords, evaluation metrics)

You must provide a structured analysis in JSON format.

Be objective and thorough. Consider:
- How well the paper aligns with the domain's research goals
- The novelty of the techniques presented
- Practical implementation requirements
- Potential challenges and risks

Your analysis should help researchers decide whether to implement the paper's techniques.`;

/**
 * Generate the paper analysis prompt
 */
export function generateAnalysisPrompt(
  paper: {
    title: string;
    authors: { name: string }[];
    abstract: string;
    categories?: string[];
    citationCount?: number;
    venue?: string;
  },
  domain?: DomainConfig
): string {
  const authorNames = paper.authors.map((a) => a.name).join(", ");
  const categories = paper.categories?.join(", ") || "Not specified";
  const venue = paper.venue || "Not specified";
  const citations = paper.citationCount ?? "Unknown";

  let domainContext = "";
  if (domain) {
    domainContext = `
## Domain Context

**Domain**: ${domain.name}
**Description**: ${domain.description}
**Research Focus**: ${domain.research.arxivCategories.join(", ")}
**Keywords**: ${domain.research.keywords.join(", ")}
${domain.evaluation?.metrics ? `**Evaluation Metrics**: ${domain.evaluation.metrics.map((m) => m.name).join(", ")}` : ""}
`;
  }

  return `# Paper Analysis Request

## Paper Information

**Title**: ${paper.title}

**Authors**: ${authorNames}

**Categories**: ${categories}

**Venue**: ${venue}

**Citations**: ${citations}

**Abstract**:
${paper.abstract}

${domainContext}

## Analysis Required

Analyze this paper and provide a JSON response with the following structure:

\`\`\`json
{
  "relevanceScore": <number 0-100>,
  "relevanceReason": "<string explaining the score>",
  "techniques": [
    {
      "name": "<technique name>",
      "description": "<brief description>",
      "isMainContribution": <boolean>,
      "relatedTo": ["<related techniques>"]
    }
  ],
  "novelty": "<string describing the main novelty/contribution>",
  "complexity": "<'simple' | 'moderate' | 'complex' | 'research'>",
  "complexityReason": "<string explaining complexity assessment>",
  "resources": [
    {
      "type": "<'gpu' | 'dataset' | 'model' | 'library' | 'hardware' | 'other'>",
      "name": "<resource name>",
      "required": <boolean>,
      "estimate": "<size/cost estimate if applicable>",
      "notes": "<additional notes>"
    }
  ],
  "taskBreakdown": {
    "research": {
      "title": "<task title>",
      "description": "<detailed description>",
      "estimatedHours": <number>
    },
    "implementation": {
      "title": "<task title>",
      "description": "<detailed description>",
      "estimatedHours": <number>,
      "dependencies": ["<dependencies>"]
    },
    "evaluation": {
      "title": "<task title>",
      "description": "<detailed description>",
      "estimatedHours": <number>,
      "metrics": ["<metrics to evaluate>"]
    }
  },
  "risks": ["<potential risks or challenges>"],
  "relatedWork": ["<related papers or techniques to explore>"]
}
\`\`\`

## Scoring Guidelines

**Relevance Score (0-100)**:
- 80-100: Directly addresses domain goals, highly applicable
- 60-79: Related to domain, useful techniques
- 40-59: Tangentially related, some applicable concepts
- 20-39: Loosely related, limited applicability
- 0-19: Not relevant to the domain

**Complexity Levels**:
- simple: Can implement in 1-4 hours, minimal dependencies
- moderate: 4-16 hours, some setup required
- complex: 1-3 days, significant infrastructure needed
- research: 3+ days, requires experimentation and iteration

Provide ONLY the JSON response, no additional text.`;
}

/**
 * Parse the analysis response from the AI
 */
export function parseAnalysisResponse(response: string): {
  success: boolean;
  analysis?: {
    relevanceScore: number;
    relevanceReason: string;
    techniques: {
      name: string;
      description: string;
      isMainContribution: boolean;
      relatedTo?: string[];
    }[];
    novelty: string;
    complexity: "simple" | "moderate" | "complex" | "research";
    complexityReason: string;
    resources: {
      type: "gpu" | "dataset" | "model" | "library" | "hardware" | "other";
      name: string;
      required: boolean;
      estimate?: string;
      notes?: string;
    }[];
    taskBreakdown: {
      research: {
        title: string;
        description: string;
        estimatedHours?: number;
      };
      implementation: {
        title: string;
        description: string;
        estimatedHours?: number;
        dependencies?: string[];
      };
      evaluation: {
        title: string;
        description: string;
        estimatedHours?: number;
        metrics?: string[];
      };
    };
    risks?: string[];
    relatedWork?: string[];
  };
  error?: string;
} {
  try {
    // Try to extract JSON from the response
    let jsonStr = response;

    // Check for markdown code block
    const codeBlockMatch = response.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (codeBlockMatch) {
      jsonStr = codeBlockMatch[1];
    }

    // Parse JSON
    const parsed = JSON.parse(jsonStr.trim());

    // Validate required fields
    if (
      typeof parsed.relevanceScore !== "number" ||
      parsed.relevanceScore < 0 ||
      parsed.relevanceScore > 100
    ) {
      return {
        success: false,
        error: "Invalid relevance score",
      };
    }

    if (!parsed.relevanceReason || typeof parsed.relevanceReason !== "string") {
      return {
        success: false,
        error: "Missing relevance reason",
      };
    }

    if (!Array.isArray(parsed.techniques)) {
      return {
        success: false,
        error: "Missing techniques array",
      };
    }

    if (!parsed.novelty || typeof parsed.novelty !== "string") {
      return {
        success: false,
        error: "Missing novelty description",
      };
    }

    const validComplexities = ["simple", "moderate", "complex", "research"];
    if (!validComplexities.includes(parsed.complexity)) {
      return {
        success: false,
        error: "Invalid complexity level",
      };
    }

    if (!parsed.taskBreakdown) {
      return {
        success: false,
        error: "Missing task breakdown",
      };
    }

    return {
      success: true,
      analysis: parsed,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? `Failed to parse response: ${error.message}`
          : "Unknown error parsing response",
    };
  }
}

/**
 * Generate a quick relevance check prompt (cheaper, faster)
 */
export function generateQuickRelevancePrompt(
  paperTitle: string,
  paperAbstract: string,
  domainKeywords: string[]
): string {
  return `Quickly assess the relevance of this paper to the research domain.

**Paper Title**: ${paperTitle}

**Abstract**: ${paperAbstract}

**Domain Keywords**: ${domainKeywords.join(", ")}

Respond with ONLY a JSON object:
{
  "relevanceScore": <number 0-100>,
  "reason": "<one sentence explanation>"
}`;
}

/**
 * Generate a task description from paper analysis
 */
export function generateTaskDescription(
  phase: "research" | "implementation" | "evaluation",
  paper: {
    title: string;
    url: string;
    authors: { name: string }[];
  },
  analysis: {
    techniques: { name: string; description: string }[];
    novelty: string;
    complexity: string;
    resources: { name: string; required: boolean }[];
    taskBreakdown: {
      research: { title: string; description: string };
      implementation: { title: string; description: string; dependencies?: string[] };
      evaluation: { title: string; description: string; metrics?: string[] };
    };
  }
): string {
  const authorNames = paper.authors.slice(0, 3).map((a) => a.name).join(", ");
  const techniques = analysis.techniques.map((t) => t.name).join(", ");

  switch (phase) {
    case "research":
      return `## Research Task: ${analysis.taskBreakdown.research.title}

**Source Paper**: [${paper.title}](${paper.url})
**Authors**: ${authorNames}
**Key Techniques**: ${techniques}

### Objective
${analysis.taskBreakdown.research.description}

### Background
${analysis.novelty}

### Research Questions
1. What are the core innovations in this paper?
2. How do the proposed techniques compare to existing methods?
3. What assumptions does the approach make?
4. What are the theoretical foundations?

### Deliverables
- [ ] Summary of key contributions
- [ ] Comparison with related work
- [ ] Notes on potential adaptations for our domain
- [ ] Identified prerequisites for implementation`;

    case "implementation":
      const requiredResources = analysis.resources
        .filter((r) => r.required)
        .map((r) => `- ${r.name}`)
        .join("\n");

      return `## Implementation Task: ${analysis.taskBreakdown.implementation.title}

**Source Paper**: [${paper.title}](${paper.url})
**Complexity**: ${analysis.complexity}
**Key Techniques**: ${techniques}

### Objective
${analysis.taskBreakdown.implementation.description}

${analysis.taskBreakdown.implementation.dependencies?.length ? `### Dependencies
${analysis.taskBreakdown.implementation.dependencies.map((d) => `- ${d}`).join("\n")}` : ""}

### Required Resources
${requiredResources || "- No specific resources required"}

### Implementation Steps
1. Set up development environment
2. Implement core algorithm/model
3. Create configuration system
4. Add logging and monitoring
5. Write unit tests
6. Document the implementation

### Success Criteria
- Code runs without errors
- Produces outputs matching paper's examples
- Unit tests pass
- Documentation is complete`;

    case "evaluation":
      const metrics = analysis.taskBreakdown.evaluation.metrics || [];

      return `## Evaluation Task: ${analysis.taskBreakdown.evaluation.title}

**Source Paper**: [${paper.title}](${paper.url})
**Key Techniques**: ${techniques}

### Objective
${analysis.taskBreakdown.evaluation.description}

### Metrics to Evaluate
${metrics.length > 0 ? metrics.map((m) => `- ${m}`).join("\n") : "- To be determined based on domain metrics"}

### Evaluation Plan
1. Prepare test dataset
2. Run baseline comparisons
3. Execute technique on test cases
4. Compute evaluation metrics
5. Analyze results and edge cases
6. Document findings

### Deliverables
- [ ] Evaluation script
- [ ] Results report with metrics
- [ ] Comparison with baseline
- [ ] Analysis of strengths/weaknesses
- [ ] Recommendations for production use`;

    default:
      return "";
  }
}
