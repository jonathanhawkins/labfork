/**
 * Task Generation from Paper Analysis
 *
 * Converts analyzed papers into actionable tasks using the existing task API.
 */

import type { Paper, PaperAnalysis } from "./types";
import { generateTaskDescription } from "./prompts";

/**
 * Task creation request
 */
export interface TaskCreateRequest {
  subject: string;
  description: string;
  activeForm?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Generated task set from a paper
 */
export interface GeneratedTasks {
  research: TaskCreateRequest;
  implementation: TaskCreateRequest;
  evaluation: TaskCreateRequest;
}

/**
 * Generate tasks from an analyzed paper
 */
export function generateTasksFromPaper(
  paper: Paper,
  domainSlug?: string
): GeneratedTasks | null {
  if (!paper.analysis) {
    console.error("Cannot generate tasks: paper has no analysis");
    return null;
  }

  const analysis = paper.analysis;
  const effectiveDomain = domainSlug || paper.domainSlug;

  // Generate research task
  const researchDescription = generateTaskDescription(
    "research",
    {
      title: paper.metadata.title,
      url: paper.metadata.url,
      authors: paper.metadata.authors,
    },
    analysis
  );

  const researchTask: TaskCreateRequest = {
    subject: `[Research] ${analysis.taskBreakdown.research.title}`,
    description: researchDescription,
    activeForm: "Researching paper techniques",
    metadata: {
      paperId: paper.id,
      paperTitle: paper.metadata.title,
      paperUrl: paper.metadata.url,
      phase: "research",
      domainSlug: effectiveDomain,
      complexity: analysis.complexity,
      relevanceScore: analysis.relevanceScore,
    },
  };

  // Generate implementation task
  const implementationDescription = generateTaskDescription(
    "implementation",
    {
      title: paper.metadata.title,
      url: paper.metadata.url,
      authors: paper.metadata.authors,
    },
    analysis
  );

  const implementationTask: TaskCreateRequest = {
    subject: `[Implement] ${analysis.taskBreakdown.implementation.title}`,
    description: implementationDescription,
    activeForm: "Implementing paper techniques",
    metadata: {
      paperId: paper.id,
      paperTitle: paper.metadata.title,
      paperUrl: paper.metadata.url,
      phase: "implementation",
      domainSlug: effectiveDomain,
      complexity: analysis.complexity,
      estimatedHours: analysis.taskBreakdown.implementation.estimatedHours,
      resources: analysis.resources
        .filter((r) => r.required)
        .map((r) => r.name),
    },
  };

  // Generate evaluation task
  const evaluationDescription = generateTaskDescription(
    "evaluation",
    {
      title: paper.metadata.title,
      url: paper.metadata.url,
      authors: paper.metadata.authors,
    },
    analysis
  );

  const evaluationTask: TaskCreateRequest = {
    subject: `[Evaluate] ${analysis.taskBreakdown.evaluation.title}`,
    description: evaluationDescription,
    activeForm: "Evaluating implementation",
    metadata: {
      paperId: paper.id,
      paperTitle: paper.metadata.title,
      paperUrl: paper.metadata.url,
      phase: "evaluation",
      domainSlug: effectiveDomain,
      metrics: analysis.taskBreakdown.evaluation.metrics,
    },
  };

  return {
    research: researchTask,
    implementation: implementationTask,
    evaluation: evaluationTask,
  };
}

/**
 * Post paper to the distributed compute network (Workers D1).
 * Generates 3-5 crowd-tier compute tasks from the paper.
 */
/**
 * Post a paper to the compute network to generate tasks.
 *
 * IMPORTANT: This function should only be called server-side (API routes)
 * to avoid exposing the admin API key to the client.
 */
export async function postToComputeNetwork(
  paper: Paper,
  options: {
    adminApiKey?: string;
  } = {}
): Promise<{
  success: boolean;
  generated?: number;
  error?: string;
}> {
  const workersApiUrl = "https://labfork-agents.jonathan-hawkins.workers.dev";
  const adminApiKey = options.adminApiKey || process.env.ADMIN_API_KEY;
  try {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (adminApiKey) {
      headers["Authorization"] = `Bearer ${adminApiKey}`;
    }
    const response = await fetch(
      `${workersApiUrl}/api/compute/tasks/from-paper`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          paperId: paper.id,
          title: paper.metadata.title,
          abstract: paper.metadata.abstract || "",
          sourceUrl: paper.metadata.url,
        }),
      }
    );

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(
        (errData as Record<string, string>).error || `HTTP ${response.status}`
      );
    }

    const data = (await response.json()) as {
      success: boolean;
      generated: number;
    };
    return { success: true, generated: data.generated };
  } catch (error) {
    console.error("Error posting to compute network:", error);
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Failed to post to compute network",
    };
  }
}

/**
 * Create tasks via the API
 *
 * Returns the created task IDs or null if creation failed.
 */
export async function createTasksViaAPI(
  tasks: GeneratedTasks,
  apiBaseUrl: string = ""
): Promise<{
  success: boolean;
  taskIds?: string[];
  error?: string;
}> {
  const createdIds: string[] = [];

  try {
    // Create research task first
    const researchResponse = await fetch(`${apiBaseUrl}/api/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tasks.research),
    });

    if (!researchResponse.ok) {
      throw new Error("Failed to create research task");
    }

    const researchData = await researchResponse.json();
    const researchId = researchData.task?.id;
    if (researchId) {
      createdIds.push(researchId);
    }

    // Create implementation task (blocked by research)
    const implementationBody = {
      ...tasks.implementation,
      blockedBy: researchId ? [researchId] : [],
    };

    const implResponse = await fetch(`${apiBaseUrl}/api/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(implementationBody),
    });

    if (!implResponse.ok) {
      throw new Error("Failed to create implementation task");
    }

    const implData = await implResponse.json();
    const implId = implData.task?.id;
    if (implId) {
      createdIds.push(implId);
    }

    // Create evaluation task (blocked by implementation)
    const evaluationBody = {
      ...tasks.evaluation,
      blockedBy: implId ? [implId] : [],
    };

    const evalResponse = await fetch(`${apiBaseUrl}/api/tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(evaluationBody),
    });

    if (!evalResponse.ok) {
      throw new Error("Failed to create evaluation task");
    }

    const evalData = await evalResponse.json();
    const evalId = evalData.task?.id;
    if (evalId) {
      createdIds.push(evalId);
    }

    return {
      success: true,
      taskIds: createdIds,
    };
  } catch (error) {
    console.error("Error creating tasks:", error);
    return {
      success: false,
      taskIds: createdIds,
      error:
        error instanceof Error ? error.message : "Failed to create tasks",
    };
  }
}

/**
 * Estimate total effort for a paper
 */
export function estimateTotalEffort(analysis: PaperAnalysis): {
  hours: number;
  days: number;
  description: string;
} {
  const researchHours = analysis.taskBreakdown.research.estimatedHours || 4;
  const implHours = analysis.taskBreakdown.implementation.estimatedHours || 16;
  const evalHours = analysis.taskBreakdown.evaluation.estimatedHours || 8;

  const totalHours = researchHours + implHours + evalHours;
  const days = Math.ceil(totalHours / 8);

  let description: string;
  if (totalHours <= 8) {
    description = "Can be completed in a day";
  } else if (totalHours <= 24) {
    description = "Requires 2-3 days of focused work";
  } else if (totalHours <= 40) {
    description = "Approximately one week of work";
  } else {
    description = "Multi-week project";
  }

  return {
    hours: totalHours,
    days,
    description,
  };
}

/**
 * Get task prefix based on complexity
 */
export function getTaskPrefix(complexity: string): string {
  switch (complexity) {
    case "simple":
      return "[Quick]";
    case "moderate":
      return "";
    case "complex":
      return "[Deep]";
    case "research":
      return "[Exp]"; // Experimental
    default:
      return "";
  }
}

/**
 * Generate a summary of tasks to be created
 */
export function generateTaskSummary(
  paper: Paper,
  tasks: GeneratedTasks
): string {
  if (!paper.analysis) {
    return "No analysis available";
  }

  const analysis = paper.analysis;
  const effort = estimateTotalEffort(analysis);

  const lines = [
    `## Task Summary for: ${paper.metadata.title}`,
    "",
    `**Relevance**: ${analysis.relevanceScore}/100`,
    `**Complexity**: ${analysis.complexity}`,
    `**Estimated Effort**: ${effort.hours} hours (~${effort.days} days)`,
    "",
    "### Tasks to Create:",
    "",
    `1. **${tasks.research.subject}**`,
    `   ${analysis.taskBreakdown.research.description.split("\n")[0]}`,
    "",
    `2. **${tasks.implementation.subject}**`,
    `   ${analysis.taskBreakdown.implementation.description.split("\n")[0]}`,
    "",
    `3. **${tasks.evaluation.subject}**`,
    `   ${analysis.taskBreakdown.evaluation.description.split("\n")[0]}`,
    "",
    "### Dependencies:",
    "- Implementation blocked by Research",
    "- Evaluation blocked by Implementation",
  ];

  return lines.join("\n");
}
