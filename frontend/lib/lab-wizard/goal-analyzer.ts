/**
 * AI Goal Analysis Service
 *
 * Uses Claude to analyze research goals and generate recommendations.
 */

import type {
  ResearchGoal,
  RecommendedPaper,
  InitialTask,
} from "./types";

/**
 * Goal analysis result
 */
export interface GoalAnalysisResult {
  success: boolean;
  analysis?: {
    /** Suggested domain slug */
    suggestedDomain: string;
    /** Domain name for display */
    domainName: string;
    /** Why this domain was suggested */
    domainReason: string;
    /** Suggested arXiv categories */
    arxivCategories: string[];
    /** Suggested keywords for search */
    keywords: string[];
    /** Recommended papers to start with */
    recommendedPapers: RecommendedPaper[];
    /** Generated initial tasks */
    initialTasks: InitialTask[];
    /** Estimated total hours to achieve goal */
    estimatedHours: number;
    /** Confidence score 0-100 */
    confidence: number;
  };
  error?: string;
}

/**
 * Analyze research goal with AI
 */
export async function analyzeGoal(
  goalText: string,
  options?: {
    preferredDomain?: string;
    hardwareVram?: number;
  }
): Promise<GoalAnalysisResult> {
  try {
    const response = await fetch("/api/lab/analyze-goal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        goalText,
        preferredDomain: options?.preferredDomain,
        hardwareVram: options?.hardwareVram,
      }),
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        analysis: data.analysis,
      };
    }

    return {
      success: false,
      error: data.error || "Failed to analyze goal",
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to analyze goal",
    };
  }
}

/**
 * Generate initial tasks from goal analysis
 */
export async function generateInitialTasks(
  goalText: string,
  domainSlug: string,
  papers?: RecommendedPaper[]
): Promise<{ success: boolean; tasks?: InitialTask[]; error?: string }> {
  try {
    const response = await fetch("/api/lab/analyze-goal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        goalText,
        domainSlug,
        papers,
        generateTasks: true,
      }),
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        tasks: data.tasks,
      };
    }

    return {
      success: false,
      error: data.error || "Failed to generate tasks",
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to generate tasks",
    };
  }
}

/**
 * System prompt for goal analysis
 */
export const GOAL_ANALYSIS_SYSTEM_PROMPT = `You are an AI research lab setup assistant. Your job is to analyze a user's research goal and suggest:

1. The most appropriate research domain
2. Relevant arXiv categories to monitor
3. Keywords for paper search
4. 3-5 specific papers to start with
5. 3-5 initial tasks to get started

Be practical and focused. Suggest achievable tasks that can be completed in a few days each.

Output your analysis in the following JSON format:
{
  "suggestedDomain": "domain-slug",
  "domainName": "Human Readable Name",
  "domainReason": "Brief explanation of why this domain fits",
  "arxivCategories": ["cs.XX", "eess.XX"],
  "keywords": ["keyword1", "keyword2"],
  "recommendedPapers": [
    {
      "title": "Paper Title",
      "arxivId": "2401.XXXXX",
      "reason": "Why this paper is relevant",
      "relevanceScore": 85
    }
  ],
  "initialTasks": [
    {
      "subject": "Task title",
      "description": "What to do",
      "type": "research|implementation|evaluation|setup",
      "estimatedHours": 4,
      "priority": "high|medium|low"
    }
  ],
  "estimatedHours": 40,
  "confidence": 85
}`;

/**
 * Generate goal analysis prompt
 */
export function generateGoalPrompt(
  goalText: string,
  options?: {
    preferredDomain?: string;
    hardwareVram?: number;
  }
): string {
  let prompt = `Analyze the following research goal and provide recommendations:\n\n`;
  prompt += `Research Goal: "${goalText}"\n\n`;

  if (options?.preferredDomain) {
    prompt += `User's preferred domain: ${options.preferredDomain}\n`;
  }

  if (options?.hardwareVram) {
    prompt += `Available GPU VRAM: ${options.hardwareVram}GB\n`;
    prompt += `Consider hardware constraints when suggesting tasks.\n`;
  }

  prompt += `\nProvide your analysis in the specified JSON format.`;

  return prompt;
}

/**
 * Parse goal analysis response
 */
export function parseGoalAnalysisResponse(response: string): GoalAnalysisResult {
  try {
    // Extract JSON from response
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      return {
        success: false,
        error: "Could not parse analysis response",
      };
    }

    const analysis = JSON.parse(jsonMatch[0]);

    // Validate required fields
    if (!analysis.suggestedDomain || !analysis.domainName) {
      return {
        success: false,
        error: "Invalid analysis format",
      };
    }

    return {
      success: true,
      analysis: {
        suggestedDomain: analysis.suggestedDomain,
        domainName: analysis.domainName,
        domainReason: analysis.domainReason || "",
        arxivCategories: analysis.arxivCategories || [],
        keywords: analysis.keywords || [],
        recommendedPapers: (analysis.recommendedPapers || []).map((p: any) => ({
          title: p.title || "Unknown Paper",
          arxivId: p.arxivId,
          reason: p.reason || "",
          relevanceScore: p.relevanceScore || 50,
        })),
        initialTasks: (analysis.initialTasks || []).map((t: any) => ({
          subject: t.subject || "Task",
          description: t.description || "",
          type: t.type || "research",
          estimatedHours: t.estimatedHours || 4,
          priority: t.priority || "medium",
        })),
        estimatedHours: analysis.estimatedHours || 40,
        confidence: analysis.confidence || 70,
      },
    };
  } catch (error) {
    return {
      success: false,
      error: "Failed to parse analysis response",
    };
  }
}

/**
 * Estimate timeline from tasks
 */
export function estimateTimeline(
  tasks: InitialTask[]
): {
  totalHours: number;
  totalDays: number;
  weeklyHours: number;
  weeks: number;
  description: string;
} {
  const totalHours = tasks.reduce((sum, t) => sum + (t.estimatedHours || 4), 0);
  const weeklyHours = 20; // Assume 20 hours/week for research
  const weeks = Math.ceil(totalHours / weeklyHours);
  const totalDays = Math.ceil(totalHours / 8);

  let description: string;
  if (weeks <= 1) {
    description = "Can be completed in about a week";
  } else if (weeks <= 2) {
    description = "About 2 weeks of focused work";
  } else if (weeks <= 4) {
    description = `Around ${weeks} weeks of research`;
  } else {
    description = `Multi-month research project (${weeks}+ weeks)`;
  }

  return {
    totalHours,
    totalDays,
    weeklyHours,
    weeks,
    description,
  };
}

/**
 * Get task type display info
 */
export function getTaskTypeInfo(type: InitialTask["type"]): {
  label: string;
  color: string;
  bgColor: string;
  icon: string;
} {
  switch (type) {
    case "research":
      return {
        label: "Research",
        color: "text-blue-400",
        bgColor: "bg-blue-500/10",
        icon: "BookOpen",
      };
    case "implementation":
      return {
        label: "Implementation",
        color: "text-purple-400",
        bgColor: "bg-purple-500/10",
        icon: "Code",
      };
    case "evaluation":
      return {
        label: "Evaluation",
        color: "text-green-400",
        bgColor: "bg-green-500/10",
        icon: "TestTube",
      };
    case "setup":
      return {
        label: "Setup",
        color: "text-yellow-400",
        bgColor: "bg-yellow-500/10",
        icon: "Settings",
      };
    default:
      return {
        label: "Task",
        color: "text-foreground-muted",
        bgColor: "bg-foreground-muted/10",
        icon: "CheckCircle",
      };
  }
}

/**
 * Get priority display info
 */
export function getPriorityInfo(priority: InitialTask["priority"]): {
  label: string;
  color: string;
} {
  switch (priority) {
    case "high":
      return { label: "High", color: "text-red-400" };
    case "medium":
      return { label: "Medium", color: "text-yellow-400" };
    case "low":
      return { label: "Low", color: "text-foreground-muted" };
    default:
      return { label: "Normal", color: "text-foreground-muted" };
  }
}

/**
 * Apply analysis to research goal
 */
export function applyAnalysisToGoal(
  goal: ResearchGoal,
  analysis: GoalAnalysisResult["analysis"]
): ResearchGoal {
  if (!analysis) return goal;

  return {
    ...goal,
    suggestedDomain: analysis.suggestedDomain,
    suggestedCategories: analysis.arxivCategories,
    suggestedKeywords: analysis.keywords,
    recommendedPapers: analysis.recommendedPapers,
    initialTasks: analysis.initialTasks,
    analyzed: true,
  };
}

/**
 * Mock goal analysis for testing
 */
export const MOCK_GOAL_ANALYSIS: GoalAnalysisResult["analysis"] = {
  suggestedDomain: "voice-clone",
  domainName: "Voice Cloning Research",
  domainReason: "Your goal involves speech synthesis and prosody control",
  arxivCategories: ["cs.SD", "eess.AS", "cs.CL"],
  keywords: ["prosody control", "voice cloning", "emotion synthesis"],
  recommendedPapers: [
    {
      title: "Controllable Prosody Generation for TTS",
      arxivId: "2401.12345",
      reason: "Directly addresses prosody control in TTS",
      relevanceScore: 92,
    },
    {
      title: "Emotion Transfer in Voice Conversion",
      arxivId: "2312.54321",
      reason: "Explores emotion disentanglement",
      relevanceScore: 85,
    },
    {
      title: "Multi-Speaker Voice Cloning with Prosody",
      arxivId: "2311.11111",
      reason: "State-of-the-art voice cloning approach",
      relevanceScore: 88,
    },
  ],
  initialTasks: [
    {
      subject: "Set up voice cloning baseline",
      description: "Install and test Sesame CSM-1B model locally",
      type: "setup",
      estimatedHours: 4,
      priority: "high",
    },
    {
      subject: "Study prosody control techniques",
      description: "Read recommended papers and summarize approaches",
      type: "research",
      estimatedHours: 8,
      priority: "high",
    },
    {
      subject: "Implement prosody conditioning",
      description: "Add prosody embeddings to training pipeline",
      type: "implementation",
      estimatedHours: 16,
      priority: "medium",
    },
    {
      subject: "Evaluate prosody control quality",
      description: "Test emotion transfer and measure F0 correlation",
      type: "evaluation",
      estimatedHours: 8,
      priority: "medium",
    },
  ],
  estimatedHours: 36,
  confidence: 87,
};
