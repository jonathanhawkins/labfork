/**
 * Research Activation
 *
 * Creates research tasks in the Nudge Engine when a lab is forked.
 * The 4090 worker picks these up and executes AI research autonomously.
 */

import type { Lab } from "./types";

const NUDGE_ENGINE_URL =
  process.env.NUDGE_ENGINE_URL ||
  "https://nudge-engine.jonathan-hawkins.workers.dev";
const NUDGE_ENGINE_API_KEY = process.env.NUDGE_ENGINE_API_KEY || "";

interface TaskCreateResult {
  success: boolean;
  task?: { id: string; action: string };
}

async function createTask(params: {
  action: string;
  description: string;
  priority?: number;
  parent_task_id?: string;
  lab_id: string;
  required_capability?: string;
}): Promise<TaskCreateResult> {
  const res = await fetch(`${NUDGE_ENGINE_URL}/tasks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(NUDGE_ENGINE_API_KEY
        ? { Authorization: `Bearer ${NUDGE_ENGINE_API_KEY}` }
        : {}),
    },
    body: JSON.stringify({
      ...params,
      source: "human",
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    console.error("Failed to create task:", err);
    return { success: false };
  }

  const data = await res.json();
  return { success: true, task: data.task };
}

/**
 * Activate research for a forked lab.
 * Creates a parent task and 3 subtasks that the 4090 worker will pick up.
 */
export async function activateResearch(lab: Lab): Promise<{
  activated: boolean;
  parentTaskId?: string;
  subtaskIds?: string[];
}> {
  const slug = lab.slug;
  const labId = lab.id;
  const labName = lab.name;

  // Create parent task
  const parent = await createTask({
    action: `goal-research-${slug}`,
    description: `Research parent task for lab "${labName}". AI agents will survey literature, identify techniques, and write prototype code.`,
    priority: 8,
    lab_id: labId,
  });

  if (!parent.success || !parent.task) {
    console.error("Failed to create parent research task for lab:", labId);
    return { activated: false };
  }

  const parentId = parent.task.id;
  const subtaskIds: string[] = [];

  // Create 3 research subtasks
  const subtasks = [
    {
      action: `goal-research-${slug}-survey`,
      description: `Survey existing approaches, papers, and implementations related to "${labName}". Find the 3-5 most relevant papers and summarize key findings. Focus on practical approaches that could be implemented.`,
      priority: 7,
      required_capability: "code",
    },
    {
      action: `goal-research-${slug}-techniques`,
      description: `Based on the research topic "${labName}", identify the most promising techniques and design a concrete experiment plan. List specific tools, libraries, and datasets needed.`,
      priority: 6,
      required_capability: "code",
    },
    {
      action: `goal-research-${slug}-prototype`,
      description: `Write initial prototype code for "${labName}". Create a minimal working implementation that demonstrates the core concept. Include tests.`,
      priority: 5,
      required_capability: "code",
    },
  ];

  for (const sub of subtasks) {
    const result = await createTask({
      ...sub,
      parent_task_id: parentId,
      lab_id: labId,
    });
    if (result.success && result.task) {
      subtaskIds.push(result.task.id);
    }
  }

  return { activated: true, parentTaskId: parentId, subtaskIds };
}
