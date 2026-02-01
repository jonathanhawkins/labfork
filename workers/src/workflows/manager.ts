/**
 * Manager Workflow
 *
 * This workflow runs on a schedule (cron every 15 minutes) and orchestrates all agent work.
 * It assesses active projects, creates tasks, assigns agents, and triggers worker workflows.
 */

import {
  WorkflowEntrypoint,
  WorkflowEvent,
  WorkflowStep,
} from 'cloudflare:workers';

// =============================================================================
// Types
// =============================================================================

export interface Env {
  DB: D1Database;
  // AI binding removed - using compute network instead
  WORKER_WORKFLOW: Workflow;
}

export interface Project {
  id: string;
  name: string;
  slug: string;
  status: string;
  config: string | null; // JSON string
  created_at: string;
  updated_at: string;
}

export interface Task {
  id: string;
  project_id: string;
  title: string;
  description: string | null;
  status: 'pending' | 'in_progress' | 'completed' | 'blocked';
  priority: number;
  assigned_agent: string | null;
  blocked_by: string | null; // JSON string of task IDs
  requires_physical: number; // 0 or 1 (boolean in SQLite)
  progress: number;
  result: string | null; // JSON string
  created_at: string;
  updated_at: string;
}

export interface AgentState {
  agent_id: string;
  project_id: string;
  persona: string; // JSON string
  memory: string | null; // JSON string
  current_task_id: string | null;
  status: 'idle' | 'working' | 'blocked' | 'offline';
  tokens_used: number;
  last_active: string | null;
  created_at: string;
}

export interface AgentPersona {
  name: string;
  role: string;
  specializations: string[];
  system_prompt: string;
}

export interface ProjectAssessment {
  projectId: string;
  summary: string;
  completedTasks: number;
  pendingTasks: number;
  blockedTasks: number;
  inProgressTasks: number;
  newTasksNeeded: boolean;
  suggestedTasks: SuggestedTask[];
  priorities: string[];
  bottlenecks: string[];
}

export interface SuggestedTask {
  title: string;
  description: string;
  priority: number;
  requiredSpecializations: string[];
  estimatedDuration: string;
  requires_physical: boolean;
}

export interface WorkerTriggerResult {
  agentId: string;
  taskId: string;
  workflowInstanceId?: string;
  status: 'triggered' | 'failed';
  error?: string;
}

// =============================================================================
// Prompts
// =============================================================================

const PROJECT_ASSESSMENT_PROMPT = `You are the Manager Agent for LabFork, an autonomous AI research platform.
Your job is to assess the current state of a project and determine what work needs to be done.

Analyze the project data and provide a JSON assessment with the following structure:
{
  "summary": "Brief summary of current project state",
  "newTasksNeeded": true/false,
  "suggestedTasks": [
    {
      "title": "Task title",
      "description": "Detailed task description",
      "priority": 1-10 (10 being highest),
      "requiredSpecializations": ["specialization1", "specialization2"],
      "estimatedDuration": "e.g. 2 hours",
      "requires_physical": false
    }
  ],
  "priorities": ["Priority 1", "Priority 2"],
  "bottlenecks": ["Bottleneck 1", "Bottleneck 2"]
}

Rules:
1. Only suggest tasks that can be completed by AI agents (code, research, analysis, documentation)
2. Mark tasks as requires_physical: true if they need hardware, ordering, assembly, etc.
3. Priority 10 = critical/blocking, Priority 1 = nice-to-have
4. Consider task dependencies - don't suggest tasks whose blockers aren't complete
5. Maximum 5 suggested tasks per assessment
6. Be specific and actionable in task descriptions

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.`;

const TASK_CREATION_PROMPT = `You are creating a new task for the LabFork project management system.
Given the suggested task, create a detailed, actionable task that an AI agent can execute.

Return ONLY a JSON object with:
{
  "title": "Clear, concise title",
  "description": "Detailed description with specific steps",
  "priority": 1-10,
  "requires_physical": false,
  "estimated_steps": ["Step 1", "Step 2", "Step 3"]
}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation.`;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate a unique ID for database records
 */
function generateId(prefix: string): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `${prefix}_${timestamp}_${random}`;
}

/**
 * Get all active projects from the database
 */
async function getActiveProjects(db: D1Database): Promise<Project[]> {
  try {
    const result = await db
      .prepare("SELECT * FROM projects WHERE status = ?")
      .bind('active')
      .all<Project>();
    return result.results || [];
  } catch (error) {
    console.error('Failed to get active projects:', error);
    return [];
  }
}

/**
 * Get tasks for a specific project
 */
async function getProjectTasks(
  db: D1Database,
  projectId: string
): Promise<Task[]> {
  try {
    const result = await db
      .prepare("SELECT * FROM tasks WHERE project_id = ? ORDER BY priority DESC, created_at ASC")
      .bind(projectId)
      .all<Task>();
    return result.results || [];
  } catch (error) {
    console.error(`Failed to get tasks for project ${projectId}:`, error);
    return [];
  }
}

/**
 * Get all agents for a project
 */
async function getProjectAgents(
  db: D1Database,
  projectId: string
): Promise<AgentState[]> {
  try {
    const result = await db
      .prepare("SELECT * FROM agent_state WHERE project_id = ?")
      .bind(projectId)
      .all<AgentState>();
    return result.results || [];
  } catch (error) {
    console.error(`Failed to get agents for project ${projectId}:`, error);
    return [];
  }
}

/**
 * Get idle agents across all projects
 */
async function getIdleAgents(db: D1Database): Promise<AgentState[]> {
  try {
    const result = await db
      .prepare("SELECT * FROM agent_state WHERE status = 'idle' AND current_task_id IS NULL")
      .all<AgentState>();
    return result.results || [];
  } catch (error) {
    console.error('Failed to get idle agents:', error);
    return [];
  }
}

/**
 * Get pending tasks that are not blocked and not assigned
 */
async function getPendingTasks(
  db: D1Database,
  projectId: string
): Promise<Task[]> {
  try {
    const result = await db
      .prepare(`
        SELECT * FROM tasks
        WHERE project_id = ?
          AND status = 'pending'
          AND assigned_agent IS NULL
          AND requires_physical = 0
        ORDER BY priority DESC, created_at ASC
      `)
      .bind(projectId)
      .all<Task>();
    return result.results || [];
  } catch (error) {
    console.error(`Failed to get pending tasks for project ${projectId}:`, error);
    return [];
  }
}

/**
 * Create a new task in the database
 */
async function createTask(
  db: D1Database,
  projectId: string,
  task: SuggestedTask
): Promise<string> {
  const taskId = generateId('task');

  try {
    await db
      .prepare(`
        INSERT INTO tasks (
          id, project_id, title, description, status, priority,
          requires_physical, progress, created_at, updated_at
        ) VALUES (?, ?, ?, ?, 'pending', ?, ?, 0, datetime('now'), datetime('now'))
      `)
      .bind(
        taskId,
        projectId,
        task.title,
        task.description,
        task.priority,
        task.requires_physical ? 1 : 0
      )
      .run();

    console.log(`Created task ${taskId}: ${task.title}`);
    return taskId;
  } catch (error) {
    console.error(`Failed to create task: ${error}`);
    throw error;
  }
}

/**
 * Assign an agent to a task
 */
async function assignAgentToTask(
  db: D1Database,
  agentId: string,
  taskId: string
): Promise<void> {
  try {
    // Update task with assigned agent
    await db
      .prepare(`
        UPDATE tasks
        SET assigned_agent = ?, status = 'in_progress', updated_at = datetime('now')
        WHERE id = ?
      `)
      .bind(agentId, taskId)
      .run();

    // Update agent state with current task
    await db
      .prepare(`
        UPDATE agent_state
        SET current_task_id = ?, status = 'working', last_active = datetime('now')
        WHERE agent_id = ?
      `)
      .bind(taskId, agentId)
      .run();

    console.log(`Assigned agent ${agentId} to task ${taskId}`);
  } catch (error) {
    console.error(`Failed to assign agent ${agentId} to task ${taskId}:`, error);
    throw error;
  }
}

/**
 * Parse agent persona from JSON string
 */
function parsePersona(personaJson: string): AgentPersona | null {
  try {
    return JSON.parse(personaJson) as AgentPersona;
  } catch {
    return null;
  }
}

/**
 * Check if an agent's specializations match task requirements
 */
function agentMatchesTask(
  agent: AgentState,
  taskSpecializations: string[]
): boolean {
  const persona = parsePersona(agent.persona);
  if (!persona || !persona.specializations) {
    return false;
  }

  // If no specific specializations required, any agent can do it
  if (!taskSpecializations || taskSpecializations.length === 0) {
    return true;
  }

  // Check if agent has at least one matching specialization
  return persona.specializations.some((spec) =>
    taskSpecializations.some(
      (required) =>
        spec.toLowerCase().includes(required.toLowerCase()) ||
        required.toLowerCase().includes(spec.toLowerCase())
    )
  );
}

/**
 * Log work to the work_log table
 */
async function logWork(
  db: D1Database,
  agentId: string,
  taskId: string | null,
  action: string,
  input: object,
  output: object,
  durationMs: number,
  tokensUsed: number
): Promise<void> {
  const logId = generateId('log');

  try {
    await db
      .prepare(`
        INSERT INTO work_log (
          id, agent_id, task_id, action, input, output,
          duration_ms, tokens_used, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
      `)
      .bind(
        logId,
        agentId,
        taskId,
        action,
        JSON.stringify(input),
        JSON.stringify(output),
        durationMs,
        tokensUsed
      )
      .run();
  } catch (error) {
    console.error('Failed to log work:', error);
    // Don't throw - logging failure shouldn't stop the workflow
  }
}

// =============================================================================
// Manager Workflow
// =============================================================================

export class ManagerWorkflow extends WorkflowEntrypoint<Env, unknown> {
  /**
   * Main workflow execution
   */
  async run(event: WorkflowEvent<unknown>, step: WorkflowStep): Promise<{ processed: number; tasksCreated: number; agentsAssigned: number }> {
    console.log('Manager Workflow started at', new Date().toISOString());

    let totalTasksCreated = 0;
    let totalAgentsAssigned = 0;

    // Step 1: Get all active projects
    const projects = await step.do('get-projects', async () => {
      const activeProjects = await getActiveProjects(this.env.DB);
      console.log(`Found ${activeProjects.length} active projects`);
      return activeProjects;
    });

    if (projects.length === 0) {
      console.log('No active projects found. Manager Workflow complete.');
      return { processed: 0, tasksCreated: 0, agentsAssigned: 0 };
    }

    // Step 2: For each project, assess current state
    for (const project of projects) {
      // Get project tasks for assessment
      const tasks = await step.do(`get-tasks-${project.id}`, async () => {
        return await getProjectTasks(this.env.DB, project.id);
      });

      // Assess project state using AI
      const assessment = await step.do(`assess-${project.id}`, async () => {
        return await this.assessProject(project, tasks);
      });

      console.log(`Assessment for ${project.name}:`, assessment.summary);

      // Step 3: Create new tasks if needed
      if (assessment.newTasksNeeded && assessment.suggestedTasks.length > 0) {
        const createdCount = await step.do(`create-tasks-${project.id}`, async () => {
          let count = 0;
          for (const suggestedTask of assessment.suggestedTasks) {
            try {
              await createTask(this.env.DB, project.id, suggestedTask);
              count++;
            } catch (error) {
              console.error(`Failed to create task: ${error}`);
            }
          }
          return count;
        });
        totalTasksCreated += createdCount;
        console.log(`Created ${createdCount} tasks for project ${project.name}`);
      }

      // Step 4: Assign idle agents to pending tasks
      const assignedCount = await step.do(`assign-${project.id}`, async () => {
        return await this.assignAgentsToTasks(project.id);
      });
      totalAgentsAssigned += assignedCount;
    }

    // Step 5: Trigger worker workflows for all assigned agents
    const triggerResults = await step.do('trigger-workers', async () => {
      return await this.triggerWorkerWorkflows();
    });

    console.log(`Triggered ${triggerResults.filter((r) => r.status === 'triggered').length} worker workflows`);

    // Log manager workflow completion
    await step.do('log-completion', async () => {
      await logWork(
        this.env.DB,
        'manager',
        null,
        'workflow_complete',
        { projectCount: projects.length },
        {
          tasksCreated: totalTasksCreated,
          agentsAssigned: totalAgentsAssigned,
          workersTriggered: triggerResults.length,
        },
        0, // Duration calculated at end
        0  // Tokens not tracked for this step
      );
    });

    console.log('Manager Workflow complete:', {
      processed: projects.length,
      tasksCreated: totalTasksCreated,
      agentsAssigned: totalAgentsAssigned,
    });

    return {
      processed: projects.length,
      tasksCreated: totalTasksCreated,
      agentsAssigned: totalAgentsAssigned,
    };
  }

  /**
   * Assess a project's state using the distributed compute network.
   * This dispatches an 'assessment' task to available compute devices (4090, etc.)
   * instead of using Workers AI.
   */
  private async assessProject(
    project: Project,
    tasks: Task[]
  ): Promise<ProjectAssessment> {
    const startTime = Date.now();

    // Calculate task statistics
    const completedTasks = tasks.filter((t) => t.status === 'completed').length;
    const pendingTasks = tasks.filter((t) => t.status === 'pending').length;
    const blockedTasks = tasks.filter((t) => t.status === 'blocked').length;
    const inProgressTasks = tasks.filter((t) => t.status === 'in_progress').length;

    // Prepare context for assessment
    const projectContext = {
      project: {
        id: project.id,
        name: project.name,
        slug: project.slug,
        config: project.config ? JSON.parse(project.config) : null,
      },
      taskSummary: {
        total: tasks.length,
        completed: completedTasks,
        pending: pendingTasks,
        blocked: blockedTasks,
        inProgress: inProgressTasks,
      },
      recentTasks: tasks.slice(0, 10).map((t) => ({
        title: t.title,
        status: t.status,
        priority: t.priority,
        requires_physical: t.requires_physical === 1,
      })),
    };

    try {
      // Check if any compute devices are available
      const devicesResult = await this.env.DB
        .prepare(`
          SELECT COUNT(*) as count FROM compute_devices
          WHERE status IN ('online', 'busy')
        `)
        .first<{ count: number }>();

      const hasComputeDevices = (devicesResult?.count || 0) > 0;

      if (!hasComputeDevices) {
        // No compute devices available - use simple heuristic assessment
        console.log(`No compute devices available for project ${project.id}, using heuristic assessment`);

        return {
          projectId: project.id,
          summary: `Project ${project.name}: ${completedTasks}/${tasks.length} tasks completed`,
          completedTasks,
          pendingTasks,
          blockedTasks,
          inProgressTasks,
          // Simple heuristic: need new tasks if queue is empty and not everything is done
          newTasksNeeded: pendingTasks === 0 && completedTasks < tasks.length,
          suggestedTasks: [],
          priorities: pendingTasks > 0 ? ['Complete pending tasks'] : [],
          bottlenecks: blockedTasks > 0 ? [`${blockedTasks} tasks blocked`] : [],
        };
      }

      // Create a compute task for assessment
      const computeTaskId = generateId('ctask');
      const now = new Date().toISOString();

      await this.env.DB
        .prepare(`
          INSERT INTO compute_tasks (
            id, type, input, config, status, priority, min_tier, created_at
          ) VALUES (?, 'assessment', ?, ?, 'pending', 7, 'standard', ?)
        `)
        .bind(
          computeTaskId,
          JSON.stringify({
            systemPrompt: PROJECT_ASSESSMENT_PROMPT,
            prompt: JSON.stringify(projectContext),
          }),
          JSON.stringify({ maxTokens: 1024, temperature: 0.3 }),
          now
        )
        .run();

      console.log(`Created assessment compute task ${computeTaskId} for project ${project.id}`);

      // Wait for the compute task to complete (with timeout)
      const timeout = 60000; // 60 seconds
      const pollInterval = 1000; // 1 second
      let elapsed = 0;

      while (elapsed < timeout) {
        const task = await this.env.DB
          .prepare('SELECT status, result, error FROM compute_tasks WHERE id = ?')
          .bind(computeTaskId)
          .first<{ status: string; result: string | null; error: string | null }>();

        if (!task) {
          throw new Error('Compute task not found');
        }

        if (task.status === 'completed' && task.result) {
          const result = JSON.parse(task.result);
          const responseText = result.output || '';

          // Parse the response
          let aiAssessment: {
            summary?: string;
            newTasksNeeded?: boolean;
            suggestedTasks?: SuggestedTask[];
            priorities?: string[];
            bottlenecks?: string[];
          };

          try {
            const jsonMatch = responseText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
              aiAssessment = JSON.parse(jsonMatch[0]);
            } else {
              throw new Error('No JSON found in response');
            }
          } catch (parseError) {
            console.error('Failed to parse compute assessment:', parseError);
            aiAssessment = {
              summary: 'Unable to parse assessment',
              newTasksNeeded: pendingTasks === 0 && completedTasks < tasks.length,
              suggestedTasks: [],
              priorities: [],
              bottlenecks: [],
            };
          }

          const durationMs = Date.now() - startTime;

          await logWork(
            this.env.DB,
            'manager',
            null,
            'project_assessment',
            projectContext,
            aiAssessment,
            durationMs,
            0
          );

          return {
            projectId: project.id,
            summary: aiAssessment.summary || 'No summary available',
            completedTasks,
            pendingTasks,
            blockedTasks,
            inProgressTasks,
            newTasksNeeded: aiAssessment.newTasksNeeded ?? false,
            suggestedTasks: aiAssessment.suggestedTasks || [],
            priorities: aiAssessment.priorities || [],
            bottlenecks: aiAssessment.bottlenecks || [],
          };
        }

        if (task.status === 'failed') {
          throw new Error(task.error || 'Compute task failed');
        }

        // Wait before next poll
        await new Promise(resolve => setTimeout(resolve, pollInterval));
        elapsed += pollInterval;
      }

      // Timeout - use fallback
      console.warn(`Assessment compute task timed out for project ${project.id}`);
      throw new Error('Assessment timeout');

    } catch (error) {
      console.error(`Assessment failed for project ${project.id}:`, error);

      // Return fallback assessment
      return {
        projectId: project.id,
        summary: `Assessment completed (fallback): ${completedTasks}/${tasks.length} tasks done`,
        completedTasks,
        pendingTasks,
        blockedTasks,
        inProgressTasks,
        newTasksNeeded: pendingTasks === 0 && completedTasks < tasks.length,
        suggestedTasks: [],
        priorities: [],
        bottlenecks: error instanceof Error ? [error.message] : [],
      };
    }
  }

  /**
   * Assign idle agents to pending tasks based on specializations
   */
  private async assignAgentsToTasks(projectId: string): Promise<number> {
    // Get idle agents for this project
    const agents = await getProjectAgents(this.env.DB, projectId);
    const idleAgents = agents.filter(
      (a) => a.status === 'idle' && !a.current_task_id
    );

    if (idleAgents.length === 0) {
      console.log(`No idle agents for project ${projectId}`);
      return 0;
    }

    // Get pending tasks
    const pendingTasks = await getPendingTasks(this.env.DB, projectId);

    if (pendingTasks.length === 0) {
      console.log(`No pending tasks for project ${projectId}`);
      return 0;
    }

    let assignedCount = 0;

    // Match agents to tasks
    for (const agent of idleAgents) {
      // Find a matching task for this agent
      const matchingTaskIndex = pendingTasks.findIndex((task) => {
        // For now, assign any pending task to any idle agent
        // In the future, we could use task metadata to match specializations
        return true;
      });

      if (matchingTaskIndex !== -1) {
        const task = pendingTasks[matchingTaskIndex];

        try {
          await assignAgentToTask(this.env.DB, agent.agent_id, task.id);
          assignedCount++;

          // Remove assigned task from pending list
          pendingTasks.splice(matchingTaskIndex, 1);

          console.log(`Assigned ${agent.agent_id} to task: ${task.title}`);
        } catch (error) {
          console.error(
            `Failed to assign ${agent.agent_id} to task ${task.id}:`,
            error
          );
        }
      }
    }

    return assignedCount;
  }

  /**
   * Trigger worker workflows for all agents with assigned tasks
   */
  private async triggerWorkerWorkflows(): Promise<WorkerTriggerResult[]> {
    const results: WorkerTriggerResult[] = [];

    try {
      // Get all agents that are working (have assigned tasks)
      const workingAgents = await this.env.DB
        .prepare(`
          SELECT * FROM agent_state
          WHERE status = 'working' AND current_task_id IS NOT NULL
        `)
        .all<AgentState>();

      if (!workingAgents.results || workingAgents.results.length === 0) {
        console.log('No working agents to trigger');
        return results;
      }

      for (const agent of workingAgents.results) {
        try {
          // Trigger the worker workflow
          const instance = await this.env.WORKER_WORKFLOW.create({
            params: {
              taskId: agent.current_task_id!,
              agentId: agent.agent_id,
            },
          });

          results.push({
            agentId: agent.agent_id,
            taskId: agent.current_task_id!,
            workflowInstanceId: instance.id,
            status: 'triggered',
          });

          console.log(
            `Triggered worker workflow for agent ${agent.agent_id}, task ${agent.current_task_id}`
          );
        } catch (error) {
          console.error(
            `Failed to trigger worker for agent ${agent.agent_id}:`,
            error
          );

          results.push({
            agentId: agent.agent_id,
            taskId: agent.current_task_id!,
            status: 'failed',
            error: String(error),
          });
        }
      }
    } catch (error) {
      console.error('Failed to query working agents:', error);
    }

    return results;
  }
}

// Export for use in index.ts
export default ManagerWorkflow;
