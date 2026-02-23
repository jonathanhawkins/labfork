/**
 * Rules — heuristic task creation.
 *
 * Rules look at observations and return tasks to create.
 * No LLM. No API calls. Pure functions.
 * Fast, free, deterministic.
 */

import type { TasksObservation, WorkersObservation, ExternalObservation } from './observers';

export interface SuggestedTask {
  action: string;
  description: string;
  context?: Record<string, unknown>;
  constraints?: Record<string, unknown>;
  priority: number;
  source: 'cron';
  /** Required worker capability: 'code' | 'monitor' | 'advisory'. NULL = any worker. */
  required_capability?: string;
}

export interface RuleContext {
  tasks: TasksObservation;
  workers: WorkersObservation;
  external: ExternalObservation[];
  /** How many tasks have already been suggested this cycle (set by evaluate). */
  suggestedThisCycle: number;
  /** Goal action slugs that have already been completed (don't repeat). */
  completedGoals?: Set<string>;
}

/**
 * Run all rules and return suggested tasks.
 * Rules are evaluated in priority order. Max 3 tasks per cycle.
 */
export function evaluate(ctx: RuleContext, maxTasks: number = 3): SuggestedTask[] {
  const suggestions: SuggestedTask[] = [];
  ctx.suggestedThisCycle = 0;

  for (const rule of RULES) {
    if (suggestions.length >= maxTasks) break;
    const result = rule(ctx);
    if (result) {
      const items = Array.isArray(result) ? result : [result];
      suggestions.push(...items);
      ctx.suggestedThisCycle = suggestions.length;
    }
  }

  return suggestions.slice(0, maxTasks);
}

// ============================================================================
// Individual rules — each returns a task suggestion or null
// ============================================================================

type Rule = (ctx: RuleContext) => SuggestedTask | SuggestedTask[] | null;

/**
 * If CI is failing (from external observation), create a fix-ci task.
 */
const failingCI: Rule = (ctx) => {
  const git = ctx.external.find((o) => o.type === 'git');
  if (!git) return null;
  if (!git.data.failingCI) return null;

  return {
    action: 'fix-ci',
    description: 'CI is failing on main. Investigate the latest test failures, identify the root cause, and fix it. Run tests locally to confirm the fix before submitting.',
    context: git.data as Record<string, unknown>,
    constraints: { mustPassTests: true, createPR: true },
    priority: 9,
    source: 'cron',
    required_capability: 'code',
  };
};

/**
 * If there are new untriaged issues (from external observation), create a triage task.
 */
const newIssues: Rule = (ctx) => {
  const issues = ctx.external.find((o) => o.type === 'issues');
  if (!issues) return null;

  const newOnes = issues.data.newIssues as Array<{ id: string; title: string }> | undefined;
  if (!newOnes || newOnes.length === 0) return null;

  return {
    action: 'triage-issues',
    description: `There are ${newOnes.length} new issue(s) that need triage. For each: read the issue, add appropriate labels, assess severity, and add a comment with initial analysis or next steps.`,
    context: { issues: newOnes.slice(0, 10) },
    constraints: { maxIssues: 10 },
    priority: 7,
    source: 'cron',
  };
};

/**
 * If there are stale PRs (from external observation), create a review task.
 */
const stalePRs: Rule = (ctx) => {
  const git = ctx.external.find((o) => o.type === 'git');
  if (!git) return null;

  const prs = git.data.stalePRs as Array<{ number: number; title: string; staleDays: number }> | undefined;
  if (!prs || prs.length === 0) return null;

  return {
    action: 'review-prs',
    description: `There are ${prs.length} PR(s) that have been open without review. Review each PR: check for correctness, test coverage, and code quality. Leave constructive comments.`,
    context: { prs: prs.slice(0, 5) },
    priority: 6,
    source: 'cron',
  };
};

/**
 * If goals exist and there's capacity, create a task toward the next objective.
 * Goals are posted via: POST /observe { type: "goals", data: { objectives: ["..."] } }
 * The first objective without an active task gets picked.
 */
const pursueGoals: Rule = (ctx) => {
  const goals = ctx.external.find((o) => o.type === 'goals');
  if (!goals) return null;

  const objectives = goals.data.objectives as string[] | undefined;
  if (!objectives || objectives.length === 0) return null;

  // Don't pile on if there's already work in the queue
  if (ctx.tasks.pending + ctx.tasks.assigned >= 2) return null;
  if (ctx.workers.idle === 0) return null;

  // Find the first objective that hasn't been completed yet
  const done = ctx.completedGoals || new Set<string>();
  let objective: string | null = null;
  let actionSlug = '';

  for (const obj of objectives) {
    const slug = 'goal-' + obj
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 40);

    if (!done.has(slug)) {
      objective = obj;
      actionSlug = slug;
      break;
    }
  }

  // All objectives completed
  if (!objective) return null;

  // All goal tasks default to 'code' — they go to codex for real work.
  // Only monitor/review goals get 'monitor' capability.
  const monitorPatterns = /^goal-(verify|check|review)/;
  const capability = monitorPatterns.test(actionSlug) ? 'monitor' : 'code';

  return {
    action: actionSlug,
    description: `Work toward this objective: ${objective}\n\nBreak it into concrete steps, implement what you can, and report what was accomplished and what remains.`,
    context: { objective, allObjectives: objectives },
    priority: 5,
    source: 'cron',
    required_capability: capability,
  };
};

/**
 * If the task queue is completely empty and workers are idle, create a health check task.
 */
const emptyQueue: Rule = (ctx) => {
  if (ctx.tasks.pending > 0 || ctx.tasks.assigned > 0) return null;
  if (ctx.workers.idle === 0) return null;
  // Don't create busywork if we already suggested real work this cycle
  if (ctx.suggestedThisCycle > 0) return null;

  return {
    action: 'check-health',
    description: 'No tasks in the queue. Run a project health check: verify tests pass, check for new issues or PRs, scan for TODO comments, and report anything that needs attention.',
    priority: 3,
    source: 'cron',
  };
};

/**
 * If tasks have failed recently, create an investigation task.
 */
const recentFailures: Rule = (ctx) => {
  if (ctx.tasks.failed24h < 3) return null;

  return {
    action: 'investigate-failures',
    description: `${ctx.tasks.failed24h} tasks have failed in the last 24 hours. Investigate the failure patterns: are they the same issue? Is it a systemic problem? Report findings and suggest fixes.`,
    priority: 7,
    source: 'cron',
  };
};

/**
 * If there are external test results showing failures, create a fix task.
 */
const failingTests: Rule = (ctx) => {
  const tests = ctx.external.find((o) => o.type === 'tests');
  if (!tests) return null;

  const failures = tests.data.failures as Array<{ test: string; error: string }> | undefined;
  if (!failures || failures.length === 0) return null;

  return {
    action: 'fix-tests',
    description: `${failures.length} test(s) are failing. Fix each failing test or the underlying code. Run the full test suite to confirm no regressions.`,
    context: { failures: failures.slice(0, 5) },
    constraints: { mustPassTests: true, createPR: true },
    priority: 8,
    source: 'cron',
    required_capability: 'code',
  };
};

// Rules in priority order
const RULES: Rule[] = [
  failingCI,
  failingTests,
  newIssues,
  recentFailures,
  stalePRs,
  pursueGoals,
  emptyQueue,
];
