/**
 * Physical Barrier Detection Module
 *
 * Detects when tasks require physical action (ordering parts, assembly, testing, etc.)
 * and creates appropriate human tasks in the database.
 */

/**
 * Represents a bill of materials item
 */
export interface BOMItem {
  name: string;
  quantity: number;
  url?: string;
  price?: number;
}

/**
 * Represents a physical barrier that blocks automated task completion
 */
export interface PhysicalBarrier {
  type: 'order_parts' | 'assemble' | 'test' | 'measure' | 'photograph';
  description: string;
  items?: BOMItem[];
  instructions: string[];
  estimated_cost?: number;
  estimated_time?: string;
  blocking_tasks: string[];
}

/**
 * Keywords that indicate a task requires physical action
 */
export const PHYSICAL_KEYWORDS = [
  'order',
  'buy',
  'purchase',
  'assemble',
  'solder',
  'test with hardware',
  'measure',
  'photograph',
  'build prototype',
  'physically',
  'hardware',
  'connect',
  'wire',
  'install',
  'mount',
  'calibrate',
  'inspect',
  'document',
  'prototype',
  'fabricate',
];

/**
 * Detects if a task description indicates physical action is needed
 *
 * @param taskDescription - The task description to analyze
 * @returns PhysicalBarrier if physical action is needed, null otherwise
 */
export function detectPhysicalBarrier(
  taskDescription: string
): PhysicalBarrier | null {
  if (!taskDescription || typeof taskDescription !== 'string') {
    return null;
  }

  const lowerDescription = taskDescription.toLowerCase();

  // Check if any physical keyword is found in the description
  const foundKeyword = PHYSICAL_KEYWORDS.find((keyword) =>
    lowerDescription.includes(keyword.toLowerCase())
  );

  if (!foundKeyword) {
    return null;
  }

  // Determine barrier type based on keywords
  let barrierType: PhysicalBarrier['type'] = 'order_parts';

  if (
    lowerDescription.includes('order') ||
    lowerDescription.includes('buy') ||
    lowerDescription.includes('purchase')
  ) {
    barrierType = 'order_parts';
  } else if (
    lowerDescription.includes('assemble') ||
    lowerDescription.includes('solder') ||
    lowerDescription.includes('wire') ||
    lowerDescription.includes('connect') ||
    lowerDescription.includes('install') ||
    lowerDescription.includes('mount')
  ) {
    barrierType = 'assemble';
  } else if (
    lowerDescription.includes('test with hardware') ||
    lowerDescription.includes('test') ||
    lowerDescription.includes('calibrate')
  ) {
    barrierType = 'test';
  } else if (
    lowerDescription.includes('measure') ||
    lowerDescription.includes('inspect')
  ) {
    barrierType = 'measure';
  } else if (
    lowerDescription.includes('photograph') ||
    lowerDescription.includes('document')
  ) {
    barrierType = 'photograph';
  }

  // Create the physical barrier
  const barrier: PhysicalBarrier = {
    type: barrierType,
    description: `Physical action required: ${taskDescription.substring(0, 100)}...`,
    instructions: [
      `Complete the following task that requires physical interaction:`,
      taskDescription,
    ],
    blocking_tasks: [],
  };

  return barrier;
}

/**
 * Creates a human task in the database for a physical barrier
 *
 * @param barrier - The physical barrier that was detected
 * @param projectId - The project ID this task belongs to
 * @param db - The D1 database instance
 * @returns Promise resolving to the created task ID
 */
export async function createHumanTask(
  barrier: PhysicalBarrier,
  projectId: string,
  db: D1Database
): Promise<string> {
  // Generate a unique task ID
  const taskId = `task_${Date.now()}_${Math.random().toString(36).substring(7)}`;

  try {
    // Insert the task into the database with requires_physical = 1 (true)
    const stmt = db
      .prepare(
        `
      INSERT INTO tasks (
        id,
        project_id,
        title,
        description,
        status,
        priority,
        requires_physical,
        blocked_by
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `
      )
      .bind(
        taskId,
        projectId,
        `[PHYSICAL] ${barrier.type}: ${barrier.description}`,
        JSON.stringify({
          barrier_type: barrier.type,
          items: barrier.items,
          instructions: barrier.instructions,
          estimated_cost: barrier.estimated_cost,
          estimated_time: barrier.estimated_time,
          blocking_tasks: barrier.blocking_tasks,
        }),
        'pending',
        8, // High priority for blocking tasks
        1, // requires_physical = true
        barrier.blocking_tasks.length > 0 ? barrier.blocking_tasks[0] : null
      );

    await stmt.run();

    return taskId;
  } catch (error) {
    console.error('Failed to create human task:', error);
    throw new Error(`Failed to create human task: ${String(error)}`);
  }
}

/**
 * Notifies humans that a physical barrier has been detected
 * This is a placeholder that currently logs to console.
 *
 * TODO: Integrate with actual notification systems:
 * - Email notifications via SendGrid or similar
 * - Slack notifications via Slack API
 * - Discord notifications via Discord webhooks
 * - Push notifications to mobile app
 *
 * @param barrier - The physical barrier that was detected
 * @returns Promise that resolves when notification is sent
 */
export async function notifyHumans(barrier: PhysicalBarrier): Promise<void> {
  console.log(`
========================================
PHYSICAL BARRIER DETECTED
========================================
Type: ${barrier.type}
Description: ${barrier.description}
Instructions:
${barrier.instructions.map((inst, i) => `  ${i + 1}. ${inst}`).join('\n')}
${barrier.items ? `\nRequired Items:\n${barrier.items.map((item) => `  - ${item.quantity}x ${item.name}${item.price ? ` ($${item.price})` : ''}${item.url ? ` (${item.url})` : ''}`).join('\n')}` : ''}
${barrier.estimated_cost ? `\nEstimated Cost: $${barrier.estimated_cost}` : ''}
${barrier.estimated_time ? `\nEstimated Time: ${barrier.estimated_time}` : ''}
========================================

TODO: Implement notification channels:
- [ ] Email notification via SendGrid
- [ ] Slack notification via Slack API webhook
- [ ] Discord notification via Discord webhook
- [ ] Push notification to mobile app
- [ ] SMS via Twilio
========================================
  `);

  // Placeholder for actual notification implementation
  // Email example (when implemented):
  // await sendEmail({
  //   to: ADMIN_EMAIL,
  //   subject: `Physical action required: ${barrier.type}`,
  //   body: formatBarrierForEmail(barrier),
  // });

  // Slack example (when implemented):
  // await fetch(process.env.SLACK_WEBHOOK_URL, {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({
  //     text: `Physical action required: ${barrier.type}`,
  //     blocks: formatBarrierForSlack(barrier),
  //   }),
  // });

  // Discord example (when implemented):
  // await fetch(process.env.DISCORD_WEBHOOK_URL, {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({
  //     content: `Physical action required: ${barrier.type}`,
  //     embeds: [formatBarrierForDiscord(barrier)],
  //   }),
  // });
}

/**
 * Complete workflow: detect barrier, create task, and notify humans
 *
 * @param taskDescription - The task description to analyze
 * @param projectId - The project ID for the task
 * @param db - The D1 database instance
 * @returns Promise resolving to the task ID if a barrier was detected, null otherwise
 */
export async function handlePhysicalBarrier(
  taskDescription: string,
  projectId: string,
  db: D1Database
): Promise<string | null> {
  // Detect if there's a physical barrier
  const barrier = detectPhysicalBarrier(taskDescription);

  if (!barrier) {
    return null;
  }

  // Create a human task in the database
  const taskId = await createHumanTask(barrier, projectId, db);

  // Notify humans about the barrier
  await notifyHumans(barrier);

  return taskId;
}
