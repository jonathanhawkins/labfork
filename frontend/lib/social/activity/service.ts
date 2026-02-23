/**
 * Activity Generation Service
 *
 * Handles creating, storing, and querying activity records.
 * Supports aggregation of similar activities.
 */

import {
  findMany,
  insertOne,
  findPaginated,
  count,
} from "@/lib/db/json-store";
import {
  Activity,
  AggregatedActivity,
  ActivityType,
  ActivityActor,
  ActivityTarget,
  ActivityContext,
  ActivityFeedOptions,
  ActivityFeedResult,
  generateActivityId,
} from "./types";

/**
 * Collection name
 */
const COLLECTION = "activities";

/**
 * Create a new activity
 */
export async function createActivity(params: {
  type: ActivityType;
  actor: ActivityActor;
  target: ActivityTarget;
  context?: ActivityContext;
  labId?: string;
  isPublic?: boolean;
}): Promise<Activity> {
  const activity: Activity = {
    id: generateActivityId(),
    type: params.type,
    actor: params.actor,
    target: params.target,
    context: params.context,
    labId: params.labId,
    isPublic: params.isPublic ?? true,
    createdAt: new Date().toISOString(),
  };

  return insertOne(COLLECTION, activity);
}

/**
 * Log a result creation activity
 */
export async function logResultCreated(
  actor: ActivityActor,
  result: { id: string; title: string; type: string },
  labId: string
): Promise<Activity> {
  return createActivity({
    type: "result_created",
    actor,
    target: {
      type: "result",
      id: result.id,
      title: result.title,
    },
    context: {
      metadata: { resultType: result.type },
    },
    labId,
  });
}

/**
 * Log a result publish activity
 */
export async function logResultPublished(
  actor: ActivityActor,
  result: { id: string; title: string },
  labId: string
): Promise<Activity> {
  return createActivity({
    type: "result_published",
    actor,
    target: {
      type: "result",
      id: result.id,
      title: result.title,
    },
    labId,
  });
}

/**
 * Log a result like activity
 */
export async function logResultLiked(
  actor: ActivityActor,
  result: { id: string; title: string },
  labId?: string
): Promise<Activity> {
  return createActivity({
    type: "result_liked",
    actor,
    target: {
      type: "result",
      id: result.id,
      title: result.title,
    },
    labId,
  });
}

/**
 * Log a result comment activity
 */
export async function logResultCommented(
  actor: ActivityActor,
  result: { id: string; title: string },
  commentSnippet: string,
  labId?: string
): Promise<Activity> {
  return createActivity({
    type: "result_commented",
    actor,
    target: {
      type: "result",
      id: result.id,
      title: result.title,
    },
    context: {
      snippet: commentSnippet.substring(0, 100),
    },
    labId,
  });
}

/**
 * Log a lab star activity
 */
export async function logLabStarred(
  actor: ActivityActor,
  lab: { id: string; name: string; slug: string; ownerUsername: string }
): Promise<Activity> {
  return createActivity({
    type: "lab_starred",
    actor,
    target: {
      type: "lab",
      id: lab.id,
      title: lab.name,
      url: `/labs/${lab.ownerUsername}/${lab.slug}`,
    },
    labId: lab.id,
  });
}

/**
 * Log a lab fork activity
 */
export async function logLabForked(
  actor: ActivityActor,
  sourceLab: { id: string; name: string },
  forkedLab: { id: string; name: string; slug: string; ownerUsername: string }
): Promise<Activity> {
  return createActivity({
    type: "lab_forked",
    actor,
    target: {
      type: "lab",
      id: forkedLab.id,
      title: forkedLab.name,
      url: `/labs/${forkedLab.ownerUsername}/${forkedLab.slug}`,
    },
    context: {
      parent: {
        type: "lab",
        id: sourceLab.id,
        title: sourceLab.name,
      },
    },
    labId: forkedLab.id,
  });
}

/**
 * Log a task completion activity
 */
export async function logTaskCompleted(
  actor: ActivityActor,
  task: { id: string; title: string },
  labId: string
): Promise<Activity> {
  return createActivity({
    type: "task_completed",
    actor,
    target: {
      type: "task",
      id: task.id,
      title: task.title,
    },
    labId,
  });
}

/**
 * Log a user mention activity
 */
export async function logUserMentioned(
  actor: ActivityActor,
  mentionedUser: { id: string; username: string; displayName: string },
  context: { entityType: string; entityId: string; entityTitle: string }
): Promise<Activity> {
  return createActivity({
    type: "user_mentioned",
    actor,
    target: {
      type: "user",
      id: mentionedUser.id,
      title: mentionedUser.displayName,
    },
    context: {
      parent: {
        type: context.entityType as Activity["target"]["type"],
        id: context.entityId,
        title: context.entityTitle,
      },
    },
    isPublic: false, // Mentions are private
  });
}

/**
 * Aggregate similar activities
 */
function aggregateActivities(
  activities: Activity[],
  windowMs: number
): AggregatedActivity[] {
  const aggregated: AggregatedActivity[] = [];
  const processed = new Set<string>();

  for (const activity of activities) {
    if (processed.has(activity.id)) continue;

    // Find similar activities within the time window
    const windowStart = new Date(activity.createdAt).getTime() - windowMs;
    const similar = activities.filter(
      (a) =>
        !processed.has(a.id) &&
        a.type === activity.type &&
        a.target.id === activity.target.id &&
        new Date(a.createdAt).getTime() >= windowStart
    );

    // Mark all as processed
    similar.forEach((a) => processed.add(a.id));

    if (similar.length === 1) {
      aggregated.push({ ...activity });
    } else {
      // Get unique actors
      const actors = similar.map((a) => a.actor);
      const uniqueActors = actors.filter(
        (actor, index, self) =>
          index === self.findIndex((a) => a.id === actor.id)
      );

      aggregated.push({
        ...activity,
        otherActors: uniqueActors.slice(1),
        totalCount: similar.length,
      });
    }
  }

  return aggregated;
}

/**
 * Get activity feed
 */
export async function getActivityFeed(
  options: ActivityFeedOptions = {}
): Promise<ActivityFeedResult> {
  const {
    types,
    labId,
    userId,
    targetId,
    publicOnly = true,
    aggregate = true,
    aggregateWindow = 60, // 60 minutes default
    page = 1,
    limit = 20,
  } = options;

  // Build filter
  const filter = (activity: Activity): boolean => {
    if (types && types.length > 0 && !types.includes(activity.type)) {
      return false;
    }
    if (labId && activity.labId !== labId) return false;
    if (userId && activity.actor.id !== userId) return false;
    if (targetId && activity.target.id !== targetId) return false;
    if (publicOnly && !activity.isPublic) return false;
    return true;
  };

  // Sort by newest first
  const sort = (a: Activity, b: Activity): number =>
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();

  // Get all matching activities first (for aggregation)
  let activities: Activity[];
  let total: number;

  if (aggregate) {
    // Get more activities for aggregation
    const allActivities = await findMany<Activity>(COLLECTION, filter);
    allActivities.sort(sort);

    // Aggregate
    const aggregatedActivities = aggregateActivities(
      allActivities,
      aggregateWindow * 60 * 1000
    );

    total = aggregatedActivities.length;

    // Paginate aggregated results
    const offset = (page - 1) * limit;
    activities = aggregatedActivities.slice(offset, offset + limit);
  } else {
    // Regular pagination without aggregation
    const result = await findPaginated<Activity>(COLLECTION, {
      filter,
      sort,
      page,
      limit,
    });

    activities = result.items;
    total = result.total;
  }

  const totalPages = Math.ceil(total / limit);
  const latestTimestamp = activities.length > 0 ? activities[0].createdAt : undefined;

  return {
    activities: activities as AggregatedActivity[],
    total,
    page,
    totalPages,
    hasMore: page < totalPages,
    latestTimestamp,
  };
}

/**
 * Get activities for a specific lab
 */
export async function getLabActivityFeed(
  labId: string,
  options: Omit<ActivityFeedOptions, "labId"> = {}
): Promise<ActivityFeedResult> {
  return getActivityFeed({ ...options, labId });
}

/**
 * Get activities by a specific user
 */
export async function getUserActivityFeed(
  userId: string,
  options: Omit<ActivityFeedOptions, "userId"> = {}
): Promise<ActivityFeedResult> {
  return getActivityFeed({ ...options, userId });
}

/**
 * Get activities for a user's home feed
 * Includes activities from labs they've starred and users they follow
 */
export async function getHomeFeed(
  userId: string,
  starredLabIds: string[],
  followedUserIds: string[],
  options: Omit<ActivityFeedOptions, "labId" | "userId"> = {}
): Promise<ActivityFeedResult> {
  const {
    types,
    publicOnly = true,
    aggregate = true,
    aggregateWindow = 60,
    page = 1,
    limit = 20,
  } = options;

  // Custom filter for home feed
  const filter = (activity: Activity): boolean => {
    if (types && types.length > 0 && !types.includes(activity.type)) {
      return false;
    }
    if (publicOnly && !activity.isPublic) return false;

    // Include activities from:
    // 1. Starred labs
    // 2. Followed users
    // 3. User's own activities
    const isFromStarredLab =
      activity.labId && starredLabIds.includes(activity.labId);
    const isFromFollowedUser = followedUserIds.includes(activity.actor.id);
    const isOwnActivity = activity.actor.id === userId;

    return isFromStarredLab || isFromFollowedUser || isOwnActivity;
  };

  // Sort by newest first
  const sort = (a: Activity, b: Activity): number =>
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();

  const allActivities = await findMany<Activity>(COLLECTION, filter);
  allActivities.sort(sort);

  let activities: Activity[];
  let total: number;

  if (aggregate) {
    const aggregatedActivities = aggregateActivities(
      allActivities,
      aggregateWindow * 60 * 1000
    );

    total = aggregatedActivities.length;
    const offset = (page - 1) * limit;
    activities = aggregatedActivities.slice(offset, offset + limit);
  } else {
    total = allActivities.length;
    const offset = (page - 1) * limit;
    activities = allActivities.slice(offset, offset + limit);
  }

  const totalPages = Math.ceil(total / limit);

  return {
    activities: activities as AggregatedActivity[],
    total,
    page,
    totalPages,
    hasMore: page < totalPages,
    latestTimestamp: activities.length > 0 ? activities[0].createdAt : undefined,
  };
}

/**
 * Get new activities since a timestamp
 */
export async function getNewActivities(
  since: string,
  options: ActivityFeedOptions = {}
): Promise<Activity[]> {
  const sinceTime = new Date(since).getTime();

  const activities = await findMany<Activity>(COLLECTION, (activity) => {
    if (new Date(activity.createdAt).getTime() <= sinceTime) return false;

    if (options.labId && activity.labId !== options.labId) return false;
    if (options.userId && activity.actor.id !== options.userId) return false;
    if (options.publicOnly !== false && !activity.isPublic) return false;

    return true;
  });

  return activities.sort(
    (a, b) =>
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );
}

/**
 * Get activity count for a lab
 */
export async function getLabActivityCount(labId: string): Promise<number> {
  return count<Activity>(COLLECTION, (a) => a.labId === labId);
}

/**
 * Get activity count for a user
 */
export async function getUserActivityCount(userId: string): Promise<number> {
  return count<Activity>(COLLECTION, (a) => a.actor.id === userId);
}

/**
 * Log a simulation started activity
 */
export async function logSimulationStarted(
  simulation: { id: string; type: string; labSlug: string },
  device?: { id: string; name: string }
): Promise<Activity> {
  return createActivity({
    type: "simulation_started",
    actor: device
      ? { id: device.id, username: device.id, displayName: device.name }
      : { id: "system", username: "system", displayName: "Compute Network" },
    target: {
      type: "simulation",
      id: simulation.id,
      title: `${simulation.type} simulation`,
      url: `/labs/${simulation.labSlug}?tab=simulate`,
    },
    labId: `lab_${simulation.labSlug}`,
    isPublic: true,
  });
}

/**
 * Log a simulation completed activity
 */
export async function logSimulationCompleted(
  simulation: {
    id: string;
    type: string;
    labSlug: string;
    results?: {
      daily_yield_liters?: number;
      efficiency_percent?: number;
    };
  },
  device?: { id: string; name: string }
): Promise<Activity> {
  return createActivity({
    type: "simulation_completed",
    actor: device
      ? { id: device.id, username: device.id, displayName: device.name }
      : { id: "system", username: "system", displayName: "Compute Network" },
    target: {
      type: "simulation",
      id: simulation.id,
      title: `${simulation.type} simulation`,
      url: `/labs/${simulation.labSlug}?tab=simulate`,
    },
    context: simulation.results
      ? {
          metrics: [
            {
              name: "Daily Yield",
              value: simulation.results.daily_yield_liters || 0,
              unit: "L/day",
            },
            {
              name: "Efficiency",
              value: simulation.results.efficiency_percent || 0,
              unit: "%",
            },
          ],
        }
      : undefined,
    labId: `lab_${simulation.labSlug}`,
    isPublic: true,
  });
}
