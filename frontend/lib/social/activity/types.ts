/**
 * Activity Types and Schema
 *
 * Defines types for activity feeds in the social layer.
 * Tracks user actions across labs, results, and comments.
 */

/**
 * Activity type categories
 */
export type ActivityType =
  // Result activities
  | "result_created"
  | "result_published"
  | "result_liked"
  | "result_commented"
  // Lab activities
  | "lab_created"
  | "lab_starred"
  | "lab_forked"
  | "lab_updated"
  // Task activities
  | "task_completed"
  | "task_created"
  // User activities
  | "user_followed"
  | "user_mentioned"
  // Research activities
  | "paper_added"
  | "experiment_completed"
  | "milestone_reached"
  // Comment activities
  | "comment_added"
  // Agent activities
  | "agent_spawned"
  | "agent_completed"
  // Simulation activities (distributed compute)
  | "simulation_started"
  | "simulation_completed";

/**
 * Entity type that an activity relates to
 */
export type ActivityEntityType =
  | "result"
  | "lab"
  | "task"
  | "user"
  | "paper"
  | "comment"
  | "agent"
  | "simulation";

/**
 * Actor information (who performed the action)
 */
export interface ActivityActor {
  /** User ID */
  id: string;
  /** Username */
  username: string;
  /** Display name */
  displayName: string;
  /** Avatar URL */
  avatar?: string;
}

/**
 * Target entity information
 */
export interface ActivityTarget {
  /** Entity type */
  type: ActivityEntityType;
  /** Entity ID */
  id: string;
  /** Entity title/name for display */
  title: string;
  /** URL to the entity */
  url?: string;
  /** Preview image */
  image?: string;
}

/**
 * Additional context for the activity
 */
export interface ActivityContext {
  /** Parent entity (e.g., lab for a result) */
  parent?: {
    type: ActivityEntityType;
    id: string;
    title: string;
  };
  /** Additional metadata */
  metadata?: Record<string, unknown>;
  /** Snippet/preview of content */
  snippet?: string;
  /** Metrics if relevant */
  metrics?: {
    name: string;
    value: number;
    unit?: string;
  }[];
}

/**
 * Activity record
 */
export interface Activity {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Unique activity ID */
  id: string;
  /** Activity type */
  type: ActivityType;
  /** Actor who performed the action */
  actor: ActivityActor;
  /** Target entity */
  target: ActivityTarget;
  /** Additional context */
  context?: ActivityContext;
  /** Is this a public activity */
  isPublic: boolean;
  /** Lab ID (if lab-scoped) */
  labId?: string;
  /** When the activity occurred */
  createdAt: string;
}

/**
 * Activity with aggregation for similar activities
 */
export interface AggregatedActivity extends Activity {
  /** Other actors involved (for "X and 5 others liked...") */
  otherActors?: ActivityActor[];
  /** Total count of similar activities */
  totalCount?: number;
}

/**
 * Activity feed options
 */
export interface ActivityFeedOptions {
  /** Filter by activity types */
  types?: ActivityType[];
  /** Filter by lab ID */
  labId?: string;
  /** Filter by user ID (activities they performed) */
  userId?: string;
  /** Filter by target entity */
  targetId?: string;
  /** Only public activities */
  publicOnly?: boolean;
  /** Enable aggregation of similar activities */
  aggregate?: boolean;
  /** Aggregation window in minutes */
  aggregateWindow?: number;
  /** Page number */
  page?: number;
  /** Items per page */
  limit?: number;
}

/**
 * Paginated activity feed result
 */
export interface ActivityFeedResult {
  /** Activities */
  activities: AggregatedActivity[];
  /** Total count */
  total: number;
  /** Current page */
  page: number;
  /** Total pages */
  totalPages: number;
  /** Has more pages */
  hasMore: boolean;
  /** Latest activity timestamp (for polling) */
  latestTimestamp?: string;
}

/**
 * Activity type labels for display
 */
export const ACTIVITY_TYPE_LABELS: Record<ActivityType, string> = {
  result_created: "created a result",
  result_published: "published a result",
  result_liked: "liked a result",
  result_commented: "commented on",
  lab_created: "created a lab",
  lab_starred: "starred a lab",
  lab_forked: "forked a lab",
  lab_updated: "updated a lab",
  task_completed: "completed a task",
  task_created: "created a task",
  user_followed: "followed",
  user_mentioned: "mentioned",
  paper_added: "added a paper",
  experiment_completed: "completed an experiment",
  milestone_reached: "reached a milestone",
  comment_added: "commented on",
  agent_spawned: "spawned an agent",
  agent_completed: "agent completed task",
  simulation_started: "started a simulation",
  simulation_completed: "completed simulation",
};

/**
 * Activity type icons (Lucide icon names)
 */
export const ACTIVITY_TYPE_ICONS: Record<ActivityType, string> = {
  result_created: "Plus",
  result_published: "Send",
  result_liked: "Heart",
  result_commented: "MessageCircle",
  lab_created: "Flask",
  lab_starred: "Star",
  lab_forked: "GitFork",
  lab_updated: "RefreshCw",
  task_completed: "CheckCircle2",
  task_created: "ListPlus",
  user_followed: "UserPlus",
  user_mentioned: "AtSign",
  paper_added: "FileText",
  experiment_completed: "Beaker",
  milestone_reached: "Trophy",
  comment_added: "MessageCircle",
  agent_spawned: "Bot",
  agent_completed: "CheckCircle",
  simulation_started: "Play",
  simulation_completed: "Droplets",
};

/**
 * Generate a unique activity ID
 */
export function generateActivityId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `act_${timestamp}${random}`;
}

/**
 * Get activity description
 */
export function getActivityDescription(activity: Activity): string {
  const actorName = activity.actor.displayName;
  const action = ACTIVITY_TYPE_LABELS[activity.type];
  const targetTitle = activity.target.title;

  return `${actorName} ${action} "${targetTitle}"`;
}

/**
 * Get aggregated activity description
 */
export function getAggregatedDescription(activity: AggregatedActivity): string {
  const actorName = activity.actor.displayName;
  const action = ACTIVITY_TYPE_LABELS[activity.type];
  const targetTitle = activity.target.title;
  const otherCount = activity.totalCount ? activity.totalCount - 1 : 0;

  if (otherCount > 0) {
    return `${actorName} and ${otherCount} other${otherCount === 1 ? "" : "s"} ${action} "${targetTitle}"`;
  }

  return `${actorName} ${action} "${targetTitle}"`;
}

/**
 * Format relative time for activity
 */
export function formatActivityTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
  return date.toLocaleDateString();
}

/**
 * Group activities by date
 */
export function groupActivitiesByDate(
  activities: Activity[]
): Map<string, Activity[]> {
  const groups = new Map<string, Activity[]>();

  for (const activity of activities) {
    const date = new Date(activity.createdAt);
    const key = date.toISOString().split("T")[0]; // YYYY-MM-DD

    if (!groups.has(key)) {
      groups.set(key, []);
    }
    groups.get(key)!.push(activity);
  }

  return groups;
}

/**
 * Get date group label
 */
export function getDateGroupLabel(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const today = now.toISOString().split("T")[0];
  const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000)
    .toISOString()
    .split("T")[0];

  if (dateStr === today) return "Today";
  if (dateStr === yesterday) return "Yesterday";

  return date.toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
  });
}

/**
 * Check if activity should trigger notification
 */
export function shouldNotify(activity: Activity, userId: string): boolean {
  // Don't notify user of their own actions
  if (activity.actor.id === userId) return false;

  // Activities that trigger notifications
  const notifyTypes: ActivityType[] = [
    "result_liked",
    "result_commented",
    "lab_starred",
    "lab_forked",
    "user_followed",
    "user_mentioned",
  ];

  return notifyTypes.includes(activity.type);
}

/**
 * Type guard for Activity
 */
export function isActivity(obj: unknown): obj is Activity {
  if (!obj || typeof obj !== "object") return false;
  const activity = obj as Record<string, unknown>;
  return (
    typeof activity.id === "string" &&
    typeof activity.type === "string" &&
    activity.actor !== undefined &&
    activity.target !== undefined
  );
}
