/**
 * Notification Types and Schema
 *
 * Defines types for user notifications.
 */

/**
 * Notification type
 */
export type NotificationType =
  | "result_liked"
  | "result_commented"
  | "lab_starred"
  | "lab_forked"
  | "user_followed"
  | "user_mentioned"
  | "suggestion_status"
  | "task_assigned"
  | "system";

/**
 * Notification status
 */
export type NotificationStatus = "unread" | "read" | "archived";

/**
 * Notification actor
 */
export interface NotificationActor {
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
 * Notification target
 */
export interface NotificationTarget {
  /** Entity type */
  type: "result" | "lab" | "comment" | "suggestion" | "task" | "user";
  /** Entity ID */
  id: string;
  /** Title/name */
  title: string;
  /** URL to navigate to */
  url?: string;
}

/**
 * Notification record
 */
export interface Notification {
  /** Index signature for JSON storage compatibility */
  [key: string]: unknown;
  /** Unique notification ID */
  id: string;
  /** User ID this notification is for */
  userId: string;
  /** Notification type */
  type: NotificationType;
  /** Actor who triggered the notification (optional for system) */
  actor?: NotificationActor;
  /** Target entity */
  target: NotificationTarget;
  /** Message content */
  message: string;
  /** Status */
  status: NotificationStatus;
  /** When created */
  createdAt: string;
  /** When read */
  readAt?: string;
}

/**
 * Notification list options
 */
export interface NotificationListOptions {
  /** User ID */
  userId: string;
  /** Filter by status */
  status?: NotificationStatus;
  /** Filter by type */
  types?: NotificationType[];
  /** Page */
  page?: number;
  /** Limit */
  limit?: number;
}

/**
 * Paginated notification list result
 */
export interface NotificationListResult {
  /** Notifications */
  notifications: Notification[];
  /** Total count */
  total: number;
  /** Unread count */
  unreadCount: number;
  /** Current page */
  page: number;
  /** Total pages */
  totalPages: number;
  /** Has more */
  hasMore: boolean;
}

/**
 * Notification type labels
 */
export const NOTIFICATION_TYPE_LABELS: Record<NotificationType, string> = {
  result_liked: "liked your result",
  result_commented: "commented on your result",
  lab_starred: "starred your lab",
  lab_forked: "forked your lab",
  user_followed: "started following you",
  user_mentioned: "mentioned you",
  suggestion_status: "updated suggestion status",
  task_assigned: "assigned you a task",
  system: "System notification",
};

/**
 * Notification type icons (Lucide icon names)
 */
export const NOTIFICATION_TYPE_ICONS: Record<NotificationType, string> = {
  result_liked: "Heart",
  result_commented: "MessageCircle",
  lab_starred: "Star",
  lab_forked: "GitFork",
  user_followed: "UserPlus",
  user_mentioned: "AtSign",
  suggestion_status: "Info",
  task_assigned: "ListChecks",
  system: "Bell",
};

/**
 * Generate notification ID
 */
export function generateNotificationId(): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 8);
  return `ntf_${timestamp}${random}`;
}

/**
 * Get notification message
 */
export function getNotificationMessage(notification: Notification): string {
  if (!notification.actor) {
    return notification.message;
  }

  const action = NOTIFICATION_TYPE_LABELS[notification.type];
  return `${notification.actor.displayName} ${action}`;
}

/**
 * Type guard for Notification
 */
export function isNotification(obj: unknown): obj is Notification {
  if (!obj || typeof obj !== "object") return false;
  const notif = obj as Record<string, unknown>;
  return (
    typeof notif.id === "string" &&
    typeof notif.userId === "string" &&
    typeof notif.type === "string" &&
    notif.target !== undefined
  );
}
