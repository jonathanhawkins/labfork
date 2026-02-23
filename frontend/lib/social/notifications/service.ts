/**
 * Notification Service
 *
 * Handles creating and querying notifications.
 */

import {
  findMany,
  insertOne,
  updateOne,
  updateMany,
  count,
  findPaginated,
} from "@/lib/db/json-store";
import {
  Notification,
  NotificationType,
  NotificationStatus,
  NotificationActor,
  NotificationTarget,
  NotificationListOptions,
  NotificationListResult,
  generateNotificationId,
  NOTIFICATION_TYPE_LABELS,
} from "./types";

/**
 * Collection name
 */
const COLLECTION = "notifications";

/**
 * Create a notification
 */
export async function createNotification(params: {
  userId: string;
  type: NotificationType;
  actor?: NotificationActor;
  target: NotificationTarget;
  message?: string;
}): Promise<Notification> {
  const { userId, type, actor, target, message } = params;

  const notification: Notification = {
    id: generateNotificationId(),
    userId,
    type,
    actor,
    target,
    message: message || (actor
      ? `${actor.displayName} ${NOTIFICATION_TYPE_LABELS[type]}`
      : NOTIFICATION_TYPE_LABELS[type]),
    status: "unread",
    createdAt: new Date().toISOString(),
  };

  return insertOne(COLLECTION, notification);
}

/**
 * Create notification for result like
 */
export async function notifyResultLiked(
  resultOwnerId: string,
  actor: NotificationActor,
  result: { id: string; title: string }
): Promise<Notification | null> {
  // Don't notify self
  if (resultOwnerId === actor.id) return null;

  return createNotification({
    userId: resultOwnerId,
    type: "result_liked",
    actor,
    target: {
      type: "result",
      id: result.id,
      title: result.title,
      url: `/results/${result.id}`,
    },
  });
}

/**
 * Create notification for result comment
 */
export async function notifyResultCommented(
  resultOwnerId: string,
  actor: NotificationActor,
  result: { id: string; title: string },
  commentSnippet: string
): Promise<Notification | null> {
  if (resultOwnerId === actor.id) return null;

  return createNotification({
    userId: resultOwnerId,
    type: "result_commented",
    actor,
    target: {
      type: "result",
      id: result.id,
      title: result.title,
      url: `/results/${result.id}`,
    },
    message: `${actor.displayName} commented: "${commentSnippet.substring(0, 50)}..."`,
  });
}

/**
 * Create notification for lab star
 */
export async function notifyLabStarred(
  labOwnerId: string,
  actor: NotificationActor,
  lab: { id: string; name: string; slug: string; ownerUsername: string }
): Promise<Notification | null> {
  if (labOwnerId === actor.id) return null;

  return createNotification({
    userId: labOwnerId,
    type: "lab_starred",
    actor,
    target: {
      type: "lab",
      id: lab.id,
      title: lab.name,
      url: `/labs/${lab.ownerUsername}/${lab.slug}`,
    },
  });
}

/**
 * Create notification for lab fork
 */
export async function notifyLabForked(
  labOwnerId: string,
  actor: NotificationActor,
  sourceLab: { id: string; name: string },
  forkedLab: { slug: string; ownerUsername: string }
): Promise<Notification | null> {
  if (labOwnerId === actor.id) return null;

  return createNotification({
    userId: labOwnerId,
    type: "lab_forked",
    actor,
    target: {
      type: "lab",
      id: sourceLab.id,
      title: sourceLab.name,
      url: `/labs/${forkedLab.ownerUsername}/${forkedLab.slug}`,
    },
  });
}

/**
 * Create notification for user mention
 */
export async function notifyUserMentioned(
  mentionedUserId: string,
  actor: NotificationActor,
  context: { entityType: string; entityId: string; entityTitle: string; url: string }
): Promise<Notification | null> {
  if (mentionedUserId === actor.id) return null;

  return createNotification({
    userId: mentionedUserId,
    type: "user_mentioned",
    actor,
    target: {
      type: context.entityType as Notification["target"]["type"],
      id: context.entityId,
      title: context.entityTitle,
      url: context.url,
    },
    message: `${actor.displayName} mentioned you in "${context.entityTitle}"`,
  });
}

/**
 * List notifications for a user
 */
export async function listNotifications(
  options: NotificationListOptions
): Promise<NotificationListResult> {
  const {
    userId,
    status,
    types,
    page = 1,
    limit = 20,
  } = options;

  // Build filter
  const filter = (notification: Notification): boolean => {
    if (notification.userId !== userId) return false;
    if (status && notification.status !== status) return false;
    if (types && types.length > 0 && !types.includes(notification.type)) {
      return false;
    }
    return true;
  };

  // Sort by newest first
  const sort = (a: Notification, b: Notification): number =>
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();

  const { items, total, totalPages, hasMore } =
    await findPaginated<Notification>(COLLECTION, {
      filter,
      sort,
      page,
      limit,
    });

  // Get unread count
  const unreadCount = await count<Notification>(
    COLLECTION,
    (n) => n.userId === userId && n.status === "unread"
  );

  return {
    notifications: items,
    total,
    unreadCount,
    page,
    totalPages,
    hasMore,
  };
}

/**
 * Get unread count for a user
 */
export async function getUnreadCount(userId: string): Promise<number> {
  return count<Notification>(
    COLLECTION,
    (n) => n.userId === userId && n.status === "unread"
  );
}

/**
 * Mark a notification as read
 */
export async function markAsRead(id: string): Promise<Notification | null> {
  return updateOne<Notification>(
    COLLECTION,
    (n) => n.id === id,
    {
      status: "read",
      readAt: new Date().toISOString(),
    }
  );
}

/**
 * Mark all notifications as read for a user
 */
export async function markAllAsRead(userId: string): Promise<number> {
  const now = new Date().toISOString();

  return updateMany<Notification>(
    COLLECTION,
    (n) => n.userId === userId && n.status === "unread",
    {
      status: "read",
      readAt: now,
    }
  );
}

/**
 * Archive a notification
 */
export async function archiveNotification(id: string): Promise<Notification | null> {
  return updateOne<Notification>(
    COLLECTION,
    (n) => n.id === id,
    { status: "archived" }
  );
}

/**
 * Get recent notifications for a user
 */
export async function getRecentNotifications(
  userId: string,
  limit = 10
): Promise<Notification[]> {
  const notifications = await findMany<Notification>(
    COLLECTION,
    (n) => n.userId === userId && n.status !== "archived"
  );

  return notifications
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    )
    .slice(0, limit);
}
