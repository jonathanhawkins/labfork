"use client";

/**
 * FeedItem Component
 *
 * Displays a single activity in the feed with appropriate styling.
 */

import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Plus,
  Send,
  Heart,
  MessageCircle,
  Star,
  GitFork,
  RefreshCw,
  CheckCircle2,
  ListPlus,
  UserPlus,
  AtSign,
  FileText,
  Trophy,
  FlaskConical,
} from "lucide-react";
import type { AggregatedActivity, ActivityType } from "@/lib/social/activity/types";
import { formatActivityTime, ACTIVITY_TYPE_LABELS } from "@/lib/social/activity/types";

export interface FeedItemProps {
  /** Activity data */
  activity: AggregatedActivity;
  /** Compact mode */
  compact?: boolean;
  /** Show timestamp */
  showTime?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Get icon component for activity type
 */
function getActivityIcon(type: ActivityType) {
  switch (type) {
    case "result_created":
      return Plus;
    case "result_published":
      return Send;
    case "result_liked":
      return Heart;
    case "result_commented":
      return MessageCircle;
    case "lab_created":
      return FlaskConical;
    case "lab_starred":
      return Star;
    case "lab_forked":
      return GitFork;
    case "lab_updated":
      return RefreshCw;
    case "task_completed":
      return CheckCircle2;
    case "task_created":
      return ListPlus;
    case "user_followed":
      return UserPlus;
    case "user_mentioned":
      return AtSign;
    case "paper_added":
      return FileText;
    case "experiment_completed":
      return FlaskConical;
    case "milestone_reached":
      return Trophy;
    default:
      return Plus;
  }
}

/**
 * Get icon color for activity type
 */
function getActivityColor(type: ActivityType): string {
  switch (type) {
    case "result_created":
    case "result_published":
      return "text-green-400 bg-green-500/10";
    case "result_liked":
      return "text-red-400 bg-red-500/10";
    case "result_commented":
      return "text-blue-400 bg-blue-500/10";
    case "lab_starred":
      return "text-yellow-400 bg-yellow-500/10";
    case "lab_forked":
      return "text-purple-400 bg-purple-500/10";
    case "task_completed":
      return "text-green-400 bg-green-500/10";
    case "user_followed":
    case "user_mentioned":
      return "text-blue-400 bg-blue-500/10";
    case "milestone_reached":
      return "text-yellow-400 bg-yellow-500/10";
    default:
      return "text-foreground-muted bg-foreground-muted/10";
  }
}

export function FeedItem({
  activity,
  compact = false,
  showTime = true,
  className,
}: FeedItemProps) {
  const Icon = getActivityIcon(activity.type);
  const iconColor = getActivityColor(activity.type);

  // Build actor text
  const actorText =
    activity.totalCount && activity.totalCount > 1
      ? `${activity.actor.displayName} and ${activity.totalCount - 1} other${
          activity.totalCount - 1 === 1 ? "" : "s"
        }`
      : activity.actor.displayName;

  // Get action text
  const actionText = ACTIVITY_TYPE_LABELS[activity.type];

  return (
    <div
      className={cn(
        "group flex gap-3 transition-colors",
        compact ? "py-2" : "py-3",
        "hover:bg-foreground-muted/5 rounded-lg",
        className
      )}
    >
      {/* Icon */}
      <div
        className={cn(
          "flex-shrink-0 rounded-lg flex items-center justify-center",
          iconColor,
          compact ? "w-7 h-7" : "w-9 h-9"
        )}
      >
        <Icon className={cn(compact ? "w-3.5 h-3.5" : "w-4 h-4")} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        {/* Main text */}
        <p className={cn("text-foreground", compact ? "text-xs" : "text-sm")}>
          {/* Actor */}
          <Link
            href={`/labs/${activity.actor.username}`}
            className="font-medium hover:underline"
          >
            {actorText}
          </Link>

          {/* Action */}
          <span className="text-foreground-muted"> {actionText} </span>

          {/* Target */}
          {activity.target.url ? (
            <Link
              href={activity.target.url}
              className="font-medium hover:underline"
            >
              {activity.target.title}
            </Link>
          ) : (
            <span className="font-medium">{activity.target.title}</span>
          )}

          {/* Context (e.g., "in Lab Name") */}
          {activity.context?.parent && (
            <span className="text-foreground-muted">
              {" "}
              in{" "}
              <span className="font-medium">
                {activity.context.parent.title}
              </span>
            </span>
          )}
        </p>

        {/* Snippet preview */}
        {activity.context?.snippet && !compact && (
          <p className="mt-1 text-xs text-foreground-muted line-clamp-2 pl-0.5 border-l-2 border-foreground-muted/20">
            {activity.context.snippet}
          </p>
        )}

        {/* Metrics if present */}
        {activity.context?.metrics && activity.context.metrics.length > 0 && !compact && (
          <div className="mt-1.5 flex gap-3">
            {activity.context.metrics.slice(0, 3).map((metric, index) => (
              <span
                key={metric.name + index}
                className="text-xs px-2 py-0.5 rounded bg-foreground-muted/10 text-foreground-muted"
              >
                {metric.name}: {metric.value}
                {metric.unit ? ` ${metric.unit}` : ""}
              </span>
            ))}
          </div>
        )}

        {/* Timestamp */}
        {showTime && (
          <p
            className={cn(
              "text-foreground-subtle",
              compact ? "mt-0.5 text-[10px]" : "mt-1 text-xs"
            )}
          >
            {formatActivityTime(activity.createdAt)}
          </p>
        )}
      </div>

      {/* Aggregated avatars */}
      {activity.otherActors && activity.otherActors.length > 0 && !compact && (
        <div className="flex -space-x-2">
          {activity.otherActors.slice(0, 3).map((actor) => (
            <div
              key={actor.id}
              className="w-6 h-6 rounded-full bg-foreground-muted/20 border-2 border-background flex items-center justify-center text-[10px] font-medium text-foreground-muted overflow-hidden"
              title={actor.displayName}
            >
              {actor.avatar ? (
                <img
                  src={actor.avatar}
                  alt={actor.displayName}
                  className="w-full h-full object-cover"
                />
              ) : (
                actor.displayName.charAt(0).toUpperCase()
              )}
            </div>
          ))}
          {activity.otherActors.length > 3 && (
            <div className="w-6 h-6 rounded-full bg-foreground-muted/20 border-2 border-background flex items-center justify-center text-[10px] font-medium text-foreground-muted">
              +{activity.otherActors.length - 3}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default FeedItem;
