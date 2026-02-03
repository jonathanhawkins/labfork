"use client";

/**
 * NotificationBell Component
 *
 * Notification bell icon with badge and dropdown.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import { cn } from "@/lib/utils";
import {
  Bell,
  Check,
  X,
  Heart,
  MessageCircle,
  Star,
  GitFork,
  UserPlus,
  AtSign,
  Info,
  ListChecks,
  Loader2,
} from "lucide-react";
import type {
  Notification,
  NotificationType,
} from "@/lib/social/notifications/types";
import { NOTIFICATION_TYPE_ICONS } from "@/lib/social/notifications/types";

export interface NotificationBellProps {
  /** Current user ID */
  userId: string;
  /** Polling interval in ms (default 30000) */
  pollingInterval?: number;
  /** Custom class name */
  className?: string;
}

/**
 * Get icon component for notification type
 */
function getNotificationIcon(type: NotificationType) {
  switch (type) {
    case "result_liked":
      return Heart;
    case "result_commented":
      return MessageCircle;
    case "lab_starred":
      return Star;
    case "lab_forked":
      return GitFork;
    case "user_followed":
      return UserPlus;
    case "user_mentioned":
      return AtSign;
    case "suggestion_status":
      return Info;
    case "task_assigned":
      return ListChecks;
    case "system":
    default:
      return Bell;
  }
}

/**
 * Format relative time
 */
function formatTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMinutes < 1) return "now";
  if (diffMinutes < 60) return `${diffMinutes}m`;
  if (diffHours < 24) return `${diffHours}h`;
  if (diffDays < 7) return `${diffDays}d`;
  return date.toLocaleDateString();
}

export function NotificationBell({
  userId,
  pollingInterval = 30000,
  className,
}: NotificationBellProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch notifications
  const fetchNotifications = useCallback(async () => {
    try {
      const response = await fetch(
        `/api/notifications?userId=${userId}&limit=10`
      );
      if (!response.ok) return;

      const data = await response.json();
      setNotifications(data.notifications || []);
      setUnreadCount(data.unreadCount || 0);
    } catch (error) {
      console.error("Error fetching notifications:", error);
    }
  }, [userId]);

  // Mark notification as read
  const markAsRead = useCallback(async (id: string) => {
    try {
      await fetch(`/api/notifications/${id}`, { method: "PATCH" });
      setNotifications((prev) =>
        prev.map((n) =>
          n.id === id ? { ...n, status: "read" as const } : n
        )
      );
      setUnreadCount((prev) => Math.max(0, prev - 1));
    } catch (error) {
      console.error("Error marking as read:", error);
    }
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(async () => {
    setIsLoading(true);
    try {
      await fetch("/api/notifications", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, action: "mark_all_read" }),
      });
      setNotifications((prev) =>
        prev.map((n) => ({ ...n, status: "read" as const }))
      );
      setUnreadCount(0);
    } catch (error) {
      console.error("Error marking all as read:", error);
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  // Initial fetch
  useEffect(() => {
    fetchNotifications();
  }, [fetchNotifications]);

  // Polling
  useEffect(() => {
    const interval = setInterval(fetchNotifications, pollingInterval);
    return () => clearInterval(interval);
  }, [fetchNotifications, pollingInterval]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className={cn("relative", className)} ref={dropdownRef}>
      {/* Bell button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "relative p-2 rounded-lg transition-colors",
          isOpen
            ? "bg-foreground-muted/10 text-foreground"
            : "text-foreground-muted hover:text-foreground hover:bg-foreground-muted/10"
        )}
      >
        <Bell className="w-5 h-5" />

        {/* Unread badge */}
        {unreadCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] px-1 flex items-center justify-center rounded-full bg-red-500 text-white text-[10px] font-medium">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 max-h-[400px] overflow-hidden bg-background rounded-lg border border-border shadow-xl z-50">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-medium text-foreground">
              Notifications
            </h3>
            {unreadCount > 0 && (
              <button
                onClick={markAllAsRead}
                disabled={isLoading}
                className="text-xs text-foreground-muted hover:text-foreground transition-colors disabled:opacity-50"
              >
                {isLoading ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  "Mark all read"
                )}
              </button>
            )}
          </div>

          {/* Notifications list */}
          <div className="max-h-[300px] overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="py-8 text-center text-foreground-muted text-sm">
                No notifications yet
              </div>
            ) : (
              notifications.map((notification) => {
                const Icon = getNotificationIcon(notification.type);
                const isUnread = notification.status === "unread";

                return (
                  <div
                    key={notification.id}
                    className={cn(
                      "flex gap-3 px-4 py-3 hover:bg-foreground-muted/5 transition-colors cursor-pointer",
                      isUnread && "bg-blue-500/5"
                    )}
                    onClick={() => {
                      if (isUnread) markAsRead(notification.id);
                      if (notification.target.url) {
                        window.location.href = notification.target.url;
                      }
                      setIsOpen(false);
                    }}
                  >
                    {/* Actor avatar or icon */}
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-foreground-muted/10 flex items-center justify-center">
                      {notification.actor?.avatar ? (
                        <Image
                          src={notification.actor.avatar}
                          alt=""
                          width={32}
                          height={32}
                          className="w-full h-full rounded-full object-cover"
                          unoptimized
                        />
                      ) : (
                        <Icon className="w-4 h-4 text-foreground-muted" />
                      )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <p
                        className={cn(
                          "text-sm line-clamp-2",
                          isUnread
                            ? "text-foreground"
                            : "text-foreground-muted"
                        )}
                      >
                        {notification.message}
                      </p>
                      <p className="text-xs text-foreground-subtle mt-0.5">
                        {formatTime(notification.createdAt)}
                      </p>
                    </div>

                    {/* Unread indicator */}
                    {isUnread && (
                      <div className="flex-shrink-0 w-2 h-2 rounded-full bg-blue-500 mt-2" />
                    )}
                  </div>
                );
              })
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-border">
            <Link
              href="/notifications"
              className="block text-center text-xs text-foreground-muted hover:text-foreground transition-colors"
              onClick={() => setIsOpen(false)}
            >
              View all notifications
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}

export default NotificationBell;
