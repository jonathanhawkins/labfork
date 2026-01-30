/**
 * TrendingAlert
 *
 * Alert notification component for displaying trending alerts
 * with significance levels, actions, and dismissal.
 */

"use client";

import React from "react";
import {
  TrendingAlert as AlertType,
  AlertType as AlertTypeEnum,
  SignificanceLevel,
} from "@/lib/meta/community/types";

interface TrendingAlertProps {
  alert: AlertType;
  onMarkRead?: (alertId: string) => void;
  onDismiss?: (alertId: string) => void;
  onAction?: (alertId: string) => void;
  compact?: boolean;
}

const alertTypeIcons: Record<AlertTypeEnum, { icon: string; color: string }> = {
  synergy: { icon: "link", color: "text-blue-500" },
  breakthrough: { icon: "star", color: "text-yellow-500" },
  trend: { icon: "trending", color: "text-green-500" },
  "gap-filled": { icon: "check", color: "text-emerald-500" },
  collaboration: { icon: "users", color: "text-purple-500" },
  evolution: { icon: "dna", color: "text-indigo-500" },
  opportunity: { icon: "lightbulb", color: "text-orange-500" },
};

const significanceStyles: Record<SignificanceLevel, string> = {
  critical: "border-l-red-500 bg-red-50",
  high: "border-l-orange-500 bg-orange-50",
  medium: "border-l-blue-500 bg-blue-50",
  low: "border-l-gray-300 bg-gray-50",
};

const significanceBadges: Record<SignificanceLevel, string> = {
  critical: "bg-red-100 text-red-700",
  high: "bg-orange-100 text-orange-700",
  medium: "bg-blue-100 text-blue-700",
  low: "bg-gray-100 text-gray-600",
};

export function TrendingAlert({
  alert,
  onMarkRead,
  onDismiss,
  onAction,
  compact = false,
}: TrendingAlertProps) {
  const typeInfo = alertTypeIcons[alert.type];
  const isUnread = !alert.read;
  const timeAgo = formatTimeAgo(alert.createdAt);

  if (compact) {
    return (
      <div
        className={`flex items-center gap-3 p-3 rounded-lg transition-colors ${
          isUnread ? "bg-blue-50" : "bg-gray-50"
        } ${alert.dismissed ? "opacity-50" : ""}`}
        onClick={() => onMarkRead?.(alert.id)}
        role="button"
        tabIndex={0}
      >
        <AlertIcon type={alert.type} />
        <div className="flex-1 min-w-0">
          <p className={`text-sm truncate ${isUnread ? "font-medium" : ""}`}>
            {alert.title}
          </p>
          <p className="text-xs text-gray-500">{timeAgo}</p>
        </div>
        {isUnread && (
          <span className="w-2 h-2 bg-blue-500 rounded-full flex-shrink-0" />
        )}
      </div>
    );
  }

  return (
    <div
      className={`border-l-4 rounded-lg shadow-sm transition-all ${
        significanceStyles[alert.significance]
      } ${alert.dismissed ? "opacity-50" : ""}`}
    >
      <div className="p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-2">
            <AlertIcon type={alert.type} />
            <span className="text-sm font-medium capitalize">{alert.type}</span>
            <span
              className={`px-2 py-0.5 text-xs rounded ${
                significanceBadges[alert.significance]
              }`}
            >
              {alert.significance}
            </span>
            {isUnread && (
              <span className="w-2 h-2 bg-blue-500 rounded-full" title="Unread" />
            )}
          </div>
          <span className="text-xs text-gray-500">{timeAgo}</span>
        </div>

        {/* Content */}
        <h3 className={`font-semibold mb-1 ${isUnread ? "" : "text-gray-700"}`}>
          {alert.title}
        </h3>
        <p className="text-gray-600 text-sm mb-3">{alert.description}</p>

        {/* Metadata */}
        {(alert.metadata.sourceLabName || alert.metadata.domain) && (
          <div className="flex flex-wrap gap-2 mb-3 text-xs">
            {alert.metadata.sourceLabName && (
              <span className="px-2 py-1 bg-white rounded border">
                Lab: {alert.metadata.sourceLabName}
              </span>
            )}
            {alert.metadata.domain && (
              <span className="px-2 py-1 bg-white rounded border">
                Domain: {alert.metadata.domain}
              </span>
            )}
            {alert.metadata.impactScore && (
              <span className="px-2 py-1 bg-white rounded border">
                Impact: {alert.metadata.impactScore.toFixed(0)}
              </span>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2">
          {onAction && alert.metadata.actionUrl && (
            <button
              onClick={() => onAction(alert.id)}
              className="px-3 py-1.5 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700 transition-colors"
            >
              View Details
            </button>
          )}
          {onMarkRead && isUnread && (
            <button
              onClick={() => onMarkRead(alert.id)}
              className="px-3 py-1.5 text-gray-600 text-sm hover:bg-white rounded transition-colors"
            >
              Mark Read
            </button>
          )}
          {onDismiss && !alert.dismissed && (
            <button
              onClick={() => onDismiss(alert.id)}
              className="px-3 py-1.5 text-gray-400 text-sm hover:text-gray-600 transition-colors ml-auto"
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function AlertIcon({ type }: { type: AlertTypeEnum }) {
  const { color } = alertTypeIcons[type];

  const icons: Record<AlertTypeEnum, React.ReactNode> = {
    synergy: (
      <svg className={`w-5 h-5 ${color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
      </svg>
    ),
    breakthrough: (
      <svg className={`w-5 h-5 ${color}`} fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
      </svg>
    ),
    trend: (
      <svg className={`w-5 h-5 ${color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    "gap-filled": (
      <svg className={`w-5 h-5 ${color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    collaboration: (
      <svg className={`w-5 h-5 ${color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
      </svg>
    ),
    evolution: (
      <svg className={`w-5 h-5 ${color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    ),
    opportunity: (
      <svg className={`w-5 h-5 ${color}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  };

  return <>{icons[type]}</>;
}

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

// Alert List Component
interface TrendingAlertListProps {
  alerts: AlertType[];
  onMarkRead?: (alertId: string) => void;
  onDismiss?: (alertId: string) => void;
  onAction?: (alertId: string) => void;
  compact?: boolean;
  maxItems?: number;
  showEmpty?: boolean;
}

export function TrendingAlertList({
  alerts,
  onMarkRead,
  onDismiss,
  onAction,
  compact = false,
  maxItems,
  showEmpty = true,
}: TrendingAlertListProps) {
  const displayedAlerts = maxItems ? alerts.slice(0, maxItems) : alerts;

  if (alerts.length === 0 && showEmpty) {
    return (
      <div className="text-center py-8 text-gray-500">
        <svg
          className="w-12 h-12 mx-auto mb-3 text-gray-300"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>
        <p>No alerts to show</p>
      </div>
    );
  }

  return (
    <div className={`space-y-${compact ? "2" : "3"}`}>
      {displayedAlerts.map((alert) => (
        <TrendingAlert
          key={alert.id}
          alert={alert}
          onMarkRead={onMarkRead}
          onDismiss={onDismiss}
          onAction={onAction}
          compact={compact}
        />
      ))}
      {maxItems && alerts.length > maxItems && (
        <p className="text-center text-sm text-gray-500 pt-2">
          +{alerts.length - maxItems} more alerts
        </p>
      )}
    </div>
  );
}

export default TrendingAlert;
