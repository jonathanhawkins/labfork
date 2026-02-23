/**
 * Trending Alerts System
 *
 * Real-time detection of exciting combinations with significance
 * classification, multiple alert types, and notification preferences.
 */

import {
  TrendingAlert,
  AlertType,
  SignificanceLevel,
  AlertMetadata,
  AlertPreferences,
  AlertSubscription,
  PaginatedResult,
} from "./types";

// ============================================================================
// Alerts System Interface
// ============================================================================

export interface AlertsSystem {
  alerts: Map<string, TrendingAlert>;
  subscriptions: Map<string, AlertSubscription>;
  alertQueue: TrendingAlert[];
  lastProcessed: string;
}

export interface AlertConfig {
  maxAlertsPerHour: number;
  alertRetentionDays: number;
  defaultSignificanceThreshold: SignificanceLevel;
  enableEmailNotifications: boolean;
  batchDelayMs: number;
}

const DEFAULT_CONFIG: AlertConfig = {
  maxAlertsPerHour: 50,
  alertRetentionDays: 30,
  defaultSignificanceThreshold: "medium",
  enableEmailNotifications: true,
  batchDelayMs: 5000,
};

// ============================================================================
// Factory Functions
// ============================================================================

export function createAlertsSystem(): AlertsSystem {
  return {
    alerts: new Map(),
    subscriptions: new Map(),
    alertQueue: [],
    lastProcessed: new Date().toISOString(),
  };
}

export function createDefaultPreferences(userId: string): AlertPreferences {
  return {
    userId,
    enabled: true,
    types: ["synergy", "breakthrough", "trend", "gap-filled"],
    minSignificance: "medium",
    emailEnabled: false,
    emailDigest: "daily",
    mutedLabs: [],
    mutedDomains: [],
  };
}

// ============================================================================
// Alert Detection
// ============================================================================

export interface DetectionContext {
  type: AlertType;
  sourceLabId?: string;
  sourceLabName?: string;
  techniqueIds?: string[];
  domain?: string;
  metrics?: Record<string, number>;
}

export function detectAlert(
  system: AlertsSystem,
  context: DetectionContext,
  title: string,
  description: string,
  relatedIds: string[] = []
): TrendingAlert {
  const significance = classifySignificance(context);
  const alert = createAlert(
    context.type,
    significance,
    title,
    description,
    relatedIds,
    {
      sourceLabId: context.sourceLabId,
      sourceLabName: context.sourceLabName,
      techniqueIds: context.techniqueIds,
      domain: context.domain,
      impactScore: calculateImpactScore(context),
    }
  );

  system.alerts.set(alert.id, alert);
  system.alertQueue.push(alert);

  return alert;
}

function classifySignificance(context: DetectionContext): SignificanceLevel {
  const metrics = context.metrics || {};

  // Critical: Major breakthroughs, significant improvements
  if (context.type === "breakthrough" && (metrics.improvement || 0) > 50) {
    return "critical";
  }

  // High: Notable discoveries, strong synergies
  if (context.type === "synergy" && (metrics.synergyScore || 0) > 0.8) {
    return "high";
  }

  if (context.type === "evolution" && (metrics.fitnessImprovement || 0) > 30) {
    return "high";
  }

  // Medium: Regular discoveries, moderate trends
  if (context.type === "trend" && (metrics.growthRate || 0) > 0.2) {
    return "medium";
  }

  if (context.type === "gap-filled") {
    return "medium";
  }

  // Low: Minor updates, small changes
  return "low";
}

function calculateImpactScore(context: DetectionContext): number {
  const metrics = context.metrics || {};
  let score = 0;

  switch (context.type) {
    case "breakthrough":
      score = 80 + (metrics.improvement || 0) * 0.5;
      break;
    case "synergy":
      score = 60 + (metrics.synergyScore || 0) * 40;
      break;
    case "evolution":
      score = 50 + (metrics.fitnessImprovement || 0);
      break;
    case "trend":
      score = 40 + (metrics.growthRate || 0) * 100;
      break;
    case "gap-filled":
      score = 30 + (metrics.difficulty || 0) * 20;
      break;
    case "collaboration":
      score = 20 + (metrics.participantCount || 0) * 5;
      break;
    case "opportunity":
      score = 10 + (metrics.bountyAmount || 0) * 0.01;
      break;
    default:
      score = 10;
  }

  return Math.min(100, Math.max(0, score));
}

function createAlert(
  type: AlertType,
  significance: SignificanceLevel,
  title: string,
  description: string,
  relatedIds: string[],
  metadata: AlertMetadata
): TrendingAlert {
  const id = `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const now = new Date();

  // Set expiration based on significance
  const expirationDays = {
    critical: 7,
    high: 5,
    medium: 3,
    low: 1,
  };

  const expiresAt = new Date(now);
  expiresAt.setDate(expiresAt.getDate() + expirationDays[significance]);

  return {
    id,
    type,
    significance,
    title,
    description,
    relatedIds,
    metadata,
    createdAt: now.toISOString(),
    expiresAt: expiresAt.toISOString(),
    read: false,
    dismissed: false,
  };
}

// ============================================================================
// Alert Management
// ============================================================================

export function getAlert(
  system: AlertsSystem,
  alertId: string
): TrendingAlert | null {
  return system.alerts.get(alertId) || null;
}

export function markAlertRead(
  system: AlertsSystem,
  alertId: string
): TrendingAlert | null {
  const alert = system.alerts.get(alertId);
  if (!alert) return null;

  const updated: TrendingAlert = { ...alert, read: true };
  system.alerts.set(alertId, updated);
  return updated;
}

export function dismissAlert(
  system: AlertsSystem,
  alertId: string
): TrendingAlert | null {
  const alert = system.alerts.get(alertId);
  if (!alert) return null;

  const updated: TrendingAlert = { ...alert, dismissed: true };
  system.alerts.set(alertId, updated);
  return updated;
}

export function getRecentAlerts(
  system: AlertsSystem,
  options: {
    limit?: number;
    types?: AlertType[];
    minSignificance?: SignificanceLevel;
    unreadOnly?: boolean;
    excludeDismissed?: boolean;
  } = {}
): TrendingAlert[] {
  const {
    limit = 20,
    types,
    minSignificance,
    unreadOnly = false,
    excludeDismissed = true,
  } = options;

  const significanceOrder: Record<SignificanceLevel, number> = {
    critical: 0,
    high: 1,
    medium: 2,
    low: 3,
  };

  const minSigLevel = minSignificance ? significanceOrder[minSignificance] : 3;

  return Array.from(system.alerts.values())
    .filter((alert) => {
      if (excludeDismissed && alert.dismissed) return false;
      if (unreadOnly && alert.read) return false;
      if (types && !types.includes(alert.type)) return false;
      if (significanceOrder[alert.significance] > minSigLevel) return false;
      if (alert.expiresAt && new Date(alert.expiresAt) < new Date()) return false;
      return true;
    })
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    )
    .slice(0, limit);
}

export function getAlertsPaginated(
  system: AlertsSystem,
  page: number = 0,
  pageSize: number = 20,
  options: {
    types?: AlertType[];
    minSignificance?: SignificanceLevel;
  } = {}
): PaginatedResult<TrendingAlert> {
  const filtered = getRecentAlerts(system, {
    limit: Infinity,
    ...options,
  });

  const start = page * pageSize;
  const items = filtered.slice(start, start + pageSize);

  return {
    items,
    total: filtered.length,
    page,
    pageSize,
    hasMore: start + pageSize < filtered.length,
  };
}

export function cleanupExpiredAlerts(system: AlertsSystem): number {
  const now = new Date();
  let count = 0;

  for (const [id, alert] of Array.from(system.alerts.entries())) {
    if (alert.expiresAt && new Date(alert.expiresAt) < now) {
      system.alerts.delete(id);
      count++;
    }
  }

  return count;
}

// ============================================================================
// Subscription Management
// ============================================================================

export function subscribe(
  system: AlertsSystem,
  userId: string,
  preferences?: Partial<AlertPreferences>
): AlertSubscription {
  const existing = system.subscriptions.get(userId);
  const now = new Date().toISOString();

  const subscription: AlertSubscription = {
    id: existing?.id || `sub-${userId}-${Date.now()}`,
    userId,
    preferences: {
      ...createDefaultPreferences(userId),
      ...existing?.preferences,
      ...preferences,
    },
    createdAt: existing?.createdAt || now,
    updatedAt: now,
  };

  system.subscriptions.set(userId, subscription);
  return subscription;
}

export function unsubscribe(system: AlertsSystem, userId: string): boolean {
  return system.subscriptions.delete(userId);
}

export function getSubscription(
  system: AlertsSystem,
  userId: string
): AlertSubscription | null {
  return system.subscriptions.get(userId) || null;
}

export function updatePreferences(
  system: AlertsSystem,
  userId: string,
  updates: Partial<AlertPreferences>
): AlertSubscription | null {
  const existing = system.subscriptions.get(userId);
  if (!existing) return null;

  const updated: AlertSubscription = {
    ...existing,
    preferences: {
      ...existing.preferences,
      ...updates,
    },
    updatedAt: new Date().toISOString(),
  };

  system.subscriptions.set(userId, updated);
  return updated;
}

export function muteSource(
  system: AlertsSystem,
  userId: string,
  type: "lab" | "domain",
  sourceId: string
): AlertSubscription | null {
  const subscription = system.subscriptions.get(userId);
  if (!subscription) return null;

  const key = type === "lab" ? "mutedLabs" : "mutedDomains";
  const currentMuted = subscription.preferences[key];

  if (currentMuted.includes(sourceId)) {
    return subscription;
  }

  return updatePreferences(system, userId, {
    [key]: [...currentMuted, sourceId],
  });
}

export function unmuteSource(
  system: AlertsSystem,
  userId: string,
  type: "lab" | "domain",
  sourceId: string
): AlertSubscription | null {
  const subscription = system.subscriptions.get(userId);
  if (!subscription) return null;

  const key = type === "lab" ? "mutedLabs" : "mutedDomains";
  const currentMuted = subscription.preferences[key];

  return updatePreferences(system, userId, {
    [key]: currentMuted.filter((id) => id !== sourceId),
  });
}

// ============================================================================
// Alert Delivery
// ============================================================================

export function shouldDeliverAlert(
  alert: TrendingAlert,
  subscription: AlertSubscription
): boolean {
  const prefs = subscription.preferences;

  // Check if alerts are enabled
  if (!prefs.enabled) return false;

  // Check alert type
  if (!prefs.types.includes(alert.type)) return false;

  // Check significance
  const significanceOrder: Record<SignificanceLevel, number> = {
    critical: 0,
    high: 1,
    medium: 2,
    low: 3,
  };
  if (significanceOrder[alert.significance] > significanceOrder[prefs.minSignificance]) {
    return false;
  }

  // Check muted sources
  if (
    alert.metadata.sourceLabId &&
    prefs.mutedLabs.includes(alert.metadata.sourceLabId)
  ) {
    return false;
  }

  if (
    alert.metadata.domain &&
    prefs.mutedDomains.includes(alert.metadata.domain)
  ) {
    return false;
  }

  // Check quiet hours
  if (prefs.quietHoursStart && prefs.quietHoursEnd) {
    const now = new Date();
    const currentHour = now.getHours();
    const startHour = parseInt(prefs.quietHoursStart.split(":")[0], 10);
    const endHour = parseInt(prefs.quietHoursEnd.split(":")[0], 10);

    if (startHour <= endHour) {
      if (currentHour >= startHour && currentHour < endHour) return false;
    } else {
      if (currentHour >= startHour || currentHour < endHour) return false;
    }
  }

  return true;
}

export function getAlertsForUser(
  system: AlertsSystem,
  userId: string,
  limit: number = 20
): TrendingAlert[] {
  const subscription = system.subscriptions.get(userId);
  if (!subscription) {
    // Return all alerts for non-subscribers
    return getRecentAlerts(system, { limit });
  }

  return getRecentAlerts(system, {
    limit,
    types: subscription.preferences.types,
    minSignificance: subscription.preferences.minSignificance,
  }).filter((alert) => shouldDeliverAlert(alert, subscription));
}

export function processAlertQueue(
  system: AlertsSystem,
  config: AlertConfig = DEFAULT_CONFIG
): TrendingAlert[] {
  const toProcess = system.alertQueue.splice(0, config.maxAlertsPerHour);
  system.lastProcessed = new Date().toISOString();
  return toProcess;
}

// ============================================================================
// Alert Statistics
// ============================================================================

export interface AlertStats {
  total: number;
  unread: number;
  byType: Record<AlertType, number>;
  bySignificance: Record<SignificanceLevel, number>;
  recentCount: number;
}

export function getAlertStats(system: AlertsSystem): AlertStats {
  const alerts = Array.from(system.alerts.values());
  const now = new Date();
  const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);

  const byType: Record<AlertType, number> = {
    synergy: 0,
    breakthrough: 0,
    trend: 0,
    "gap-filled": 0,
    collaboration: 0,
    evolution: 0,
    opportunity: 0,
  };

  const bySignificance: Record<SignificanceLevel, number> = {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  };

  let unread = 0;
  let recentCount = 0;

  for (const alert of alerts) {
    byType[alert.type]++;
    bySignificance[alert.significance]++;
    if (!alert.read) unread++;
    if (new Date(alert.createdAt) > oneDayAgo) recentCount++;
  }

  return {
    total: alerts.length,
    unread,
    byType,
    bySignificance,
    recentCount,
  };
}

// ============================================================================
// Export
// ============================================================================

export const alertsConfig = DEFAULT_CONFIG;
