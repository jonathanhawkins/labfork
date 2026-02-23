/**
 * Activity Registry
 *
 * Manages activity configurations, loading from domain plugins when available.
 */

import * as yaml from 'yaml';
import * as fs from 'fs';
import * as path from 'path';
import {
  ActivityConfig,
  BUILTIN_ACTIVITIES,
  mergeActivityDefaults,
  getBuiltinActivity,
} from './types';
import { getDomainPath } from '../domain/loader';

/**
 * Cache for loaded domain activities
 */
const domainActivityCache = new Map<string, ActivityConfig[]>();

/**
 * Parse activity YAML file
 */
function parseActivityYaml(filePath: string): ActivityConfig | null {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    const parsed = yaml.parse(content);

    if (!parsed || typeof parsed !== 'object') {
      console.warn(`Invalid activity YAML: ${filePath}`);
      return null;
    }

    // Merge with defaults to ensure all required fields
    return mergeActivityDefaults(parsed as Partial<ActivityConfig>);
  } catch (error) {
    console.warn(`Failed to parse activity YAML: ${filePath}`, error);
    return null;
  }
}

/**
 * Load activities from a domain's activities directory
 */
export function loadDomainActivities(domainSlug: string): ActivityConfig[] {
  // Check cache first
  if (domainActivityCache.has(domainSlug)) {
    return domainActivityCache.get(domainSlug)!;
  }

  const domainPath = getDomainPath(domainSlug);
  const activitiesDir = path.join(domainPath, 'activities');

  const activities: ActivityConfig[] = [];

  // Try to load from domain activities directory
  if (fs.existsSync(activitiesDir)) {
    try {
      const files = fs.readdirSync(activitiesDir).filter(
        (f) => f.endsWith('.yaml') || f.endsWith('.yml')
      );

      for (const file of files) {
        const filePath = path.join(activitiesDir, file);
        const activity = parseActivityYaml(filePath);
        if (activity) {
          activities.push(activity);
        }
      }
    } catch (error) {
      console.warn(`Failed to read activities directory: ${activitiesDir}`, error);
    }
  }

  // If no domain activities found, return built-in activities
  if (activities.length === 0) {
    domainActivityCache.set(domainSlug, BUILTIN_ACTIVITIES);
    return BUILTIN_ACTIVITIES;
  }

  // Sort by priority
  activities.sort((a, b) => (a.priority ?? 50) - (b.priority ?? 50));

  // Cache and return
  domainActivityCache.set(domainSlug, activities);
  return activities;
}

/**
 * Get a specific activity config by ID, optionally from a domain
 */
export function getActivityConfig(
  activityId: string,
  domainSlug?: string
): ActivityConfig | undefined {
  if (domainSlug) {
    const domainActivities = loadDomainActivities(domainSlug);
    const domainActivity = domainActivities.find((a) => a.id === activityId);
    if (domainActivity) {
      return domainActivity;
    }
  }

  // Fall back to built-in
  return getBuiltinActivity(activityId);
}

/**
 * Get all activities for a domain (or built-in if no domain)
 */
export function getAllActivities(domainSlug?: string): ActivityConfig[] {
  if (domainSlug) {
    return loadDomainActivities(domainSlug);
  }
  return BUILTIN_ACTIVITIES;
}

/**
 * Clear the activity cache (useful for development)
 */
export function clearActivityCache(): void {
  domainActivityCache.clear();
}

/**
 * Check if domain has custom activities
 */
export function domainHasCustomActivities(domainSlug: string): boolean {
  const domainPath = getDomainPath(domainSlug);
  const activitiesDir = path.join(domainPath, 'activities');

  if (!fs.existsSync(activitiesDir)) {
    return false;
  }

  try {
    const files = fs.readdirSync(activitiesDir).filter(
      (f) => f.endsWith('.yaml') || f.endsWith('.yml')
    );
    return files.length > 0;
  } catch {
    return false;
  }
}

/**
 * Activity registry class for managing activities
 */
export class ActivityRegistry {
  private domainSlug?: string;
  private activities: ActivityConfig[];

  constructor(domainSlug?: string) {
    this.domainSlug = domainSlug;
    this.activities = getAllActivities(domainSlug);
  }

  /**
   * Get activity by ID
   */
  get(id: string): ActivityConfig | undefined {
    return this.activities.find((a) => a.id === id);
  }

  /**
   * Get all activities
   */
  getAll(): ActivityConfig[] {
    return this.activities;
  }

  /**
   * Get activities filtered by tag
   */
  getByTag(tag: string): ActivityConfig[] {
    return this.activities.filter((a) => a.tags?.includes(tag));
  }

  /**
   * Reload activities from domain
   */
  reload(): void {
    clearActivityCache();
    this.activities = getAllActivities(this.domainSlug);
  }

  /**
   * Set domain and reload
   */
  setDomain(domainSlug: string): void {
    this.domainSlug = domainSlug;
    this.activities = getAllActivities(domainSlug);
  }
}

/**
 * Default export
 */
export default ActivityRegistry;
