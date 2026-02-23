/**
 * Activity Icon Registry
 *
 * Maps activity types to Lucide icons dynamically.
 * Supports custom icon names from activity configs.
 */

import {
  Brain,
  Mic,
  Sparkles,
  BarChart,
  Search,
  Code,
  Pause,
  Play,
  Cpu,
  Server,
  Database,
  Zap,
  Flame,
  AudioWaveform,
  FileAudio,
  Volume2,
  Radio,
  Activity,
  TrendingUp,
  GitBranch,
  Terminal,
  FileCode,
  FlaskConical,
  TestTube,
  Microscope,
  Beaker,
  Lightbulb,
  Target,
  Rocket,
  Settings,
  Wrench,
  Bug,
  CheckCircle,
  XCircle,
  Clock,
  Timer,
  RefreshCw,
  Download,
  Upload,
  Cloud,
  HardDrive,
  type LucideIcon,
} from 'lucide-react';

/**
 * Map of icon names to Lucide icon components
 */
export const ICON_MAP: Record<string, LucideIcon> = {
  // Activity-specific icons
  Brain,
  Mic,
  Sparkles,
  BarChart,
  ChartBar: BarChart, // Alias for backward compatibility
  Search,
  Code,
  Pause,
  Play,

  // Hardware icons
  Cpu,
  Server,
  Database,
  HardDrive,
  Cloud,

  // Process icons
  Zap,
  Flame,
  Activity,
  TrendingUp,

  // Audio icons
  AudioWaveform,
  FileAudio,
  Volume2,
  Radio,

  // Development icons
  GitBranch,
  Terminal,
  FileCode,

  // Science icons
  FlaskConical,
  TestTube,
  Microscope,
  Beaker,

  // General icons
  Lightbulb,
  Target,
  Rocket,
  Settings,
  Wrench,
  Bug,

  // Status icons
  CheckCircle,
  XCircle,
  Clock,
  Timer,
  RefreshCw,

  // Transfer icons
  Download,
  Upload,
};

/**
 * Default icons for common activity types
 */
export const DEFAULT_ACTIVITY_ICONS: Record<string, string> = {
  training: 'Brain',
  recording: 'Mic',
  generation: 'Sparkles',
  evaluation: 'ChartBar',
  research: 'Search',
  implementation: 'Code',
  idle: 'Pause',
  inference: 'Zap',
  processing: 'Cpu',
  testing: 'TestTube',
  deployment: 'Rocket',
  debugging: 'Bug',
  optimization: 'TrendingUp',
};

/**
 * Get icon component for an activity
 *
 * @param iconName - Icon name from activity config
 * @param activityId - Activity ID for fallback lookup
 * @returns Lucide icon component
 */
export function getActivityIcon(
  iconName?: string,
  activityId?: string
): LucideIcon {
  // Try explicit icon name first
  if (iconName && ICON_MAP[iconName]) {
    return ICON_MAP[iconName];
  }

  // Try activity ID lookup
  if (activityId) {
    const defaultIconName = DEFAULT_ACTIVITY_ICONS[activityId];
    if (defaultIconName && ICON_MAP[defaultIconName]) {
      return ICON_MAP[defaultIconName];
    }
  }

  // Final fallback
  return Activity;
}

/**
 * Check if an icon name is valid
 */
export function isValidIconName(name: string): boolean {
  return name in ICON_MAP;
}

/**
 * Get all available icon names
 */
export function getAvailableIcons(): string[] {
  return Object.keys(ICON_MAP);
}

/**
 * Icon with color wrapper props
 */
export interface IconProps {
  icon: LucideIcon;
  color?: string;
  size?: number;
  className?: string;
}

/**
 * Get icon props for an activity
 */
export function getActivityIconProps(
  iconName?: string,
  activityId?: string,
  color?: string,
  size?: number
): IconProps {
  return {
    icon: getActivityIcon(iconName, activityId),
    color,
    size: size ?? 20,
  };
}

export default getActivityIcon;
