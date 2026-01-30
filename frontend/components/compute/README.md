# Compute Contributor Dashboard

A real-time compute contribution tracking dashboard for LabFork, built with Next.js, TypeScript, Framer Motion, and Tailwind CSS.

## Overview

The ContributorDashboard system provides users with real-time feedback on their compute contributions, including task progress, credits earned, network statistics, and contribution history.

## Components

### ContributorDashboard (Main Component)

The primary dashboard component that orchestrates all sub-components.

**Features:**
- Real-time contribution status (active/paused/idle)
- Animated counters for tasks completed and credits earned
- Session duration tracking
- Pause/Resume controls with 44px touch targets
- Settings access
- Live task progress display
- Network-wide statistics
- 7-day contribution history chart

**Usage:**
```tsx
import { ContributorDashboard } from "@/components/compute";

<ContributorDashboard onSettingsClick={() => console.log("Settings")} />
```

**Props:**
- `onSettingsClick?: () => void` - Callback when settings button is clicked

### TaskProgress

Displays current task execution with animated progress bar.

**Features:**
- Task name display
- Real-time progress percentage
- Animated progress bar with gradient
- Time remaining estimate
- Loading spinner for active tasks
- Enter/exit animations

**Usage:**
```tsx
import { TaskProgress } from "@/components/compute";

<TaskProgress
  taskName="Voice Model Training - Batch 47"
  progress={67}
  timeRemaining="3m 24s"
  isActive={true}
/>
```

**Props:**
- `taskName: string` - Name of the current task
- `progress: number` - Progress percentage (0-100)
- `timeRemaining?: string` - Estimated time remaining
- `isActive: boolean` - Whether the task is currently processing

### NetworkStats

Displays network-wide compute statistics.

**Features:**
- Total active contributors count
- Network compute power in TFLOPS
- User rank and contribution percentage
- Staggered entrance animations
- Live connection indicator

**Usage:**
```tsx
import { NetworkStats } from "@/components/compute";

<NetworkStats
  activeContributors={1247}
  networkComputePower={847.3}
  yourRank={23}
  contributionPercent={1.8}
/>
```

**Props:**
- `activeContributors: number` - Total number of active contributors
- `networkComputePower: number` - Total network compute power in TFLOPS
- `yourRank: number` - User's rank among contributors
- `contributionPercent: number` - User's contribution as percentage

### ContributionChart

Bar chart showing 7-day contribution history.

**Features:**
- Daily task completion visualization
- Credits earned tracking
- Interactive hover tooltips
- Animated bar entrance
- Summary statistics
- Responsive grid layout

**Usage:**
```tsx
import { ContributionChart } from "@/components/compute";

const historyData = [
  { day: "Mon", tasksCompleted: 18, creditsEarned: 360 },
  { day: "Tue", tasksCompleted: 23, creditsEarned: 460 },
  // ... more days
];

<ContributionChart data={historyData} />
```

**Props:**
- `data: Array<{ day: string; tasksCompleted: number; creditsEarned: number }>` - 7 days of history data

## Design System

### Color Semantics

- **Active State**: Green (`bg-green-500`, `text-green-400`)
- **Paused State**: Amber (`bg-amber-500`, `text-amber-400`)
- **Idle State**: Muted gray (`bg-foreground-subtle`, `text-muted-foreground`)
- **Accent**: Green for credits and progress
- **Borders**: `border-border` with subtle variants

### Typography

- **Headings**: `text-foreground-bright` with `font-medium` or `font-semibold`
- **Body**: `text-foreground` or `text-muted-foreground`
- **Stats**: `font-mono` for numerical values
- **Labels**: `text-xs` or `text-sm` with `text-muted-foreground`

### Spacing & Layout

- **Container Padding**: `p-4` or `p-3` for cards
- **Gaps**: `gap-2`, `gap-3`, `gap-4` for consistent spacing rhythm
- **Touch Targets**: Minimum `min-h-[44px]` and `min-w-[44px]` for buttons
- **Border Radius**: `rounded-lg` for cards, `rounded-full` for status indicators

### Animations

All animations use Framer Motion with performance-optimized settings:

- **Duration**: 0.3s for most transitions
- **Easing**: `ease: "easeOut"` for natural feel
- **Stagger**: 0.05-0.1s delays for sequential elements
- **Counter**: 1s smooth counter animation with 60 steps

### Mobile-First Design

- **Responsive Grid**: `grid-cols-1 sm:grid-cols-3` for stats
- **Touch Targets**: All interactive elements meet 44px minimum
- **No Hover Dependencies**: All interactions work without hover
- **Readable Text**: Minimum `text-xs` size, optimized for small screens

## Mock Data

The dashboard currently uses mock data for demonstration. The `getMockData()` function generates realistic sample data including:

- Task completion count: 147
- Credits earned: 2941
- Current task progress: 67%
- Session duration: 2h 14m
- Network stats: 1247 contributors, 847.3 TFLOPS
- 7-day history with varying task counts

## Future Integration

To connect to real data:

1. Replace `getMockData()` with WebSocket connection
2. Update state based on incoming messages
3. Handle connection status and errors
4. Implement settings dialog
5. Add persistence for user preferences

## Accessibility

- **Keyboard Navigation**: All controls are keyboard accessible
- **ARIA Labels**: Buttons include descriptive aria-labels
- **Focus States**: Clear focus indicators with ring outlines
- **Screen Readers**: Semantic HTML with proper labels
- **Color Independence**: Information not conveyed by color alone

## Demo

View the component in action:
```
/demo/contributor-dashboard
```

## File Structure

```
components/compute/
├── ContributorDashboard.tsx  # Main dashboard component
├── TaskProgress.tsx          # Current task progress display
├── NetworkStats.tsx          # Network-wide statistics
├── ContributionChart.tsx     # 7-day history chart
├── index.ts                  # Component exports
└── README.md                 # This file
```

## Dependencies

- `framer-motion` - Animation library
- `lucide-react` - Icon set
- `react` / `react-dom` - React framework
- `tailwindcss` - Utility-first CSS

## Browser Support

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari, Chrome Android)

## Error Boundaries and Loading States

### ComputeErrorBoundary

A React error boundary component that catches errors in compute components and displays a user-friendly error UI with recovery options.

#### Usage

**As a wrapper component:**

```tsx
import { ComputeErrorBoundary } from "@/components/compute/ComputeErrorBoundary";

export function MyComponent() {
  return (
    <ComputeErrorBoundary>
      <YourComputeComponent />
    </ComputeErrorBoundary>
  );
}
```

**As a HOC (Higher-Order Component):**

```tsx
import { withComputeErrorBoundary } from "@/components/compute/ComputeErrorBoundary";

function MyComponent() {
  // Your component code
}

export default withComputeErrorBoundary(MyComponent, {
  showReportLink: true,
  onError: (error, errorInfo) => {
    console.error("Error caught:", error, errorInfo);
  },
});
```

**Props:**

- `children`: ReactNode - Child components to render
- `fallback?`: ReactNode - Optional custom fallback component
- `onError?`: (error: Error, errorInfo: React.ErrorInfo) => void - Optional error callback
- `showReportLink?`: boolean - Show "Report Issue" button (default: true)
- `className?`: string - Additional CSS classes

### ComputeLoading

Skeleton loader components for compute pages and components.

#### Available Skeletons

- **Skeleton** - Base skeleton component
- **GPUInfoSkeleton** - Matches GPU detection card layout
- **NetworkStatsSkeleton** - Matches LiveDashboard network stats
- **TaskListSkeleton** - Generic task/item list skeleton
- **NetworkHealthSkeleton** - Matches NetworkHealth component
- **ContributorMapSkeleton** - Matches ContributorMap component
- **PageLoadingSkeleton** - Full page loading state

#### Usage

```tsx
import {
  GPUInfoSkeleton,
  NetworkStatsSkeleton,
} from "@/components/compute/ComputeLoading";

export function MyLoadingState() {
  return (
    <div className="space-y-6">
      <NetworkStatsSkeleton />
      <GPUInfoSkeleton />
    </div>
  );
}
```

### Next.js Route Error Boundaries

Error boundaries are automatically set up for the following routes:

- `/contribute/error.tsx` - Error boundary for contribute page
- `/analytics/error.tsx` - Error boundary for analytics page

### Next.js Loading States

Loading states are automatically displayed for:

- `/contribute/loading.tsx` - Loading skeleton for contribute page
- `/analytics/loading.tsx` - Loading skeleton for analytics page

## Example: Full Error Handling Flow

```tsx
"use client";

import { Suspense } from "react";
import { ComputeErrorBoundary } from "@/components/compute/ComputeErrorBoundary";
import { NetworkStatsSkeleton } from "@/components/compute/ComputeLoading";
import { LiveDashboard } from "@/components/compute/LiveDashboard";

export default function MyPage() {
  return (
    <ComputeErrorBoundary
      onError={(error) => {
        // Send to error tracking service
        console.error("Page error:", error);
      }}
    >
      <Suspense fallback={<NetworkStatsSkeleton />}>
        <LiveDashboard />
      </Suspense>
    </ComputeErrorBoundary>
  );
}
```

## Testing Error Boundaries

To test error boundaries in development:

```tsx
function TestError() {
  throw new Error("Test error for error boundary");
}

export default function TestPage() {
  return (
    <ComputeErrorBoundary>
      <TestError />
    </ComputeErrorBoundary>
  );
}
```
