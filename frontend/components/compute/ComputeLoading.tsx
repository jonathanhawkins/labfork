"use client";

/**
 * Compute Loading Component
 *
 * Skeleton loaders for compute components.
 * Provides visual feedback while data is loading.
 */

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

/**
 * Props for skeleton components
 */
export interface SkeletonProps {
  /** Additional CSS classes */
  className?: string;
}

/**
 * Base Skeleton Component
 */
export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        "bg-muted/50 rounded-lg animate-pulse",
        className
      )}
    />
  );
}

/**
 * GPU Info Skeleton Loader
 *
 * Matches the layout of the GPU detection card
 */
export function GPUInfoSkeleton({ className }: SkeletonProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("bg-card border border-border rounded-xl p-6", className)}
    >
      <div className="flex items-start gap-4">
        {/* Icon Skeleton */}
        <Skeleton className="w-12 h-12 rounded-lg flex-shrink-0" />

        <div className="flex-1 min-w-0 space-y-4">
          {/* Title and Description */}
          <div className="space-y-2">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-4 w-full max-w-sm" />
          </div>

          {/* GPU Info */}
          <div className="space-y-2">
            <Skeleton className="h-4 w-48" />

            {/* Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
              <div className="bg-background/50 rounded-lg p-3 border border-border">
                <Skeleton className="h-3 w-12 mb-2" />
                <Skeleton className="h-4 w-20" />
              </div>
              <div className="bg-background/50 rounded-lg p-3 border border-border">
                <Skeleton className="h-3 w-24 mb-2" />
                <Skeleton className="h-4 w-16" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Network Stats Skeleton Loader
 *
 * Matches the layout of the LiveDashboard network stats grid
 */
export function NetworkStatsSkeleton({ className }: SkeletonProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("space-y-4", className)}
    >
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Skeleton className="w-2 h-2 rounded-full" />
          <Skeleton className="h-3 w-16" />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="bg-card border border-border rounded-lg p-4"
          >
            <div className="flex items-center gap-2 mb-2">
              <Skeleton className="w-4 h-4" />
              <Skeleton className="h-3 w-16" />
            </div>
            <Skeleton className="h-7 w-12 mb-1" />
            <Skeleton className="h-3 w-20" />
          </div>
        ))}
      </div>

      {/* Device Distribution */}
      <div className="bg-card border border-border rounded-lg p-4">
        <Skeleton className="h-4 w-32 mb-3" />
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <div className="flex items-center justify-between">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-16" />
              </div>
              <Skeleton className="h-1.5 w-full rounded-full" />
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Task List Skeleton Loader
 *
 * Generic skeleton for task/item lists
 */
export function TaskListSkeleton({ className }: SkeletonProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("bg-card border border-border rounded-xl p-6", className)}
    >
      {/* Header */}
      <div className="flex items-start gap-4 mb-6">
        <Skeleton className="w-12 h-12 rounded-lg flex-shrink-0" />
        <div className="flex-1 space-y-2">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-full max-w-md" />
        </div>
      </div>

      {/* Task Items */}
      <div className="space-y-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <div
            key={i}
            className="border border-border rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-3 w-12" />
            </div>
            <Skeleton className="h-3 w-full max-w-xs" />
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/**
 * Network Health Skeleton Loader
 *
 * Matches the NetworkHealth component layout
 */
export function NetworkHealthSkeleton({ className }: SkeletonProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("space-y-6", className)}
    >
      {/* Health Score Card */}
      <div className="bg-card border border-border rounded-xl p-6">
        {/* Header */}
        <div className="flex items-start justify-between mb-6">
          <div className="space-y-2">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-4 w-48" />
          </div>
          <div className="flex items-center gap-2">
            <Skeleton className="w-6 h-6 rounded-full" />
            <div className="text-right space-y-1">
              <Skeleton className="h-7 w-12" />
              <Skeleton className="h-3 w-16" />
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <Skeleton className="h-2 w-full rounded-full mb-6" />

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <div className="flex items-center gap-2">
                <Skeleton className="w-4 h-4" />
                <Skeleton className="h-3 w-20" />
              </div>
              <Skeleton className="h-6 w-16" />
            </div>
          ))}
        </div>
      </div>

      {/* Contributor Tier Breakdown */}
      <div className="bg-card border border-border rounded-xl p-6">
        <Skeleton className="h-4 w-40 mb-4" />
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Skeleton className="w-3 h-3 rounded-full" />
                  <Skeleton className="h-3 w-24" />
                </div>
                <Skeleton className="h-3 w-16" />
              </div>
              <Skeleton className="h-1.5 w-full rounded-full" />
              <div className="flex items-center justify-between">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-16" />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Task Throughput */}
      <div className="bg-card border border-border rounded-xl p-6">
        <Skeleton className="h-4 w-32 mb-4" />
        <div className="grid grid-cols-3 gap-4 mb-4">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="space-y-1">
              <Skeleton className="h-3 w-12" />
              <Skeleton className="h-5 w-16" />
            </div>
          ))}
        </div>
        {/* Chart Skeleton */}
        <div className="h-24 flex items-end gap-1">
          {Array.from({ length: 24 }).map((_, i) => (
            <div
              key={i}
              className="flex-1 bg-muted/50 rounded-sm animate-pulse"
              style={{ height: `${20 + (i % 5) * 15}%` }}
            />
          ))}
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Contributor Map Skeleton Loader
 *
 * Matches the ContributorMap component
 */
export function ContributorMapSkeleton({ className }: SkeletonProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn("bg-card border border-border rounded-xl p-6", className)}
    >
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-5 w-8" />
        </div>

        {/* Map Skeleton */}
        <Skeleton className="h-64 w-full rounded-lg" />

        <div className="text-center">
          <Skeleton className="h-3 w-48 mx-auto" />
        </div>
      </div>
    </motion.div>
  );
}

/**
 * Full Page Loading Skeleton
 *
 * Complete loading state for entire page
 */
export function PageLoadingSkeleton({ className }: SkeletonProps) {
  return (
    <div className={cn("min-h-screen bg-background", className)}>
      {/* Hero Section */}
      <section className="relative pt-24 pb-16 px-4">
        <div className="max-w-4xl mx-auto text-center space-y-6">
          <Skeleton className="h-6 w-48 mx-auto rounded-full" />
          <Skeleton className="h-12 w-96 mx-auto" />
          <Skeleton className="h-12 w-72 mx-auto" />
          <Skeleton className="h-4 w-full max-w-2xl mx-auto" />
          <Skeleton className="h-4 w-full max-w-xl mx-auto" />
        </div>
      </section>

      {/* Content Section */}
      <section className="px-4 pb-24">
        <div className="max-w-4xl mx-auto space-y-6">
          <NetworkStatsSkeleton />
          <GPUInfoSkeleton />
          <TaskListSkeleton />
        </div>
      </section>
    </div>
  );
}
