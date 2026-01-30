"use client";

/**
 * Network Health Component
 *
 * Visual health indicator showing network status, TFLOPS meter,
 * contributor breakdown, and task throughput.
 */

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Activity,
  TrendingUp,
  Users,
  Zap,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import {
  formatTFLOPS,
  getHealthStatus,
  getTierColor,
  getTierLabel,
  type NetworkAnalytics,
  type TierStats,
} from "@/lib/compute/analytics";

/**
 * Props for NetworkHealth
 */
export interface NetworkHealthProps {
  /** Network analytics data */
  analytics: NetworkAnalytics | null;
  /** Show detailed breakdown */
  detailed?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Network Health Component
 */
export function NetworkHealth({
  analytics,
  detailed = false,
  className,
}: NetworkHealthProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!analytics) {
    return (
      <div className={cn("bg-card border border-border rounded-xl p-6", className)}>
        <div className="flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading analytics...</p>
        </div>
      </div>
    );
  }

  const healthStatus = getHealthStatus(analytics.healthScore);

  return (
    <div className={cn("space-y-6", className)}>
      {/* Health Score Header */}
      <div className="bg-card border border-border rounded-xl p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-lg font-semibold text-foreground-bright mb-1">
              Network Health
            </h2>
            <p className="text-sm text-muted-foreground">
              Real-time network status and performance
            </p>
          </div>
          <div className="flex items-center gap-2">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3, type: "spring" }}
            >
              {analytics.healthScore >= 80 ? (
                <CheckCircle2 className="w-6 h-6 text-green-400" />
              ) : (
                <AlertCircle className="w-6 h-6 text-yellow-400" />
              )}
            </motion.div>
            <div className="text-right">
              <p className="text-2xl font-bold text-foreground-bright">
                {analytics.healthScore}%
              </p>
              <p className={cn("text-xs font-medium", healthStatus.color)}>
                {healthStatus.label}
              </p>
            </div>
          </div>
        </div>

        {/* Health Score Bar */}
        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: mounted ? `${analytics.healthScore}%` : 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className={cn(
              "h-full rounded-full transition-colors",
              analytics.healthScore >= 80
                ? "bg-green-500"
                : analytics.healthScore >= 60
                ? "bg-yellow-500"
                : "bg-red-500"
            )}
          />
        </div>

        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
          {/* Total TFLOPS */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Compute Power</span>
            </div>
            <p className="text-xl font-semibold text-foreground-bright">
              {formatTFLOPS(analytics.totalTFLOPS)}FLOPS
            </p>
          </motion.div>

          {/* Active Contributors */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Users className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Contributors</span>
            </div>
            <p className="text-xl font-semibold text-foreground-bright">
              {analytics.activeContributors}
            </p>
          </motion.div>

          {/* Tasks Per Hour */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Tasks/Hour</span>
            </div>
            <p className="text-xl font-semibold text-foreground-bright">
              {analytics.throughput.current}
            </p>
          </motion.div>

          {/* Completion Rate */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Success Rate</span>
            </div>
            <p className="text-xl font-semibold text-foreground-bright">
              {analytics.completionRate}%
            </p>
          </motion.div>
        </div>
      </div>

      {/* Contributor Tier Breakdown */}
      {detailed && (
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="text-sm font-semibold text-foreground-bright mb-4">
            Contributors by Tier
          </h3>
          <div className="space-y-4">
            {analytics.contributorsByTier.map((tier, index) => (
              <TierCard key={tier.tier} tier={tier} index={index} />
            ))}
          </div>
        </div>
      )}

      {/* Task Throughput */}
      {detailed && (
        <div className="bg-card border border-border rounded-xl p-6">
          <h3 className="text-sm font-semibold text-foreground-bright mb-4">
            Task Throughput
          </h3>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Current</p>
              <p className="text-lg font-semibold text-foreground-bright">
                {analytics.throughput.current}/hr
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Average</p>
              <p className="text-lg font-semibold text-foreground-bright">
                {analytics.throughput.average}/hr
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Peak</p>
              <p className="text-lg font-semibold text-foreground-bright">
                {analytics.throughput.peak}/hr
              </p>
            </div>
          </div>

          {/* Simple throughput visualization */}
          <div className="mt-4 h-24 flex items-end gap-1">
            {Array.from({ length: 24 }).map((_, i) => {
              const height = Math.random() * 100; // Mock data - would be real historical data
              return (
                <motion.div
                  key={i}
                  initial={{ height: 0 }}
                  animate={{ height: `${height}%` }}
                  transition={{ delay: i * 0.02, duration: 0.3 }}
                  className="flex-1 bg-muted-foreground/20 rounded-t-sm min-h-[4px]"
                />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Tier Card Component
 */
function TierCard({ tier, index }: { tier: TierStats; index: number }) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const percentage = tier.count > 0 ? 100 : 0; // Would calculate relative percentage in production

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
      className="space-y-2"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={cn("w-3 h-3 rounded-full", getTierColor(tier.tier))} />
          <span className="text-sm font-medium text-foreground">
            {getTierLabel(tier.tier)}
          </span>
        </div>
        <div className="text-right">
          <span className="text-sm font-semibold text-foreground-bright">
            {tier.count}
          </span>
          <span className="text-xs text-muted-foreground ml-1">devices</span>
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: mounted ? `${percentage}%` : 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className={cn("h-full rounded-full", getTierColor(tier.tier))}
        />
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{formatTFLOPS(tier.totalCompute)} TFLOPS</span>
        <span>{tier.tasksCompleted} tasks</span>
      </div>
    </motion.div>
  );
}
