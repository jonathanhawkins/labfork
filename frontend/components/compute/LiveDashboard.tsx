"use client";

/**
 * Live Network Dashboard Component
 *
 * Real-time visualization of the distributed compute network
 * Shows network stats, device activity, and recent task completions
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Activity,
  Cpu,
  Zap,
  Award,
  TrendingUp,
  Signal,
  AlertCircle,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import {
  useNetworkEvents,
  formatTierBreakdown,
  calculateTasksPerHour,
  calculateNetworkHealth,
} from "@/lib/compute/useNetworkEvents";
import { useDeviceInfo, formatCredits } from "@/lib/compute/useDeviceAgent";

/**
 * Props for LiveDashboard
 */
export interface LiveDashboardProps {
  /** Whether to show personal stats (if user is contributing) */
  showPersonalStats?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Live Dashboard Component
 */
export function LiveDashboard({ showPersonalStats = false, className }: LiveDashboardProps) {
  const { isConnected, networkStats, recentCompletion, error, isSupported } = useNetworkEvents();
  const device = useDeviceInfo();

  const [pulseKey, setPulseKey] = useState(0);

  // Trigger pulse animation on new completion
  useEffect(() => {
    if (recentCompletion) {
      setPulseKey((k) => k + 1);
    }
  }, [recentCompletion]);

  if (!isSupported) {
    return (
      <div className={cn("bg-background-card border border-border rounded-xl p-6", className)}>
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-foreground mb-1">
              Real-time updates unavailable
            </p>
            <p className="text-xs text-foreground-muted">
              Your browser doesn't support EventSource. Try a modern browser like Chrome, Firefox, or Edge.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const tierBreakdown = networkStats ? formatTierBreakdown(networkStats.devicesByTier) : [];
  const tasksPerHour = networkStats ? calculateTasksPerHour(networkStats.completedToday) : 0;
  const networkHealth = calculateNetworkHealth(networkStats);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "w-2 h-2 rounded-full transition-colors",
              isConnected ? "bg-green-400 animate-pulse" : "bg-gray-400"
            )}
          />
          <span className="text-sm text-foreground-muted">
            {isConnected ? "Live" : "Connecting..."}
          </span>
        </div>

        {error && (
          <div className="flex items-center gap-2 text-xs text-red-400">
            <AlertCircle className="w-3 h-3" />
            Connection error
          </div>
        )}
      </div>

      {/* Network Stats Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Online Devices */}
        <motion.div
          key={`devices-${networkStats?.onlineDevices}`}
          initial={{ scale: 0.95 }}
          animate={{ scale: 1 }}
          className="bg-background-card border border-border rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Cpu className="w-4 h-4 text-foreground-muted" />
            <span className="text-xs text-foreground-muted">Devices</span>
          </div>
          <p className="text-2xl font-medium text-foreground-bright">
            {networkStats?.onlineDevices ?? 0}
          </p>
          <p className="text-xs text-foreground-muted mt-1">
            {networkStats?.totalDevices ?? 0} total
          </p>
        </motion.div>

        {/* Tasks Per Hour */}
        <motion.div
          key={`tasks-${tasksPerHour}`}
          initial={{ scale: 0.95 }}
          animate={{ scale: 1 }}
          className="bg-background-card border border-border rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-foreground-muted" />
            <span className="text-xs text-foreground-muted">Tasks/Hour</span>
          </div>
          <p className="text-2xl font-medium text-foreground-bright">
            {tasksPerHour}
          </p>
          <p className="text-xs text-foreground-muted mt-1">
            {networkStats?.processingTasks ?? 0} processing
          </p>
        </motion.div>

        {/* Credits Distributed */}
        <motion.div
          key={`credits-${networkStats?.creditsToday}`}
          initial={{ scale: 0.95 }}
          animate={{ scale: 1 }}
          className="bg-background-card border border-border rounded-lg p-4"
        >
          <div className="flex items-center gap-2 mb-2">
            <Award className="w-4 h-4 text-foreground-muted" />
            <span className="text-xs text-foreground-muted">Credits Today</span>
          </div>
          <p className="text-2xl font-medium text-foreground-bright">
            {formatCredits(networkStats?.creditsToday ?? 0)}
          </p>
          <p className="text-xs text-foreground-muted mt-1">
            {networkStats?.completedToday ?? 0} tasks
          </p>
        </motion.div>

        {/* Network Health */}
        <div className="bg-background-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Signal className="w-4 h-4 text-foreground-muted" />
            <span className="text-xs text-foreground-muted">Health</span>
          </div>
          <div className="flex items-center gap-2">
            <p className="text-2xl font-medium text-foreground-bright">
              {networkHealth}%
            </p>
            {networkHealth >= 80 && (
              <CheckCircle2 className="w-4 h-4 text-green-400" />
            )}
          </div>
          <div className="w-full h-1.5 bg-background rounded-full mt-2 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${networkHealth}%` }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className={cn(
                "h-full rounded-full",
                networkHealth >= 80 ? "bg-green-500" :
                networkHealth >= 50 ? "bg-yellow-500" : "bg-red-500"
              )}
            />
          </div>
        </div>
      </div>

      {/* Device Tier Breakdown */}
      {networkStats && networkStats.onlineDevices > 0 && (
        <div className="bg-background-card border border-border rounded-lg p-4">
          <h3 className="text-sm font-medium text-foreground mb-3">Device Distribution</h3>
          <div className="space-y-2">
            {tierBreakdown.map((tier) => (
              <div key={tier.tier}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-foreground-muted">{tier.label}</span>
                  <span className="text-foreground">{tier.count} devices</span>
                </div>
                <div className="w-full h-1.5 bg-background rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{
                      width: networkStats.onlineDevices > 0
                        ? `${(tier.count / networkStats.onlineDevices) * 100}%`
                        : "0%"
                    }}
                    transition={{ duration: 0.3, ease: "easeOut" }}
                    className={cn("h-full rounded-full", tier.color)}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Completion Alert */}
      <AnimatePresence>
        {recentCompletion && (
          <motion.div
            key={pulseKey}
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="bg-green-500/10 border border-green-500/20 rounded-lg p-4"
          >
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
                <CheckCircle2 className="w-4 h-4 text-green-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-green-400 mb-1">
                  Task Completed!
                </p>
                <p className="text-xs text-green-400/80">
                  {recentCompletion.count} {recentCompletion.count === 1 ? "task" : "tasks"} completed
                  {" • "}
                  {formatCredits(recentCompletion.creditsAwarded)} credits distributed
                </p>
              </div>
              <motion.div
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 0.5, repeat: 3 }}
              >
                <Zap className="w-4 h-4 text-green-400" />
              </motion.div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Personal Stats (if contributing) */}
      {showPersonalStats && device && (
        <div className="bg-background-card border border-border rounded-lg p-4">
          <h3 className="text-sm font-medium text-foreground mb-3">Your Contribution</h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-foreground-muted mb-1">Device Tier</p>
              <p className="text-sm font-medium text-foreground capitalize">
                {device.tier}
              </p>
            </div>
            <div>
              <p className="text-xs text-foreground-muted mb-1">Status</p>
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
                <p className="text-sm font-medium text-foreground">Online</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {networkStats && networkStats.onlineDevices === 0 && (
        <div className="bg-background-card border border-border rounded-lg p-6 text-center">
          <Loader2 className="w-8 h-8 text-foreground-muted mx-auto mb-3 animate-spin" />
          <p className="text-sm font-medium text-foreground mb-1">
            Waiting for devices...
          </p>
          <p className="text-xs text-foreground-muted">
            Be the first to contribute compute power to the network
          </p>
        </div>
      )}
    </div>
  );
}
