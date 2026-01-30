"use client";

/**
 * Analytics Dashboard Page
 *
 * Comprehensive network analytics and health monitoring dashboard.
 * Shows real-time network health, contributor distribution, and performance metrics.
 */

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { NetworkHealth } from "@/components/compute/NetworkHealth";
import { ContributorMap } from "@/components/compute/ContributorMap";
import { LiveDashboard } from "@/components/compute/LiveDashboard";
import { useNetworkEvents } from "@/lib/compute/useNetworkEvents";
import type { NetworkAnalytics } from "@/lib/compute/analytics";
import { generateNetworkAnalytics } from "@/lib/compute/analytics";
import { BarChart3, RefreshCw, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/**
 * Time period options
 */
type TimePeriod = "1h" | "24h" | "7d" | "30d";

const PERIOD_LABELS: Record<TimePeriod, string> = {
  "1h": "Last Hour",
  "24h": "Last 24 Hours",
  "7d": "Last 7 Days",
  "30d": "Last 30 Days",
};

/**
 * Analytics Dashboard Page
 */
export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<NetworkAnalytics | null>(null);
  const [period, setPeriod] = useState<TimePeriod>("24h");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Subscribe to real-time network events
  const { networkStats, recentCompletion } = useNetworkEvents();

  // Fetch analytics data
  const fetchAnalytics = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/compute/analytics?period=${period}`);

      if (!response.ok) {
        throw new Error("Failed to fetch analytics");
      }

      const data = await response.json();
      setAnalytics(data);
      setLastUpdate(new Date());
    } catch (err) {
      console.error("Error fetching analytics:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch and period changes
  useEffect(() => {
    fetchAnalytics();
  }, [period]);

  // Update analytics when network stats change
  useEffect(() => {
    if (networkStats && !loading) {
      // Generate analytics from real-time stats
      const updatedAnalytics = generateNetworkAnalytics(networkStats, [], period);
      setAnalytics(updatedAnalytics);
      setLastUpdate(new Date());
    }
  }, [networkStats, period, loading]);

  // Show new contributor join animation
  const [showNewJoin, setShowNewJoin] = useState(false);
  useEffect(() => {
    if (
      networkStats &&
      analytics &&
      networkStats.onlineDevices > analytics.activeContributors
    ) {
      setShowNewJoin(true);
      const timer = setTimeout(() => setShowNewJoin(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [networkStats, analytics]);

  return (
    <div className="min-h-screen bg-background">
      {/* Page Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <BarChart3 className="w-6 h-6 text-foreground-bright" />
                <h1 className="text-2xl font-bold text-foreground-bright">
                  Network Analytics
                </h1>
              </div>
              <p className="text-sm text-muted-foreground">
                Real-time insights into the distributed compute network
              </p>
            </div>

            <div className="flex items-center gap-3">
              {/* Period Selector */}
              <Select
                value={period}
                onValueChange={(value) => setPeriod(value as TimePeriod)}
              >
                <SelectTrigger className="w-[160px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(Object.keys(PERIOD_LABELS) as TimePeriod[]).map((p) => (
                    <SelectItem key={p} value={p}>
                      {PERIOD_LABELS[p]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Refresh Button */}
              <Button
                variant="outline"
                size="sm"
                onClick={fetchAnalytics}
                disabled={loading}
              >
                <RefreshCw
                  className={`w-4 h-4 mr-2 ${loading ? "animate-spin" : ""}`}
                />
                Refresh
              </Button>
            </div>
          </div>

          {/* Last Updated */}
          <div className="flex items-center gap-2 mt-4">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-xs text-muted-foreground">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Error State */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-destructive/10 border border-destructive/20 rounded-xl p-4 mb-6"
          >
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
              <div>
                <p className="text-sm font-medium text-destructive">
                  Failed to load analytics
                </p>
                <p className="text-xs text-destructive/80 mt-1">{error}</p>
              </div>
            </div>
          </motion.div>
        )}

        {/* Analytics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Network Health (spans 2 cols on large screens) */}
          <div className="lg:col-span-2 space-y-6">
            <NetworkHealth analytics={analytics} detailed={true} />

            {/* Live Dashboard */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="bg-card border border-border rounded-xl p-6">
                <h3 className="text-sm font-semibold text-foreground-bright mb-4">
                  Live Activity
                </h3>
                <LiveDashboard showPersonalStats={false} />
              </div>
            </motion.div>
          </div>

          {/* Right Column - Contributor Map */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <ContributorMap
                activeContributors={analytics?.activeContributors || 0}
                showNewJoin={showNewJoin}
              />
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-card border border-border rounded-xl p-6"
            >
              <h3 className="text-sm font-semibold text-foreground-bright mb-4">
                Today's Activity
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Tasks Completed
                  </span>
                  <span className="text-lg font-semibold text-foreground-bright">
                    {analytics?.totalTasksCompleted || 0}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Credits Distributed
                  </span>
                  <span className="text-lg font-semibold text-foreground-bright">
                    {analytics?.totalCreditsDistributed.toFixed(1) || "0.0"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Average Latency
                  </span>
                  <span className="text-lg font-semibold text-foreground-bright">
                    {analytics?.averageLatency.toFixed(0) || "0"}ms
                  </span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Performance Insights */}
        {analytics && analytics.latencyPercentiles && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-6 bg-card border border-border rounded-xl p-6"
          >
            <h3 className="text-sm font-semibold text-foreground-bright mb-4">
              Latency Distribution
            </h3>
            <div className="grid grid-cols-3 gap-6">
              <div>
                <p className="text-xs text-muted-foreground mb-1">
                  50th Percentile (Median)
                </p>
                <p className="text-2xl font-bold text-foreground-bright">
                  {analytics.latencyPercentiles.p50.toFixed(0)}ms
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">
                  95th Percentile
                </p>
                <p className="text-2xl font-bold text-foreground-bright">
                  {analytics.latencyPercentiles.p95.toFixed(0)}ms
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">
                  99th Percentile
                </p>
                <p className="text-2xl font-bold text-foreground-bright">
                  {analytics.latencyPercentiles.p99.toFixed(0)}ms
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
