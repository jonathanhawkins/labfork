"use client";

/**
 * Admin Compute Dashboard
 *
 * A simple admin interface to:
 * - View connected devices
 * - Submit demo tasks
 * - Monitor task queue
 * - View network statistics
 */

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  RefreshCw,
  Plus,
  Cpu,
  Zap,
  CheckCircle2,
  Clock,
  Users,
  Activity,
  Loader2,
  Play,
  BarChart3,
  Server,
} from "lucide-react";

interface NetworkStats {
  totalDevices: number;
  onlineDevices: number;
  devicesByTier: {
    power: number;
    standard: number;
    crowd: number;
  };
  totalCompute: number;
  pendingTasks: number;
  processingTasks: number;
  completedToday: number;
  creditsToday: number;
}

interface Device {
  id: string;
  name: string;
  tier: string;
  status: string;
  compute: number;
  platform: string;
  stats: {
    tasksCompleted: number;
    creditsEarned: number;
    totalComputeTime: number;
  };
}

interface Task {
  id: string;
  type: string;
  status: string;
  priority: number;
  reward: number;
  createdAt: string;
  assignedAt?: string;
  completedAt?: string;
  assignedDeviceId?: string;
}

export default function AdminComputePage() {
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [devices, setDevices] = useState<Device[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [taskCount, setTaskCount] = useState(5);
  const [selectedTier, setSelectedTier] = useState<string>("all");
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Fetch all data
  const fetchData = useCallback(async () => {
    try {
      const [statsRes, devicesRes, tasksRes] = await Promise.all([
        fetch("/api/compute/stats"),
        fetch("/api/compute/devices"),
        fetch("/api/compute/tasks"),
      ]);

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data.stats);
      }

      if (devicesRes.ok) {
        const data = await devicesRes.json();
        setDevices(data.devices || []);
      }

      if (tasksRes.ok) {
        const data = await tasksRes.json();
        setTasks(data.tasks || []);
      }

      setLastRefresh(new Date());
    } catch (error) {
      console.error("Failed to fetch data:", error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial load and auto-refresh
  useEffect(() => {
    fetchData();

    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Generate demo tasks
  const handleGenerateTasks = async () => {
    setIsGenerating(true);
    try {
      const body: { count: number; tier?: string } = { count: taskCount };
      if (selectedTier !== "all") {
        body.tier = selectedTier;
      }

      const response = await fetch("/api/compute/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Generated tasks:", data);
        // Refresh data to show new tasks
        await fetchData();
      }
    } catch (error) {
      console.error("Failed to generate tasks:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-foreground-muted" />
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-medium text-foreground-bright">
              Compute Network Admin
            </h1>
            <p className="text-sm text-foreground-muted">
              Monitor and manage the distributed compute network
            </p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs text-foreground-muted">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </span>
            <button
              onClick={fetchData}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-foreground-muted/10 hover:bg-foreground-muted/20 transition-colors min-h-[44px]"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
        </div>

        {/* Network Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            icon={<Users className="w-5 h-5" />}
            label="Online Devices"
            value={stats?.onlineDevices || 0}
            subValue={`${stats?.totalDevices || 0} total`}
            color="text-green-400"
          />
          <StatCard
            icon={<Zap className="w-5 h-5" />}
            label="Total Compute"
            value={`${(stats?.totalCompute || 0).toFixed(1)} TFLOPS`}
            color="text-yellow-400"
          />
          <StatCard
            icon={<Clock className="w-5 h-5" />}
            label="Pending Tasks"
            value={stats?.pendingTasks || 0}
            subValue={`${stats?.processingTasks || 0} processing`}
            color="text-blue-400"
          />
          <StatCard
            icon={<CheckCircle2 className="w-5 h-5" />}
            label="Completed Today"
            value={stats?.completedToday || 0}
            subValue={`${stats?.creditsToday || 0} credits`}
            color="text-purple-400"
          />
        </div>

        {/* Tier Breakdown */}
        <div className="bg-background-card border border-border rounded-xl p-6">
          <h2 className="text-lg font-medium text-foreground-bright mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Device Tier Breakdown
          </h2>
          <div className="grid grid-cols-3 gap-4">
            <TierCard
              tier="Power"
              count={stats?.devicesByTier.power || 0}
              color="bg-purple-500"
              description="RTX 4090+, A100"
            />
            <TierCard
              tier="Standard"
              count={stats?.devicesByTier.standard || 0}
              color="bg-blue-500"
              description="Mac M1-M4, RTX 3060+"
            />
            <TierCard
              tier="Crowd"
              count={stats?.devicesByTier.crowd || 0}
              color="bg-green-500"
              description="Browser, Mobile"
            />
          </div>
        </div>

        {/* Task Generator */}
        <div className="bg-background-card border border-border rounded-xl p-6">
          <h2 className="text-lg font-medium text-foreground-bright mb-4 flex items-center gap-2">
            <Plus className="w-5 h-5" />
            Generate Demo Tasks
          </h2>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-foreground-muted">Count:</label>
              <input
                type="number"
                min={1}
                max={20}
                value={taskCount}
                onChange={(e) => setTaskCount(Number(e.target.value))}
                className="w-20 px-3 py-2 rounded-lg bg-background border border-border text-foreground"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-foreground-muted">Tier:</label>
              <select
                value={selectedTier}
                onChange={(e) => setSelectedTier(e.target.value)}
                className="px-3 py-2 rounded-lg bg-background border border-border text-foreground"
              >
                <option value="all">All Tiers</option>
                <option value="crowd">Crowd</option>
                <option value="standard">Standard</option>
                <option value="power">Power</option>
              </select>
            </div>
            <button
              onClick={handleGenerateTasks}
              disabled={isGenerating}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors font-medium min-h-[44px] disabled:opacity-50"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Generate Tasks
                </>
              )}
            </button>
          </div>
          <p className="text-xs text-foreground-muted mt-3">
            Generate demo tasks to test the distributed compute network. Tasks will be assigned to available devices.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Connected Devices */}
          <div className="bg-background-card border border-border rounded-xl p-6">
            <h2 className="text-lg font-medium text-foreground-bright mb-4 flex items-center gap-2">
              <Server className="w-5 h-5" />
              Connected Devices ({devices.length})
            </h2>
            {devices.length === 0 ? (
              <div className="text-center py-8 text-foreground-muted">
                <Cpu className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No devices connected</p>
                <p className="text-xs mt-1">
                  Open /contribute in another tab to connect a device
                </p>
              </div>
            ) : (
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {devices.map((device) => (
                  <div
                    key={device.id}
                    className="flex items-center justify-between p-3 rounded-lg bg-background/50 border border-border"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "w-2 h-2 rounded-full",
                          device.status === "online"
                            ? "bg-green-400"
                            : device.status === "busy"
                            ? "bg-blue-400"
                            : "bg-gray-400"
                        )}
                      />
                      <div>
                        <p className="text-sm text-foreground font-medium">
                          {device.name}
                        </p>
                        <p className="text-xs text-foreground-muted">
                          {device.tier} | {device.compute.toFixed(2)} TFLOPS
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-foreground">
                        {device.stats.tasksCompleted} tasks
                      </p>
                      <p className="text-xs text-foreground-muted">
                        {device.stats.creditsEarned.toFixed(1)} credits
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Task Queue */}
          <div className="bg-background-card border border-border rounded-xl p-6">
            <h2 className="text-lg font-medium text-foreground-bright mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Recent Tasks ({tasks.length})
            </h2>
            {tasks.length === 0 ? (
              <div className="text-center py-8 text-foreground-muted">
                <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No tasks in queue</p>
                <p className="text-xs mt-1">
                  Click "Generate Tasks" to add demo tasks
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-80 overflow-y-auto">
                {tasks.slice(0, 20).map((task) => (
                  <div
                    key={task.id}
                    className="flex items-center justify-between p-3 rounded-lg bg-background/50 border border-border"
                  >
                    <div className="flex items-center gap-3">
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded text-xs font-medium",
                          task.status === "completed"
                            ? "bg-green-500/20 text-green-400"
                            : task.status === "processing"
                            ? "bg-blue-500/20 text-blue-400"
                            : task.status === "assigned"
                            ? "bg-yellow-500/20 text-yellow-400"
                            : "bg-gray-500/20 text-gray-400"
                        )}
                      >
                        {task.status}
                      </span>
                      <div>
                        <p className="text-sm text-foreground">{task.type}</p>
                        <p className="text-xs text-foreground-muted">
                          {task.id.slice(0, 12)}...
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-foreground">
                        {task.reward} credits
                      </p>
                      <p className="text-xs text-foreground-muted">
                        P{task.priority}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-background-card border border-border rounded-xl p-6">
          <h2 className="text-lg font-medium text-foreground-bright mb-3">
            Testing Instructions
          </h2>
          <ol className="list-decimal list-inside space-y-2 text-sm text-foreground-muted">
            <li>
              Open <code className="px-1.5 py-0.5 rounded bg-background">/contribute</code> in another browser tab
            </li>
            <li>Run the benchmark and click "Start Contributing"</li>
            <li>Come back here and click "Generate Tasks"</li>
            <li>Watch the tasks get assigned and processed in real-time</li>
            <li>Check the /contribute tab to see credits being earned</li>
          </ol>
        </div>
      </div>
    </main>
  );
}

function StatCard({
  icon,
  label,
  value,
  subValue,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subValue?: string;
  color: string;
}) {
  return (
    <div className="bg-background-card border border-border rounded-xl p-4">
      <div className={cn("mb-2", color)}>{icon}</div>
      <p className="text-2xl font-medium text-foreground-bright">{value}</p>
      <p className="text-sm text-foreground-muted">{label}</p>
      {subValue && (
        <p className="text-xs text-foreground-muted/70 mt-1">{subValue}</p>
      )}
    </div>
  );
}

function TierCard({
  tier,
  count,
  color,
  description,
}: {
  tier: string;
  count: number;
  color: string;
  description: string;
}) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-background/50 border border-border">
      <div className={cn("w-3 h-3 rounded-full", color)} />
      <div>
        <p className="text-sm text-foreground font-medium">
          {tier}: {count}
        </p>
        <p className="text-xs text-foreground-muted">{description}</p>
      </div>
    </div>
  );
}
