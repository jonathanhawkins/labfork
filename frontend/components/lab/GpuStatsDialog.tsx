"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Cpu, Thermometer, Zap, HardDrive, Activity, RefreshCw, WifiOff } from "lucide-react";

interface GpuProcess {
  pid: string;
  name: string;
  memoryUsed: string;
  script?: string;
  session?: string;
  config?: string;
  progress?: string;
}

interface GpuStats {
  connected: boolean;
  error?: string;
  timestamp: string;
  gpu: {
    name: string;
    driverVersion: string;
    cudaVersion: string;
    utilization: number;
    memoryUsed: number;
    memoryTotal: number;
    memoryPercent: number;
    temperature: number;
    powerDraw: number;
    powerLimit: number;
    fanSpeed?: number;
  } | null;
  processes: GpuProcess[];
}

interface GpuStatsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function StatBar({ value, max, color = "foreground" }: { value: number; max: number; color?: string }) {
  const percent = Math.min(100, (value / max) * 100);
  return (
    <div className="h-1.5 bg-border rounded-full overflow-hidden">
      <div
        className={`h-full transition-all duration-500 ${
          color === "temperature"
            ? percent > 80 ? "bg-red-500" : percent > 60 ? "bg-yellow-500" : "bg-foreground"
            : "bg-foreground"
        }`}
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

function StatRow({
  icon: Icon,
  label,
  value,
  subValue,
  bar,
}: {
  icon: typeof Cpu;
  label: string;
  value: string | number;
  subValue?: string;
  bar?: { value: number; max: number; color?: string };
}) {
  return (
    <div className="py-2">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Icon className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
        <div className="text-right">
          <span className="text-sm text-foreground-bright font-mono">{value}</span>
          {subValue && (
            <span className="text-xs text-muted-foreground ml-1">{subValue}</span>
          )}
        </div>
      </div>
      {bar && <StatBar value={bar.value} max={bar.max} color={bar.color} />}
    </div>
  );
}

export function GpuStatsDialog({ open, onOpenChange }: GpuStatsDialogProps) {
  const [stats, setStats] = useState<GpuStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchStats = useCallback(async () => {
    setLoading(true);
    try {
      // Call the local Next.js API route (which proxies to backend)
      const response = await fetch("/api/lab/gpu-stats");
      const data = await response.json();
      setStats(data);
      setLastUpdate(new Date());
    } catch (error) {
      setStats({
        connected: false,
        error: "Failed to connect to backend server",
        timestamp: new Date().toISOString(),
        gpu: null,
        processes: [],
      });
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch on open and set up polling
  useEffect(() => {
    if (open) {
      fetchStats();
      const interval = setInterval(fetchStats, 5000);
      return () => clearInterval(interval);
    }
  }, [open, fetchStats]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md bg-background-elevated border-border">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Cpu className="w-4 h-4" />
            RTX 4090 Training Machine
          </DialogTitle>
          <DialogDescription className="font-mono text-xs">
            {process.env.NEXT_PUBLIC_REMOTE_GPU_HOST || "Not configured"} (Remote)
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Connection status */}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              {stats?.connected ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                  <span className="text-foreground">Connected</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3 h-3 text-muted-foreground" />
                  <span className="text-muted-foreground">Disconnected</span>
                </>
              )}
            </div>
            <button
              onClick={fetchStats}
              disabled={loading}
              className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
            >
              <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
          </div>

          {/* Error state */}
          {stats && !stats.connected && stats.error && (
            <div className="p-3 bg-background border border-border rounded">
              <div className="text-xs text-muted-foreground font-mono">
                <span className="text-foreground-bright">$</span> ssh {process.env.NEXT_PUBLIC_REMOTE_GPU_USER || "user"}@{process.env.NEXT_PUBLIC_REMOTE_GPU_HOST || "your-gpu-host"}
              </div>
              <div className="text-xs text-red-400 mt-1">{stats.error}</div>
            </div>
          )}

          {/* GPU Stats */}
          {stats?.connected && stats.gpu && (
            <>
              {/* GPU Name */}
              <div className="p-3 bg-background border border-border rounded">
                <div className="text-xs text-muted-foreground mb-1">GPU</div>
                <div className="text-sm text-foreground-bright font-mono">
                  {stats.gpu.name}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  Driver {stats.gpu.driverVersion} | CUDA {stats.gpu.cudaVersion}
                </div>
              </div>

              {/* Stats Grid */}
              <div className="space-y-1 p-3 bg-background border border-border rounded">
                <StatRow
                  icon={Activity}
                  label="GPU Utilization"
                  value={`${stats.gpu.utilization}%`}
                  bar={{ value: stats.gpu.utilization, max: 100 }}
                />
                <StatRow
                  icon={HardDrive}
                  label="VRAM"
                  value={`${(stats.gpu.memoryUsed / 1024).toFixed(1)} GB`}
                  subValue={`/ ${(stats.gpu.memoryTotal / 1024).toFixed(0)} GB`}
                  bar={{ value: stats.gpu.memoryUsed, max: stats.gpu.memoryTotal }}
                />
                <StatRow
                  icon={Thermometer}
                  label="Temperature"
                  value={`${stats.gpu.temperature}°C`}
                  bar={{ value: stats.gpu.temperature, max: 90, color: "temperature" }}
                />
                <StatRow
                  icon={Zap}
                  label="Power"
                  value={`${stats.gpu.powerDraw}W`}
                  subValue={`/ ${stats.gpu.powerLimit}W`}
                  bar={{ value: stats.gpu.powerDraw, max: stats.gpu.powerLimit }}
                />
              </div>

              {/* Running Processes */}
              {stats.processes.length > 0 && (
                <div className="p-3 bg-background border border-border rounded">
                  <div className="text-xs text-muted-foreground mb-2">Active Training</div>
                  <div className="space-y-3">
                    {stats.processes.map((proc, idx) => (
                      <div key={idx} className="space-y-1.5">
                        {/* Script name with tmux session badge */}
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-foreground-bright font-mono">
                            {proc.script || proc.name}
                          </span>
                          {proc.session && (
                            <span className="px-1.5 py-0.5 bg-green-500/20 text-green-400 text-[10px] font-mono rounded">
                              tmux:{proc.session}
                            </span>
                          )}
                        </div>
                        {/* Config file */}
                        {proc.config && (
                          <div className="text-xs text-muted-foreground font-mono">
                            config/{proc.config}
                          </div>
                        )}
                        {/* Progress */}
                        {proc.progress && (
                          <div className="flex items-center gap-2 mt-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                            <span className="text-xs text-green-400 font-mono">
                              {proc.progress}
                            </span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {stats.processes.length === 0 && (
                <div className="p-3 bg-background border border-border rounded text-center">
                  <div className="text-xs text-muted-foreground">No GPU processes running</div>
                  <div className="text-xs text-foreground-subtle mt-1">GPU is idle</div>
                </div>
              )}
            </>
          )}

          {/* Loading state */}
          {loading && !stats && (
            <div className="p-6 text-center">
              <div className="text-sm text-muted-foreground">Connecting to training machine...</div>
            </div>
          )}

          {/* Footer */}
          {lastUpdate && (
            <div className="text-xs text-foreground-subtle text-center">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
