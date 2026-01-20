"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

const TRAINING_API = process.env.NEXT_PUBLIC_TRAINING_API || "http://localhost:8001";

// ============== Types ==============

interface TrainingMetrics {
  step: number;
  epoch: number;
  epoch_progress: number;
  train_loss: number;
  val_loss: number;
  mtp_loss: number;
  learning_rate: number;
  samples_per_second: number;
  tokens_per_second: number;
  memory_used_gb: number;
  memory_peak_gb: number;
  memory_allocated_gb: number;
  grad_norm: number;
  grad_norm_clipped: boolean;
  elapsed_seconds: number;
  eta_seconds: number;
  loss_history: Array<{ step: number; train_loss: number; val_loss: number }>;
  lr_history: Array<{ step: number; lr: number }>;
  memory_history: Array<{ step: number; used: number; peak: number }>;
  errors: string[];
  warnings: string[];
  status: string;
}

interface DataSample {
  id: string;
  text: string;
  duration: number;
  prosody?: {
    semantic?: {
      emotion?: string;
      tone?: string;
      energy_level?: string;
    };
  };
}

// ============== Utility Functions ==============

function formatTime(seconds: number | undefined | null): string {
  if (seconds === undefined || seconds === null || isNaN(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

function formatNumber(num: number | undefined | null, decimals: number = 2): string {
  if (num === undefined || num === null || isNaN(num)) return "—";
  if (num < 0.001 && num > 0) return num.toExponential(decimals);
  return num.toFixed(decimals);
}

// ============== Components ==============

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    initializing: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    training: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    validating: "bg-blue-500/20 text-blue-400 border-blue-500/30",
    saving: "bg-purple-500/20 text-purple-400 border-purple-500/30",
    error: "bg-red-500/20 text-red-400 border-red-500/30",
    complete: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  };

  const icons: Record<string, string> = {
    initializing: "◌",
    training: "●",
    validating: "◐",
    saving: "↓",
    error: "✕",
    complete: "✓",
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border ${colors[status] || colors.initializing}`}>
      <span className={status === "training" ? "animate-pulse" : ""}>{icons[status] || "○"}</span>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function MetricCard({
  title,
  value,
  subtitle,
  trend,
  icon,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  icon?: string;
}) {
  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">{title}</span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold text-white">{value}</span>
        {trend && (
          <span className={`text-sm ${
            trend === "up" ? "text-red-400" : 
            trend === "down" ? "text-emerald-400" : 
            "text-slate-400"
          }`}>
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}
          </span>
        )}
      </div>
      {subtitle && (
        <span className="text-xs text-slate-500 mt-1 block">{subtitle}</span>
      )}
    </div>
  );
}

function ProgressRing({ progress, size = 80, strokeWidth = 6 }: { progress: number; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          className="text-slate-700"
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className="text-emerald-500 transition-all duration-300"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-bold text-white">{Math.round(progress)}%</span>
      </div>
    </div>
  );
}

function LossChart({ data }: { data?: Array<{ step: number; train_loss: number; val_loss: number }> }) {
  if (!data || data.length < 2) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        Waiting for data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={256}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="trainGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="valGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis 
          dataKey="step" 
          stroke="#64748b" 
          fontSize={11}
          tickFormatter={(v) => v.toLocaleString()}
        />
        <YAxis 
          stroke="#64748b" 
          fontSize={11}
          tickFormatter={(v) => v.toFixed(3)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            fontSize: "12px",
          }}
          labelStyle={{ color: "#94a3b8" }}
        />
        <Legend />
        <Area
          type="monotone"
          dataKey="train_loss"
          name="Train Loss"
          stroke="#3b82f6"
          fill="url(#trainGradient)"
          strokeWidth={2}
        />
        <Area
          type="monotone"
          dataKey="val_loss"
          name="Val Loss"
          stroke="#f59e0b"
          fill="url(#valGradient)"
          strokeWidth={2}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function LearningRateChart({ data }: { data?: Array<{ step: number; lr: number }> }) {
  if (!data || data.length < 2) return null;

  return (
    <ResponsiveContainer width="100%" height={120}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="step" stroke="#64748b" fontSize={10} />
        <YAxis 
          stroke="#64748b" 
          fontSize={10}
          tickFormatter={(v) => v.toExponential(0)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            fontSize: "11px",
          }}
        />
        <Line
          type="monotone"
          dataKey="lr"
          name="Learning Rate"
          stroke="#8b5cf6"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function MemoryChart({ data }: { data?: Array<{ step: number; used: number; peak: number }> }) {
  if (!data || data.length < 2) return null;

  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="memGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis dataKey="step" stroke="#64748b" fontSize={10} />
        <YAxis 
          stroke="#64748b" 
          fontSize={10}
          tickFormatter={(v) => `${v.toFixed(0)}GB`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            fontSize: "11px",
          }}
          formatter={(v: number) => `${v.toFixed(2)} GB`}
        />
        <Area
          type="monotone"
          dataKey="used"
          name="Memory Used"
          stroke="#10b981"
          fill="url(#memGradient)"
          strokeWidth={2}
        />
        <Line
          type="monotone"
          dataKey="peak"
          name="Peak"
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="5 5"
          dot={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function ErrorsPanel({ errors, warnings }: { errors?: string[]; warnings?: string[] }) {
  const errorList = errors || [];
  const warningList = warnings || [];
  if (errorList.length === 0 && warningList.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500">
        <span className="text-2xl">✓</span>
        <p className="mt-2 text-sm">No errors or warnings</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-48 overflow-y-auto">
      {errorList.map((error, i) => (
        <div key={`error-${i}`} className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm">
          <span className="text-red-400 font-medium">Error: </span>
          <span className="text-red-300">{error}</span>
        </div>
      ))}
      {warningList.map((warning, i) => (
        <div key={`warn-${i}`} className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 text-sm">
          <span className="text-yellow-400 font-medium">Warning: </span>
          <span className="text-yellow-300">{warning}</span>
        </div>
      ))}
    </div>
  );
}

function DataViewer({ samples }: { samples: DataSample[] }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<string>("all");

  const emotions = [...new Set(samples.map(s => s.prosody?.semantic?.emotion).filter(Boolean))];
  
  const filtered = samples.filter(s => {
    if (search && !s.text.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter !== "all" && s.prosody?.semantic?.emotion !== filter) return false;
    return true;
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <input
          type="text"
          placeholder="Search samples..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
        />
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/50"
        >
          <option value="all">All emotions</option>
          {emotions.map(e => (
            <option key={e} value={e}>{e}</option>
          ))}
        </select>
      </div>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {filtered.slice(0, 50).map((sample) => (
          <div
            key={sample.id}
            className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-3 hover:border-slate-600 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm text-white flex-1">{sample.text}</p>
              <span className="text-xs text-slate-500 whitespace-nowrap">
                {sample.duration.toFixed(1)}s
              </span>
            </div>
            {sample.prosody?.semantic && (
              <div className="flex gap-2 mt-2">
                {sample.prosody.semantic.emotion && (
                  <span className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded">
                    {sample.prosody.semantic.emotion}
                  </span>
                )}
                {sample.prosody.semantic.tone && (
                  <span className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded">
                    {sample.prosody.semantic.tone}
                  </span>
                )}
                {sample.prosody.semantic.energy_level && (
                  <span className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded">
                    {sample.prosody.semantic.energy_level}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
        {filtered.length > 50 && (
          <p className="text-center text-sm text-slate-500 py-2">
            Showing 50 of {filtered.length} samples
          </p>
        )}
      </div>
    </div>
  );
}

function EmotionDistribution({ data }: { data: Record<string, number> }) {
  const chartData = Object.entries(data).map(([emotion, count]) => ({
    emotion,
    count,
  }));

  if (chartData.length === 0) return null;

  const colors: Record<string, string> = {
    neutral: "#94a3b8",
    happy: "#fbbf24",
    sad: "#60a5fa",
    angry: "#f87171",
    friendly: "#34d399",
    excited: "#fb923c",
    thoughtful: "#818cf8",
    concerned: "#fcd34d",
    confident: "#22d3ee",
  };

  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={chartData} layout="vertical">
        <XAxis type="number" stroke="#64748b" fontSize={10} />
        <YAxis 
          type="category" 
          dataKey="emotion" 
          stroke="#64748b" 
          fontSize={11}
          width={80}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            fontSize: "11px",
          }}
        />
        <Bar 
          dataKey="count" 
          fill="#3b82f6"
          radius={[0, 4, 4, 0]}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ============== Main Dashboard ==============

export default function TrainingDashboard() {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [connected, setConnected] = useState(false);
  const [samples, setSamples] = useState<DataSample[]>([]);
  const [activeTab, setActiveTab] = useState<"overview" | "data" | "logs">("overview");
  const wsRef = useRef<WebSocket | null>(null);

  // Connect to WebSocket
  useEffect(() => {
    const connect = () => {
      // Use env var or default to localhost
      const apiHost = process.env.NEXT_PUBLIC_TRAINING_API || "localhost:8001";
      const ws = new WebSocket(`ws://${apiHost}/ws`);
      
      ws.onopen = () => {
        console.log("Connected to training API");
        setConnected(true);
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setMetrics(data);
      };
      
      ws.onclose = () => {
        setConnected(false);
        // Reconnect after 3 seconds
        setTimeout(connect, 3000);
      };
      
      ws.onerror = () => {
        setConnected(false);
      };
      
      wsRef.current = ws;
    };

    connect();

    // Fallback: Poll for metrics
    const pollInterval = setInterval(async () => {
      if (!connected) {
        try {
          const res = await fetch(`${TRAINING_API}/metrics`);
          if (res.ok) {
            const data = await res.json();
            setMetrics(data);
          }
        } catch (e) {
          // Ignore
        }
      }
    }, 2000);

    // Load sample data
    fetch("http://localhost:8000/samples")
      .then(res => res.json())
      .then(data => setSamples(data.samples || []))
      .catch(() => {});

    return () => {
      clearInterval(pollInterval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connected]);

  const epochProgress = metrics ? (metrics.epoch_progress * 100) : 0;
  const totalProgress = metrics 
    ? ((metrics.epoch - 1 + metrics.epoch_progress) / 50 * 100) // Assuming 50 epochs
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Page Header with Status */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            {metrics && <StatusBadge status={metrics.status} />}
          </div>
          <div className="flex items-center gap-3">
            <span className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-500" : "bg-red-500"}`} />
            <span className="text-xs text-slate-500">
              {connected ? "Live" : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6">
            {(["overview", "data", "logs"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  activeTab === tab
                    ? "bg-slate-800 text-white"
                    : "text-slate-400 hover:text-white hover:bg-slate-800/50"
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
        </div>
        {!metrics ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="w-12 h-12 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mb-4" />
            <p className="text-slate-400">Waiting for training to start...</p>
            <p className="text-xs text-slate-600 mt-2">
              Run: python train_deepseek.py --config config/m4_pro_deepseek.yaml --dashboard
            </p>
          </div>
        ) : activeTab === "overview" ? (
          <div className="space-y-6">
            {/* Progress Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-lg font-semibold">Training Progress</h2>
                    <p className="text-sm text-slate-400">
                      Epoch {metrics.epoch} • Step {metrics.step.toLocaleString()}
                    </p>
                  </div>
                  <ProgressRing progress={totalProgress} size={64} />
                </div>

                {/* Epoch Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">Epoch Progress</span>
                    <span className="text-white font-medium">{epochProgress.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 transition-all duration-300"
                      style={{ width: `${epochProgress}%` }}
                    />
                  </div>
                </div>

                {/* Time Stats */}
                <div className="grid grid-cols-2 gap-4 mt-6">
                  <div>
                    <span className="text-xs text-slate-500 uppercase tracking-wider">Elapsed</span>
                    <p className="text-lg font-semibold text-white">{formatTime(metrics.elapsed_seconds)}</p>
                  </div>
                  <div>
                    <span className="text-xs text-slate-500 uppercase tracking-wider">ETA</span>
                    <p className="text-lg font-semibold text-white">{formatTime(metrics.eta_seconds)}</p>
                  </div>
                </div>
              </div>

              {/* Key Metrics */}
              <div className="space-y-4">
                <MetricCard
                  title="Train Loss"
                  value={formatNumber(metrics.train_loss, 4)}
                  subtitle={`Val: ${formatNumber(metrics.val_loss, 4)}`}
                  trend={metrics.train_loss < (metrics.loss_history[metrics.loss_history.length - 2]?.train_loss || metrics.train_loss) ? "down" : "neutral"}
                  icon="📉"
                />
                <MetricCard
                  title="Learning Rate"
                  value={metrics.learning_rate.toExponential(2)}
                  icon="📈"
                />
                <MetricCard
                  title="Speed"
                  value={`${metrics.samples_per_second.toFixed(1)}/s`}
                  subtitle="samples per second"
                  icon="⚡"
                />
              </div>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Loss Chart */}
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Loss Over Time</h3>
                <LossChart data={metrics.loss_history} />
              </div>

              {/* Memory & LR Charts */}
              <div className="space-y-6">
                <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-slate-300">Memory Usage</h3>
                    <span className="text-xs text-slate-500">
                      {metrics.memory_used_gb.toFixed(1)} / 64 GB
                    </span>
                  </div>
                  <MemoryChart data={metrics.memory_history} />
                </div>

                <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                  <h3 className="text-sm font-semibold text-slate-300 mb-4">Learning Rate Schedule</h3>
                  <LearningRateChart data={metrics.lr_history} />
                </div>
              </div>
            </div>

            {/* Bottom Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Training Stats */}
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Training Stats</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-400">Gradient Norm</span>
                    <span className={`text-sm font-medium ${metrics.grad_norm_clipped ? "text-yellow-400" : "text-white"}`}>
                      {formatNumber(metrics.grad_norm, 4)}
                      {metrics.grad_norm_clipped && " (clipped)"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-400">MTP Loss</span>
                    <span className="text-sm font-medium text-white">{formatNumber(metrics.mtp_loss, 4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-slate-400">Batch Size</span>
                    <span className="text-sm font-medium text-white">12 × 2 = 24</span>
                  </div>
                </div>
              </div>

              {/* DeepSeek Techniques */}
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">DeepSeek Techniques</h3>
                <div className="space-y-2">
                  {[
                    { name: "Multi-Token Prediction", status: true },
                    { name: "DeepSeek LR Schedule", status: true },
                    { name: "Gradient Checkpointing", status: true },
                    { name: "torch.compile (MPS)", status: true },
                  ].map((tech) => (
                    <div key={tech.name} className="flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full ${tech.status ? "bg-emerald-500" : "bg-slate-600"}`} />
                      <span className="text-sm text-slate-300">{tech.name}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Errors & Warnings */}
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Errors & Warnings</h3>
                <ErrorsPanel errors={metrics.errors} warnings={metrics.warnings} />
              </div>
            </div>
          </div>
        ) : activeTab === "data" ? (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Stats */}
              <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Dataset Stats</h3>
                <div className="space-y-4">
                  <MetricCard
                    title="Total Samples"
                    value={samples.length}
                    icon="📊"
                  />
                  <MetricCard
                    title="Total Duration"
                    value={`${(samples.reduce((a, s) => a + s.duration, 0) / 60).toFixed(1)} min`}
                    icon="⏱️"
                  />
                </div>
              </div>

              {/* Emotion Distribution */}
              <div className="lg:col-span-2 bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
                <h3 className="text-sm font-semibold text-slate-300 mb-4">Emotion Distribution</h3>
                <EmotionDistribution
                  data={samples.reduce((acc, s) => {
                    const e = s.prosody?.semantic?.emotion;
                    if (e) acc[e] = (acc[e] || 0) + 1;
                    return acc;
                  }, {} as Record<string, number>)}
                />
              </div>
            </div>

            {/* Sample Browser */}
            <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
              <h3 className="text-sm font-semibold text-slate-300 mb-4">Training Samples</h3>
              <DataViewer samples={samples} />
            </div>
          </div>
        ) : (
          <div className="bg-slate-800/30 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50">
            <h3 className="text-sm font-semibold text-slate-300 mb-4">Training Logs</h3>
            <div className="font-mono text-xs space-y-1 max-h-[600px] overflow-y-auto bg-slate-900 rounded-lg p-4">
              {metrics?.errors.map((e, i) => (
                <div key={`e-${i}`} className="text-red-400">{e}</div>
              ))}
              {metrics?.warnings.map((w, i) => (
                <div key={`w-${i}`} className="text-yellow-400">{w}</div>
              ))}
              <div className="text-slate-500">
                [Step {metrics?.step}] Loss: {metrics?.train_loss.toFixed(4)} | LR: {metrics?.learning_rate.toExponential(2)} | Mem: {metrics?.memory_used_gb.toFixed(1)}GB
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
