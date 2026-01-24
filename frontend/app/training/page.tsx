"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

const TRAINING_API = process.env.NEXT_PUBLIC_TRAINING_API || "http://localhost:8001";
const BACKEND_API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";

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

interface VoiceDataStats {
  has_training_data: boolean;
  manifest_path?: string;
  sample_count?: number;
  total_duration_seconds?: number;
  total_duration_minutes?: number;
  emotion_distribution?: Record<string, number>;
  samples_preview?: DataSample[];
  sessions_available?: Array<{ session_id: string; recording_count: number }>;
  message?: string;
}

interface TrainingStatus {
  status: string;
  pid?: number;
  log_tail?: string;
  exit_code?: number;
  message?: string;
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

// ============== Section Component ==============

function Section({
  title,
  defaultOpen = false,
  badge,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-foreground-bright hover:text-foreground transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm">{title}</span>
          {badge}
        </div>
        <span className="text-muted-foreground">{isOpen ? "-" : "+"}</span>
      </button>
      {isOpen && <div className="px-4 pb-4 animate-fade-in">{children}</div>}
    </div>
  );
}

// ============== StatRow Component ==============

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

// ============== Components ==============

function StatusBadge({ status }: { status: string }) {
  const icons: Record<string, string> = {
    initializing: "○",
    training: "●",
    validating: "◐",
    saving: "↓",
    error: "×",
    complete: "✓",
  };

  const isActive = status === "training" || status === "validating";

  return (
    <span className={`inline-flex items-center gap-1.5 text-xs ${isActive ? "text-foreground-bright" : "text-muted-foreground"}`}>
      <span className={status === "training" ? "animate-pulse" : ""}>{icons[status] || "○"}</span>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function ProgressRing({ progress, size = 64, strokeWidth = 4 }: { progress: number; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg className="transform -rotate-90" width={size} height={size}>
        <circle
          className="text-border"
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className="text-foreground transition-all duration-300"
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
        <span className="text-xs text-foreground-bright">{Math.round(progress)}%</span>
      </div>
    </div>
  );
}

function LossChart({ data }: { data?: Array<{ step: number; train_loss: number; val_loss: number }> }) {
  if (!data || data.length < 2) {
    return (
      <div className="h-48 flex items-center justify-center text-muted-foreground text-sm">
        Waiting for data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={192}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis
          dataKey="step"
          stroke="hsl(var(--muted-foreground))"
          fontSize={10}
          tickFormatter={(v) => v.toLocaleString()}
        />
        <YAxis
          stroke="hsl(var(--muted-foreground))"
          fontSize={10}
          tickFormatter={(v) => v.toFixed(3)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--background-elevated))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "4px",
            fontSize: "11px",
          }}
          labelStyle={{ color: "hsl(var(--foreground))" }}
        />
        <Area
          type="monotone"
          dataKey="train_loss"
          name="Train"
          stroke="hsl(var(--foreground))"
          fill="hsl(var(--foreground))"
          fillOpacity={0.1}
          strokeWidth={1}
        />
        <Area
          type="monotone"
          dataKey="val_loss"
          name="Val"
          stroke="hsl(var(--muted-foreground))"
          fill="hsl(var(--muted-foreground))"
          fillOpacity={0.05}
          strokeWidth={1}
          strokeDasharray="3 3"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function LearningRateChart({ data }: { data?: Array<{ step: number; lr: number }> }) {
  if (!data || data.length < 2) return null;

  return (
    <ResponsiveContainer width="100%" height={100}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={9} />
        <YAxis
          stroke="hsl(var(--muted-foreground))"
          fontSize={9}
          tickFormatter={(v) => v.toExponential(0)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--background-elevated))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "4px",
            fontSize: "10px",
          }}
        />
        <Line
          type="monotone"
          dataKey="lr"
          name="LR"
          stroke="hsl(var(--foreground))"
          strokeWidth={1}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

function MemoryChart({ data }: { data?: Array<{ step: number; used: number; peak: number }> }) {
  if (!data || data.length < 2) return null;

  return (
    <ResponsiveContainer width="100%" height={100}>
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
        <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={9} />
        <YAxis
          stroke="hsl(var(--muted-foreground))"
          fontSize={9}
          tickFormatter={(v) => `${v.toFixed(0)}GB`}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--background-elevated))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "4px",
            fontSize: "10px",
          }}
          formatter={(v: number) => `${v.toFixed(2)} GB`}
        />
        <Area
          type="monotone"
          dataKey="used"
          name="Used"
          stroke="hsl(var(--foreground))"
          fill="hsl(var(--foreground))"
          fillOpacity={0.1}
          strokeWidth={1}
        />
        <Line
          type="monotone"
          dataKey="peak"
          name="Peak"
          stroke="hsl(var(--muted-foreground))"
          strokeWidth={1}
          strokeDasharray="3 3"
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
      <div className="text-center py-4 text-muted-foreground">
        <span className="text-lg">✓</span>
        <p className="mt-1 text-xs">No errors or warnings</p>
      </div>
    );
  }

  return (
    <div className="space-y-2 max-h-32 overflow-y-auto">
      {errorList.map((error, i) => (
        <div key={`error-${i}`} className="text-xs p-2 border border-border rounded">
          <span className="text-foreground-bright">Error: </span>
          <span className="text-foreground">{error}</span>
        </div>
      ))}
      {warningList.map((warning, i) => (
        <div key={`warn-${i}`} className="text-xs p-2 border border-border rounded">
          <span className="text-muted-foreground">Warning: </span>
          <span className="text-foreground">{warning}</span>
        </div>
      ))}
    </div>
  );
}

function DataViewer({ samples }: { samples: DataSample[] }) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<string>("all");

  const emotions = Array.from(new Set(samples.map(s => s.prosody?.semantic?.emotion).filter(Boolean)));

  const filtered = samples.filter(s => {
    if (search && !s.text.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter !== "all" && s.prosody?.semantic?.emotion !== filter) return false;
    return true;
  });

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <input
          type="text"
          placeholder="Search samples..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 bg-background border border-border rounded px-3 py-1.5 text-sm text-foreground placeholder-muted-foreground focus:outline-none focus:border-foreground"
        />
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-background border border-border rounded px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-foreground"
        >
          <option value="all">All</option>
          {emotions.map(e => (
            <option key={e} value={e}>{e}</option>
          ))}
        </select>
      </div>

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {filtered.slice(0, 50).map((sample) => (
          <div
            key={sample.id}
            className="border border-border rounded p-3 hover:border-foreground/30 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              <p className="text-sm text-foreground flex-1">{sample.text}</p>
              <span className="text-xs text-muted-foreground whitespace-nowrap">
                {sample.duration.toFixed(1)}s
              </span>
            </div>
            {sample.prosody?.semantic && (
              <div className="flex gap-2 mt-2">
                {sample.prosody.semantic.emotion && (
                  <span className="text-xs text-muted-foreground">
                    {sample.prosody.semantic.emotion}
                  </span>
                )}
                {sample.prosody.semantic.tone && (
                  <span className="text-xs text-foreground-subtle">
                    {sample.prosody.semantic.tone}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
        {filtered.length > 50 && (
          <p className="text-center text-xs text-muted-foreground py-2">
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

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={chartData} layout="vertical">
        <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={10} />
        <YAxis
          type="category"
          dataKey="emotion"
          stroke="hsl(var(--muted-foreground))"
          fontSize={10}
          width={60}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--background-elevated))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "4px",
            fontSize: "10px",
          }}
        />
        <Bar
          dataKey="count"
          fill="hsl(var(--foreground))"
          fillOpacity={0.6}
          radius={[0, 2, 2, 0]}
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

  // Voice data and training state
  const [voiceStats, setVoiceStats] = useState<VoiceDataStats | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 50,
    batchSize: 4,
    config: "m4_pro.yaml",
  });

  // Connect to WebSocket
  useEffect(() => {
    const connect = () => {
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
        setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        setConnected(false);
      };

      wsRef.current = ws;
    };

    connect();

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

    fetch(`${BACKEND_API}/samples`)
      .then(res => res.json())
      .then(data => setSamples(data.samples || []))
      .catch(() => {});

    const fetchVoiceStats = async () => {
      try {
        const res = await fetch(`${BACKEND_API}/training/data-stats`);
        if (res.ok) {
          const data = await res.json();
          setVoiceStats(data);
        }
      } catch (e) {
        console.error("Failed to fetch voice stats:", e);
      }
    };
    fetchVoiceStats();

    const statusInterval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_API}/training/status`);
        if (res.ok) {
          const data = await res.json();
          setTrainingStatus(data);
        }
      } catch (e) {
        // Ignore
      }
    }, 3000);

    return () => {
      clearInterval(statusInterval);
      clearInterval(pollInterval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connected]);

  const epochProgress = metrics ? (metrics.epoch_progress * 100) : 0;
  const totalProgress = metrics
    ? ((metrics.epoch - 1 + metrics.epoch_progress) / 50 * 100)
    : 0;

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Status & Config */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        {/* Connection Status */}
        <Section title="Status" defaultOpen>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Connection</span>
              <div className="flex items-center gap-2">
                <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-foreground" : "bg-muted-foreground"}`} />
                <span className="text-xs text-foreground">{connected ? "Live" : "Disconnected"}</span>
              </div>
            </div>
            {metrics && (
              <>
                <StatRow label="Status" value={metrics.status} />
                <StatRow label="Epoch" value={`${metrics.epoch} / 50`} />
                <StatRow label="Step" value={metrics.step.toLocaleString()} />
              </>
            )}
          </div>
        </Section>

        {/* Training Config */}
        <Section title="Configuration" defaultOpen={!metrics}>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Epochs</label>
              <input
                type="number"
                value={trainingConfig.epochs}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) || 50 }))}
                className="w-full bg-background border border-border rounded px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-foreground"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Batch Size</label>
              <input
                type="number"
                value={trainingConfig.batchSize}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) || 4 }))}
                className="w-full bg-background border border-border rounded px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-foreground"
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Config</label>
              <select
                value={trainingConfig.config}
                onChange={(e) => setTrainingConfig(prev => ({ ...prev, config: e.target.value }))}
                className="w-full bg-background border border-border rounded px-3 py-1.5 text-sm text-foreground focus:outline-none focus:border-foreground"
              >
                <option value="m4_pro.yaml">M4 Pro (Mac)</option>
                <option value="rtx_4090.yaml">RTX 4090</option>
              </select>
            </div>
          </div>
        </Section>

        {/* DeepSeek Techniques */}
        <Section title="DeepSeek Techniques">
          <div className="space-y-1">
            {[
              { name: "Multi-Token Prediction", status: true },
              { name: "DeepSeek LR Schedule", status: true },
              { name: "Gradient Checkpointing", status: true },
              { name: "torch.compile (MPS)", status: true },
            ].map((tech) => (
              <div key={tech.name} className="flex items-center gap-2 py-1">
                <span className={`w-1.5 h-1.5 rounded-full ${tech.status ? "bg-foreground" : "bg-muted-foreground"}`} />
                <span className="text-xs text-foreground">{tech.name}</span>
              </div>
            ))}
          </div>
        </Section>

        {/* Voice Data Stats */}
        {voiceStats && voiceStats.has_training_data && (
          <Section title="Training Data">
            <div className="space-y-2">
              <StatRow label="Samples" value={voiceStats.sample_count || 0} />
              <StatRow label="Duration" value={`${(voiceStats.total_duration_minutes || 0).toFixed(1)} min`} />
              {voiceStats.emotion_distribution && (
                <div className="pt-2">
                  <div className="text-xs text-muted-foreground mb-2">Emotion Distribution</div>
                  <EmotionDistribution data={voiceStats.emotion_distribution} />
                </div>
              )}
            </div>
          </Section>
        )}
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
        <div className="p-6">
          {/* Tabs */}
          <div className="flex gap-px mb-6 border border-border rounded overflow-hidden w-fit">
            {(["overview", "data", "logs"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 text-sm transition-colors ${
                  activeTab === tab
                    ? "bg-foreground text-background"
                    : "bg-background-elevated text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {!metrics && trainingStatus?.status !== "running" ? (
            /* No Training Running - Show Setup */
            <div className="space-y-6">
              <div className="border border-border rounded p-6">
                <div className="text-center mb-6">
                  <h2 className="text-lg text-foreground-bright mb-1">Train Your Voice Model</h2>
                  <p className="text-sm text-muted-foreground">Fine-tune CSM-1B on your recorded voice samples</p>
                </div>

                {voiceStats ? (
                  voiceStats.has_training_data ? (
                    <div className="space-y-4">
                      <div className="border border-foreground/20 rounded p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-foreground-bright">✓</span>
                          <span className="text-sm text-foreground-bright">Training Data Ready</span>
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {voiceStats.sample_count} samples | {voiceStats.total_duration_minutes?.toFixed(1)} minutes
                        </p>
                      </div>

                      <button
                        onClick={async () => {
                          setIsStarting(true);
                          try {
                            const res = await fetch(`${BACKEND_API}/training/start?config=${trainingConfig.config}&epochs=${trainingConfig.epochs}&batch_size=${trainingConfig.batchSize}`, {
                              method: "POST",
                            });
                            const data = await res.json();
                            if (res.ok) {
                              setTrainingStatus(data);
                            } else {
                              alert(data.detail || "Failed to start training");
                            }
                          } catch (e) {
                            alert("Failed to start training");
                          } finally {
                            setIsStarting(false);
                          }
                        }}
                        disabled={isStarting}
                        className="w-full py-3 bg-foreground text-background rounded hover:bg-foreground-bright transition-colors disabled:opacity-50"
                      >
                        {isStarting ? "Starting Training..." : "Start Training"}
                      </button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="border border-border rounded p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-muted-foreground">○</span>
                          <span className="text-sm text-foreground">No Training Data</span>
                        </div>
                        <p className="text-xs text-muted-foreground">{voiceStats.message}</p>
                      </div>

                      {voiceStats.sessions_available && voiceStats.sessions_available.length > 0 && (
                        <>
                          <div className="space-y-2">
                            {voiceStats.sessions_available.map((session) => (
                              <div key={session.session_id} className="flex items-center justify-between border border-border rounded p-3">
                                <span className="text-sm text-foreground">{session.session_id}</span>
                                <span className="text-xs text-muted-foreground">{session.recording_count} recordings</span>
                              </div>
                            ))}
                          </div>

                          <button
                            onClick={async () => {
                              setIsPreparing(true);
                              try {
                                const res = await fetch(`${BACKEND_API}/training/prepare`, {
                                  method: "POST",
                                });
                                const data = await res.json();
                                if (res.ok) {
                                  const statsRes = await fetch(`${BACKEND_API}/training/data-stats`);
                                  if (statsRes.ok) {
                                    setVoiceStats(await statsRes.json());
                                  }
                                } else {
                                  alert(data.detail || "Failed to prepare training data");
                                }
                              } catch (e) {
                                alert("Failed to prepare training data");
                              } finally {
                                setIsPreparing(false);
                              }
                            }}
                            disabled={isPreparing}
                            className="w-full py-3 border border-border text-foreground rounded hover:bg-accent transition-colors disabled:opacity-50"
                          >
                            {isPreparing ? "Preparing..." : "Prepare Training Data"}
                          </button>
                        </>
                      )}

                      <div className="text-center pt-2">
                        <p className="text-xs text-muted-foreground">
                          Go to the <a href="/perform" className="text-foreground hover:text-foreground-bright underline">Perform page</a> to record voice samples first.
                        </p>
                      </div>
                    </div>
                  )
                ) : (
                  <div className="flex justify-center py-8">
                    <div className="w-6 h-6 border-2 border-border border-t-foreground rounded-full animate-spin" />
                  </div>
                )}
              </div>

              {/* Training Status */}
              {trainingStatus && trainingStatus.status !== "not_started" && (
                <div className="border border-border rounded p-6">
                  <h3 className="text-sm text-foreground-bright mb-4">Training Status</h3>
                  <div className="flex items-center gap-3 mb-4">
                    <span className={`w-2 h-2 rounded-full ${
                      trainingStatus.status === "running" ? "bg-foreground animate-pulse" :
                      trainingStatus.status === "completed" ? "bg-foreground" :
                      trainingStatus.status === "failed" ? "bg-muted-foreground" : "bg-muted-foreground"
                    }`} />
                    <span className="text-sm text-foreground capitalize">{trainingStatus.status}</span>
                    {trainingStatus.pid && <span className="text-xs text-muted-foreground">PID: {trainingStatus.pid}</span>}
                  </div>

                  {trainingStatus.log_tail && (
                    <pre className="bg-background rounded p-4 text-xs text-foreground overflow-auto max-h-48 font-mono border border-border">
                      {trainingStatus.log_tail}
                    </pre>
                  )}

                  {trainingStatus.status === "running" && (
                    <button
                      onClick={async () => {
                        if (confirm("Are you sure you want to stop training?")) {
                          await fetch(`${BACKEND_API}/training/stop`, { method: "POST" });
                        }
                      }}
                      className="mt-4 px-4 py-2 border border-border text-foreground rounded hover:bg-accent transition-colors"
                    >
                      Stop Training
                    </button>
                  )}
                </div>
              )}
            </div>
          ) : activeTab === "overview" ? (
            /* Training Overview */
            <div className="space-y-6">
              {/* Progress Section */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 border border-border rounded p-6">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-sm text-foreground-bright">Training Progress</h2>
                      <p className="text-xs text-muted-foreground mt-1">
                        Epoch {metrics?.epoch} | Step {metrics?.step.toLocaleString()}
                      </p>
                    </div>
                    {metrics && <ProgressRing progress={totalProgress} />}
                  </div>

                  {/* Epoch Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-muted-foreground">Epoch Progress</span>
                      <span className="text-foreground">{epochProgress.toFixed(1)}%</span>
                    </div>
                    <div className="h-1 bg-border rounded-full overflow-hidden">
                      <div
                        className="h-full bg-foreground transition-all duration-300"
                        style={{ width: `${epochProgress}%` }}
                      />
                    </div>
                  </div>

                  {/* Time Stats */}
                  {metrics && (
                    <div className="grid grid-cols-2 gap-4 mt-6">
                      <div>
                        <span className="text-xs text-muted-foreground">Elapsed</span>
                        <p className="text-sm text-foreground-bright">{formatTime(metrics.elapsed_seconds)}</p>
                      </div>
                      <div>
                        <span className="text-xs text-muted-foreground">ETA</span>
                        <p className="text-sm text-foreground-bright">{formatTime(metrics.eta_seconds)}</p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Key Metrics */}
                <div className="space-y-4">
                  <div className="border border-border rounded p-4">
                    <div className="text-xs text-muted-foreground mb-1">Train Loss</div>
                    <div className="text-lg text-foreground-bright">{formatNumber(metrics?.train_loss, 4)}</div>
                    <div className="text-xs text-muted-foreground mt-1">Val: {formatNumber(metrics?.val_loss, 4)}</div>
                  </div>
                  <div className="border border-border rounded p-4">
                    <div className="text-xs text-muted-foreground mb-1">Learning Rate</div>
                    <div className="text-lg text-foreground-bright">{metrics?.learning_rate.toExponential(2)}</div>
                  </div>
                  <div className="border border-border rounded p-4">
                    <div className="text-xs text-muted-foreground mb-1">Speed</div>
                    <div className="text-lg text-foreground-bright">{metrics?.samples_per_second.toFixed(1)}/s</div>
                    <div className="text-xs text-muted-foreground mt-1">samples per second</div>
                  </div>
                </div>
              </div>

              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Loss Chart */}
                <div className="border border-border rounded p-6">
                  <h3 className="text-xs text-muted-foreground mb-4">Loss Over Time</h3>
                  <LossChart data={metrics?.loss_history} />
                </div>

                {/* Memory & LR Charts */}
                <div className="space-y-6">
                  <div className="border border-border rounded p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-xs text-muted-foreground">Memory Usage</h3>
                      <span className="text-xs text-foreground">
                        {metrics?.memory_used_gb.toFixed(1)} / 64 GB
                      </span>
                    </div>
                    <MemoryChart data={metrics?.memory_history} />
                  </div>

                  <div className="border border-border rounded p-6">
                    <h3 className="text-xs text-muted-foreground mb-4">Learning Rate Schedule</h3>
                    <LearningRateChart data={metrics?.lr_history} />
                  </div>
                </div>
              </div>

              {/* Bottom Row */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Training Stats */}
                <div className="border border-border rounded p-6">
                  <h3 className="text-xs text-muted-foreground mb-4">Training Stats</h3>
                  <div className="space-y-2">
                    <StatRow
                      label="Gradient Norm"
                      value={`${formatNumber(metrics?.grad_norm, 4)}${metrics?.grad_norm_clipped ? " (clipped)" : ""}`}
                    />
                    <StatRow label="MTP Loss" value={formatNumber(metrics?.mtp_loss, 4)} />
                    <StatRow label="Batch Size" value="12 x 2 = 24" />
                  </div>
                </div>

                {/* Errors & Warnings */}
                <div className="lg:col-span-2 border border-border rounded p-6">
                  <h3 className="text-xs text-muted-foreground mb-4">Errors & Warnings</h3>
                  <ErrorsPanel errors={metrics?.errors} warnings={metrics?.warnings} />
                </div>
              </div>
            </div>
          ) : activeTab === "data" ? (
            /* Data Tab */
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Stats */}
                <div className="border border-border rounded p-6">
                  <h3 className="text-xs text-muted-foreground mb-4">Dataset Stats</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Total Samples</div>
                      <div className="text-lg text-foreground-bright">{samples.length}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted-foreground mb-1">Total Duration</div>
                      <div className="text-lg text-foreground-bright">
                        {(samples.reduce((a, s) => a + s.duration, 0) / 60).toFixed(1)} min
                      </div>
                    </div>
                  </div>
                </div>

                {/* Emotion Distribution */}
                <div className="lg:col-span-2 border border-border rounded p-6">
                  <h3 className="text-xs text-muted-foreground mb-4">Emotion Distribution</h3>
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
              <div className="border border-border rounded p-6">
                <h3 className="text-xs text-muted-foreground mb-4">Training Samples</h3>
                <DataViewer samples={samples} />
              </div>
            </div>
          ) : (
            /* Logs Tab */
            <div className="border border-border rounded p-6">
              <h3 className="text-xs text-muted-foreground mb-4">Training Logs</h3>
              <div className="font-mono text-xs space-y-1 max-h-[500px] overflow-y-auto bg-background rounded p-4 border border-border">
                {metrics?.errors.map((e, i) => (
                  <div key={`e-${i}`} className="text-foreground">[ERROR] {e}</div>
                ))}
                {metrics?.warnings.map((w, i) => (
                  <div key={`w-${i}`} className="text-muted-foreground">[WARN] {w}</div>
                ))}
                <div className="text-muted-foreground">
                  [Step {metrics?.step}] Loss: {metrics?.train_loss.toFixed(4)} | LR: {metrics?.learning_rate.toExponential(2)} | Mem: {metrics?.memory_used_gb.toFixed(1)}GB
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Right Sidebar - Quick Stats */}
      <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Live Metrics" defaultOpen>
          {metrics ? (
            <div className="space-y-2">
              <StatRow label="Train Loss" value={formatNumber(metrics.train_loss, 4)} />
              <StatRow label="Val Loss" value={formatNumber(metrics.val_loss, 4)} />
              <StatRow label="LR" value={metrics.learning_rate.toExponential(2)} />
              <StatRow label="Speed" value={`${metrics.samples_per_second.toFixed(1)}/s`} />
              <StatRow label="Memory" value={`${metrics.memory_used_gb.toFixed(1)} GB`} />
            </div>
          ) : (
            <div className="text-xs text-muted-foreground text-center py-4">
              No training in progress
            </div>
          )}
        </Section>

        <Section title="Hardware">
          <div className="space-y-2">
            <StatRow label="Training" value="RTX 4090" />
            <StatRow label="Inference" value="M4 Pro" />
            <StatRow label="VRAM" value="24GB" />
          </div>
        </Section>

        <Section title="Actions">
          <div className="space-y-2">
            {trainingStatus?.status === "running" && (
              <button
                onClick={async () => {
                  if (confirm("Stop training?")) {
                    await fetch(`${BACKEND_API}/training/stop`, { method: "POST" });
                  }
                }}
                className="w-full py-2 text-sm border border-border text-foreground rounded hover:bg-accent transition-colors"
              >
                Stop Training
              </button>
            )}
            <a
              href="/perform"
              className="block w-full py-2 text-sm text-center border border-border text-foreground rounded hover:bg-accent transition-colors"
            >
              Record More Samples
            </a>
          </div>
        </Section>
      </aside>
    </div>
  );
}
