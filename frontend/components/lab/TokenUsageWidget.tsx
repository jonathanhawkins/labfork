"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Zap,
  DollarSign,
  PiggyBank,
  RefreshCw,
  TrendingUp,
  Clock,
} from "lucide-react";

interface TokenUsage {
  service: string;
  name: string;
  description: string;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  cost: number;
  sessions: number;
  totalMinutes: number;
}

interface HypotheticalCost {
  codex: number;
  opus: number;
}

interface TokenUsageData {
  services: {
    ollama: TokenUsage;
  };
  totals: {
    totalTokens: number;
    totalCost: number;
    hypotheticalCost: HypotheticalCost;
  };
  timeRange: {
    start: string;
    end: string;
    days: number;
  };
  lastUpdated: string;
}

function formatTokens(tokens: number): string {
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(2)}M`;
  }
  if (tokens >= 1_000) {
    return `${(tokens / 1_000).toFixed(1)}K`;
  }
  return tokens.toString();
}

function formatMinutes(minutes: number): string {
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  }
  return `${Math.round(minutes)}m`;
}

function formatCost(cost: number): string {
  if (cost < 0.01) return "$0.00";
  return `$${cost.toFixed(2)}`;
}

export function TokenUsageWidget() {
  const [data, setData] = useState<TokenUsageData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/lab/token-usage");
      if (!response.ok) {
        throw new Error("Failed to fetch token usage");
      }
      const result = await response.json();
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Poll every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (error) {
    return (
      <div className="text-xs text-muted-foreground text-center py-4">
        Failed to load token usage
      </div>
    );
  }

  if (!data && loading) {
    return (
      <div className="text-xs text-muted-foreground text-center py-4">
        Loading token usage...
      </div>
    );
  }

  if (!data) {
    return (
      <div className="text-xs text-muted-foreground text-center py-4">
        No usage data available
      </div>
    );
  }

  const hasUsage = data.totals.totalTokens > 0;
  const ollama = data.services.ollama;

  return (
    <div className="space-y-3">
      {/* Header with refresh */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-muted-foreground" />
          <span className="text-xs text-muted-foreground">
            Last {data.timeRange.days} day{data.timeRange.days !== 1 ? "s" : ""}
          </span>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="p-1 rounded hover:bg-accent transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-3 h-3 text-muted-foreground ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {/* Ollama usage - always FREE */}
      {hasUsage ? (
        <div className="border border-border rounded">
          <div className="p-2.5 border-b border-border">
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded bg-purple-500/20 text-purple-400">
                  <Zap className="w-3.5 h-3.5" />
                </div>
                <div>
                  <div className="text-sm text-foreground-bright">{ollama.name}</div>
                  <div className="text-[10px] text-muted-foreground">{ollama.description}</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-mono text-green-500">FREE</div>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{formatTokens(ollama.totalTokens)} tokens</span>
              <span>{ollama.sessions} sessions</span>
              <span>{formatMinutes(ollama.totalMinutes)}</span>
            </div>
          </div>

          {/* Hypothetical costs - what it WOULD have cost */}
          <div className="p-2.5 space-y-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wide">
              Would have cost with:
            </div>

            {/* OpenAI Codex */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                <span className="text-xs text-muted-foreground">OpenAI Codex</span>
                <span className="text-[10px] text-foreground-subtle">$1.50/$6 per M</span>
              </div>
              <span className="text-xs font-mono text-orange-400">
                {formatCost(data.totals.hypotheticalCost.codex)}
              </span>
            </div>

            {/* Claude Opus 4.5 */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                <span className="text-xs text-muted-foreground">Claude Opus 4.5</span>
                <span className="text-[10px] text-foreground-subtle">$5/$25 per M</span>
              </div>
              <span className="text-xs font-mono text-blue-400">
                {formatCost(data.totals.hypotheticalCost.opus)}
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground text-center py-4 border border-border rounded">
          No AI usage recorded yet
        </div>
      )}

      {/* Totals and Savings */}
      {hasUsage && (
        <div className="space-y-2">
          {/* Total Cost */}
          <div className="flex items-center justify-between p-2.5 border border-border rounded">
            <div className="flex items-center gap-2">
              <DollarSign className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Total Cost</span>
            </div>
            <span className="text-sm font-mono text-green-500">FREE</span>
          </div>

          {/* Savings */}
          <div className="flex items-center justify-between p-2.5 border border-green-500/30 bg-green-500/5 rounded">
            <div className="flex items-center gap-2">
              <PiggyBank className="w-4 h-4 text-green-500" />
              <div>
                <span className="text-sm text-green-500">Saved vs Opus</span>
                <div className="text-[10px] text-muted-foreground">
                  Using local 4090
                </div>
              </div>
            </div>
            <span className="text-sm font-mono text-green-500">
              {formatCost(data.totals.hypotheticalCost.opus)}
            </span>
          </div>

          {/* Total tokens */}
          <div className="flex items-center justify-between text-xs text-muted-foreground px-1">
            <span>Total tokens processed</span>
            <span className="font-mono">{formatTokens(data.totals.totalTokens)}</span>
          </div>
        </div>
      )}

      {/* Last updated */}
      <div className="flex items-center justify-center gap-1 text-[10px] text-foreground-subtle">
        <Clock className="w-3 h-3" />
        <span>Updated {new Date(data.lastUpdated).toLocaleTimeString()}</span>
      </div>
    </div>
  );
}
