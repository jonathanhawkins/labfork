"use client";

/**
 * EvolutionViewer Component
 *
 * Real-time visualization of evolution progress with generation stats,
 * fitness trends, and population diversity.
 */

import React, { useState, useEffect, useCallback } from "react";

interface GenerationData {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  worstFitness: number;
  diversity: number;
  mutationCount: number;
  crossoverCount: number;
  eliteCount: number;
  timestamp: Date;
}

interface ChromosomePreview {
  id: string;
  name: string;
  fitness: number;
  isElite: boolean;
}

interface EvolutionViewerProps {
  populationId?: string;
  populationName?: string;
  generationHistory: GenerationData[];
  currentGeneration: number;
  status: "idle" | "running" | "paused" | "completed" | "converged";
  topChromosomes?: ChromosomePreview[];
  onPause?: () => void;
  onResume?: () => void;
  onStop?: () => void;
  className?: string;
}

export function EvolutionViewer({
  populationName = "Evolution",
  generationHistory,
  currentGeneration,
  status,
  topChromosomes = [],
  onPause,
  onResume,
  onStop,
  className = "",
}: EvolutionViewerProps) {
  const [selectedMetric, setSelectedMetric] = useState<"fitness" | "diversity" | "operators">("fitness");

  // Get latest stats
  const latestStats = generationHistory[generationHistory.length - 1];
  const improvement = generationHistory.length >= 2
    ? latestStats.bestFitness - generationHistory[0].bestFitness
    : 0;

  return (
    <div className={`bg-white border rounded-lg shadow-sm ${className}`}>
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between">
        <div>
          <h3 className="font-semibold text-lg">{populationName}</h3>
          <p className="text-sm text-gray-500">
            Generation {currentGeneration} | {status}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <StatusBadge status={status} />
          {status === "running" && onPause && (
            <button
              onClick={onPause}
              className="px-3 py-1 text-sm bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200"
            >
              Pause
            </button>
          )}
          {status === "paused" && onResume && (
            <button
              onClick={onResume}
              className="px-3 py-1 text-sm bg-green-100 text-green-700 rounded hover:bg-green-200"
            >
              Resume
            </button>
          )}
          {(status === "running" || status === "paused") && onStop && (
            <button
              onClick={onStop}
              className="px-3 py-1 text-sm bg-red-100 text-red-700 rounded hover:bg-red-200"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-4 gap-4 p-4 bg-gray-50">
        <StatCard
          label="Best Fitness"
          value={latestStats ? `${(latestStats.bestFitness * 100).toFixed(1)}%` : "—"}
          trend={improvement > 0 ? "up" : improvement < 0 ? "down" : "neutral"}
        />
        <StatCard
          label="Average Fitness"
          value={latestStats ? `${(latestStats.averageFitness * 100).toFixed(1)}%` : "—"}
        />
        <StatCard
          label="Diversity"
          value={latestStats ? `${(latestStats.diversity * 100).toFixed(1)}%` : "—"}
        />
        <StatCard
          label="Improvement"
          value={`${improvement >= 0 ? "+" : ""}${(improvement * 100).toFixed(1)}%`}
          trend={improvement > 0 ? "up" : improvement < 0 ? "down" : "neutral"}
        />
      </div>

      {/* Metric Selector */}
      <div className="px-4 pt-4 flex gap-2">
        <MetricButton
          label="Fitness"
          selected={selectedMetric === "fitness"}
          onClick={() => setSelectedMetric("fitness")}
        />
        <MetricButton
          label="Diversity"
          selected={selectedMetric === "diversity"}
          onClick={() => setSelectedMetric("diversity")}
        />
        <MetricButton
          label="Operators"
          selected={selectedMetric === "operators"}
          onClick={() => setSelectedMetric("operators")}
        />
      </div>

      {/* Chart */}
      <div className="p-4">
        <EvolutionChart
          data={generationHistory}
          metric={selectedMetric}
          height={200}
        />
      </div>

      {/* Top Chromosomes */}
      {topChromosomes.length > 0 && (
        <div className="p-4 border-t">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Top Performers</h4>
          <div className="space-y-2">
            {topChromosomes.slice(0, 5).map((chr, idx) => (
              <div
                key={chr.id}
                className="flex items-center justify-between text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 flex items-center justify-center bg-gray-100 rounded text-xs font-medium">
                    {idx + 1}
                  </span>
                  <span className="truncate max-w-[150px]">{chr.name}</span>
                  {chr.isElite && (
                    <span className="px-1.5 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded">
                      Elite
                    </span>
                  )}
                </div>
                <span className="font-mono">
                  {(chr.fitness * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Generation Log */}
      <div className="p-4 border-t max-h-32 overflow-y-auto">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Recent Activity</h4>
        <div className="space-y-1 text-xs text-gray-600">
          {generationHistory.slice(-5).reverse().map((gen) => (
            <div key={gen.generation} className="flex justify-between">
              <span>Gen {gen.generation}</span>
              <span>
                Best: {(gen.bestFitness * 100).toFixed(1)}% |
                Mutations: {gen.mutationCount} |
                Crossovers: {gen.crossoverCount}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Sub-components

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    idle: "bg-gray-100 text-gray-700",
    running: "bg-blue-100 text-blue-700",
    paused: "bg-yellow-100 text-yellow-700",
    completed: "bg-green-100 text-green-700",
    converged: "bg-purple-100 text-purple-700",
  };

  return (
    <span className={`px-2 py-1 text-xs rounded-full ${colors[status] || colors.idle}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function StatCard({
  label,
  value,
  trend,
}: {
  label: string;
  value: string;
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <div className="text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="font-semibold flex items-center justify-center gap-1">
        {value}
        {trend === "up" && <span className="text-green-500 text-xs">^</span>}
        {trend === "down" && <span className="text-red-500 text-xs">v</span>}
      </div>
    </div>
  );
}

function MetricButton({
  label,
  selected,
  onClick,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 text-sm rounded ${
        selected
          ? "bg-blue-100 text-blue-700"
          : "bg-gray-100 text-gray-600 hover:bg-gray-200"
      }`}
    >
      {label}
    </button>
  );
}

function EvolutionChart({
  data,
  metric,
  height = 200,
}: {
  data: GenerationData[];
  metric: "fitness" | "diversity" | "operators";
  height?: number;
}) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-400"
        style={{ height }}
      >
        No data yet
      </div>
    );
  }

  const width = 500;
  const padding = { top: 20, right: 20, bottom: 30, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Determine data series based on metric
  let series: Array<{ label: string; color: string; values: number[] }> = [];

  if (metric === "fitness") {
    series = [
      { label: "Best", color: "#3b82f6", values: data.map((d) => d.bestFitness) },
      { label: "Avg", color: "#22c55e", values: data.map((d) => d.averageFitness) },
      { label: "Worst", color: "#ef4444", values: data.map((d) => d.worstFitness) },
    ];
  } else if (metric === "diversity") {
    series = [
      { label: "Diversity", color: "#8b5cf6", values: data.map((d) => d.diversity) },
    ];
  } else {
    const maxOps = Math.max(
      ...data.map((d) => Math.max(d.mutationCount, d.crossoverCount))
    );
    series = [
      {
        label: "Mutations",
        color: "#f59e0b",
        values: data.map((d) => d.mutationCount / (maxOps || 1)),
      },
      {
        label: "Crossovers",
        color: "#06b6d4",
        values: data.map((d) => d.crossoverCount / (maxOps || 1)),
      },
    ];
  }

  const maxValue = Math.max(...series.flatMap((s) => s.values), 0.1);
  const xScale = chartWidth / Math.max(data.length - 1, 1);
  const yScale = chartHeight / maxValue;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <g transform={`translate(${padding.left}, ${padding.top})`}>
        {/* Y-axis */}
        <line x1={0} y1={0} x2={0} y2={chartHeight} stroke="#e5e7eb" />
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick} transform={`translate(0, ${chartHeight - tick * chartHeight})`}>
            <line x1={-5} x2={chartWidth} stroke="#e5e7eb" />
            <text x={-10} y={4} textAnchor="end" className="text-xs fill-gray-500">
              {metric === "operators"
                ? Math.round(tick * maxValue)
                : `${(tick * maxValue * 100).toFixed(0)}%`}
            </text>
          </g>
        ))}

        {/* X-axis */}
        <line
          x1={0}
          y1={chartHeight}
          x2={chartWidth}
          y2={chartHeight}
          stroke="#e5e7eb"
        />

        {/* Series */}
        {series.map((s) => (
          <g key={s.label}>
            <path
              d={s.values
                .map((v, i) => {
                  const x = i * xScale;
                  const y = chartHeight - v * yScale;
                  return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
                })
                .join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
            />
          </g>
        ))}

        {/* Legend */}
        <g transform={`translate(${chartWidth - 100}, 0)`}>
          {series.map((s, i) => (
            <g key={s.label} transform={`translate(0, ${i * 16})`}>
              <rect width={12} height={12} fill={s.color} rx={2} />
              <text x={16} y={10} className="text-xs fill-gray-600">
                {s.label}
              </text>
            </g>
          ))}
        </g>
      </g>
    </svg>
  );
}

/**
 * GenerationCompare Component
 *
 * Side-by-side comparison of two generations.
 */

interface GenerationCompareProps {
  generation1: GenerationData & { chromosomes?: ChromosomePreview[] };
  generation2: GenerationData & { chromosomes?: ChromosomePreview[] };
  className?: string;
}

export function GenerationCompare({
  generation1,
  generation2,
  className = "",
}: GenerationCompareProps) {
  const improvement = generation2.bestFitness - generation1.bestFitness;

  return (
    <div className={`bg-white border rounded-lg ${className}`}>
      <div className="p-4 border-b">
        <h3 className="font-semibold">Generation Comparison</h3>
        <p className="text-sm text-gray-500">
          Gen {generation1.generation} vs Gen {generation2.generation}
        </p>
      </div>

      <div className="grid grid-cols-2 divide-x">
        {/* Generation 1 */}
        <div className="p-4">
          <h4 className="text-sm font-medium text-gray-700 mb-3">
            Generation {generation1.generation}
          </h4>
          <div className="space-y-2 text-sm">
            <MetricRow label="Best Fitness" value={`${(generation1.bestFitness * 100).toFixed(1)}%`} />
            <MetricRow label="Avg Fitness" value={`${(generation1.averageFitness * 100).toFixed(1)}%`} />
            <MetricRow label="Diversity" value={`${(generation1.diversity * 100).toFixed(1)}%`} />
            <MetricRow label="Mutations" value={generation1.mutationCount.toString()} />
            <MetricRow label="Crossovers" value={generation1.crossoverCount.toString()} />
          </div>
        </div>

        {/* Generation 2 */}
        <div className="p-4">
          <h4 className="text-sm font-medium text-gray-700 mb-3">
            Generation {generation2.generation}
          </h4>
          <div className="space-y-2 text-sm">
            <MetricRow
              label="Best Fitness"
              value={`${(generation2.bestFitness * 100).toFixed(1)}%`}
              diff={improvement}
            />
            <MetricRow
              label="Avg Fitness"
              value={`${(generation2.averageFitness * 100).toFixed(1)}%`}
              diff={generation2.averageFitness - generation1.averageFitness}
            />
            <MetricRow
              label="Diversity"
              value={`${(generation2.diversity * 100).toFixed(1)}%`}
              diff={generation2.diversity - generation1.diversity}
            />
            <MetricRow
              label="Mutations"
              value={generation2.mutationCount.toString()}
            />
            <MetricRow
              label="Crossovers"
              value={generation2.crossoverCount.toString()}
            />
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="p-4 bg-gray-50 border-t">
        <div className="text-center">
          <span className="text-sm text-gray-500">Improvement: </span>
          <span
            className={`font-semibold ${
              improvement > 0
                ? "text-green-600"
                : improvement < 0
                ? "text-red-600"
                : "text-gray-600"
            }`}
          >
            {improvement >= 0 ? "+" : ""}
            {(improvement * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );
}

function MetricRow({
  label,
  value,
  diff,
}: {
  label: string;
  value: string;
  diff?: number;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-600">{label}</span>
      <span className="font-mono flex items-center gap-1">
        {value}
        {diff !== undefined && diff !== 0 && (
          <span
            className={`text-xs ${diff > 0 ? "text-green-500" : "text-red-500"}`}
          >
            ({diff >= 0 ? "+" : ""}{(diff * 100).toFixed(1)}%)
          </span>
        )}
      </span>
    </div>
  );
}

export default EvolutionViewer;
