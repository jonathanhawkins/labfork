"use client";

/**
 * FitnessChart Component
 *
 * Visualizes fitness evolution over generations, including best, average,
 * and population diversity metrics.
 */

import React, { useMemo } from "react";
import {
  GenerationStats,
  FitnessComponents,
  Population,
  formatFitness,
} from "@/lib/meta/evolution";

interface FitnessChartProps {
  generationHistory: GenerationStats[];
  width?: number;
  height?: number;
  showAverage?: boolean;
  showDiversity?: boolean;
  title?: string;
  className?: string;
}

export function FitnessChart({
  generationHistory,
  width = 600,
  height = 300,
  showAverage = true,
  showDiversity = false,
  title = "Fitness Evolution",
  className = "",
}: FitnessChartProps) {
  const padding = { top: 30, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const { xScale, yScale, paths, points } = useMemo(() => {
    if (generationHistory.length === 0) {
      return { xScale: () => 0, yScale: () => 0, paths: {}, points: {} };
    }

    const maxGen = Math.max(...generationHistory.map((s) => s.generation));
    const xScale = (gen: number) => (gen / maxGen) * chartWidth;
    const yScale = (fitness: number) => chartHeight - fitness * chartHeight;

    // Generate path data
    const bestPath = generationHistory
      .map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s.generation)} ${yScale(s.bestFitness)}`)
      .join(" ");

    const avgPath = generationHistory
      .map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s.generation)} ${yScale(s.averageFitness)}`)
      .join(" ");

    const diversityPath = generationHistory
      .map((s, i) => `${i === 0 ? "M" : "L"} ${xScale(s.generation)} ${yScale(s.fitnessStdDev * 5)}`)
      .join(" ");

    // Generate point data
    const bestPoints = generationHistory.map((s) => ({
      x: xScale(s.generation),
      y: yScale(s.bestFitness),
      value: s.bestFitness,
      gen: s.generation,
    }));

    return {
      xScale,
      yScale,
      paths: { best: bestPath, avg: avgPath, diversity: diversityPath },
      points: { best: bestPoints },
    };
  }, [generationHistory, chartWidth, chartHeight]);

  if (generationHistory.length === 0) {
    return (
      <div className={`bg-white border rounded-lg p-4 ${className}`}>
        <div className="text-center text-gray-500 py-8">No generation data available</div>
      </div>
    );
  }

  const latestStats = generationHistory[generationHistory.length - 1];

  return (
    <div className={`bg-white border rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between">
        <h3 className="font-semibold">{title}</h3>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span>Best: {formatFitness(latestStats.bestFitness)}</span>
          </div>
          {showAverage && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span>Avg: {formatFitness(latestStats.averageFitness)}</span>
            </div>
          )}
          {showDiversity && (
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500" />
              <span>Diversity: {latestStats.fitnessStdDev.toFixed(3)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Chart */}
      <div className="p-4">
        <svg width={width} height={height} className="overflow-visible">
          {/* Y-axis */}
          <g transform={`translate(${padding.left}, ${padding.top})`}>
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
              <g key={tick} transform={`translate(0, ${yScale(tick)})`}>
                <line x1={-5} x2={chartWidth} y1={0} y2={0} stroke="#e5e7eb" />
                <text
                  x={-10}
                  y={4}
                  textAnchor="end"
                  className="text-xs fill-gray-500"
                >
                  {(tick * 100).toFixed(0)}%
                </text>
              </g>
            ))}
          </g>

          {/* X-axis */}
          <g transform={`translate(${padding.left}, ${padding.top + chartHeight})`}>
            {generationHistory
              .filter((_, i) => i % Math.ceil(generationHistory.length / 5) === 0)
              .map((s) => (
                <g key={s.generation} transform={`translate(${xScale(s.generation)}, 0)`}>
                  <line y1={0} y2={5} stroke="#9ca3af" />
                  <text
                    y={20}
                    textAnchor="middle"
                    className="text-xs fill-gray-500"
                  >
                    {s.generation}
                  </text>
                </g>
              ))}
            <text
              x={chartWidth / 2}
              y={35}
              textAnchor="middle"
              className="text-xs fill-gray-500"
            >
              Generation
            </text>
          </g>

          {/* Chart area */}
          <g transform={`translate(${padding.left}, ${padding.top})`}>
            {/* Diversity line */}
            {showDiversity && (
              <path
                d={paths.diversity}
                fill="none"
                stroke="#a855f7"
                strokeWidth={1.5}
                strokeDasharray="4,4"
                opacity={0.6}
              />
            )}

            {/* Average line */}
            {showAverage && (
              <path
                d={paths.avg}
                fill="none"
                stroke="#22c55e"
                strokeWidth={2}
                opacity={0.8}
              />
            )}

            {/* Best line */}
            <path
              d={paths.best}
              fill="none"
              stroke="#3b82f6"
              strokeWidth={2.5}
            />

            {/* Best points */}
            {points.best.map((point, idx) => (
              <circle
                key={idx}
                cx={point.x}
                cy={point.y}
                r={idx === points.best.length - 1 ? 5 : 3}
                fill="#3b82f6"
                className="cursor-pointer hover:r-5"
              >
                <title>
                  Gen {point.gen}: {formatFitness(point.value)}
                </title>
              </circle>
            ))}
          </g>
        </svg>
      </div>

      {/* Stats footer */}
      <div className="px-4 pb-4">
        <div className="grid grid-cols-4 gap-4 text-center text-sm">
          <div>
            <div className="text-gray-500">Generations</div>
            <div className="font-semibold">{generationHistory.length}</div>
          </div>
          <div>
            <div className="text-gray-500">Mutations</div>
            <div className="font-semibold">
              {generationHistory.reduce((sum, s) => sum + s.mutationCount, 0)}
            </div>
          </div>
          <div>
            <div className="text-gray-500">Crossovers</div>
            <div className="font-semibold">
              {generationHistory.reduce((sum, s) => sum + s.crossoverCount, 0)}
            </div>
          </div>
          <div>
            <div className="text-gray-500">Improvement</div>
            <div className="font-semibold text-green-600">
              +{((latestStats.bestFitness - generationHistory[0].bestFitness) * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Fitness components radar chart
 */
interface FitnessRadarProps {
  components: FitnessComponents;
  compareWith?: FitnessComponents;
  size?: number;
  className?: string;
}

export function FitnessRadar({
  components,
  compareWith,
  size = 200,
  className = "",
}: FitnessRadarProps) {
  const center = size / 2;
  const radius = (size - 40) / 2;
  const angles = [0, 72, 144, 216, 288].map((deg) => (deg * Math.PI) / 180);
  const labels = ["Quality", "Efficiency", "Novelty", "Feasibility", "Compatibility"];
  const values = [
    components.quality,
    components.efficiency,
    components.novelty,
    components.feasibility,
    components.compatibility,
  ];

  const getPoint = (angle: number, value: number) => ({
    x: center + Math.sin(angle) * radius * value,
    y: center - Math.cos(angle) * radius * value,
  });

  const mainPath = values
    .map((v, i) => {
      const point = getPoint(angles[i], v);
      return `${i === 0 ? "M" : "L"} ${point.x} ${point.y}`;
    })
    .join(" ") + " Z";

  let comparePath = "";
  if (compareWith) {
    const compareValues = [
      compareWith.quality,
      compareWith.efficiency,
      compareWith.novelty,
      compareWith.feasibility,
      compareWith.compatibility,
    ];
    comparePath = compareValues
      .map((v, i) => {
        const point = getPoint(angles[i], v);
        return `${i === 0 ? "M" : "L"} ${point.x} ${point.y}`;
      })
      .join(" ") + " Z";
  }

  return (
    <div className={`bg-white border rounded-lg p-4 ${className}`}>
      <h4 className="font-medium mb-3 text-center">Fitness Components</h4>
      <svg width={size} height={size} className="mx-auto">
        {/* Grid circles */}
        {[0.25, 0.5, 0.75, 1].map((r) => (
          <circle
            key={r}
            cx={center}
            cy={center}
            r={radius * r}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth={1}
          />
        ))}

        {/* Axis lines */}
        {angles.map((angle, i) => {
          const endPoint = getPoint(angle, 1);
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={endPoint.x}
              y2={endPoint.y}
              stroke="#e5e7eb"
              strokeWidth={1}
            />
          );
        })}

        {/* Compare area */}
        {comparePath && (
          <path
            d={comparePath}
            fill="#f59e0b"
            fillOpacity={0.2}
            stroke="#f59e0b"
            strokeWidth={1.5}
            strokeDasharray="4,4"
          />
        )}

        {/* Main area */}
        <path
          d={mainPath}
          fill="#3b82f6"
          fillOpacity={0.3}
          stroke="#3b82f6"
          strokeWidth={2}
        />

        {/* Points */}
        {values.map((v, i) => {
          const point = getPoint(angles[i], v);
          return (
            <circle
              key={i}
              cx={point.x}
              cy={point.y}
              r={4}
              fill="#3b82f6"
              stroke="white"
              strokeWidth={2}
            />
          );
        })}

        {/* Labels */}
        {labels.map((label, i) => {
          const point = getPoint(angles[i], 1.2);
          return (
            <text
              key={i}
              x={point.x}
              y={point.y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-xs fill-gray-600"
            >
              {label}
            </text>
          );
        })}
      </svg>

      {/* Values list */}
      <div className="mt-3 grid grid-cols-5 gap-1 text-center text-xs">
        {labels.map((label, i) => (
          <div key={label}>
            <div className="font-semibold text-blue-600">
              {Math.round(values[i] * 100)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Population overview stats
 */
interface PopulationStatsProps {
  population: Population;
  className?: string;
}

export function PopulationStats({ population, className = "" }: PopulationStatsProps) {
  const latestStats = population.generationHistory[population.generationHistory.length - 1];

  return (
    <div className={`bg-white border rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">{population.name}</h3>
        <span
          className={`px-2 py-1 text-xs rounded-full ${
            population.status === "evolving"
              ? "bg-blue-100 text-blue-800"
              : population.status === "converged"
              ? "bg-green-100 text-green-800"
              : "bg-gray-100 text-gray-800"
          }`}
        >
          {population.status}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-blue-600">
            {population.generation}
          </div>
          <div className="text-xs text-gray-500">Generation</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-green-600">
            {formatFitness(latestStats.bestFitness)}
          </div>
          <div className="text-xs text-gray-500">Best Fitness</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-yellow-600">
            {formatFitness(latestStats.averageFitness)}
          </div>
          <div className="text-xs text-gray-500">Avg Fitness</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-purple-600">
            {population.size}
          </div>
          <div className="text-xs text-gray-500">Population</div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t">
        <div className="flex justify-between text-sm text-gray-500">
          <span>Created: {population.createdAt.toLocaleDateString()}</span>
          <span>Last evolved: {population.lastEvolved.toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Mini sparkline for fitness
 */
interface FitnessSparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  className?: string;
}

export function FitnessSparkline({
  data,
  width = 100,
  height = 30,
  color = "#3b82f6",
  className = "",
}: FitnessSparklineProps) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className={className}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default FitnessChart;
