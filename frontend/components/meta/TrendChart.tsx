"use client";

/**
 * TrendChart Component
 *
 * Visualizes research trends over time with interactive features.
 */

import React, { useState, useMemo } from "react";

interface TrendDataPoint {
  timestamp: Date | string;
  value: number;
  breakdown?: Record<string, number>;
}

interface TrendData {
  id: string;
  name: string;
  category?: string;
  color?: string;
  timeSeries: TrendDataPoint[];
  strength?: number;
  momentum?: number;
}

interface TrendChartProps {
  trends: TrendData[];
  title?: string;
  height?: number;
  showLegend?: boolean;
  showMomentum?: boolean;
  onTrendClick?: (trendId: string) => void;
  className?: string;
}

// Default colors for trends
const TREND_COLORS = [
  "#3B82F6", // blue
  "#10B981", // green
  "#F59E0B", // amber
  "#EF4444", // red
  "#8B5CF6", // purple
  "#EC4899", // pink
  "#06B6D4", // cyan
  "#F97316", // orange
];

/**
 * Format date for display
 */
function formatDate(date: Date | string): string {
  const d = typeof date === "string" ? new Date(date) : date;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/**
 * Momentum indicator
 */
function MomentumIndicator({ momentum }: { momentum: number }) {
  const isPositive = momentum > 0;
  const isNeutral = Math.abs(momentum) < 0.1;

  return (
    <div
      className={`flex items-center gap-1 text-xs font-medium ${
        isNeutral
          ? "text-gray-500"
          : isPositive
            ? "text-green-600"
            : "text-red-600"
      }`}
    >
      {!isNeutral && (
        <svg
          className={`w-3 h-3 ${isPositive ? "" : "rotate-180"}`}
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"
            clipRule="evenodd"
          />
        </svg>
      )}
      <span>
        {isNeutral ? "Stable" : isPositive ? "Growing" : "Declining"}
      </span>
    </div>
  );
}

/**
 * Main TrendChart component
 */
export function TrendChart({
  trends,
  title = "Research Trends",
  height = 300,
  showLegend = true,
  showMomentum = true,
  onTrendClick,
  className = "",
}: TrendChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<{
    trendId: string;
    pointIndex: number;
    x: number;
    y: number;
  } | null>(null);
  const [selectedTrends, setSelectedTrends] = useState<Set<string>>(
    new Set(trends.map((t) => t.id))
  );

  // Compute chart dimensions and scaling
  const chartData = useMemo(() => {
    if (trends.length === 0) return null;

    // Find global min/max across all trends
    let minValue = Infinity;
    let maxValue = -Infinity;
    let allDates: Date[] = [];

    for (const trend of trends) {
      for (const point of trend.timeSeries) {
        const date =
          typeof point.timestamp === "string"
            ? new Date(point.timestamp)
            : point.timestamp;
        allDates.push(date);
        minValue = Math.min(minValue, point.value);
        maxValue = Math.max(maxValue, point.value);
      }
    }

    // Sort dates
    allDates = Array.from(new Set(allDates.map((d) => d.getTime())))
      .sort((a, b) => a - b)
      .map((t) => new Date(t));

    // Add padding to value range
    const valueRange = maxValue - minValue || 1;
    const paddedMin = Math.max(0, minValue - valueRange * 0.1);
    const paddedMax = maxValue + valueRange * 0.1;

    return {
      dates: allDates,
      minValue: paddedMin,
      maxValue: paddedMax,
      valueRange: paddedMax - paddedMin,
    };
  }, [trends]);

  // Toggle trend visibility
  const toggleTrend = (trendId: string) => {
    const newSelected = new Set(selectedTrends);
    if (newSelected.has(trendId)) {
      newSelected.delete(trendId);
    } else {
      newSelected.add(trendId);
    }
    setSelectedTrends(newSelected);
  };

  if (!chartData || trends.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-50 rounded-lg ${className}`}
        style={{ height }}
      >
        <p className="text-gray-500">No trend data available</p>
      </div>
    );
  }

  const chartWidth = 100; // percentage
  const chartHeight = height - 40; // Leave room for x-axis labels

  // Calculate point positions
  const getPointPosition = (
    date: Date | string,
    value: number
  ): { x: number; y: number } => {
    const d = typeof date === "string" ? new Date(date) : date;
    const firstDate = chartData.dates[0].getTime();
    const lastDate = chartData.dates[chartData.dates.length - 1].getTime();
    const dateRange = lastDate - firstDate || 1;

    const x = ((d.getTime() - firstDate) / dateRange) * 100;
    const y =
      ((chartData.maxValue - value) / chartData.valueRange) * chartHeight;

    return { x, y };
  };

  // Generate SVG path for a trend
  const generatePath = (timeSeries: TrendDataPoint[]): string => {
    if (timeSeries.length === 0) return "";

    const points = timeSeries.map((point) =>
      getPointPosition(point.timestamp, point.value)
    );

    return points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
      .join(" ");
  };

  return (
    <div className={`bg-white rounded-lg border border-gray-200 p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
        {showMomentum && trends.length === 1 && trends[0].momentum !== undefined && (
          <MomentumIndicator momentum={trends[0].momentum} />
        )}
      </div>

      {/* Chart */}
      <div className="relative" style={{ height: chartHeight }}>
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full w-10 flex flex-col justify-between text-xs text-gray-500">
          <span>{Math.round(chartData.maxValue)}</span>
          <span>
            {Math.round((chartData.maxValue + chartData.minValue) / 2)}
          </span>
          <span>{Math.round(chartData.minValue)}</span>
        </div>

        {/* Chart area */}
        <div className="ml-12 h-full relative">
          {/* Grid lines */}
          <div className="absolute inset-0 flex flex-col justify-between pointer-events-none">
            {[0, 1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="border-t border-gray-100"
                style={{ height: "1px" }}
              />
            ))}
          </div>

          {/* SVG chart */}
          <svg
            className="absolute inset-0 w-full h-full overflow-visible"
            preserveAspectRatio="none"
            viewBox={`0 0 100 ${chartHeight}`}
          >
            {trends.map((trend, trendIndex) => {
              if (!selectedTrends.has(trend.id)) return null;

              const color = trend.color || TREND_COLORS[trendIndex % TREND_COLORS.length];
              const path = generatePath(trend.timeSeries);

              return (
                <g key={trend.id}>
                  {/* Line */}
                  <path
                    d={path}
                    fill="none"
                    stroke={color}
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="transition-opacity"
                    style={{
                      opacity:
                        hoveredPoint && hoveredPoint.trendId !== trend.id
                          ? 0.3
                          : 1,
                    }}
                  />

                  {/* Area fill */}
                  <path
                    d={`${path} L 100 ${chartHeight} L 0 ${chartHeight} Z`}
                    fill={color}
                    fillOpacity="0.1"
                  />

                  {/* Data points */}
                  {trend.timeSeries.map((point, pointIndex) => {
                    const pos = getPointPosition(point.timestamp, point.value);

                    return (
                      <circle
                        key={pointIndex}
                        cx={pos.x}
                        cy={pos.y}
                        r={3}
                        fill={color}
                        stroke="white"
                        strokeWidth="1.5"
                        className="cursor-pointer transition-all"
                        style={{
                          transform:
                            hoveredPoint?.trendId === trend.id &&
                            hoveredPoint?.pointIndex === pointIndex
                              ? "scale(1.5)"
                              : "scale(1)",
                          transformOrigin: `${pos.x}% ${pos.y}px`,
                        }}
                        onMouseEnter={() =>
                          setHoveredPoint({
                            trendId: trend.id,
                            pointIndex,
                            x: pos.x,
                            y: pos.y,
                          })
                        }
                        onMouseLeave={() => setHoveredPoint(null)}
                        onClick={() => onTrendClick?.(trend.id)}
                      />
                    );
                  })}
                </g>
              );
            })}
          </svg>

          {/* Tooltip */}
          {hoveredPoint && (
            <div
              className="absolute z-10 px-3 py-2 bg-gray-900 text-white text-xs rounded shadow-lg pointer-events-none"
              style={{
                left: `${hoveredPoint.x}%`,
                top: hoveredPoint.y - 10,
                transform: "translate(-50%, -100%)",
              }}
            >
              {(() => {
                const trend = trends.find((t) => t.id === hoveredPoint.trendId);
                if (!trend) return null;
                const point = trend.timeSeries[hoveredPoint.pointIndex];
                return (
                  <>
                    <div className="font-medium">{trend.name}</div>
                    <div className="text-gray-300">
                      {formatDate(point.timestamp)}: {point.value.toFixed(1)}
                    </div>
                  </>
                );
              })()}
            </div>
          )}
        </div>
      </div>

      {/* X-axis labels */}
      <div className="ml-12 mt-2 flex justify-between text-xs text-gray-500">
        {chartData.dates
          .filter((_, i, arr) => i % Math.ceil(arr.length / 6) === 0)
          .map((date, i) => (
            <span key={i}>{formatDate(date)}</span>
          ))}
      </div>

      {/* Legend */}
      {showLegend && trends.length > 1 && (
        <div className="mt-4 flex flex-wrap gap-3">
          {trends.map((trend, i) => {
            const color = trend.color || TREND_COLORS[i % TREND_COLORS.length];
            const isSelected = selectedTrends.has(trend.id);

            return (
              <button
                key={trend.id}
                onClick={() => toggleTrend(trend.id)}
                className={`flex items-center gap-2 px-2 py-1 rounded transition-opacity ${
                  isSelected ? "opacity-100" : "opacity-50"
                }`}
              >
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-sm text-gray-700">{trend.name}</span>
                {showMomentum && trend.momentum !== undefined && (
                  <MomentumIndicator momentum={trend.momentum} />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

/**
 * Compact trend indicator for dashboards
 */
export function TrendIndicator({
  name,
  value,
  change,
  changeLabel,
  className = "",
}: {
  name: string;
  value: number;
  change?: number;
  changeLabel?: string;
  className?: string;
}) {
  const isPositive = (change || 0) > 0;
  const isNeutral = Math.abs(change || 0) < 0.01;

  return (
    <div
      className={`flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200 ${className}`}
    >
      <div>
        <div className="text-sm text-gray-600">{name}</div>
        <div className="text-xl font-semibold text-gray-900">{value}</div>
      </div>

      {change !== undefined && (
        <div
          className={`flex items-center gap-1 px-2 py-1 rounded text-sm font-medium ${
            isNeutral
              ? "bg-gray-100 text-gray-600"
              : isPositive
                ? "bg-green-100 text-green-700"
                : "bg-red-100 text-red-700"
          }`}
        >
          {!isNeutral && (
            <svg
              className={`w-3 h-3 ${isPositive ? "" : "rotate-180"}`}
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"
                clipRule="evenodd"
              />
            </svg>
          )}
          <span>
            {isPositive ? "+" : ""}
            {(change * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {changeLabel && <div className="text-xs text-gray-500">{changeLabel}</div>}
    </div>
  );
}

/**
 * Sparkline for inline trend display
 */
export function TrendSparkline({
  data,
  width = 100,
  height = 30,
  color = "#3B82F6",
  className = "",
}: {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  className?: string;
}) {
  if (data.length < 2) {
    return null;
  }

  const minValue = Math.min(...data);
  const maxValue = Math.max(...data);
  const range = maxValue - minValue || 1;

  const points = data
    .map((value, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((value - minValue) / range) * height;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      className={className}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default TrendChart;
