"use client";

import { Card, CardContent } from "@/components/ui/card";
import { CheckCircle2, XCircle, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricRowProps {
  label: string;
  groundTruth: string | number;
  ourValue: string | number;
  match?: boolean;
  error?: number;
  unit?: string;
}

function MetricRow({
  label,
  groundTruth,
  ourValue,
  match,
  error,
  unit = "",
}: MetricRowProps) {
  const getStatusIcon = () => {
    if (match === undefined) return null;
    if (match) {
      return <CheckCircle2 className="w-4 h-4 text-green-500" />;
    }
    if (error !== undefined && error < 5) {
      return <AlertCircle className="w-4 h-4 text-yellow-500" />;
    }
    return <XCircle className="w-4 h-4 text-red-500" />;
  };

  const formatValue = (v: string | number) => {
    if (typeof v === "number") {
      return v.toFixed(1) + unit;
    }
    return v;
  };

  return (
    <div className="flex items-center justify-between py-2 border-b border-zinc-800 last:border-0">
      <span className="text-sm text-zinc-400">{label}</span>
      <div className="flex items-center gap-4">
        <div className="text-right">
          <div className="text-xs text-zinc-500">Ground Truth</div>
          <div className="text-sm font-medium text-white">
            {formatValue(groundTruth)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-zinc-500">Our Analysis</div>
          <div
            className={cn(
              "text-sm font-medium",
              match === true
                ? "text-green-400"
                : match === false
                ? "text-red-400"
                : "text-white"
            )}
          >
            {formatValue(ourValue)}
          </div>
        </div>
        <div className="w-6 flex justify-center">{getStatusIcon()}</div>
        {error !== undefined && (
          <div className="w-16 text-right">
            <span
              className={cn(
                "text-xs",
                error < 1 ? "text-green-400" : error < 5 ? "text-yellow-400" : "text-red-400"
              )}
            >
              {error.toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

interface MetricsCardProps {
  groundTruth: {
    pitch_category: string;
    pitch_mean_hz: number;
    pitch_std_hz: number;
    speaking_rate: string;
    monotony: string;
    gender: string;
  };
  ourAnalysis: {
    pitch_mean_hz: number;
    pitch_std_hz: number;
    pitch_category: string;
    speaking_rate_sps: number;
    speaking_rate_category: string;
    gender_inferred: string;
  };
  metrics: {
    pitch_mean_error_hz: number;
    pitch_mean_error_pct: number;
    pitch_category_match: boolean;
    speaking_rate_match: boolean;
    gender_match: boolean;
  };
}

export function MetricsCard({
  groundTruth,
  ourAnalysis,
  metrics,
}: MetricsCardProps) {
  const overallScore =
    ((metrics.pitch_category_match ? 1 : 0) +
      (metrics.speaking_rate_match ? 1 : 0) +
      (metrics.gender_match ? 1 : 0)) /
    3;

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardContent className="pt-6">
        {/* Overall Score */}
        <div className="flex items-center justify-between mb-4 pb-4 border-b border-zinc-700">
          <span className="text-sm font-medium text-zinc-300">
            Overall Match Score
          </span>
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "text-2xl font-bold",
                overallScore >= 0.8
                  ? "text-green-400"
                  : overallScore >= 0.5
                  ? "text-yellow-400"
                  : "text-red-400"
              )}
            >
              {(overallScore * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        {/* Detailed Metrics */}
        <div className="space-y-1">
          <MetricRow
            label="Pitch (Hz)"
            groundTruth={groundTruth.pitch_mean_hz}
            ourValue={ourAnalysis.pitch_mean_hz}
            error={metrics.pitch_mean_error_pct}
            unit=" Hz"
          />
          <MetricRow
            label="Pitch Category"
            groundTruth={groundTruth.pitch_category}
            ourValue={ourAnalysis.pitch_category}
            match={metrics.pitch_category_match}
          />
          <MetricRow
            label="Speaking Rate"
            groundTruth={groundTruth.speaking_rate}
            ourValue={ourAnalysis.speaking_rate_category}
            match={metrics.speaking_rate_match}
          />
          <MetricRow
            label="Gender"
            groundTruth={groundTruth.gender}
            ourValue={ourAnalysis.gender_inferred}
            match={metrics.gender_match}
          />
          <MetricRow
            label="Pitch Std"
            groundTruth={groundTruth.pitch_std_hz}
            ourValue={ourAnalysis.pitch_std_hz}
            unit=" Hz"
          />
          <MetricRow
            label="Rate (SPS)"
            groundTruth="-"
            ourValue={ourAnalysis.speaking_rate_sps}
            unit=" sps"
          />
        </div>
      </CardContent>
    </Card>
  );
}
