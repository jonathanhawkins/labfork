"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Play,
  Loader2,
  BarChart3,
  CheckCircle2,
  XCircle,
  TrendingUp,
  Target,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

interface BatchResults {
  summary: {
    total_samples: number;
    successful: number;
    failed: number;
    pitch: {
      mean_error_hz: number;
      median_error_hz: number;
      max_error_hz: number;
      mean_error_pct: number;
      category_accuracy: number;
    };
    speaking_rate: {
      category_accuracy: number;
    };
    gender: {
      accuracy: number;
    };
    overall_accuracy: number;
  };
  pitch_category_breakdown: Record<string, { total: number; correct: number }>;
  rate_category_breakdown: Record<string, { total: number; correct: number }>;
  individual_results: Array<{
    sample_id: string;
    pitch_error_pct: number;
    pitch_category_match: boolean;
    speaking_rate_match: boolean;
    gender_match: boolean;
  }>;
  errors: Array<{ sample_id: string; error: string }>;
}

export function BatchResultsPanel() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BatchResults | null>(null);
  const [sampleCount, setSampleCount] = useState("30");
  const [genderFilter, setGenderFilter] = useState("all");
  const [useRealAudio, setUseRealAudio] = useState(true);

  const runBatchComparison = async () => {
    setLoading(true);
    setResults(null);

    try {
      const res = await fetch(`${API_URL}/libritts/batch-compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_count: parseInt(sampleCount),
          gender_filter: genderFilter,
          use_real_audio: useRealAudio,
        }),
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || "Batch comparison failed");
      }

      const data = await res.json();
      setResults(data);
      toast.success(
        `Analyzed ${data.summary.successful} samples with ${useRealAudio ? "real" : "synthetic"} audio`
      );
    } catch (err) {
      console.error("Batch comparison error:", err);
      toast.error(err instanceof Error ? err.message : "Batch comparison failed");
    } finally {
      setLoading(false);
    }
  };

  const pitchBreakdownData = results
    ? Object.entries(results.pitch_category_breakdown).map(([category, data]) => ({
        category: category.replace(" pitch", ""),
        accuracy: Math.round((data.correct / data.total) * 100),
        total: data.total,
      }))
    : [];

  const rateBreakdownData = results
    ? Object.entries(results.rate_category_breakdown).map(([category, data]) => ({
        category: category.replace(" speed", "").replace("ly", ""),
        accuracy: Math.round((data.correct / data.total) * 100),
        total: data.total,
      }))
    : [];

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-yellow-400" />
            Batch Comparison
          </CardTitle>
          {results && (
            <Badge
              variant="outline"
              className={cn(
                results.summary.overall_accuracy >= 70
                  ? "text-green-400 border-green-400/50"
                  : results.summary.overall_accuracy >= 40
                  ? "text-yellow-400 border-yellow-400/50"
                  : "text-red-400 border-red-400/50"
              )}
            >
              {results.summary.overall_accuracy}% Overall
            </Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Controls */}
        <div className="flex gap-2 flex-wrap">
          <Select value={sampleCount} onValueChange={setSampleCount}>
            <SelectTrigger className="w-[120px] bg-zinc-800 border-zinc-700 text-white">
              <SelectValue placeholder="Samples" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="20">20 samples</SelectItem>
              <SelectItem value="50">50 samples</SelectItem>
              <SelectItem value="100">100 samples</SelectItem>
              <SelectItem value="200">200 samples</SelectItem>
              <SelectItem value="500">500 samples</SelectItem>
            </SelectContent>
          </Select>

          <Select value={genderFilter} onValueChange={setGenderFilter}>
            <SelectTrigger className="w-[120px] bg-zinc-800 border-zinc-700 text-white">
              <SelectValue placeholder="Gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="male">Male</SelectItem>
              <SelectItem value="female">Female</SelectItem>
            </SelectContent>
          </Select>

          <button
            onClick={() => setUseRealAudio(!useRealAudio)}
            className={cn(
              "px-3 py-2 rounded-md text-sm font-medium border transition-colors",
              useRealAudio
                ? "bg-green-500/20 border-green-500/50 text-green-400"
                : "bg-zinc-800 border-zinc-700 text-zinc-400"
            )}
          >
            {useRealAudio ? "Real Audio" : "Synthetic"}
          </button>

          <Button
            onClick={runBatchComparison}
            disabled={loading}
            className="flex-1 bg-gradient-to-r from-yellow-500 to-orange-600 hover:from-yellow-600 hover:to-orange-700"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing {sampleCount} samples...
              </>
            ) : (
              <>
                <Zap className="w-4 h-4 mr-2" />
                Run Batch Test
              </>
            )}
          </Button>
        </div>

        {/* Results */}
        {results && (
          <div className="space-y-4">
            {/* Summary Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
                  <Target className="w-3 h-3" />
                  Pitch Accuracy
                </div>
                <div
                  className={cn(
                    "text-xl font-bold",
                    results.summary.pitch.category_accuracy >= 70
                      ? "text-green-400"
                      : results.summary.pitch.category_accuracy >= 40
                      ? "text-yellow-400"
                      : "text-red-400"
                  )}
                >
                  {results.summary.pitch.category_accuracy}%
                </div>
              </div>

              <div className="bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
                  <TrendingUp className="w-3 h-3" />
                  Rate Accuracy
                </div>
                <div
                  className={cn(
                    "text-xl font-bold",
                    results.summary.speaking_rate.category_accuracy >= 70
                      ? "text-green-400"
                      : results.summary.speaking_rate.category_accuracy >= 40
                      ? "text-yellow-400"
                      : "text-red-400"
                  )}
                >
                  {results.summary.speaking_rate.category_accuracy}%
                </div>
              </div>

              <div className="bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
                  <CheckCircle2 className="w-3 h-3" />
                  Gender Accuracy
                </div>
                <div
                  className={cn(
                    "text-xl font-bold",
                    results.summary.gender.accuracy >= 70
                      ? "text-green-400"
                      : results.summary.gender.accuracy >= 40
                      ? "text-yellow-400"
                      : "text-red-400"
                  )}
                >
                  {results.summary.gender.accuracy}%
                </div>
              </div>

              <div className="bg-zinc-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 text-xs text-zinc-500 mb-1">
                  <BarChart3 className="w-3 h-3" />
                  Pitch Error
                </div>
                <div className="text-xl font-bold text-blue-400">
                  {results.summary.pitch.mean_error_pct}%
                </div>
              </div>
            </div>

            {/* Pitch Breakdown Chart */}
            {pitchBreakdownData.length > 0 && (
              <div className="bg-zinc-800/30 rounded-lg p-4">
                <h4 className="text-sm font-medium text-zinc-300 mb-3">
                  Pitch Category Accuracy
                </h4>
                <div className="h-[150px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={pitchBreakdownData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        type="number"
                        domain={[0, 100]}
                        tick={{ fill: "#9ca3af", fontSize: 10 }}
                      />
                      <YAxis
                        dataKey="category"
                        type="category"
                        tick={{ fill: "#9ca3af", fontSize: 10 }}
                        width={80}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1f2937",
                          border: "1px solid #374151",
                          borderRadius: "8px",
                          color: "#fff",
                        }}
                        formatter={(value: number, name: string, props: any) => [
                          `${value}% (${props.payload.total} samples)`,
                          "Accuracy",
                        ]}
                      />
                      <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}>
                        {pitchBreakdownData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={
                              entry.accuracy >= 70
                                ? "#22c55e"
                                : entry.accuracy >= 40
                                ? "#eab308"
                                : "#ef4444"
                            }
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Samples Processed */}
            <div className="flex items-center justify-between text-sm">
              <span className="text-zinc-500">
                Processed: {results.summary.successful} / {results.summary.total_samples}
              </span>
              {results.summary.failed > 0 && (
                <span className="text-red-400">
                  {results.summary.failed} failed
                </span>
              )}
            </div>

            {/* Individual Results Preview */}
            <div className="border-t border-zinc-800 pt-3">
              <h4 className="text-xs font-medium text-zinc-500 mb-2">
                Sample Results (first 10)
              </h4>
              <div className="flex flex-wrap gap-1">
                {results.individual_results.slice(0, 10).map((r) => (
                  <div
                    key={r.sample_id}
                    className={cn(
                      "w-6 h-6 rounded flex items-center justify-center text-xs",
                      r.pitch_category_match &&
                        r.speaking_rate_match &&
                        r.gender_match
                        ? "bg-green-500/20 text-green-400"
                        : r.pitch_category_match || r.speaking_rate_match
                        ? "bg-yellow-500/20 text-yellow-400"
                        : "bg-red-500/20 text-red-400"
                    )}
                    title={`${r.sample_id}: Pitch ${r.pitch_category_match ? "Y" : "N"}, Rate ${r.speaking_rate_match ? "Y" : "N"}, Gender ${r.gender_match ? "Y" : "N"}`}
                  >
                    {r.pitch_category_match && r.speaking_rate_match && r.gender_match
                      ? "3"
                      : (r.pitch_category_match ? 1 : 0) +
                        (r.speaking_rate_match ? 1 : 0) +
                        (r.gender_match ? 1 : 0)}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {!results && !loading && (
          <div className="text-center py-8 text-zinc-500">
            <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>Run a batch test to measure aggregate accuracy</p>
            <p className="text-xs mt-1">
              Compares our prosody analysis against LibriTTS ground truth
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
