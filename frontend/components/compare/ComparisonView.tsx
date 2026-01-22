"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MetricsCard } from "./MetricsCard";
import { PitchContourChart } from "./PitchContourChart";
import { Play, Loader2, GitCompare, FileText, Music } from "lucide-react";
import { toast } from "sonner";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8002";

interface LibriTTSSample {
  id: string;
  text: string;
  speaker_id: string;
  gender: string;
  pitch: string;
  speaking_rate: string;
  speech_monotony: string;
  utterance_pitch_mean: number;
  utterance_pitch_std: number;
}

interface ComparisonResult {
  sample_id: string;
  audio_path: string;
  ground_truth: {
    pitch_category: string;
    pitch_mean_hz: number;
    pitch_std_hz: number;
    speaking_rate: string;
    monotony: string;
    gender: string;
    text: string;
  };
  our_analysis: {
    pitch_mean_hz: number;
    pitch_std_hz: number;
    pitch_category: string;
    speaking_rate_sps: number;
    speaking_rate_category: string;
    gender_inferred: string;
    contour: {
      times: number[];
      values: number[];
    };
  };
  metrics: {
    pitch_mean_error_hz: number;
    pitch_mean_error_pct: number;
    pitch_category_match: boolean;
    speaking_rate_match: boolean;
    gender_match: boolean;
  };
}

interface ComparisonViewProps {
  sample: LibriTTSSample | null;
}

export function ComparisonView({ sample }: ComparisonViewProps) {
  const [comparing, setComparing] = useState(false);
  const [result, setResult] = useState<ComparisonResult | null>(null);

  const runComparison = async () => {
    if (!sample) return;

    setComparing(true);
    try {
      const res = await fetch(
        `${API_URL}/libritts/compare/${sample.id}?use_synthetic=true`,
        { method: "POST" }
      );

      if (!res.ok) {
        throw new Error("Comparison failed");
      }

      const data = await res.json();
      setResult(data);
      toast.success("Comparison complete");
    } catch (err) {
      console.error("Comparison error:", err);
      toast.error("Failed to run comparison");
    } finally {
      setComparing(false);
    }
  };

  if (!sample) {
    return (
      <Card className="bg-zinc-900/50 border-zinc-800 h-full">
        <CardContent className="flex flex-col items-center justify-center h-full min-h-[400px] text-zinc-500">
          <GitCompare className="w-12 h-12 mb-4 opacity-50" />
          <p className="text-lg">Select a sample to compare</p>
          <p className="text-sm mt-1">
            Choose a sample from the browser to run prosody analysis
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Sample Info */}
      <Card className="bg-zinc-900/50 border-zinc-800">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-white flex items-center gap-2">
              <FileText className="w-5 h-5 text-green-400" />
              Selected Sample
            </CardTitle>
            <Badge variant="outline" className="text-zinc-400">
              {sample.id}
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-white text-sm leading-relaxed">{sample.text}</p>

          <div className="flex flex-wrap gap-2">
            <Badge className="bg-blue-500/20 text-blue-400">
              {sample.gender}
            </Badge>
            <Badge className="bg-purple-500/20 text-purple-400">
              {sample.pitch}
            </Badge>
            <Badge className="bg-green-500/20 text-green-400">
              {sample.speaking_rate}
            </Badge>
            <Badge className="bg-yellow-500/20 text-yellow-400">
              {sample.speech_monotony}
            </Badge>
          </div>

          <div className="flex items-center gap-4 text-sm text-zinc-400">
            <span className="flex items-center gap-1">
              <Music className="w-4 h-4" />
              {sample.utterance_pitch_mean.toFixed(1)} Hz (mean)
            </span>
            <span>
              &plusmn; {sample.utterance_pitch_std.toFixed(1)} Hz (std)
            </span>
          </div>

          <Button
            onClick={runComparison}
            disabled={comparing}
            className="w-full bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700"
          >
            {comparing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Run Prosody Comparison
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <>
          <MetricsCard
            groundTruth={result.ground_truth}
            ourAnalysis={result.our_analysis}
            metrics={result.metrics}
          />

          {result.our_analysis.contour.times.length > 0 && (
            <PitchContourChart
              contour={result.our_analysis.contour}
              groundTruthPitch={result.ground_truth.pitch_mean_hz}
            />
          )}
        </>
      )}
    </div>
  );
}
