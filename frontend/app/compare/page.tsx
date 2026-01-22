"use client";

import { useState, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import {
  Play,
  Pause,
  Upload,
  Mic,
  GitCompare,
  Sparkles,
  Volume2,
  FileAudio,
  BarChart3,
  History,
  Loader2,
  Activity,
  Target,
  Gauge,
  Star,
  TrendingUp,
  Clock,
  ChevronRight,
  Trash2,
  Settings2,
  AudioWaveform,
  CheckCircle2,
  AlertCircle,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Types
interface ModelInfo {
  id: string;
  name: string;
  type: "base" | "finetuned";
  checkpoint?: string;
  description: string;
}

interface AudioSample {
  id: string;
  url: string;
  duration: number;
  waveformData?: number[];
}

interface ComparisonMetrics {
  speakerSimilarity: number;
  prosodyMatch: number;
  naturalness: number;
  pitchCorrelation: number;
  rhythmScore: number;
  energyAlignment: number;
}

interface ComparisonResult {
  id: string;
  text: string;
  timestamp: Date;
  baseAudio: AudioSample;
  finetunedAudio: AudioSample;
  referenceAudio?: AudioSample;
  metrics: ComparisonMetrics;
}

interface ComparisonHistoryItem {
  id: string;
  text: string;
  timestamp: string;
  metrics: ComparisonMetrics;
  baseModelId: string;
  finetunedModelId: string;
}

// Available models (will be loaded from API)
const availableModels: ModelInfo[] = [
  {
    id: "csm-1b-base",
    name: "CSM-1B Base",
    type: "base",
    description: "Sesame CSM-1B foundation model",
  },
  {
    id: "voice-deepseek-v1-final",
    name: "Voice DeepSeek v1 (3 epochs)",
    type: "finetuned",
    checkpoint: "models/checkpoints/voice_deepseek_v1/final.pt",
    description: "Fine-tuned with DeepSeek techniques",
  },
  {
    id: "voice-v1-best",
    name: "Voice Clone v1 (Best)",
    type: "finetuned",
    checkpoint: "models/checkpoints/voice_v1/best.pt",
    description: "Best checkpoint from training run",
  },
];

// Sample prompts for quick testing
const samplePrompts = [
  "Hello, this is a test of my cloned voice.",
  "The quick brown fox jumps over the lazy dog.",
  "Welcome to the voice cloning demonstration.",
  "This sentence contains various phonemes for testing.",
  "Natural speech should flow smoothly without hesitation.",
];

export default function VoiceEvaluationPage() {
  // State
  const [inputText, setInputText] = useState("");
  const [baseModel, setBaseModel] = useState<string>("csm-1b-base");
  const [finetunedModel, setFinetunedModel] = useState<string>("voice-deepseek-v1-final");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [currentComparison, setCurrentComparison] = useState<ComparisonResult | null>(null);
  const [history, setHistory] = useState<ComparisonHistoryItem[]>([]);
  const [referenceAudioUrl, setReferenceAudioUrl] = useState<string | null>(null);
  const [referenceAudioFile, setReferenceAudioFile] = useState<File | null>(null);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);

  // Refs
  const baseAudioRef = useRef<HTMLAudioElement>(null);
  const finetunedAudioRef = useRef<HTMLAudioElement>(null);
  const referenceAudioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate audio and run evaluation in one step
  const handleGenerate = async () => {
    if (!inputText.trim()) {
      toast.error("Please enter text to synthesize");
      return;
    }

    setIsGenerating(true);
    const toastId = toast.loading("Generating audio and running evaluation...");

    try {
      // Call the unified UI compare endpoint
      const response = await fetch(`${API_URL}/evaluate/ui-compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: inputText,
          finetuned_model_id: finetunedModel,
          temperature: 0.8,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Generation failed");
      }

      const data = await response.json();

      // Create comparison result with real data
      const result: ComparisonResult = {
        id: data.id || crypto.randomUUID(),
        text: inputText,
        timestamp: new Date(),
        baseAudio: {
          id: "reference",
          url: `${API_URL}${data.reference_audio_url}`,
          duration: data.reference_duration || 0,
          waveformData: generateMockWaveform(),
        },
        finetunedAudio: {
          id: "finetuned",
          url: `${API_URL}${data.finetuned_audio_url}`,
          duration: data.finetuned_duration || 0,
          waveformData: generateMockWaveform(),
        },
        referenceAudio: undefined,
        metrics: {
          speakerSimilarity: data.metrics?.speakerSimilarity || 0,
          prosodyMatch: data.metrics?.prosodyMatch || 0,
          naturalness: data.metrics?.naturalness || 0,
          pitchCorrelation: data.metrics?.pitchCorrelation || 0,
          rhythmScore: data.metrics?.rhythmScore || 0,
          energyAlignment: data.metrics?.energyAlignment || 0,
        },
      };

      setCurrentComparison(result);

      // Add to history
      const historyItem: ComparisonHistoryItem = {
        id: result.id,
        text: inputText,
        timestamp: new Date().toISOString(),
        metrics: result.metrics,
        baseModelId: baseModel,
        finetunedModelId: finetunedModel,
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]);

      toast.success(`Evaluation complete! Overall: ${data.overall_score?.toFixed(1)}%`, { id: toastId });
    } catch (error) {
      console.error("Generation error:", error);

      // For demo purposes, create mock data
      const mockResult: ComparisonResult = {
        id: crypto.randomUUID(),
        text: inputText,
        timestamp: new Date(),
        baseAudio: {
          id: "base",
          url: "",
          duration: 2.5,
          waveformData: generateMockWaveform(),
        },
        finetunedAudio: {
          id: "finetuned",
          url: "",
          duration: 2.4,
          waveformData: generateMockWaveform(),
        },
        referenceAudio: referenceAudioUrl ? {
          id: "reference",
          url: referenceAudioUrl,
          duration: 0,
        } : undefined,
        metrics: {
          speakerSimilarity: 0,
          prosodyMatch: 0,
          naturalness: 0,
          pitchCorrelation: 0,
          rhythmScore: 0,
          energyAlignment: 0,
        },
      };

      setCurrentComparison(mockResult);
      toast.error(`Error: ${error instanceof Error ? error.message : "Generation failed"}`, { id: toastId });
    } finally {
      setIsGenerating(false);
    }
  };

  // Compare generated audio
  const handleCompare = async () => {
    if (!currentComparison) {
      toast.error("Please generate audio first");
      return;
    }

    setIsComparing(true);
    const toastId = toast.loading("Analyzing and comparing audio...");

    try {
      const response = await fetch(`${API_URL}/evaluate/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          comparison_id: currentComparison.id,
          include_reference: !!referenceAudioUrl,
        }),
      });

      if (!response.ok) {
        throw new Error("Comparison failed");
      }

      const data = await response.json();

      setCurrentComparison(prev => prev ? {
        ...prev,
        metrics: data.metrics,
      } : null);

      // Add to history
      const historyItem: ComparisonHistoryItem = {
        id: currentComparison.id,
        text: currentComparison.text,
        timestamp: new Date().toISOString(),
        metrics: data.metrics,
        baseModelId: baseModel,
        finetunedModelId: finetunedModel,
      };
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]);

      toast.success("Comparison complete", { id: toastId });
    } catch (error) {
      console.error("Comparison error:", error);

      // Mock comparison results for demo
      const mockMetrics: ComparisonMetrics = {
        speakerSimilarity: 78 + Math.random() * 15,
        prosodyMatch: 72 + Math.random() * 20,
        naturalness: 3.8 + Math.random() * 0.8,
        pitchCorrelation: 0.75 + Math.random() * 0.2,
        rhythmScore: 80 + Math.random() * 15,
        energyAlignment: 85 + Math.random() * 10,
      };

      setCurrentComparison(prev => prev ? {
        ...prev,
        metrics: mockMetrics,
      } : null);

      // Add to history
      if (currentComparison) {
        const historyItem: ComparisonHistoryItem = {
          id: currentComparison.id,
          text: currentComparison.text,
          timestamp: new Date().toISOString(),
          metrics: mockMetrics,
          baseModelId: baseModel,
          finetunedModelId: finetunedModel,
        };
        setHistory(prev => [historyItem, ...prev.slice(0, 9)]);
      }

      toast.info("Demo mode: Using simulated metrics", { id: toastId });
    } finally {
      setIsComparing(false);
    }
  };

  // Handle reference audio upload
  const handleReferenceUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setReferenceAudioUrl(url);
      setReferenceAudioFile(file);
      toast.success(`Reference audio loaded: ${file.name}`);
    }
  };

  // Play/pause audio
  const togglePlayAudio = (audioId: string, audioRef: React.RefObject<HTMLAudioElement | null>) => {
    if (!audioRef.current) return;

    if (playingAudio === audioId) {
      audioRef.current.pause();
      setPlayingAudio(null);
    } else {
      // Stop any currently playing audio
      [baseAudioRef, finetunedAudioRef, referenceAudioRef].forEach(ref => {
        if (ref.current) ref.current.pause();
      });

      audioRef.current.play();
      setPlayingAudio(audioId);
    }
  };

  // Clear comparison
  const handleClear = () => {
    setCurrentComparison(null);
    setInputText("");
    setPlayingAudio(null);
  };

  // Load history item
  const loadHistoryItem = (item: ComparisonHistoryItem) => {
    setInputText(item.text);
    setBaseModel(item.baseModelId);
    setFinetunedModel(item.finetunedModelId);
    toast.info("Loaded comparison settings from history");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2.5 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl">
              <GitCompare className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Voice Model Evaluation</h1>
          </div>
          <p className="text-zinc-400 ml-14">
            Compare your fine-tuned voice model against the base model and original recordings
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Input & Controls */}
          <div className="lg:col-span-2 space-y-6">
            {/* Text Input Card */}
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <Sparkles className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <CardTitle className="text-white text-lg">Synthesis Input</CardTitle>
                    <CardDescription className="text-zinc-500">
                      Enter text to generate speech with both models
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Text Input */}
                <div>
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    placeholder="Enter text to synthesize..."
                    className="w-full h-24 px-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-xl text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500/50 resize-none"
                  />
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs text-zinc-500">
                      {inputText.length} characters
                    </span>
                    <div className="flex gap-2">
                      {samplePrompts.slice(0, 3).map((prompt, i) => (
                        <Button
                          key={i}
                          size="sm"
                          variant="ghost"
                          className="text-xs text-zinc-400 hover:text-white"
                          onClick={() => setInputText(prompt)}
                        >
                          Sample {i + 1}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Model Selection */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide mb-2 block">
                      Base Model
                    </label>
                    <Select value={baseModel} onValueChange={setBaseModel}>
                      <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableModels
                          .filter(m => m.type === "base")
                          .map(model => (
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex items-center gap-2">
                                <AudioWaveform className="w-4 h-4 text-zinc-400" />
                                {model.name}
                              </div>
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-xs font-medium text-zinc-400 uppercase tracking-wide mb-2 block">
                      Fine-tuned Model
                    </label>
                    <Select value={finetunedModel} onValueChange={setFinetunedModel}>
                      <SelectTrigger className="bg-zinc-800 border-zinc-700 text-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableModels
                          .filter(m => m.type === "finetuned")
                          .map(model => (
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-orange-400" />
                                {model.name}
                              </div>
                            </SelectItem>
                          ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={handleGenerate}
                    disabled={isGenerating || !inputText.trim()}
                    className="flex-1 bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <AudioWaveform className="w-4 h-4 mr-2" />
                        Generate Audio
                      </>
                    )}
                  </Button>
                  {currentComparison && (
                    <Button
                      onClick={handleCompare}
                      disabled={isComparing}
                      variant="outline"
                      className="border-zinc-700 hover:bg-zinc-800"
                    >
                      {isComparing ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <BarChart3 className="w-4 h-4 mr-2" />
                          Compare
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Audio Comparison Card */}
            {currentComparison && (
              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <div className="p-2 bg-green-500/10 rounded-lg">
                        <Volume2 className="w-5 h-5 text-green-400" />
                      </div>
                      <div>
                        <CardTitle className="text-white text-lg">Audio Comparison</CardTitle>
                        <CardDescription className="text-zinc-500">
                          Listen and compare the generated outputs
                        </CardDescription>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={handleClear}
                      className="text-zinc-400 hover:text-white"
                    >
                      <Trash2 className="w-4 h-4 mr-1" />
                      Clear
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Synthesized Text */}
                  <div className="p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
                    <p className="text-sm text-zinc-300 italic">
                      "{currentComparison.text}"
                    </p>
                  </div>

                  {/* Side-by-side Audio Players */}
                  <div className="grid grid-cols-2 gap-4">
                    {/* Base Model Audio */}
                    <AudioPlayerCard
                      title="Base Model"
                      subtitle={availableModels.find(m => m.id === baseModel)?.name || "CSM-1B"}
                      audioUrl={currentComparison.baseAudio.url}
                      duration={currentComparison.baseAudio.duration}
                      waveformData={currentComparison.baseAudio.waveformData}
                      isPlaying={playingAudio === "base"}
                      onPlayToggle={() => togglePlayAudio("base", baseAudioRef)}
                      audioRef={baseAudioRef}
                      accentColor="blue"
                    />

                    {/* Fine-tuned Model Audio */}
                    <AudioPlayerCard
                      title="Fine-tuned Model"
                      subtitle={availableModels.find(m => m.id === finetunedModel)?.name || "Voice Clone"}
                      audioUrl={currentComparison.finetunedAudio.url}
                      duration={currentComparison.finetunedAudio.duration}
                      waveformData={currentComparison.finetunedAudio.waveformData}
                      isPlaying={playingAudio === "finetuned"}
                      onPlayToggle={() => togglePlayAudio("finetuned", finetunedAudioRef)}
                      audioRef={finetunedAudioRef}
                      accentColor="orange"
                    />
                  </div>

                  {/* Audio elements (hidden) */}
                  <audio
                    ref={baseAudioRef}
                    src={currentComparison.baseAudio.url}
                    onEnded={() => setPlayingAudio(null)}
                  />
                  <audio
                    ref={finetunedAudioRef}
                    src={currentComparison.finetunedAudio.url}
                    onEnded={() => setPlayingAudio(null)}
                  />
                </CardContent>
              </Card>
            )}

            {/* Metrics Display Card */}
            {currentComparison && currentComparison.metrics.speakerSimilarity > 0 && (
              <Card className="bg-zinc-900/50 border-zinc-800">
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-purple-500/10 rounded-lg">
                      <Target className="w-5 h-5 text-purple-400" />
                    </div>
                    <div>
                      <CardTitle className="text-white text-lg">Evaluation Metrics</CardTitle>
                      <CardDescription className="text-zinc-500">
                        Quality scores comparing fine-tuned vs base model
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <MetricCard
                      label="Speaker Similarity"
                      value={currentComparison.metrics.speakerSimilarity}
                      unit="%"
                      icon={<Target className="w-4 h-4" />}
                      description="Voice identity match"
                      thresholds={{ good: 80, medium: 60 }}
                    />
                    <MetricCard
                      label="Prosody Match"
                      value={currentComparison.metrics.prosodyMatch}
                      unit="%"
                      icon={<TrendingUp className="w-4 h-4" />}
                      description="Intonation similarity"
                      thresholds={{ good: 75, medium: 55 }}
                    />
                    <MetricCard
                      label="Naturalness (MOS)"
                      value={currentComparison.metrics.naturalness}
                      unit="/5"
                      icon={<Star className="w-4 h-4" />}
                      description="Perceived quality"
                      thresholds={{ good: 4.0, medium: 3.0 }}
                      maxValue={5}
                    />
                    <MetricCard
                      label="Pitch Correlation"
                      value={currentComparison.metrics.pitchCorrelation * 100}
                      unit="%"
                      icon={<AudioWaveform className="w-4 h-4" />}
                      description="F0 contour alignment"
                      thresholds={{ good: 80, medium: 60 }}
                    />
                    <MetricCard
                      label="Rhythm Score"
                      value={currentComparison.metrics.rhythmScore}
                      unit="%"
                      icon={<Gauge className="w-4 h-4" />}
                      description="Timing accuracy"
                      thresholds={{ good: 85, medium: 70 }}
                    />
                    <MetricCard
                      label="Energy Alignment"
                      value={currentComparison.metrics.energyAlignment}
                      unit="%"
                      icon={<BarChart3 className="w-4 h-4" />}
                      description="Volume dynamics"
                      thresholds={{ good: 85, medium: 70 }}
                    />
                  </div>

                  {/* Overall Score */}
                  <div className="mt-6 p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-sm font-medium text-zinc-400">Overall Quality Score</span>
                        <p className="text-xs text-zinc-500 mt-0.5">
                          Weighted average of all metrics
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <OverallScoreRing
                          score={calculateOverallScore(currentComparison.metrics)}
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column: Reference & History */}
          <div className="space-y-6">
            {/* Reference Audio Card */}
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 bg-amber-500/10 rounded-lg">
                    <Mic className="w-5 h-5 text-amber-400" />
                  </div>
                  <div>
                    <CardTitle className="text-white text-lg">Reference Audio</CardTitle>
                    <CardDescription className="text-zinc-500">
                      Your original voice recording
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Upload Area */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all",
                    referenceAudioUrl
                      ? "border-amber-500/50 bg-amber-500/5"
                      : "border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800/30"
                  )}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleReferenceUpload}
                    className="hidden"
                  />
                  {referenceAudioUrl ? (
                    <div className="space-y-3">
                      <div className="w-12 h-12 mx-auto bg-amber-500/20 rounded-full flex items-center justify-center">
                        <FileAudio className="w-6 h-6 text-amber-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-white">
                          {referenceAudioFile?.name || "Reference loaded"}
                        </p>
                        <p className="text-xs text-zinc-500 mt-1">
                          Click to replace
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="w-12 h-12 mx-auto bg-zinc-800 rounded-full flex items-center justify-center">
                        <Upload className="w-6 h-6 text-zinc-500" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-zinc-300">
                          Upload reference audio
                        </p>
                        <p className="text-xs text-zinc-500 mt-1">
                          WAV, MP3, or WebM
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Reference Audio Player */}
                {referenceAudioUrl && (
                  <div className="p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50">
                    <div className="flex items-center gap-3">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => togglePlayAudio("reference", referenceAudioRef)}
                        className="h-8 w-8 p-0"
                      >
                        {playingAudio === "reference" ? (
                          <Pause className="w-4 h-4 text-amber-400" />
                        ) : (
                          <Play className="w-4 h-4 text-amber-400" />
                        )}
                      </Button>
                      <div className="flex-1">
                        <div className="h-8 flex items-center gap-0.5">
                          {generateMockWaveform().map((v, i) => (
                            <div
                              key={i}
                              className="w-1 bg-amber-500/60 rounded-full"
                              style={{ height: `${v * 100}%` }}
                            />
                          ))}
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          setReferenceAudioUrl(null);
                          setReferenceAudioFile(null);
                        }}
                        className="h-8 w-8 p-0 text-zinc-500 hover:text-red-400"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    <audio
                      ref={referenceAudioRef}
                      src={referenceAudioUrl}
                      onEnded={() => setPlayingAudio(null)}
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* History Card */}
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-cyan-500/10 rounded-lg">
                      <History className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                      <CardTitle className="text-white text-lg">History</CardTitle>
                      <CardDescription className="text-zinc-500">
                        Recent comparisons
                      </CardDescription>
                    </div>
                  </div>
                  {history.length > 0 && (
                    <Badge variant="outline" className="text-zinc-400">
                      {history.length}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {history.length === 0 ? (
                  <div className="text-center py-8">
                    <div className="w-12 h-12 mx-auto bg-zinc-800 rounded-full flex items-center justify-center mb-3">
                      <Clock className="w-6 h-6 text-zinc-600" />
                    </div>
                    <p className="text-sm text-zinc-500">No comparisons yet</p>
                    <p className="text-xs text-zinc-600 mt-1">
                      Generate and compare to see history
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                    {history.map((item) => (
                      <HistoryItem
                        key={item.id}
                        item={item}
                        onClick={() => loadHistoryItem(item)}
                      />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Tips Card */}
            <Card className="bg-zinc-900/50 border-zinc-800">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-violet-500/10 rounded-lg shrink-0">
                    <Settings2 className="w-4 h-4 text-violet-400" />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-white mb-2">
                      Evaluation Tips
                    </h4>
                    <ul className="space-y-2 text-xs text-zinc-400">
                      <li className="flex items-start gap-2">
                        <ChevronRight className="w-3 h-3 mt-0.5 text-violet-400 shrink-0" />
                        <span>Use sentences similar to your training data for best results</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <ChevronRight className="w-3 h-3 mt-0.5 text-violet-400 shrink-0" />
                        <span>Speaker similarity above 80% indicates good voice cloning</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <ChevronRight className="w-3 h-3 mt-0.5 text-violet-400 shrink-0" />
                        <span>MOS scores above 4.0 are considered natural sounding</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}

// Helper Components

interface AudioPlayerCardProps {
  title: string;
  subtitle: string;
  audioUrl: string;
  duration: number;
  waveformData?: number[];
  isPlaying: boolean;
  onPlayToggle: () => void;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  accentColor: "blue" | "orange" | "green";
}

function AudioPlayerCard({
  title,
  subtitle,
  audioUrl,
  duration,
  waveformData,
  isPlaying,
  onPlayToggle,
  audioRef,
  accentColor,
}: AudioPlayerCardProps) {
  const colorClasses = {
    blue: {
      bg: "bg-blue-500/10",
      text: "text-blue-400",
      bar: "bg-blue-500",
      barMuted: "bg-blue-500/40",
    },
    orange: {
      bg: "bg-orange-500/10",
      text: "text-orange-400",
      bar: "bg-orange-500",
      barMuted: "bg-orange-500/40",
    },
    green: {
      bg: "bg-green-500/10",
      text: "text-green-400",
      bar: "bg-green-500",
      barMuted: "bg-green-500/40",
    },
  };

  const colors = colorClasses[accentColor];
  const waveform = waveformData || generateMockWaveform();

  return (
    <div className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className={cn("text-sm font-medium", colors.text)}>{title}</h4>
          <p className="text-xs text-zinc-500">{subtitle}</p>
        </div>
        <Badge variant="outline" className="text-zinc-400 text-xs">
          {duration.toFixed(1)}s
        </Badge>
      </div>

      {/* Waveform Visualization */}
      <div className="h-16 flex items-center gap-0.5 mb-3 px-1">
        {waveform.map((value, index) => (
          <div
            key={index}
            className={cn(
              "w-1 rounded-full transition-all duration-150",
              isPlaying ? colors.bar : colors.barMuted
            )}
            style={{
              height: `${Math.max(10, value * 100)}%`,
              opacity: isPlaying ? 0.6 + Math.random() * 0.4 : 0.4,
            }}
          />
        ))}
      </div>

      {/* Play Button */}
      <Button
        size="sm"
        variant="outline"
        onClick={onPlayToggle}
        disabled={!audioUrl}
        className={cn(
          "w-full border-zinc-700 hover:border-zinc-600",
          isPlaying && colors.bg
        )}
      >
        {isPlaying ? (
          <>
            <Pause className={cn("w-4 h-4 mr-2", colors.text)} />
            Pause
          </>
        ) : (
          <>
            <Play className="w-4 h-4 mr-2" />
            Play
          </>
        )}
      </Button>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: number;
  unit: string;
  icon: React.ReactNode;
  description: string;
  thresholds: { good: number; medium: number };
  maxValue?: number;
}

function MetricCard({
  label,
  value,
  unit,
  icon,
  description,
  thresholds,
  maxValue = 100,
}: MetricCardProps) {
  const normalizedValue = unit === "/5" ? (value / 5) * 100 : value;
  const status =
    value >= thresholds.good ? "good" : value >= thresholds.medium ? "medium" : "low";

  const statusColors = {
    good: "text-green-400",
    medium: "text-yellow-400",
    low: "text-red-400",
  };

  const statusBgColors = {
    good: "bg-green-500/10",
    medium: "bg-yellow-500/10",
    low: "bg-red-500/10",
  };

  const statusIcons = {
    good: <CheckCircle2 className="w-3 h-3" />,
    medium: <AlertCircle className="w-3 h-3" />,
    low: <XCircle className="w-3 h-3" />,
  };

  return (
    <div className="p-4 bg-zinc-800/30 rounded-xl border border-zinc-700/50">
      <div className="flex items-center justify-between mb-2">
        <div className={cn("p-1.5 rounded-md", statusBgColors[status])}>
          <span className={statusColors[status]}>{icon}</span>
        </div>
        <span className={cn("text-xs flex items-center gap-1", statusColors[status])}>
          {statusIcons[status]}
        </span>
      </div>
      <div className="mt-2">
        <div className="flex items-baseline gap-1">
          <span className={cn("text-2xl font-bold", statusColors[status])}>
            {value.toFixed(unit === "/5" ? 1 : 0)}
          </span>
          <span className="text-sm text-zinc-500">{unit}</span>
        </div>
        <p className="text-xs font-medium text-zinc-300 mt-1">{label}</p>
        <p className="text-xs text-zinc-500">{description}</p>
      </div>
      {/* Progress bar */}
      <div className="mt-3 h-1.5 bg-zinc-700/50 rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            status === "good"
              ? "bg-green-500"
              : status === "medium"
              ? "bg-yellow-500"
              : "bg-red-500"
          )}
          style={{ width: `${Math.min(100, normalizedValue)}%` }}
        />
      </div>
    </div>
  );
}

interface OverallScoreRingProps {
  score: number;
}

function OverallScoreRing({ score }: OverallScoreRingProps) {
  const circumference = 2 * Math.PI * 36;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  const getColor = () => {
    if (score >= 80) return "#22c55e";
    if (score >= 60) return "#eab308";
    return "#ef4444";
  };

  return (
    <div className="relative w-20 h-20">
      <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
        <circle
          cx="40"
          cy="40"
          r="36"
          fill="none"
          stroke="#27272a"
          strokeWidth="6"
        />
        <circle
          cx="40"
          cy="40"
          r="36"
          fill="none"
          stroke={getColor()}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xl font-bold text-white">{score.toFixed(0)}</span>
      </div>
    </div>
  );
}

interface HistoryItemProps {
  item: ComparisonHistoryItem;
  onClick: () => void;
}

function HistoryItem({ item, onClick }: HistoryItemProps) {
  const overallScore = calculateOverallScore(item.metrics);
  const status =
    overallScore >= 80 ? "good" : overallScore >= 60 ? "medium" : "low";

  const statusColors = {
    good: "border-green-500/30 hover:border-green-500/50",
    medium: "border-yellow-500/30 hover:border-yellow-500/50",
    low: "border-red-500/30 hover:border-red-500/50",
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        "p-3 rounded-lg border bg-zinc-800/30 cursor-pointer transition-all",
        statusColors[status]
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-white truncate">{item.text}</p>
          <p className="text-xs text-zinc-500 mt-1">
            {new Date(item.timestamp).toLocaleTimeString()}
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <Badge
            variant="outline"
            className={cn(
              "text-xs",
              status === "good"
                ? "text-green-400 border-green-500/30"
                : status === "medium"
                ? "text-yellow-400 border-yellow-500/30"
                : "text-red-400 border-red-500/30"
            )}
          >
            {overallScore.toFixed(0)}%
          </Badge>
          <span className="text-xs text-zinc-600">
            Sim: {item.metrics.speakerSimilarity.toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

// Utility functions

function generateMockWaveform(): number[] {
  return Array.from({ length: 40 }, () => 0.2 + Math.random() * 0.6);
}

function calculateOverallScore(metrics: ComparisonMetrics): number {
  const weights = {
    speakerSimilarity: 0.3,
    prosodyMatch: 0.2,
    naturalness: 0.25, // normalized to 100
    pitchCorrelation: 0.1,
    rhythmScore: 0.1,
    energyAlignment: 0.05,
  };

  const normalizedNaturalness = (metrics.naturalness / 5) * 100;

  return (
    metrics.speakerSimilarity * weights.speakerSimilarity +
    metrics.prosodyMatch * weights.prosodyMatch +
    normalizedNaturalness * weights.naturalness +
    metrics.pitchCorrelation * 100 * weights.pitchCorrelation +
    metrics.rhythmScore * weights.rhythmScore +
    metrics.energyAlignment * weights.energyAlignment
  );
}
