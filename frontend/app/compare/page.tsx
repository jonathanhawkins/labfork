"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
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
  Target,
  Clock,
  ChevronRight,
  Trash2,
  Circle,
} from "lucide-react";

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

// Available models
const availableModels: ModelInfo[] = [
  {
    id: "csm-1b-base",
    name: "CSM-1B Base",
    type: "base",
    description: "Sesame CSM-1B foundation model",
  },
  {
    id: "voice-deepseek-v1-final",
    name: "Voice DeepSeek v1",
    type: "finetuned",
    checkpoint: "models/checkpoints/voice_deepseek_v1/final.pt",
    description: "Fine-tuned with DeepSeek techniques",
  },
  {
    id: "voice-v1-best",
    name: "Voice Clone v1",
    type: "finetuned",
    checkpoint: "models/checkpoints/voice_v1/best.pt",
    description: "Best checkpoint from training",
  },
];

// Sample prompts
const samplePrompts = [
  "Hello, this is a test of my cloned voice.",
  "The quick brown fox jumps over the lazy dog.",
  "Welcome to the voice cloning demonstration.",
];

// Collapsible Section
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

// Stat Row
function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

// Metric Display
function MetricRow({
  label,
  value,
  unit = "%",
  threshold = 70,
}: {
  label: string;
  value: number;
  unit?: string;
  threshold?: number;
}) {
  const isGood = value >= threshold;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className={`text-sm ${isGood ? "text-foreground-bright" : "text-foreground"}`}>
          {value.toFixed(unit === "/5" ? 1 : 0)}{unit}
        </span>
      </div>
      <div className="h-1 bg-border rounded-full overflow-hidden">
        <div
          className={`h-full transition-all ${isGood ? "bg-foreground" : "bg-foreground/50"}`}
          style={{ width: `${Math.min(100, unit === "/5" ? value * 20 : value)}%` }}
        />
      </div>
    </div>
  );
}

export default function VoiceEvaluationPage() {
  const [inputText, setInputText] = useState("");
  const [baseModel, setBaseModel] = useState<string>("csm-1b-base");
  const [finetunedModel, setFinetunedModel] = useState<string>("voice-deepseek-v1-final");
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentComparison, setCurrentComparison] = useState<ComparisonResult | null>(null);
  const [history, setHistory] = useState<ComparisonHistoryItem[]>([]);
  const [referenceAudioUrl, setReferenceAudioUrl] = useState<string | null>(null);
  const [referenceAudioFile, setReferenceAudioFile] = useState<File | null>(null);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);

  const baseAudioRef = useRef<HTMLAudioElement>(null);
  const finetunedAudioRef = useRef<HTMLAudioElement>(null);
  const referenceAudioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleGenerate = async () => {
    if (!inputText.trim()) {
      toast.error("Please enter text to synthesize");
      return;
    }

    setIsGenerating(true);
    const toastId = toast.loading("Generating audio...");

    try {
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

  const handleReferenceUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setReferenceAudioUrl(url);
      setReferenceAudioFile(file);
      toast.success(`Reference audio loaded: ${file.name}`);
    }
  };

  const togglePlayAudio = (audioId: string, audioRef: React.RefObject<HTMLAudioElement | null>) => {
    if (!audioRef.current) return;

    if (playingAudio === audioId) {
      audioRef.current.pause();
      setPlayingAudio(null);
    } else {
      [baseAudioRef, finetunedAudioRef, referenceAudioRef].forEach(ref => {
        if (ref.current) ref.current.pause();
      });

      audioRef.current.play();
      setPlayingAudio(audioId);
    }
  };

  const handleClear = () => {
    setCurrentComparison(null);
    setInputText("");
    setPlayingAudio(null);
  };

  const loadHistoryItem = (item: ComparisonHistoryItem) => {
    setInputText(item.text);
    setBaseModel(item.baseModelId);
    setFinetunedModel(item.finetunedModelId);
    toast.info("Loaded comparison settings");
  };

  const overallScore = currentComparison?.metrics
    ? calculateOverallScore(currentComparison.metrics)
    : 0;

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Input & Models */}
      <aside className="w-[320px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Text Input" defaultOpen>
          <div className="space-y-3">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Enter text to synthesize..."
              className="w-full h-24 px-3 py-2 bg-background border border-border rounded text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-foreground-muted resize-none"
            />
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                {inputText.length} chars
              </span>
              <div className="flex gap-1">
                {samplePrompts.map((prompt, i) => (
                  <button
                    key={i}
                    onClick={() => setInputText(prompt)}
                    className="text-xs text-muted-foreground hover:text-foreground px-2 py-1"
                  >
                    #{i + 1}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </Section>

        <Section title="Models" defaultOpen>
          <div className="space-y-4">
            <div>
              <label className="text-xs text-muted-foreground mb-2 block">
                Base Model
              </label>
              <select
                value={baseModel}
                onChange={(e) => setBaseModel(e.target.value)}
                className="w-full bg-background border border-border text-foreground text-sm py-2 px-3 rounded focus:outline-none focus:border-foreground-muted"
              >
                {availableModels
                  .filter(m => m.type === "base")
                  .map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-2 block">
                Fine-tuned Model
              </label>
              <select
                value={finetunedModel}
                onChange={(e) => setFinetunedModel(e.target.value)}
                className="w-full bg-background border border-border text-foreground text-sm py-2 px-3 rounded focus:outline-none focus:border-foreground-muted"
              >
                {availableModels
                  .filter(m => m.type === "finetuned")
                  .map(model => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
              </select>
            </div>
          </div>
        </Section>

        <Section title="Reference Audio">
          <div
            onClick={() => fileInputRef.current?.click()}
            className={`border border-dashed rounded p-6 text-center cursor-pointer transition-colors ${
              referenceAudioUrl
                ? "border-foreground/30 bg-foreground/5"
                : "border-border hover:border-foreground-muted"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleReferenceUpload}
              className="hidden"
            />
            {referenceAudioUrl ? (
              <div className="space-y-2">
                <FileAudio className="w-6 h-6 text-foreground-bright mx-auto" />
                <p className="text-sm text-foreground">
                  {referenceAudioFile?.name || "Reference loaded"}
                </p>
                <p className="text-xs text-muted-foreground">Click to replace</p>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-6 h-6 text-muted-foreground mx-auto" />
                <p className="text-sm text-muted-foreground">
                  Upload reference audio
                </p>
                <p className="text-xs text-foreground-subtle">WAV, MP3, WebM</p>
              </div>
            )}
          </div>

          {referenceAudioUrl && (
            <div className="mt-3 p-3 bg-background border border-border rounded">
              <div className="flex items-center gap-3">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => togglePlayAudio("reference", referenceAudioRef)}
                  className="h-8 w-8 p-0"
                >
                  {playingAudio === "reference" ? (
                    <Pause className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                </Button>
                <div className="flex-1 h-6 flex items-center gap-0.5">
                  {generateMockWaveform().map((v, i) => (
                    <div
                      key={i}
                      className="w-1 bg-foreground/40 rounded-full"
                      style={{ height: `${v * 100}%` }}
                    />
                  ))}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setReferenceAudioUrl(null);
                    setReferenceAudioFile(null);
                  }}
                  className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
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
        </Section>

        {/* Generate Button */}
        <div className="px-4 py-4 border-t border-border">
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !inputText.trim()}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <GitCompare className="w-4 h-4 mr-2" />
                Generate & Compare
              </>
            )}
          </Button>
        </div>

        <Section title="Tips">
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <ChevronRight className="w-3 h-3 mt-1 text-foreground-subtle" />
              <span>Use sentences similar to training data</span>
            </li>
            <li className="flex items-start gap-2">
              <ChevronRight className="w-3 h-3 mt-1 text-foreground-subtle" />
              <span>Speaker similarity above 80% is good</span>
            </li>
            <li className="flex items-start gap-2">
              <ChevronRight className="w-3 h-3 mt-1 text-foreground-subtle" />
              <span>MOS above 4.0 is natural sounding</span>
            </li>
          </ul>
        </Section>
      </aside>

      {/* Main Content - Audio Comparison */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
        <div className="p-8 max-w-4xl mx-auto">
          {!currentComparison ? (
            <div className="text-center py-16">
              <GitCompare className="w-8 h-8 text-muted-foreground mx-auto mb-4" />
              <h2 className="text-lg text-foreground-bright mb-2">
                Voice Model Evaluation
              </h2>
              <p className="text-sm text-muted-foreground max-w-md mx-auto">
                Compare your fine-tuned voice model against the base model.
                Enter text and click Generate to start.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Synthesized Text */}
              <div>
                <div className="text-xs text-muted-foreground mb-2">Synthesized Text</div>
                <div className="bg-background-card border border-border rounded p-4">
                  <p className="text-sm text-foreground-bright italic">
                    "{currentComparison.text}"
                  </p>
                </div>
              </div>

              {/* Side-by-side Audio */}
              <div className="grid grid-cols-2 gap-4">
                {/* Base Model */}
                <div className="bg-background-card border border-border rounded p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="text-sm text-foreground-bright">Base Model</h4>
                      <p className="text-xs text-muted-foreground">
                        {availableModels.find(m => m.id === baseModel)?.name}
                      </p>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {currentComparison.baseAudio.duration.toFixed(1)}s
                    </span>
                  </div>

                  <div className="h-12 flex items-center gap-0.5 mb-3 px-1">
                    {currentComparison.baseAudio.waveformData?.map((value, index) => (
                      <div
                        key={index}
                        className={`w-1 rounded-full transition-all ${
                          playingAudio === "base" ? "bg-foreground" : "bg-foreground/40"
                        }`}
                        style={{ height: `${Math.max(10, value * 100)}%` }}
                      />
                    ))}
                  </div>

                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => togglePlayAudio("base", baseAudioRef)}
                    disabled={!currentComparison.baseAudio.url}
                    className="w-full"
                  >
                    {playingAudio === "base" ? (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
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

                {/* Fine-tuned Model */}
                <div className="bg-background-card border border-border rounded p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="text-sm text-foreground-bright">Fine-tuned</h4>
                      <p className="text-xs text-muted-foreground">
                        {availableModels.find(m => m.id === finetunedModel)?.name}
                      </p>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {currentComparison.finetunedAudio.duration.toFixed(1)}s
                    </span>
                  </div>

                  <div className="h-12 flex items-center gap-0.5 mb-3 px-1">
                    {currentComparison.finetunedAudio.waveformData?.map((value, index) => (
                      <div
                        key={index}
                        className={`w-1 rounded-full transition-all ${
                          playingAudio === "finetuned" ? "bg-foreground" : "bg-foreground/40"
                        }`}
                        style={{ height: `${Math.max(10, value * 100)}%` }}
                      />
                    ))}
                  </div>

                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => togglePlayAudio("finetuned", finetunedAudioRef)}
                    disabled={!currentComparison.finetunedAudio.url}
                    className="w-full"
                  >
                    {playingAudio === "finetuned" ? (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
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

              {/* Metrics */}
              {currentComparison.metrics.speakerSimilarity > 0 && (
                <div className="space-y-4">
                  <div className="text-xs text-muted-foreground">Evaluation Metrics</div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-background-card border border-border rounded p-4 space-y-3">
                      <MetricRow
                        label="Speaker Similarity"
                        value={currentComparison.metrics.speakerSimilarity}
                        threshold={80}
                      />
                      <MetricRow
                        label="Prosody Match"
                        value={currentComparison.metrics.prosodyMatch}
                        threshold={75}
                      />
                      <MetricRow
                        label="Naturalness (MOS)"
                        value={currentComparison.metrics.naturalness}
                        unit="/5"
                        threshold={4}
                      />
                    </div>
                    <div className="bg-background-card border border-border rounded p-4 space-y-3">
                      <MetricRow
                        label="Pitch Correlation"
                        value={currentComparison.metrics.pitchCorrelation * 100}
                        threshold={80}
                      />
                      <MetricRow
                        label="Rhythm Score"
                        value={currentComparison.metrics.rhythmScore}
                        threshold={85}
                      />
                      <MetricRow
                        label="Energy Alignment"
                        value={currentComparison.metrics.energyAlignment}
                        threshold={85}
                      />
                    </div>
                  </div>

                  {/* Overall Score */}
                  <div className="bg-background-card border border-border rounded p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-sm text-muted-foreground">Overall Score</span>
                        <p className="text-xs text-foreground-subtle mt-0.5">
                          Weighted average of all metrics
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <OverallScoreRing score={overallScore} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Clear Button */}
              <div className="flex justify-end">
                <Button variant="outline" size="sm" onClick={handleClear}>
                  <Trash2 className="w-4 h-4 mr-2" />
                  Clear
                </Button>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Right Sidebar - History */}
      <aside className="w-[280px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section
          title="History"
          defaultOpen
          badge={
            history.length > 0 ? (
              <span className="text-xs text-muted-foreground">{history.length}</span>
            ) : null
          }
        >
          {history.length === 0 ? (
            <div className="text-center py-8">
              <Clock className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No comparisons yet</p>
              <p className="text-xs text-foreground-subtle mt-1">
                Generate and compare to see history
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {history.map((item) => {
                const score = calculateOverallScore(item.metrics);
                return (
                  <button
                    key={item.id}
                    onClick={() => loadHistoryItem(item)}
                    className="w-full text-left p-3 rounded border border-border bg-background hover:border-foreground-muted transition-colors"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-foreground truncate">{item.text}</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span className="text-xs text-foreground-bright">
                          {score.toFixed(0)}%
                        </span>
                        <span className="text-xs text-foreground-subtle">
                          Sim: {item.metrics.speakerSimilarity.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </Section>

        {/* Quick Stats */}
        {currentComparison && currentComparison.metrics.speakerSimilarity > 0 && (
          <Section title="Quick Stats" defaultOpen>
            <div className="space-y-2">
              <StatRow label="Overall" value={`${overallScore.toFixed(0)}%`} />
              <StatRow
                label="Speaker Sim"
                value={`${currentComparison.metrics.speakerSimilarity.toFixed(0)}%`}
              />
              <StatRow
                label="Naturalness"
                value={`${currentComparison.metrics.naturalness.toFixed(1)}/5`}
              />
            </div>
          </Section>
        )}

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border mt-auto">
          <div className="text-xs text-muted-foreground">Model Comparison</div>
          <div className="text-xxs text-foreground-subtle mt-0.5">
            Compare base vs fine-tuned models
          </div>
        </div>
      </aside>
    </div>
  );
}

// Overall Score Ring
function OverallScoreRing({ score }: { score: number }) {
  const circumference = 2 * Math.PI * 28;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="relative w-16 h-16">
      <svg className="w-16 h-16 -rotate-90" viewBox="0 0 64 64">
        <circle
          cx="32"
          cy="32"
          r="28"
          fill="none"
          stroke="hsl(var(--border))"
          strokeWidth="4"
        />
        <circle
          cx="32"
          cy="32"
          r="28"
          fill="none"
          stroke="hsl(var(--foreground))"
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-sm font-medium text-foreground-bright">{score.toFixed(0)}</span>
      </div>
    </div>
  );
}

// Utilities
function generateMockWaveform(): number[] {
  return Array.from({ length: 40 }, () => 0.2 + Math.random() * 0.6);
}

function calculateOverallScore(metrics: ComparisonMetrics): number {
  const weights = {
    speakerSimilarity: 0.3,
    prosodyMatch: 0.2,
    naturalness: 0.25,
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
