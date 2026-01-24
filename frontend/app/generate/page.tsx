"use client";

import React, { useState, useRef, useEffect } from "react";
import { toast } from "sonner";
import {
  Mic,
  Upload,
  Play,
  Pause,
  Download,
  Loader2,
  Sparkles,
  Volume2,
  Music,
  Circle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";

interface PredictedProsody {
  emotion: string;
  emotion_intensity: number;
  energy: number;
  pace: string;
  pitch_tendency: string;
  emphasis_words: string[];
  tone: string;
}

interface VoiceSample {
  id: string;
  path: string;
  filename: string;
  emotion: string;
  session_id: string;
}

// Emotion presets - design system compatible
const EMOTIONS = [
  { id: null, label: "Auto", description: "AI predicts from text" },
  { id: "neutral", label: "Neutral", description: "Balanced, natural speech" },
  { id: "happy", label: "Happy", description: "Upbeat, enthusiastic tone" },
  { id: "sad", label: "Sad", description: "Slower, lower energy" },
  { id: "angry", label: "Angry", description: "Intense, emphatic delivery" },
  { id: "calm", label: "Calm", description: "Relaxed, soothing pace" },
];

// Collapsible Section Component
function Section({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-foreground-bright hover:text-foreground transition-colors"
      >
        <span className="text-sm">{title}</span>
        <span className="text-muted-foreground">{isOpen ? "-" : "+"}</span>
      </button>
      {isOpen && <div className="px-4 pb-4 animate-fade-in">{children}</div>}
    </div>
  );
}

// Stat Row Component
function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

export default function GeneratePage() {
  // Input state
  const [text, setText] = useState("");
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);
  const [intensity, setIntensity] = useState(0.8);
  const [temperature, setTemperature] = useState(0.8);

  // Auto-predicted prosody
  const [predictedProsody, setPredictedProsody] = useState<PredictedProsody | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Reference audio for style transfer
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [referenceUrl, setReferenceUrl] = useState<string | null>(null);

  // Voice samples from recordings
  const [voiceSamples, setVoiceSamples] = useState<VoiceSample[]>([]);
  const [voiceSamplesByEmotion, setVoiceSamplesByEmotion] = useState<Record<string, VoiceSample[]>>({});
  const [selectedVoiceSample, setSelectedVoiceSample] = useState<VoiceSample | null>(null);
  const [isLoadingSamples, setIsLoadingSamples] = useState(true);

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch voice samples on mount
  useEffect(() => {
    const fetchVoiceSamples = async () => {
      try {
        const res = await fetch(`${API_BASE}/voice-samples`);
        if (res.ok) {
          const data = await res.json();
          setVoiceSamples(data.samples || []);
          setVoiceSamplesByEmotion(data.by_emotion || {});
          if (data.samples?.length > 0) {
            setSelectedVoiceSample(data.samples[0]);
          }
        }
      } catch (err) {
        console.error("Failed to fetch voice samples:", err);
      } finally {
        setIsLoadingSamples(false);
      }
    };
    fetchVoiceSamples();
  }, []);

  // Auto-predict prosody when text changes (debounced)
  useEffect(() => {
    if (!text.trim() || selectedEmotion) {
      setPredictedProsody(null);
      return;
    }

    const timer = setTimeout(async () => {
      setIsPredicting(true);
      try {
        const response = await fetch(`${API_BASE}/predict-prosody?text=${encodeURIComponent(text)}`, {
          method: "POST",
        });
        if (response.ok) {
          const data = await response.json();
          setPredictedProsody(data.predicted);
        }
      } catch (err) {
        console.error("Prosody prediction failed:", err);
      } finally {
        setIsPredicting(false);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [text, selectedEmotion]);

  const handleReferenceUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setReferenceAudio(file);
      setReferenceUrl(URL.createObjectURL(file));
      toast.success("Reference audio loaded - prosody will be extracted");
    }
  };

  const clearReference = () => {
    setReferenceAudio(null);
    setReferenceUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleGenerate = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text to generate");
      return;
    }

    setIsGenerating(true);

    const useVoiceSample = !!selectedVoiceSample;
    const useReferenceAudio = !!referenceAudio;
    const usePocketTTS = useVoiceSample || useReferenceAudio;

    const toastId = toast.loading(
      usePocketTTS ? "Cloning your voice..." : "Generating speech..."
    );

    try {
      const formData = new FormData();
      formData.append("text", text);

      let endpoint = `${API_BASE}/generate`;

      if (useVoiceSample && selectedVoiceSample) {
        endpoint = `${API_BASE}/generate-with-voice-sample`;
        formData.append("voice_sample_path", selectedVoiceSample.path);
      } else if (useReferenceAudio && referenceAudio) {
        endpoint = `${API_BASE}/generate-pocket-tts`;
        formData.append("reference_audio", referenceAudio);
      } else {
        formData.append("emotion", selectedEmotion || "neutral");
        formData.append("intensity", intensity.toString());
        formData.append("temperature", temperature.toString());
      }

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Generation failed");
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      setGeneratedAudioUrl(audioUrl);

      toast.success(
        usePocketTTS ? "Voice cloned successfully!" : "Speech generated!",
        { id: toastId }
      );
    } catch (err) {
      console.error("Generation failed:", err);
      toast.error("Generation failed. Is the backend running?", { id: toastId });
    } finally {
      setIsGenerating(false);
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const downloadAudio = () => {
    if (generatedAudioUrl) {
      const a = document.createElement("a");
      a.href = generatedAudioUrl;
      a.download = `generated_${selectedEmotion}_${Date.now()}.wav`;
      a.click();
    }
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Controls */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Emotion" defaultOpen>
          <div className="space-y-2">
            {EMOTIONS.map((emotion) => {
              const isSelected = selectedEmotion === emotion.id;

              return (
                <button
                  key={emotion.id ?? "auto"}
                  onClick={() => setSelectedEmotion(emotion.id)}
                  className={`
                    w-full p-2.5 rounded text-left transition-colors border
                    ${isSelected
                      ? "border-foreground-bright bg-background-card"
                      : "border-border hover:border-muted-foreground"
                    }
                  `}
                >
                  <div className="flex items-center gap-2">
                    <Circle
                      className={`w-2 h-2 ${
                        isSelected ? "fill-foreground-bright text-foreground-bright" : "fill-transparent text-muted-foreground"
                      }`}
                    />
                    <span className={`text-sm ${isSelected ? "text-foreground-bright" : "text-foreground"}`}>
                      {emotion.label}
                    </span>
                    {emotion.id === null && isPredicting && (
                      <Loader2 className="w-3 h-3 animate-spin text-muted-foreground ml-auto" />
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 ml-4">
                    {emotion.description}
                  </p>
                </button>
              );
            })}
          </div>

          {/* Auto-predicted prosody display */}
          {selectedEmotion === null && predictedProsody && (
            <div className="mt-4 border border-border rounded p-3 bg-background-card">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-3 h-3 text-foreground-bright" />
                <span className="text-xs text-foreground-bright">AI Prediction</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-muted-foreground">Emotion:</span>{" "}
                  <span className="text-foreground capitalize">{predictedProsody.emotion}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Energy:</span>{" "}
                  <span className="text-foreground">{Math.round(predictedProsody.energy * 100)}%</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Pace:</span>{" "}
                  <span className="text-foreground capitalize">{predictedProsody.pace}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Tone:</span>{" "}
                  <span className="text-foreground capitalize">{predictedProsody.tone}</span>
                </div>
              </div>
              {predictedProsody.emphasis_words?.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  <span className="text-xxs text-muted-foreground">Emphasis:</span>
                  {predictedProsody.emphasis_words.map((word, i) => (
                    <span key={i} className="text-xxs px-1.5 py-0.5 border border-border rounded text-foreground">
                      {word}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </Section>

        <Section title="Intensity" defaultOpen>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Emotion Level</span>
              <span className="text-sm text-foreground">{Math.round(intensity * 100)}%</span>
            </div>
            <Slider
              value={[intensity]}
              onValueChange={([v]) => setIntensity(v)}
              min={0}
              max={1}
              step={0.1}
            />
          </div>
        </Section>

        <Section title="Temperature">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Variation</span>
              <span className="text-sm text-foreground">{temperature.toFixed(1)}</span>
            </div>
            <Slider
              value={[temperature]}
              onValueChange={([v]) => setTemperature(v)}
              min={0.1}
              max={1.5}
              step={0.1}
            />
            <p className="text-xs text-muted-foreground">
              Lower = consistent, Higher = varied
            </p>
          </div>
        </Section>

        <Section title="Voice Samples">
          {isLoadingSamples ? (
            <div className="flex items-center justify-center py-4">
              <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            </div>
          ) : voiceSamples.length > 0 ? (
            <div className="space-y-3">
              {/* Emotion filter buttons */}
              <div className="flex flex-wrap gap-1">
                {Object.keys(voiceSamplesByEmotion).map((emotion) => (
                  <button
                    key={emotion}
                    onClick={() => {
                      const samples = voiceSamplesByEmotion[emotion];
                      if (samples?.length > 0) {
                        setSelectedVoiceSample(samples[0]);
                      }
                    }}
                    className={`px-2 py-1 rounded text-xs transition-colors ${
                      selectedVoiceSample?.emotion === emotion
                        ? "bg-foreground text-background"
                        : "bg-background-card text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {emotion} ({voiceSamplesByEmotion[emotion]?.length})
                  </button>
                ))}
              </div>

              {/* Selected sample display */}
              {selectedVoiceSample && (
                <div className="border border-border rounded p-3 bg-background-card">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-foreground">
                        {selectedVoiceSample.emotion}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {selectedVoiceSample.filename}
                      </p>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-muted-foreground">Ready</span>
                      <div className="w-1.5 h-1.5 bg-foreground-bright rounded-full" />
                    </div>
                  </div>
                </div>
              )}

              {/* Sample selector dropdown */}
              <select
                value={selectedVoiceSample?.id || ""}
                onChange={(e) => {
                  const sample = voiceSamples.find(s => s.id === e.target.value);
                  setSelectedVoiceSample(sample || null);
                }}
                className="w-full bg-background-card border border-border rounded px-3 py-2 text-sm text-foreground"
              >
                {voiceSamples.map((sample) => (
                  <option key={sample.id} value={sample.id}>
                    {sample.emotion} - {sample.filename}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-muted-foreground text-sm mb-2">No voice samples recorded</p>
              <a href="/perform" className="text-foreground-bright hover:underline text-sm">
                Go to Perform page to record
              </a>
            </div>
          )}
        </Section>

        <Section title="Reference Audio">
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleReferenceUpload}
            className="hidden"
          />

          {!referenceUrl ? (
            <Button
              onClick={() => fileInputRef.current?.click()}
              variant="outline"
              className="w-full h-16 border-dashed"
            >
              <div className="flex flex-col items-center gap-1">
                <Upload className="w-4 h-4" />
                <span className="text-xs">Upload audio file</span>
              </div>
            </Button>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between bg-background-card border border-border rounded p-3">
                <div className="flex items-center gap-2">
                  <Mic className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-foreground truncate">
                    {referenceAudio?.name}
                  </span>
                </div>
                <Button
                  onClick={clearReference}
                  variant="ghost"
                  size="sm"
                  className="text-muted-foreground hover:text-foreground"
                >
                  Remove
                </Button>
              </div>
              <audio src={referenceUrl} controls className="w-full h-10" />
              <p className="text-xs text-muted-foreground">
                Prosody will be extracted from this audio
              </p>
            </div>
          )}
        </Section>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border mt-auto">
          <div className="text-xs text-muted-foreground">
            Voice Generation
          </div>
          <div className="text-xxs text-foreground-subtle mt-0.5">
            Prosody-controlled synthesis
          </div>
        </div>
      </aside>

      {/* Main Content - Text Input & Generate */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto p-8">
        <div className="max-w-2xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-5 h-5 text-foreground-bright" />
              <h1 className="text-lg text-foreground-bright">Voice Generation</h1>
            </div>
            <p className="text-sm text-muted-foreground">
              Generate speech with prosody control - emotion, intensity, and style transfer
            </p>
          </div>

          {/* Text Input */}
          <div className="border border-border rounded p-4 bg-background-elevated">
            <div className="flex items-center gap-2 mb-3">
              <Music className="w-4 h-4 text-muted-foreground" />
              <Label className="text-sm text-foreground-bright">Text to Speak</Label>
            </div>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to generate as speech..."
              className="min-h-[120px] bg-background border-border text-foreground placeholder:text-muted-foreground resize-none"
            />
            <p className="text-xs text-muted-foreground mt-2">
              {text.length} characters
            </p>
          </div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
            className="w-full h-14"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Generate Speech
              </>
            )}
          </Button>

          <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
            <span>Emotion: {selectedEmotion || "Auto"}</span>
            <span>|</span>
            <span>Intensity: {Math.round(intensity * 100)}%</span>
            <span>|</span>
            <span>Temp: {temperature.toFixed(1)}</span>
          </div>

          {/* Output Player */}
          <div className="border border-border rounded p-6 bg-background-elevated">
            <div className="flex items-center gap-2 mb-4">
              <Volume2 className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-foreground-bright">Generated Audio</span>
            </div>

            {generatedAudioUrl ? (
              <div className="space-y-4">
                <audio
                  ref={audioRef}
                  src={generatedAudioUrl}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />

                {/* Custom Player */}
                <div className="bg-background-card rounded p-6 flex flex-col items-center">
                  <Button
                    onClick={togglePlayback}
                    size="lg"
                    className="w-16 h-16 rounded-full"
                  >
                    {isPlaying ? (
                      <Pause className="w-6 h-6" />
                    ) : (
                      <Play className="w-6 h-6 ml-1" />
                    )}
                  </Button>

                  {/* Standard audio controls */}
                  <audio
                    src={generatedAudioUrl}
                    controls
                    className="w-full mt-4"
                  />
                </div>

                {/* Download */}
                <Button
                  onClick={downloadAudio}
                  variant="outline"
                  className="w-full"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Audio
                </Button>

                {/* Generation Info */}
                <div className="flex flex-wrap gap-2">
                  <span className="text-xs px-2 py-1 border border-border rounded text-muted-foreground">
                    Emotion: {selectedEmotion || "Auto"}
                  </span>
                  <span className="text-xs px-2 py-1 border border-border rounded text-muted-foreground">
                    Intensity: {Math.round(intensity * 100)}%
                  </span>
                  {referenceAudio && (
                    <span className="text-xs px-2 py-1 border border-border rounded text-foreground">
                      Style transferred
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-background-card rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Volume2 className="w-8 h-8 text-muted-foreground" />
                </div>
                <p className="text-foreground mb-1">No audio generated yet</p>
                <p className="text-muted-foreground text-sm">
                  Enter text and click Generate to create speech
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Right Panel - Tips */}
      <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Current Settings" defaultOpen>
          <StatRow label="Emotion" value={selectedEmotion || "Auto"} />
          <StatRow label="Intensity" value={`${Math.round(intensity * 100)}%`} />
          <StatRow label="Temperature" value={temperature.toFixed(1)} />
          <StatRow label="Characters" value={text.length.toString()} />
        </Section>

        <Section title="Voice Source" defaultOpen>
          <StatRow
            label="Sample"
            value={selectedVoiceSample ? selectedVoiceSample.emotion : "None"}
          />
          <StatRow
            label="Reference"
            value={referenceAudio ? "Uploaded" : "None"}
          />
        </Section>

        <Section title="Tips" defaultOpen>
          <ul className="space-y-2 text-xs text-muted-foreground">
            <li>Use "Happy" for greetings and positive messages</li>
            <li>"Calm" works well for instructions and announcements</li>
            <li>Upload reference audio to match a specific speaking style</li>
            <li>Lower temperature for consistent results, higher for variety</li>
          </ul>
        </Section>
      </aside>
    </div>
  );
}
