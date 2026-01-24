"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import { toast } from "sonner";
import {
  Play,
  Pause,
  Download,
  Loader2,
  Sparkles,
  Volume2,
  Film,
  RotateCcw,
  Save,
  ChevronLeft,
  ChevronRight,
  Circle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import KeyframeTimeline, { Keyframe } from "@/components/KeyframeTimeline";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Example movie scene presets
const SCENE_PRESETS = [
  {
    name: "Dramatic Revelation",
    text: "I can't believe you did this. After everything we've been through together.",
    keyframes: [
      { id: "1", time: 0.0, emotion: "neutral", intensity: 50 },
      { id: "2", time: 0.25, emotion: "surprised", intensity: 80 },
      { id: "3", time: 0.5, emotion: "sad", intensity: 70 },
      { id: "4", time: 0.85, emotion: "angry", intensity: 60 },
    ],
    duration: 4.0,
  },
  {
    name: "Building Excitement",
    text: "Wait, are you saying... we actually won? We won the championship!",
    keyframes: [
      { id: "1", time: 0.0, emotion: "neutral", intensity: 40 },
      { id: "2", time: 0.3, emotion: "surprised", intensity: 70 },
      { id: "3", time: 0.6, emotion: "excited", intensity: 90 },
      { id: "4", time: 1.0, emotion: "happy", intensity: 100 },
    ],
    duration: 3.5,
  },
  {
    name: "Calming Down",
    text: "It's okay. Take a deep breath. Everything is going to be fine.",
    keyframes: [
      { id: "1", time: 0.0, emotion: "calm", intensity: 60 },
      { id: "2", time: 0.4, emotion: "calm", intensity: 80 },
      { id: "3", time: 1.0, emotion: "calm", intensity: 90 },
    ],
    duration: 3.0,
  },
  {
    name: "Villain Monologue",
    text: "You think you've won? This is only the beginning. I will return.",
    keyframes: [
      { id: "1", time: 0.0, emotion: "angry", intensity: 60 },
      { id: "2", time: 0.35, emotion: "calm", intensity: 70 },
      { id: "3", time: 0.7, emotion: "angry", intensity: 90 },
      { id: "4", time: 1.0, emotion: "neutral", intensity: 50 },
    ],
    duration: 4.0,
  },
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

export default function AuthorPage() {
  // UI state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Input state
  const [text, setText] = useState(SCENE_PRESETS[0].text);
  const [keyframes, setKeyframes] = useState<Keyframe[]>(SCENE_PRESETS[0].keyframes);
  const [duration, setDuration] = useState(SCENE_PRESETS[0].duration);
  const [temperature, setTemperature] = useState(0.8);

  // Generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState<string | null>(null);
  const [previousAudioUrl, setPreviousAudioUrl] = useState<string | null>(null);
  const [previousKeyframes, setPreviousKeyframes] = useState<Keyframe[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPreviousPlaying, setIsPreviousPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [generationHistory, setGenerationHistory] = useState<
    Array<{ url: string; keyframes: Keyframe[]; text: string }>
  >([]);

  const audioRef = useRef<HTMLAudioElement>(null);
  const previousAudioRef = useRef<HTMLAudioElement>(null);

  // Track audio playback position
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      if (audio.duration > 0) {
        setPlaybackTime(audio.currentTime / audio.duration);
      }
    };

    const handleDurationChange = () => {
      setAudioDuration(audio.duration);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setPlaybackTime(0);
    };

    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("durationchange", handleDurationChange);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("durationchange", handleDurationChange);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [generatedAudioUrl]);

  // Load a preset
  const loadPreset = useCallback((preset: typeof SCENE_PRESETS[0]) => {
    setText(preset.text);
    setKeyframes(preset.keyframes);
    setDuration(preset.duration);
    setGeneratedAudioUrl(null);
    toast.success(`Loaded "${preset.name}" preset`);
  }, []);

  // Reset timeline
  const handleReset = useCallback(() => {
    setKeyframes([
      { id: "start", time: 0.0, emotion: "neutral", intensity: 50 },
      { id: "end", time: 1.0, emotion: "neutral", intensity: 50 },
    ]);
  }, []);

  // Generate with keyframes
  const handleGenerate = async () => {
    if (!text.trim()) {
      toast.error("Please enter some text to generate");
      return;
    }

    if (keyframes.length < 2) {
      toast.error("Please add at least 2 keyframes");
      return;
    }

    const wordCount = text.split(/\s+/).filter((word) => word.length > 0).length;

    setIsGenerating(true);
    const toastId = toast.loading("Generating with keyframe prosody...");

    try {
      const apiKeyframes = keyframes.map((kf) => {
        if (kf.anchor === "word" && typeof kf.wordIndex === "number") {
          if (kf.wordIndex < 0 || kf.wordIndex >= wordCount) {
            throw new Error(
              `Keyframe word index ${kf.wordIndex} is out of range for ${wordCount} words`
            );
          }
          return {
            word_index: kf.wordIndex,
            emotion: kf.emotion,
            intensity: kf.intensity / 100,
          };
        }
        return {
          time: kf.time * duration,
          emotion: kf.emotion,
          intensity: kf.intensity / 100,
        };
      });

      const response = await fetch(`${API_BASE}/generate-keyframes`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          keyframes: apiKeyframes,
          temperature,
        }),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "Generation failed");
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      if (generatedAudioUrl) {
        setPreviousAudioUrl(generatedAudioUrl);
        setPreviousKeyframes([...keyframes]);
      }

      setGeneratedAudioUrl(audioUrl);

      setGenerationHistory((prev) => [
        { url: audioUrl, keyframes: [...keyframes], text },
        ...prev.slice(0, 4),
      ]);

      toast.success("Speech generated with keyframe prosody!", { id: toastId });
    } catch (err) {
      console.error("Generation failed:", err);
      toast.error(
        err instanceof Error ? err.message : "Generation failed. Is the backend running?",
        { id: toastId }
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (previousAudioRef.current && isPreviousPlaying) {
        previousAudioRef.current.pause();
        setIsPreviousPlaying(false);
      }
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const togglePreviousPlayback = () => {
    if (previousAudioRef.current) {
      if (audioRef.current && isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      }
      if (isPreviousPlaying) {
        previousAudioRef.current.pause();
      } else {
        previousAudioRef.current.play();
      }
      setIsPreviousPlaying(!isPreviousPlaying);
    }
  };

  const downloadAudio = () => {
    if (generatedAudioUrl) {
      const a = document.createElement("a");
      a.href = generatedAudioUrl;
      a.download = `authored_${Date.now()}.wav`;
      a.click();
    }
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Presets & History */}
      <aside className={`flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto transition-all duration-200 ${sidebarCollapsed ? "w-12" : "w-[280px]"}`}>
        {/* Collapse toggle */}
        <div className="border-b border-border">
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="w-full flex items-center justify-between py-3 px-4 text-muted-foreground hover:text-foreground transition-colors"
          >
            {!sidebarCollapsed && <span className="text-sm">Collapse</span>}
            {sidebarCollapsed ? <ChevronRight className="w-4 h-4 mx-auto" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {!sidebarCollapsed && (
          <>
            <Section title="Scene Presets" defaultOpen>
              <div className="space-y-2">
                {SCENE_PRESETS.map((preset, i) => (
                  <button
                    key={i}
                    onClick={() => loadPreset(preset)}
                    className="w-full text-left p-3 rounded border border-border bg-background-card hover:border-muted-foreground transition-colors"
                  >
                    <div className="text-sm text-foreground-bright">{preset.name}</div>
                    <div className="text-xs text-muted-foreground mt-1 truncate">
                      {preset.text}
                    </div>
                    <div className="flex gap-1 mt-2">
                      {preset.keyframes.map((kf, j) => (
                        <div
                          key={j}
                          className="w-2 h-2 rounded-full bg-foreground"
                          style={{ opacity: 0.3 + (kf.intensity / 100) * 0.7 }}
                        />
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </Section>

            {generationHistory.length > 0 && (
              <Section title="History">
                <div className="space-y-2">
                  {generationHistory.map((item, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        setGeneratedAudioUrl(item.url);
                        setText(item.text);
                        setKeyframes(item.keyframes);
                      }}
                      className="w-full text-left p-2 rounded bg-background-card hover:bg-background transition-colors"
                    >
                      <div className="text-xs text-muted-foreground truncate">
                        {item.text}
                      </div>
                      <div className="flex gap-1 mt-1">
                        {item.keyframes.slice(0, 6).map((kf, j) => (
                          <div
                            key={j}
                            className="w-1.5 h-1.5 rounded-full bg-foreground"
                          />
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              </Section>
            )}

            {/* Footer */}
            <div className="px-4 py-3 border-t border-border mt-auto">
              <div className="text-xs text-muted-foreground">
                Prosody Director
              </div>
              <div className="text-xxs text-foreground-subtle mt-0.5">
                Keyframe emotion control
              </div>
            </div>
          </>
        )}
      </aside>

      {/* Main Content - Timeline Editor */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto p-8">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Header */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Film className="w-5 h-5 text-foreground-bright" />
              <h1 className="text-lg text-foreground-bright">Prosody Director</h1>
            </div>
            <p className="text-sm text-muted-foreground">
              Author voice delivery like a director - place emotion keyframes on a timeline
            </p>
          </div>

          {/* Script Input */}
          <div className="border border-border rounded p-4 bg-background-elevated">
            <div className="flex items-center justify-between mb-3">
              <Label className="text-sm text-foreground-bright">Script</Label>
              <div className="flex items-center gap-2">
                <Label className="text-xs text-muted-foreground">Duration (s)</Label>
                <input
                  type="number"
                  value={duration}
                  onChange={(e) => setDuration(parseFloat(e.target.value) || 1)}
                  className="w-16 h-8 px-2 text-sm bg-background border border-border rounded text-foreground"
                  min={0.5}
                  max={30}
                  step={0.5}
                />
              </div>
            </div>
            <Textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter the text you want to speak..."
              className="min-h-[80px] bg-background border-border text-foreground placeholder:text-muted-foreground resize-none"
            />
            <p className="text-xs text-muted-foreground mt-2">{text.length} characters</p>
          </div>

          {/* Keyframe Timeline */}
          <div className="border border-border rounded p-4 bg-background-elevated">
            <div className="flex items-center justify-between mb-3">
              <div>
                <h3 className="text-sm text-foreground-bright">Emotion Timeline</h3>
                <p className="text-xs text-muted-foreground">
                  Double-click to add keyframes, drag to move
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={handleReset}
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Reset
              </Button>
            </div>
            <KeyframeTimeline
              text={text}
              keyframes={keyframes}
              onChange={setKeyframes}
              duration={duration}
              playbackTime={playbackTime}
              isPlaying={isPlaying}
            />
          </div>

          {/* Generation Controls */}
          <div className="border border-border rounded p-4 bg-background-elevated space-y-4">
            {/* Temperature */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label className="text-sm text-muted-foreground">Temperature</Label>
                <span className="text-sm text-foreground">{temperature.toFixed(1)}</span>
              </div>
              <Slider
                value={[temperature]}
                onValueChange={([v]) => setTemperature(v)}
                min={0.1}
                max={1.5}
                step={0.1}
              />
            </div>

            {/* Generate Button */}
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || !text.trim() || keyframes.length < 2}
              className="w-full h-14"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Generating with Keyframes...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 mr-2" />
                  Generate Authored Speech
                </>
              )}
            </Button>

            {/* Info */}
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
              <span>{keyframes.length} keyframes</span>
              <span>|</span>
              <span>{duration}s duration</span>
              <span>|</span>
              <span>Catmull-Rom interpolation</span>
            </div>
          </div>

          {/* Output Players */}
          {generatedAudioUrl && (
            <div className={`grid gap-4 ${previousAudioUrl ? "grid-cols-2" : "grid-cols-1"}`}>
              {/* Current Generation */}
              <div className="border border-border rounded p-4 bg-background-elevated">
                <div className="flex items-center gap-2 mb-4">
                  <Volume2 className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-foreground-bright">Current</span>
                  <span className="text-xs px-2 py-0.5 border border-border rounded text-muted-foreground ml-auto">A</span>
                </div>

                <audio
                  ref={audioRef}
                  src={generatedAudioUrl}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />

                <div className="bg-background-card rounded p-4 flex flex-col items-center">
                  <Button
                    onClick={togglePlayback}
                    size="lg"
                    className={`w-14 h-14 rounded-full ${isPlaying ? "ring-2 ring-foreground-bright/30" : ""}`}
                  >
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
                  </Button>
                  <audio src={generatedAudioUrl} controls className="w-full mt-3 h-8" />
                </div>

                <div className="flex gap-2 mt-3">
                  <Button onClick={downloadAudio} variant="outline" size="sm" className="flex-1">
                    <Download className="w-3 h-3 mr-1" />
                    Download
                  </Button>
                  <Button variant="outline" size="sm">
                    <Save className="w-3 h-3 mr-1" />
                    Save
                  </Button>
                </div>

                <div className="flex flex-wrap gap-1 mt-3">
                  {keyframes.slice(0, 4).map((kf, i) => (
                    <span key={i} className="text-xs px-2 py-0.5 border border-border rounded text-muted-foreground">
                      {kf.emotion}
                    </span>
                  ))}
                  {keyframes.length > 4 && (
                    <span className="text-xs px-2 py-0.5 border border-border rounded text-muted-foreground">
                      +{keyframes.length - 4}
                    </span>
                  )}
                </div>
              </div>

              {/* Previous Generation */}
              {previousAudioUrl && (
                <div className="border border-border rounded p-4 bg-background-elevated">
                  <div className="flex items-center gap-2 mb-4">
                    <Volume2 className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-foreground">Previous</span>
                    <span className="text-xs px-2 py-0.5 border border-border rounded text-muted-foreground ml-auto">B</span>
                  </div>

                  <audio
                    ref={previousAudioRef}
                    src={previousAudioUrl}
                    onEnded={() => setIsPreviousPlaying(false)}
                    className="hidden"
                  />

                  <div className="bg-background-card rounded p-4 flex flex-col items-center">
                    <Button
                      onClick={togglePreviousPlayback}
                      variant="outline"
                      size="lg"
                      className={`w-14 h-14 rounded-full ${isPreviousPlaying ? "ring-2 ring-foreground/30" : ""}`}
                    >
                      {isPreviousPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
                    </Button>
                    <audio src={previousAudioUrl} controls className="w-full mt-3 h-8" />
                  </div>

                  <div className="flex flex-wrap gap-1 mt-3">
                    {previousKeyframes.slice(0, 4).map((kf, i) => (
                      <span key={i} className="text-xs px-2 py-0.5 border border-border rounded text-muted-foreground">
                        {kf.emotion}
                      </span>
                    ))}
                    {previousKeyframes.length > 4 && (
                      <span className="text-xs px-2 py-0.5 border border-border rounded text-muted-foreground">
                        +{previousKeyframes.length - 4}
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-muted-foreground mt-3 text-center">
                    Compare with current to hear the difference
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Right Panel - Settings */}
      <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Current Settings" defaultOpen>
          <StatRow label="Keyframes" value={keyframes.length.toString()} />
          <StatRow label="Duration" value={`${duration}s`} />
          <StatRow label="Temperature" value={temperature.toFixed(1)} />
          <StatRow label="Characters" value={text.length.toString()} />
        </Section>

        <Section title="Keyframe Info" defaultOpen>
          <div className="space-y-2">
            {keyframes.map((kf, i) => (
              <div key={kf.id} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <Circle className="w-2 h-2 fill-foreground text-foreground" />
                  <span className="text-muted-foreground">{kf.emotion}</span>
                </div>
                <span className="text-foreground">{(kf.time * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Tips" defaultOpen>
          <ul className="space-y-2 text-xs text-muted-foreground">
            <li>Double-click timeline to add keyframes</li>
            <li>Drag keyframes to reposition</li>
            <li>Click keyframe to edit emotion</li>
            <li>Use presets for common patterns</li>
          </ul>
        </Section>
      </aside>
    </div>
  );
}
