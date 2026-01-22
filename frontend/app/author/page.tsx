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
  PanelLeftClose,
  PanelLeft,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
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
  const [playbackTime, setPlaybackTime] = useState(0); // 0.0 to 1.0 normalized
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

    setIsGenerating(true);
    const toastId = toast.loading("Generating with keyframe prosody...");

    try {
      // Convert keyframes to API format (time in seconds)
      const apiKeyframes = keyframes.map((kf) => ({
        time: kf.time * duration,
        emotion: kf.emotion,
        intensity: kf.intensity / 100, // API expects 0-1
      }));

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

      // Get audio blob
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      // Save current as previous before setting new
      if (generatedAudioUrl) {
        setPreviousAudioUrl(generatedAudioUrl);
        setPreviousKeyframes([...keyframes]);
      }

      setGeneratedAudioUrl(audioUrl);

      // Add to history
      setGenerationHistory((prev) => [
        { url: audioUrl, keyframes: [...keyframes], text },
        ...prev.slice(0, 4), // Keep last 5
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
      // Stop previous if playing
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
      // Stop current if playing
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMyMDIwMjAiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRoLTJ2LTRoMnY0em0wLTZ2LTRoLTJ2NGgyem0tNiA2aC00djJoNHYtMnptLTYgMGgtNHYyaDR2LTJ6bTEyLTEydi00aC0ydjRoMnptLTYgMGgtNHYyaDR2LTJ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-50 pointer-events-none" />

      <div className="relative z-10 container mx-auto px-6 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-2 bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl">
              <Film className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Prosody Director</h1>
          </div>
          <p className="text-slate-400">
            Author voice delivery like a director - place emotion keyframes on a timeline
          </p>
        </div>

        <div className="flex gap-6">
          {/* Left: Collapsible Sidebar */}
          <div
            className={`transition-all duration-300 ease-in-out ${
              sidebarCollapsed ? "w-12" : "w-80"
            } flex-shrink-0`}
          >
            {/* Collapse toggle button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="w-full mb-4 justify-start text-slate-400 hover:text-white hover:bg-slate-800"
            >
              {sidebarCollapsed ? (
                <>
                  <ChevronRight className="w-4 h-4" />
                </>
              ) : (
                <>
                  <ChevronLeft className="w-4 h-4 mr-2" />
                  <span>Collapse</span>
                </>
              )}
            </Button>

            {/* Sidebar content - hidden when collapsed */}
            <div className={`space-y-6 ${sidebarCollapsed ? "hidden" : "block"}`}>
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg text-white">Scene Presets</CardTitle>
                  <CardDescription className="text-slate-400">
                    Start with a template
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  {SCENE_PRESETS.map((preset, i) => (
                    <button
                      key={i}
                      onClick={() => loadPreset(preset)}
                      className="w-full text-left p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-slate-600 transition-all"
                    >
                      <div className="font-medium text-white text-sm">{preset.name}</div>
                      <div className="text-xs text-slate-500 mt-1 truncate">
                        {preset.text}
                      </div>
                      <div className="flex gap-1 mt-2">
                        {preset.keyframes.map((kf, j) => (
                          <div
                            key={j}
                            className={`w-2 h-2 rounded-full ${
                              kf.emotion === "happy" ? "bg-yellow-500" :
                              kf.emotion === "sad" ? "bg-blue-500" :
                              kf.emotion === "angry" ? "bg-red-500" :
                              kf.emotion === "surprised" ? "bg-orange-500" :
                              kf.emotion === "calm" ? "bg-emerald-500" :
                              kf.emotion === "excited" ? "bg-pink-500" :
                              "bg-slate-500"
                            }`}
                            style={{ opacity: 0.3 + (kf.intensity / 100) * 0.7 }}
                          />
                        ))}
                      </div>
                    </button>
                  ))}
                </CardContent>
              </Card>

              {/* Generation History */}
              {generationHistory.length > 0 && (
                <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-white">History</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {generationHistory.map((item, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setGeneratedAudioUrl(item.url);
                          setText(item.text);
                          setKeyframes(item.keyframes);
                        }}
                        className="w-full text-left p-2 rounded bg-slate-800/30 hover:bg-slate-800/50 transition-all"
                      >
                        <div className="text-xs text-slate-400 truncate">
                          {item.text}
                        </div>
                        <div className="flex gap-1 mt-1">
                          {item.keyframes.slice(0, 6).map((kf, j) => (
                            <div
                              key={j}
                              className={`w-1.5 h-1.5 rounded-full ${
                                kf.emotion === "happy" ? "bg-yellow-500" :
                                kf.emotion === "sad" ? "bg-blue-500" :
                                kf.emotion === "angry" ? "bg-red-500" :
                                "bg-slate-500"
                              }`}
                            />
                          ))}
                        </div>
                      </button>
                    ))}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          {/* Main: Timeline Editor - takes remaining space */}
          <div className="flex-1 min-w-0 space-y-6">
            {/* Text Input */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg text-white">Script</CardTitle>
                <CardDescription className="text-slate-400">
                  Enter the dialogue or narration to synthesize
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter the text you want to speak..."
                  className="min-h-[80px] bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 resize-none"
                />
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">{text.length} characters</span>
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-slate-400">Duration (s)</Label>
                    <input
                      type="number"
                      value={duration}
                      onChange={(e) => setDuration(parseFloat(e.target.value) || 1)}
                      className="w-16 h-8 px-2 text-sm bg-slate-800 border border-slate-700 rounded text-white"
                      min={0.5}
                      max={30}
                      step={0.5}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Keyframe Timeline */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-lg text-white">Emotion Timeline</CardTitle>
                  <CardDescription className="text-slate-400">
                    Place keyframes to control emotion arc (double-click to add)
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  className="border-slate-700 text-slate-400 hover:text-white"
                >
                  <RotateCcw className="w-4 h-4 mr-1" />
                  Reset
                </Button>
              </CardHeader>
              <CardContent>
                <KeyframeTimeline
                  text={text}
                  keyframes={keyframes}
                  onChange={setKeyframes}
                  duration={duration}
                  playbackTime={playbackTime}
                  isPlaying={isPlaying}
                />
              </CardContent>
            </Card>

            {/* Generation Controls */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardContent className="pt-6 space-y-4">
                {/* Temperature */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-slate-400">Variation (Temperature)</Label>
                    <span className="text-sm text-white">{temperature.toFixed(1)}</span>
                  </div>
                  <Slider
                    value={[temperature]}
                    onValueChange={([v]) => setTemperature(v)}
                    min={0.1}
                    max={1.5}
                    step={0.1}
                    className="py-2"
                  />
                </div>

                {/* Generate Button */}
                <Button
                  onClick={handleGenerate}
                  disabled={isGenerating || !text.trim() || keyframes.length < 2}
                  className="w-full h-14 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 disabled:from-slate-700 disabled:to-slate-800 text-white font-semibold text-lg shadow-lg shadow-amber-500/20"
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

                {/* Info badges */}
                <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
                  <span>{keyframes.length} keyframes</span>
                  <span>•</span>
                  <span>{duration}s duration</span>
                  <span>•</span>
                  <span>Catmull-Rom interpolation</span>
                </div>
              </CardContent>
            </Card>

            {/* Output Players - A/B Comparison */}
            {generatedAudioUrl && (
              <div className={`grid gap-4 ${previousAudioUrl ? "grid-cols-2" : "grid-cols-1"}`}>
                {/* Current Generation */}
                <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800 border-amber-500/30">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      <Volume2 className="w-5 h-5 text-amber-400" />
                      Current Generation
                      <Badge className="ml-auto bg-amber-500/20 text-amber-400 border-amber-500/30">
                        A
                      </Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <audio
                      ref={audioRef}
                      src={generatedAudioUrl}
                      onEnded={() => setIsPlaying(false)}
                      className="hidden"
                    />

                    {/* Custom Player */}
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <div className="flex items-center justify-center gap-4">
                        <Button
                          onClick={togglePlayback}
                          size="lg"
                          className={`w-14 h-14 rounded-full transition-all ${
                            isPlaying
                              ? "bg-amber-500 hover:bg-amber-600 ring-4 ring-amber-500/30"
                              : "bg-amber-500 hover:bg-amber-600"
                          }`}
                        >
                          {isPlaying ? (
                            <Pause className="w-5 h-5" />
                          ) : (
                            <Play className="w-5 h-5 ml-0.5" />
                          )}
                        </Button>
                      </div>

                      {/* Standard audio controls */}
                      <audio
                        src={generatedAudioUrl}
                        controls
                        className="w-full mt-3 h-8"
                      />
                    </div>

                    {/* Actions */}
                    <div className="flex gap-2 mt-3">
                      <Button
                        onClick={downloadAudio}
                        variant="outline"
                        size="sm"
                        className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-800"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        Download
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="border-slate-600 text-slate-300 hover:bg-slate-800"
                      >
                        <Save className="w-3 h-3 mr-1" />
                        Save
                      </Button>
                    </div>

                    {/* Keyframe summary */}
                    <div className="flex flex-wrap gap-1 mt-3">
                      {keyframes.slice(0, 4).map((kf, i) => (
                        <Badge
                          key={i}
                          variant="outline"
                          className={`text-xs ${
                            kf.emotion === "happy" ? "text-yellow-400 border-yellow-500/30" :
                            kf.emotion === "sad" ? "text-blue-400 border-blue-500/30" :
                            kf.emotion === "angry" ? "text-red-400 border-red-500/30" :
                            kf.emotion === "surprised" ? "text-orange-400 border-orange-500/30" :
                            kf.emotion === "calm" ? "text-emerald-400 border-emerald-500/30" :
                            kf.emotion === "excited" ? "text-pink-400 border-pink-500/30" :
                            "text-slate-400 border-slate-500/30"
                          }`}
                        >
                          {kf.emotion}
                        </Badge>
                      ))}
                      {keyframes.length > 4 && (
                        <Badge variant="outline" className="text-xs text-slate-400 border-slate-500/30">
                          +{keyframes.length - 4}
                        </Badge>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Previous Generation - Only shown if exists */}
                {previousAudioUrl && (
                  <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800 border-slate-600/30">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg text-white flex items-center gap-2">
                        <Volume2 className="w-5 h-5 text-slate-400" />
                        Previous Generation
                        <Badge className="ml-auto bg-slate-500/20 text-slate-400 border-slate-500/30">
                          B
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <audio
                        ref={previousAudioRef}
                        src={previousAudioUrl}
                        onEnded={() => setIsPreviousPlaying(false)}
                        className="hidden"
                      />

                      {/* Custom Player */}
                      <div className="bg-slate-800/50 rounded-xl p-4">
                        <div className="flex items-center justify-center gap-4">
                          <Button
                            onClick={togglePreviousPlayback}
                            size="lg"
                            className={`w-14 h-14 rounded-full transition-all ${
                              isPreviousPlaying
                                ? "bg-slate-500 hover:bg-slate-600 ring-4 ring-slate-500/30"
                                : "bg-slate-600 hover:bg-slate-500"
                            }`}
                          >
                            {isPreviousPlaying ? (
                              <Pause className="w-5 h-5" />
                            ) : (
                              <Play className="w-5 h-5 ml-0.5" />
                            )}
                          </Button>
                        </div>

                        {/* Standard audio controls */}
                        <audio
                          src={previousAudioUrl}
                          controls
                          className="w-full mt-3 h-8"
                        />
                      </div>

                      {/* Keyframe summary for previous */}
                      <div className="flex flex-wrap gap-1 mt-3">
                        {previousKeyframes.slice(0, 4).map((kf, i) => (
                          <Badge
                            key={i}
                            variant="outline"
                            className={`text-xs ${
                              kf.emotion === "happy" ? "text-yellow-400 border-yellow-500/30" :
                              kf.emotion === "sad" ? "text-blue-400 border-blue-500/30" :
                              kf.emotion === "angry" ? "text-red-400 border-red-500/30" :
                              kf.emotion === "surprised" ? "text-orange-400 border-orange-500/30" :
                              kf.emotion === "calm" ? "text-emerald-400 border-emerald-500/30" :
                              kf.emotion === "excited" ? "text-pink-400 border-pink-500/30" :
                              "text-slate-400 border-slate-500/30"
                            }`}
                          >
                            {kf.emotion}
                          </Badge>
                        ))}
                        {previousKeyframes.length > 4 && (
                          <Badge variant="outline" className="text-xs text-slate-400 border-slate-500/30">
                            +{previousKeyframes.length - 4}
                          </Badge>
                        )}
                      </div>

                      {/* Hint */}
                      <p className="text-xs text-slate-500 mt-3 text-center">
                        Compare with current to hear the difference
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
