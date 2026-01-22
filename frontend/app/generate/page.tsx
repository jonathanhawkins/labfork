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
  Sliders,
  Music,
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

// Emotion presets with descriptions
const EMOTIONS = [
  { id: null, label: "Auto", color: "bg-gradient-to-r from-violet-500 to-purple-500", description: "AI predicts from text" },
  { id: "neutral", label: "Neutral", color: "bg-slate-500", description: "Balanced, natural speech" },
  { id: "happy", label: "Happy", color: "bg-yellow-500", description: "Upbeat, enthusiastic tone" },
  { id: "sad", label: "Sad", color: "bg-blue-500", description: "Slower, lower energy" },
  { id: "angry", label: "Angry", color: "bg-red-500", description: "Intense, emphatic delivery" },
  { id: "calm", label: "Calm", color: "bg-green-500", description: "Relaxed, soothing pace" },
];

export default function GeneratePage() {
  // Input state
  const [text, setText] = useState("");
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null); // null = auto
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
          // Auto-select first sample if available
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
    }, 500); // Debounce 500ms

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

    // Determine which method to use
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
        // Use recorded voice sample
        endpoint = `${API_BASE}/generate-with-voice-sample`;
        formData.append("voice_sample_path", selectedVoiceSample.path);
      } else if (useReferenceAudio && referenceAudio) {
        // Use uploaded reference audio
        endpoint = `${API_BASE}/generate-pocket-tts`;
        formData.append("reference_audio", referenceAudio);
      } else {
        // Use CSM with prosody control
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

      // Get audio blob
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMyMDIwMjAiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRoLTJ2LTRoMnY0em0wLTZ2LTRoLTJ2NGgyem0tNiA2aC00djJoNHYtMnptLTYgMGgtNHYyaDR2LTJ6bTEyLTEydi00aC0ydjRoMnptLTYgMGgtNHYyaDR2LTJ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-50 pointer-events-none" />

      <div className="relative z-10 container mx-auto px-6 py-8 max-w-5xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-2 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Voice Generation</h1>
          </div>
          <p className="text-slate-400">
            Generate speech with prosody control - emotion, intensity, and style transfer
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Input Controls */}
          <div className="space-y-6">
            {/* Text Input */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Music className="w-5 h-5 text-violet-400" />
                  Text to Speak
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Enter the text you want to generate as speech..."
                  className="min-h-[120px] bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500 resize-none"
                />
                <p className="text-xs text-slate-500 mt-2">
                  {text.length} characters
                </p>
              </CardContent>
            </Card>

            {/* Emotion Selector */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-amber-400" />
                  Emotion Control
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Select the emotional tone for the generated speech
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Emotion Grid */}
                <div className="grid grid-cols-3 gap-2">
                  {EMOTIONS.map((emotion) => (
                    <button
                      key={emotion.id ?? "auto"}
                      onClick={() => setSelectedEmotion(emotion.id)}
                      className={`p-3 rounded-xl border transition-all ${
                        selectedEmotion === emotion.id
                          ? "border-violet-500 bg-violet-500/20"
                          : "border-slate-700 bg-slate-800/50 hover:border-slate-600"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${emotion.color}`} />
                        <span className="text-sm font-medium text-white">
                          {emotion.label}
                        </span>
                        {emotion.id === null && isPredicting && (
                          <Loader2 className="w-3 h-3 animate-spin text-violet-400" />
                        )}
                      </div>
                      <p className="text-xs text-slate-500 mt-1 text-left">
                        {emotion.description}
                      </p>
                    </button>
                  ))}
                </div>

                {/* Auto-predicted prosody display */}
                {selectedEmotion === null && predictedProsody && (
                  <div className="bg-violet-500/10 border border-violet-500/30 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="w-4 h-4 text-violet-400" />
                      <span className="text-sm font-medium text-violet-300">
                        AI Prediction
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-slate-400">Emotion:</span>{" "}
                        <span className="text-white capitalize">{predictedProsody.emotion}</span>
                      </div>
                      <div>
                        <span className="text-slate-400">Energy:</span>{" "}
                        <span className="text-white">{Math.round(predictedProsody.energy * 100)}%</span>
                      </div>
                      <div>
                        <span className="text-slate-400">Pace:</span>{" "}
                        <span className="text-white capitalize">{predictedProsody.pace}</span>
                      </div>
                      <div>
                        <span className="text-slate-400">Tone:</span>{" "}
                        <span className="text-white capitalize">{predictedProsody.tone}</span>
                      </div>
                    </div>
                    {predictedProsody.emphasis_words?.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        <span className="text-xs text-slate-400">Emphasis:</span>
                        {predictedProsody.emphasis_words.map((word, i) => (
                          <Badge key={i} variant="outline" className="text-xs text-violet-300 border-violet-500/30">
                            {word}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Intensity Slider */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-slate-400">Emotion Intensity</Label>
                    <span className="text-sm text-white">{Math.round(intensity * 100)}%</span>
                  </div>
                  <Slider
                    value={[intensity]}
                    onValueChange={([v]) => setIntensity(v)}
                    min={0}
                    max={1}
                    step={0.1}
                    className="py-2"
                  />
                </div>

                {/* Temperature Slider */}
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
                  <p className="text-xs text-slate-500">
                    Lower = more consistent, Higher = more varied
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Voice Cloning - Your Recorded Samples */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800 border-emerald-500/30">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Mic className="w-5 h-5 text-emerald-400" />
                  Your Voice
                  {voiceSamples.length > 0 && (
                    <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                      {voiceSamples.length} samples
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Use your recorded voice samples for instant voice cloning
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {isLoadingSamples ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader2 className="w-6 h-6 animate-spin text-emerald-400" />
                  </div>
                ) : voiceSamples.length > 0 ? (
                  <>
                    {/* Emotion filter buttons */}
                    <div className="flex flex-wrap gap-2">
                      {Object.keys(voiceSamplesByEmotion).map((emotion) => (
                        <button
                          key={emotion}
                          onClick={() => {
                            const samples = voiceSamplesByEmotion[emotion];
                            if (samples?.length > 0) {
                              setSelectedVoiceSample(samples[0]);
                            }
                          }}
                          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                            selectedVoiceSample?.emotion === emotion
                              ? "bg-emerald-500 text-white"
                              : "bg-slate-700 text-slate-300 hover:bg-slate-600"
                          }`}
                        >
                          {emotion} ({voiceSamplesByEmotion[emotion]?.length})
                        </button>
                      ))}
                    </div>

                    {/* Selected sample display */}
                    {selectedVoiceSample && (
                      <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-white">
                              Selected: {selectedVoiceSample.emotion}
                            </p>
                            <p className="text-xs text-slate-400">
                              {selectedVoiceSample.filename}
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-emerald-400">Ready to clone</span>
                            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
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
                      className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-sm text-white"
                    >
                      {voiceSamples.map((sample) => (
                        <option key={sample.id} value={sample.id}>
                          {sample.emotion} - {sample.filename}
                        </option>
                      ))}
                    </select>
                  </>
                ) : (
                  <div className="text-center py-4">
                    <p className="text-slate-400 text-sm mb-2">No voice samples recorded yet</p>
                    <a href="/perform" className="text-emerald-400 hover:underline text-sm">
                      Go to Perform page to record samples →
                    </a>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Reference Audio (Style Transfer) - Alternative */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Upload className="w-5 h-5 text-cyan-400" />
                  Upload Reference (Alternative)
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Or upload any audio file to clone that voice
                </CardDescription>
              </CardHeader>
              <CardContent>
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
                    className="w-full h-16 border-dashed border-slate-600 bg-slate-800/30 hover:bg-slate-800/50 text-slate-400"
                  >
                    <div className="flex flex-col items-center gap-1">
                      <Upload className="w-5 h-5" />
                      <span className="text-xs">Upload audio file</span>
                    </div>
                  </Button>
                ) : (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between bg-slate-800/50 rounded-lg p-3">
                      <div className="flex items-center gap-2">
                        <Mic className="w-4 h-4 text-cyan-400" />
                        <span className="text-sm text-white truncate">
                          {referenceAudio?.name}
                        </span>
                      </div>
                      <Button
                        onClick={clearReference}
                        variant="ghost"
                        size="sm"
                        className="text-slate-400 hover:text-white"
                      >
                        Remove
                      </Button>
                    </div>
                    <audio src={referenceUrl} controls className="w-full h-10" />
                    <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30">
                      Prosody will be extracted from this audio
                    </Badge>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right: Generate & Output */}
          <div className="space-y-6">
            {/* Generate Button */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardContent className="pt-6">
                <Button
                  onClick={handleGenerate}
                  disabled={isGenerating || !text.trim()}
                  className="w-full h-14 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 disabled:from-slate-700 disabled:to-slate-800 text-white font-semibold text-lg shadow-lg shadow-violet-500/20"
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

                <div className="flex items-center justify-center gap-4 mt-4 text-xs text-slate-500">
                  <span>Emotion: {selectedEmotion}</span>
                  <span>•</span>
                  <span>Intensity: {Math.round(intensity * 100)}%</span>
                  <span>•</span>
                  <span>Temp: {temperature.toFixed(1)}</span>
                </div>
              </CardContent>
            </Card>

            {/* Output Player */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader>
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Volume2 className="w-5 h-5 text-emerald-400" />
                  Generated Audio
                </CardTitle>
              </CardHeader>
              <CardContent>
                {generatedAudioUrl ? (
                  <div className="space-y-4">
                    <audio
                      ref={audioRef}
                      src={generatedAudioUrl}
                      onEnded={() => setIsPlaying(false)}
                      className="hidden"
                    />

                    {/* Custom Player */}
                    <div className="bg-slate-800/50 rounded-xl p-6">
                      <div className="flex items-center justify-center gap-4">
                        <Button
                          onClick={togglePlayback}
                          size="lg"
                          className="w-16 h-16 rounded-full bg-emerald-500 hover:bg-emerald-600"
                        >
                          {isPlaying ? (
                            <Pause className="w-6 h-6" />
                          ) : (
                            <Play className="w-6 h-6 ml-1" />
                          )}
                        </Button>
                      </div>

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
                      className="w-full border-slate-600 text-slate-300 hover:bg-slate-800"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download Audio
                    </Button>

                    {/* Generation Info */}
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline" className="text-slate-400 border-slate-600">
                        Emotion: {selectedEmotion}
                      </Badge>
                      <Badge variant="outline" className="text-slate-400 border-slate-600">
                        Intensity: {Math.round(intensity * 100)}%
                      </Badge>
                      {referenceAudio && (
                        <Badge variant="outline" className="text-cyan-400 border-cyan-500/30">
                          Style transferred
                        </Badge>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <Volume2 className="w-8 h-8 text-slate-600" />
                    </div>
                    <p className="text-slate-400 font-medium mb-1">
                      No audio generated yet
                    </p>
                    <p className="text-slate-500 text-sm">
                      Enter text and click Generate to create speech
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Tips */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardContent className="pt-6">
                <h3 className="text-sm font-medium text-slate-400 mb-3">Tips</h3>
                <ul className="space-y-2 text-sm text-slate-500">
                  <li className="flex gap-2">
                    <span className="text-violet-400">•</span>
                    Use "Happy" for greetings and positive messages
                  </li>
                  <li className="flex gap-2">
                    <span className="text-violet-400">•</span>
                    "Calm" works well for instructions and announcements
                  </li>
                  <li className="flex gap-2">
                    <span className="text-violet-400">•</span>
                    Upload reference audio to match a specific speaking style
                  </li>
                  <li className="flex gap-2">
                    <span className="text-violet-400">•</span>
                    Lower temperature for consistent results, higher for variety
                  </li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
