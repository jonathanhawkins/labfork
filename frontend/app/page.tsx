"use client";

import React, { useState, useRef, useEffect } from "react";
import dynamic from "next/dynamic";
import { toast } from "sonner";
import {
  Mic,
  Square,
  Upload,
  Check,
  X,
  Package,
  Clock,
  FileAudio,
  AudioWaveform,
  Activity,
  Zap,
  Target,
  Volume2,
  Loader2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

// Dynamic import for 3D visualizer (no SSR)
const ProsodyVisualizer = dynamic(
  () => import("../components/ProsodyMatrixVisualizer"),
  { ssr: false }
);

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Sample {
  id: string;
  duration: number;
  transcript?: string;
  prosody?: {
    acoustic?: Record<string, number>;
    rhythm?: Record<string, number>;
    semantic?: {
      emotion?: string;
      tone?: string;
      energy_level?: string;
      pace_category?: string;
      emphasis_words?: string[];
    };
  };
  approved: boolean;
}

export default function VoiceRecorderPage() {
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // Current sample
  const [currentSample, setCurrentSample] = useState<Sample | null>(null);

  // All samples
  const [samples, setSamples] = useState<Sample[]>([]);
  const [stats, setStats] = useState<{
    total_samples: number;
    approved_samples: number;
    total_duration_minutes: number;
    approved_duration_minutes: number;
  } | null>(null);

  // Audio refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Visualizer data
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [prosodyData, setProsodyData] = useState<any>(null);

  // Load samples on mount
  useEffect(() => {
    fetchSamples();
    fetchStats();
  }, []);

  const fetchSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/samples`);
      const data = await res.json();
      setSamples(data.samples || []);
    } catch (err) {
      console.error("Failed to fetch samples:", err);
      toast.error("Failed to fetch samples");
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Setup audio context for visualization
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      setAnalyserNode(analyser);

      // Setup media recorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));

        // Cleanup
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setIsRecording(true);
      toast.success("Recording started");
    } catch (err) {
      console.error("Failed to start recording:", err);
      toast.error("Could not access microphone. Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      toast.info("Recording stopped");
    }
  };

  const uploadAndProcess = async () => {
    if (!audioBlob) return;

    setIsProcessing(true);
    const toastId = toast.loading("Processing recording...");

    try {
      // Upload
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");

      const uploadRes = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!uploadRes.ok) throw new Error("Upload failed");

      const uploadData = await uploadRes.json();
      const sampleId = uploadData.id;

      // Process (transcribe + analyze)
      const processRes = await fetch(`${API_BASE}/process/${sampleId}`, {
        method: "POST",
      });

      if (!processRes.ok) throw new Error("Processing failed");

      const processData = await processRes.json();
      setCurrentSample(processData.sample);

      // Update prosody data for visualizer
      if (processData.sample?.prosody) {
        setProsodyData(processData.sample.prosody);
      }

      // Refresh samples list
      fetchSamples();
      fetchStats();
      toast.success("Recording processed successfully", { id: toastId });
    } catch (err) {
      console.error("Processing failed:", err);
      toast.error("Processing failed. Check console for details.", { id: toastId });
    } finally {
      setIsProcessing(false);
    }
  };

  const approveSample = async () => {
    if (!currentSample) return;

    try {
      await fetch(`${API_BASE}/sample/${currentSample.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved: true }),
      });

      toast.success("Sample approved and added to dataset");

      // Reset for next recording
      setCurrentSample(null);
      setAudioBlob(null);
      setAudioUrl(null);
      setProsodyData(null);

      fetchSamples();
      fetchStats();
    } catch (err) {
      console.error("Approval failed:", err);
      toast.error("Failed to approve sample");
    }
  };

  const discardSample = async () => {
    if (!currentSample) return;

    try {
      await fetch(`${API_BASE}/sample/${currentSample.id}`, {
        method: "DELETE",
      });

      toast.info("Sample discarded");

      // Reset
      setCurrentSample(null);
      setAudioBlob(null);
      setAudioUrl(null);
      setProsodyData(null);

      fetchSamples();
      fetchStats();
    } catch (err) {
      console.error("Delete failed:", err);
      toast.error("Failed to discard sample");
    }
  };

  const exportDataset = async () => {
    const toastId = toast.loading("Exporting dataset...");
    try {
      const res = await fetch(`${API_BASE}/export`, { method: "POST" });
      const data = await res.json();
      toast.success(
        `Exported ${data.count} samples (${data.total_duration_minutes.toFixed(1)} min) to ${data.path}`,
        { id: toastId, duration: 5000 }
      );
    } catch (err) {
      console.error("Export failed:", err);
      toast.error("Export failed. Make sure you have approved samples.", { id: toastId });
    }
  };

  const progressPercentage = stats
    ? Math.min(100, (stats.approved_duration_minutes / 60) * 100)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Subtle background pattern */}
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMyMDIwMjAiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRoLTJ2LTRoMnY0em0wLTZ2LTRoLTJ2NGgyem0tNiA2aC00djJoNHYtMnptLTYgMGgtNHYyaDR2LTJ6bTEyLTEydi00aC0ydjRoMnptLTYgMGgtNHYyaDR2LTJ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-50 pointer-events-none" />

      <div className="relative z-10 container mx-auto px-6 py-8 max-w-7xl">

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Recording & Controls */}
          <div className="lg:col-span-5 space-y-6">
            {/* Dataset Progress Card */}
            {stats && (
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <div className="p-2 bg-emerald-500/10 rounded-lg">
                        <Target className="w-5 h-5 text-emerald-400" />
                      </div>
                      <CardTitle className="text-lg text-white">
                        Dataset Progress
                      </CardTitle>
                    </div>
                    <span className="text-2xl font-bold text-emerald-400">
                      {progressPercentage.toFixed(0)}%
                    </span>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Progress Bar */}
                  <div className="relative">
                    <Progress
                      value={progressPercentage}
                      className="h-3 bg-slate-800"
                      indicatorClassName="bg-gradient-to-r from-emerald-500 to-emerald-400"
                    />
                    <div className="absolute -top-1 left-0 w-full h-5 flex items-center">
                      <div
                        className="h-5 w-0.5 bg-amber-400/50 absolute"
                        style={{ left: "100%" }}
                        title="Target: 60 min"
                      />
                    </div>
                  </div>

                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-1">
                        <Check className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                          Approved
                        </span>
                      </div>
                      <p className="text-2xl font-bold text-white">
                        {stats.approved_samples}
                        <span className="text-sm font-normal text-slate-500 ml-1">
                          samples
                        </span>
                      </p>
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-1">
                        <Clock className="w-4 h-4 text-violet-400" />
                        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                          Duration
                        </span>
                      </div>
                      <p className="text-2xl font-bold text-white">
                        {stats.approved_duration_minutes.toFixed(1)}
                        <span className="text-sm font-normal text-slate-500 ml-1">
                          / 60 min
                        </span>
                      </p>
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-1">
                        <FileAudio className="w-4 h-4 text-blue-400" />
                        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                          Total
                        </span>
                      </div>
                      <p className="text-2xl font-bold text-white">
                        {stats.total_samples}
                        <span className="text-sm font-normal text-slate-500 ml-1">
                          samples
                        </span>
                      </p>
                    </div>
                    <div className="bg-slate-800/50 rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-1">
                        <Activity className="w-4 h-4 text-amber-400" />
                        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                          Total Time
                        </span>
                      </div>
                      <p className="text-2xl font-bold text-white">
                        {stats.total_duration_minutes.toFixed(1)}
                        <span className="text-sm font-normal text-slate-500 ml-1">
                          min
                        </span>
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recording Controls Card */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2.5">
                  <div
                    className={`p-2 rounded-lg transition-colors ${
                      isRecording ? "bg-red-500/20" : "bg-violet-500/10"
                    }`}
                  >
                    <Mic
                      className={`w-5 h-5 ${
                        isRecording ? "text-red-400" : "text-violet-400"
                      }`}
                    />
                  </div>
                  <CardTitle className="text-lg text-white">
                    Recording Studio
                  </CardTitle>
                  {isRecording && (
                    <span className="ml-auto flex items-center gap-2">
                      <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                      <span className="text-sm text-red-400 font-medium">
                        Recording
                      </span>
                    </span>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Recording Button */}
                {!isRecording ? (
                  <Button
                    onClick={startRecording}
                    disabled={isProcessing}
                    className="w-full h-12 bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600 text-white font-semibold shadow-lg shadow-red-500/20 hover:shadow-red-500/30 transition-all duration-200"
                  >
                    <Mic className="w-5 h-5 mr-2" />
                    Start Recording
                  </Button>
                ) : (
                  <Button
                    onClick={stopRecording}
                    className="w-full h-12 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-semibold shadow-lg shadow-amber-500/20 hover:shadow-amber-500/30 transition-all duration-200"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Stop Recording
                  </Button>
                )}

                {/* Audio Playback */}
                {audioUrl && !isRecording && (
                  <div className="bg-slate-800/50 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Volume2 className="w-4 h-4 text-slate-400" />
                      <span className="text-sm font-medium text-slate-400">
                        Preview Recording
                      </span>
                    </div>
                    <audio
                      src={audioUrl}
                      controls
                      className="w-full h-10 [&::-webkit-media-controls-panel]:bg-slate-700 [&::-webkit-media-controls-current-time-display]:text-white [&::-webkit-media-controls-time-remaining-display]:text-white"
                    />
                  </div>
                )}

                {/* Process Button */}
                {audioBlob && !isRecording && !currentSample && (
                  <Button
                    onClick={uploadAndProcess}
                    disabled={isProcessing}
                    className="w-full h-12 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-semibold shadow-lg shadow-blue-500/20 hover:shadow-blue-500/30 transition-all duration-200"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Upload className="w-5 h-5 mr-2" />
                        Process Recording
                      </>
                    )}
                  </Button>
                )}
              </CardContent>
            </Card>

            {/* Current Sample Review Card */}
            {currentSample && (
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="p-2 bg-cyan-500/10 rounded-lg">
                      <FileAudio className="w-5 h-5 text-cyan-400" />
                    </div>
                    <CardTitle className="text-lg text-white">
                      Review Sample
                    </CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="space-y-5">
                  {/* Transcript */}
                  <div>
                    <label className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2 block">
                      Transcript
                    </label>
                    <div className="bg-slate-800/70 border border-slate-700/50 p-4 rounded-xl">
                      <p className="text-white leading-relaxed">
                        {currentSample.transcript || "No transcript available"}
                      </p>
                    </div>
                  </div>

                  {/* Prosody Labels */}
                  {currentSample.prosody?.semantic && (
                    <div>
                      <label className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-3 block">
                        Prosody Analysis
                      </label>
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-3">
                          <span className="text-xs text-slate-500 block mb-1">
                            Emotion
                          </span>
                          <span className="text-white font-medium capitalize">
                            {currentSample.prosody.semantic.emotion || "neutral"}
                          </span>
                        </div>
                        <div className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-3">
                          <span className="text-xs text-slate-500 block mb-1">
                            Tone
                          </span>
                          <span className="text-white font-medium capitalize">
                            {currentSample.prosody.semantic.tone || "conversational"}
                          </span>
                        </div>
                        <div className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-3">
                          <span className="text-xs text-slate-500 block mb-1">
                            Energy
                          </span>
                          <span className="text-white font-medium capitalize">
                            {currentSample.prosody.semantic.energy_level || "medium"}
                          </span>
                        </div>
                        <div className="bg-slate-800/50 border border-slate-700/30 rounded-xl p-3">
                          <span className="text-xs text-slate-500 block mb-1">
                            Pace
                          </span>
                          <span className="text-white font-medium capitalize">
                            {currentSample.prosody.semantic.pace_category || "normal"}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Emphasis Words */}
                  {currentSample.prosody?.semantic?.emphasis_words &&
                    currentSample.prosody.semantic.emphasis_words.length > 0 && (
                      <div>
                        <label className="text-xs font-medium text-slate-400 uppercase tracking-wide mb-3 block">
                          Emphasis Words
                        </label>
                        <div className="flex flex-wrap gap-2">
                          {currentSample.prosody.semantic.emphasis_words.map(
                            (word, i) => (
                              <Badge
                                key={i}
                                variant="outline"
                                className="bg-violet-500/20 border-violet-500/30 text-violet-300 px-3 py-1.5"
                              >
                                {word}
                              </Badge>
                            )
                          )}
                        </div>
                      </div>
                    )}

                  {/* Actions */}
                  <div className="flex gap-3 pt-2">
                    <Button
                      onClick={approveSample}
                      className="flex-1 h-11 bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600 text-white font-semibold shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30 transition-all duration-200"
                    >
                      <Check className="w-5 h-5 mr-2" />
                      Approve
                    </Button>
                    <Button
                      onClick={discardSample}
                      variant="outline"
                      className="flex-1 h-11 bg-slate-700 hover:bg-slate-600 border-slate-600 text-white font-semibold transition-all duration-200"
                    >
                      <X className="w-5 h-5 mr-2" />
                      Discard
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Export Button */}
            <Button
              onClick={exportDataset}
              disabled={!stats || stats.approved_samples === 0}
              className="w-full h-12 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 disabled:from-slate-700 disabled:to-slate-800 disabled:cursor-not-allowed text-white font-semibold shadow-lg shadow-violet-500/20 hover:shadow-violet-500/30 disabled:shadow-none transition-all duration-200"
            >
              <Package className="w-5 h-5 mr-2" />
              Export Dataset for Training
            </Button>
          </div>

          {/* Right Column: Visualizer & Samples */}
          <div className="lg:col-span-7 space-y-6">
            {/* Prosody Visualizer Card */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 bg-blue-500/10 rounded-lg">
                    <Activity className="w-5 h-5 text-blue-400" />
                  </div>
                  <CardTitle className="text-lg text-white">
                    Prosody Visualizer
                  </CardTitle>
                  <Badge
                    variant="secondary"
                    className="ml-auto bg-slate-800 text-slate-400 hover:bg-slate-800"
                  >
                    Interactive 3D
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="h-[380px] bg-slate-950/50 rounded-xl overflow-hidden border border-slate-800/50">
                  <ProsodyVisualizer
                    analyserNode={analyserNode}
                    isRecording={isRecording}
                    isProcessing={isProcessing}
                    prosodyData={prosodyData}
                  />
                </div>
                <p className="text-xs text-slate-500 mt-3 text-center">
                  Drag to rotate, scroll to zoom. Visualizes pitch contour,
                  rhythm, and semantic features.
                </p>
              </CardContent>
            </Card>

            {/* Recent Samples Card */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader className="pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 bg-amber-500/10 rounded-lg">
                    <Zap className="w-5 h-5 text-amber-400" />
                  </div>
                  <CardTitle className="text-lg text-white">
                    Recent Samples
                  </CardTitle>
                  {samples.length > 0 && (
                    <Badge
                      variant="secondary"
                      className="ml-auto bg-slate-800 text-slate-400 hover:bg-slate-800"
                    >
                      {samples.length} total
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                {samples.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-4">
                      <FileAudio className="w-8 h-8 text-slate-600" />
                    </div>
                    <p className="text-slate-400 font-medium mb-1">
                      No samples yet
                    </p>
                    <p className="text-slate-500 text-sm">
                      Start recording to build your dataset
                    </p>
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[320px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                    {samples.slice(0, 10).map((sample) => (
                      <div
                        key={sample.id}
                        className={`group p-4 rounded-xl border transition-all duration-200 ${
                          sample.approved
                            ? "bg-emerald-500/5 border-emerald-500/20 hover:border-emerald-500/40"
                            : "bg-slate-800/50 border-slate-700/50 hover:border-slate-600"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 min-w-0">
                            <p className="text-white text-sm leading-relaxed line-clamp-2">
                              {sample.transcript || "No transcript"}
                            </p>
                            {sample.prosody?.semantic && (
                              <div className="flex flex-wrap gap-1.5 mt-2">
                                {sample.prosody.semantic.emotion && (
                                  <Badge
                                    variant="secondary"
                                    className="text-xs bg-slate-700/70 text-slate-300 hover:bg-slate-700/70"
                                  >
                                    {sample.prosody.semantic.emotion}
                                  </Badge>
                                )}
                                {sample.prosody.semantic.tone && (
                                  <Badge
                                    variant="secondary"
                                    className="text-xs bg-slate-700/70 text-slate-300 hover:bg-slate-700/70"
                                  >
                                    {sample.prosody.semantic.tone}
                                  </Badge>
                                )}
                              </div>
                            )}
                          </div>
                          <div className="flex flex-col items-end gap-1">
                            <span className="text-xs text-slate-400 font-medium tabular-nums">
                              {sample.duration?.toFixed(1)}s
                            </span>
                            {sample.approved && (
                              <Badge
                                variant="outline"
                                className="text-xs text-emerald-400 border-emerald-500/30 bg-emerald-500/10"
                              >
                                <Check className="w-3 h-3 mr-1" />
                                Approved
                              </Badge>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 text-center text-slate-500 text-sm">
          <p>
            Voice Clone Pipeline — Built for high-quality voice model training
          </p>
        </footer>
      </div>
    </div>
  );
}
