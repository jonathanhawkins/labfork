"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
import {
  Mic,
  Square,
  Play,
  Pause,
  RotateCcw,
  Check,
  X,
  ChevronRight,
  ChevronLeft,
  Sparkles,
  Volume2,
  Download,
  Film,
  Theater,
  Scroll,
  Speech,
  Loader2,
  Target,
  Zap,
  Clock,
  AlertCircle,
  CheckCircle2,
  XCircle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";

// Emotion colors and icons
const EMOTION_CONFIG: Record<string, { color: string; bgColor: string; borderColor: string; icon: string }> = {
  neutral: { color: "text-slate-400", bgColor: "bg-slate-500/20", borderColor: "border-slate-500/30", icon: "O" },
  calm: { color: "text-emerald-400", bgColor: "bg-emerald-500/20", borderColor: "border-emerald-500/30", icon: "~" },
  happy: { color: "text-yellow-400", bgColor: "bg-yellow-500/20", borderColor: "border-yellow-500/30", icon: ":)" },
  sad: { color: "text-blue-400", bgColor: "bg-blue-500/20", borderColor: "border-blue-500/30", icon: ":(" },
  angry: { color: "text-red-400", bgColor: "bg-red-500/20", borderColor: "border-red-500/30", icon: ">:" },
  fearful: { color: "text-purple-400", bgColor: "bg-purple-500/20", borderColor: "border-purple-500/30", icon: "!" },
  surprised: { color: "text-orange-400", bgColor: "bg-orange-500/20", borderColor: "border-orange-500/30", icon: "O!" },
  excited: { color: "text-pink-400", bgColor: "bg-pink-500/20", borderColor: "border-pink-500/30", icon: "!!" },
};

interface ScriptLine {
  id: string;
  text: string;
  source: string;
  emotion: string;
  difficulty: string;
}

interface Recording {
  recording_id: string;
  line_id: string;
  script_text: string;
  expected_emotion: string;
  duration: number;
  prosody_analysis?: {
    predicted_emotion: string;
    confidence: number;
    emotion_scores?: Record<string, number>;
    acoustic?: Record<string, number>;
    rhythm?: Record<string, number>;
    model_used?: string;
  };
  emotion_match?: boolean;
}

interface Session {
  session_id: string;
  created_at: string;
  recorded_count: number;
  recordings: Recording[];
}

export default function RecordPage() {
  // Script lines state
  const [scriptLines, setScriptLines] = useState<Record<string, ScriptLine[]>>({});
  const [emotions, setEmotions] = useState<string[]>([]);
  const [totalLines, setTotalLines] = useState(0);
  const [selectedEmotion, setSelectedEmotion] = useState<string | null>(null);
  const [currentLineIndex, setCurrentLineIndex] = useState(0);

  // Session state
  const [session, setSession] = useState<Session | null>(null);
  const [isStartingSession, setIsStartingSession] = useState(false);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [lastResult, setLastResult] = useState<Recording | null>(null);

  // Waveform visualization
  const [waveformData, setWaveformData] = useState<number[]>([]);
  const waveformInterval = useRef<NodeJS.Timeout | null>(null);

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Fetch script lines on mount
  useEffect(() => {
    fetchScriptLines();
  }, []);

  // Restore session from localStorage on mount
  useEffect(() => {
    const savedSessionId = localStorage.getItem("voice_recording_session_id");
    if (savedSessionId && !session) {
      // Try to fetch the session from the backend
      fetch(`${API_BASE}/voice-recording/sessions`)
        .then(res => res.json())
        .then(data => {
          const existingSession = data.sessions?.find(
            (s: Session) => s.session_id === savedSessionId
          );
          if (existingSession) {
            setSession(existingSession);
            console.log("Restored session:", savedSessionId);
          } else {
            // Session no longer exists on backend, clear localStorage
            localStorage.removeItem("voice_recording_session_id");
          }
        })
        .catch(err => {
          console.error("Failed to restore session:", err);
          localStorage.removeItem("voice_recording_session_id");
        });
    }
  }, []);

  const fetchScriptLines = async () => {
    try {
      const res = await fetch(`${API_BASE}/script-lines`);
      const data = await res.json();
      setScriptLines(data.script_lines || {});
      setEmotions(data.emotions || []);
      setTotalLines(data.total_lines || 0);
      // Auto-select first emotion
      if (data.emotions?.length > 0) {
        setSelectedEmotion(data.emotions[0]);
      }
    } catch (err) {
      console.error("Failed to fetch script lines:", err);
      toast.error("Failed to load script lines");
    }
  };

  const startSession = async (): Promise<Session | null> => {
    setIsStartingSession(true);
    try {
      const res = await fetch(`${API_BASE}/voice-recording/start-session`, {
        method: "POST",
      });
      const data = await res.json();
      setSession(data);
      // Persist session ID to localStorage
      localStorage.setItem("voice_recording_session_id", data.session_id);
      toast.success("Recording session started");
      return data;
    } catch (err) {
      console.error("Failed to start session:", err);
      toast.error("Failed to start session");
      return null;
    } finally {
      setIsStartingSession(false);
    }
  };

  const currentLines = selectedEmotion ? scriptLines[selectedEmotion] || [] : [];
  const currentLine = currentLines[currentLineIndex];

  // Recording functions
  const startRecording = async () => {
    let currentSession = session;
    if (!currentSession) {
      currentSession = await startSession();
      if (!currentSession) {
        toast.error("Failed to create recording session");
        return;
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Setup audio context for visualization
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      // Start waveform visualization
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      waveformInterval.current = setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        const normalized = Array.from(dataArray.slice(0, 32)).map(v => v / 255);
        setWaveformData(normalized);
      }, 50);

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
        stream.getTracks().forEach((track) => track.stop());
        if (waveformInterval.current) {
          clearInterval(waveformInterval.current);
        }
        setWaveformData([]);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setIsRecording(true);
      setLastResult(null);
    } catch (err) {
      console.error("Failed to start recording:", err);
      toast.error("Could not access microphone");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const uploadRecording = async () => {
    if (!audioBlob || !currentLine) {
      toast.error("No recording to save");
      return;
    }

    if (!session) {
      toast.error("No active session. Please start recording first.");
      return;
    }

    setIsProcessing(true);
    const toastId = toast.loading("Analyzing your recording...");

    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");
      formData.append("line_id", currentLine.id);
      formData.append("expected_emotion", currentLine.emotion);
      formData.append("script_text", currentLine.text);

      const res = await fetch(
        `${API_BASE}/voice-recording/upload/${session.session_id}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      // Update session with new recording count
      setSession(prev => prev ? {
        ...prev,
        recorded_count: data.session_progress?.recorded_count || prev.recorded_count + 1,
      } : null);

      setLastResult({
        recording_id: data.recording_id,
        line_id: currentLine.id,
        script_text: currentLine.text,
        expected_emotion: currentLine.emotion,
        duration: data.duration,
        prosody_analysis: data.prosody_analysis,
        emotion_match: data.emotion_match,
      });

      if (data.emotion_match) {
        toast.success("Recording saved with matching emotion!", { id: toastId });
      } else {
        toast.info(`Recording saved. Detected: ${data.prosody_analysis?.predicted_emotion || "unknown"}`, { id: toastId });
      }
    } catch (err) {
      console.error("Upload failed:", err);
      toast.error("Failed to upload recording", { id: toastId });
    } finally {
      setIsProcessing(false);
    }
  };

  const discardRecording = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setLastResult(null);
  };

  const nextLine = async (skipSave: boolean = false) => {
    // Auto-save recording if there's one and we have a session
    if (!skipSave && audioBlob && session && !isProcessing) {
      setIsProcessing(true);
      const toastId = toast.loading("Saving and moving to next...");
      try {
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");
        formData.append("line_id", currentLine!.id);
        formData.append("expected_emotion", currentLine!.emotion);
        formData.append("script_text", currentLine!.text);

        const res = await fetch(
          `${API_BASE}/voice-recording/upload/${session.session_id}`,
          { method: "POST", body: formData }
        );

        if (!res.ok) throw new Error("Upload failed");
        const data = await res.json();

        // Update session with new recording count
        setSession(prev => prev ? {
          ...prev,
          recorded_count: data.session_progress?.recorded_count || prev.recorded_count + 1,
        } : null);

        toast.success(`Saved! Detected: ${data.prosody_analysis?.predicted_emotion || "unknown"}`, { id: toastId });
      } catch (err) {
        console.error("Auto-save failed:", err);
        toast.error("Failed to save recording", { id: toastId });
      } finally {
        setIsProcessing(false);
      }
    }

    discardRecording();
    if (currentLineIndex < currentLines.length - 1) {
      setCurrentLineIndex(currentLineIndex + 1);
    } else {
      // Move to next emotion
      const currentEmotionIndex = emotions.indexOf(selectedEmotion || "");
      if (currentEmotionIndex < emotions.length - 1) {
        setSelectedEmotion(emotions[currentEmotionIndex + 1]);
        setCurrentLineIndex(0);
      }
    }
  };

  const prevLine = async () => {
    // Auto-save if there's a recording
    if (audioBlob && session && !isProcessing && currentLine) {
      setIsProcessing(true);
      const toastId = toast.loading("Saving before going back...");
      try {
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");
        formData.append("line_id", currentLine.id);
        formData.append("expected_emotion", currentLine.emotion);
        formData.append("script_text", currentLine.text);

        const res = await fetch(
          `${API_BASE}/voice-recording/upload/${session.session_id}`,
          { method: "POST", body: formData }
        );

        if (res.ok) {
          const data = await res.json();
          setSession(prev => prev ? {
            ...prev,
            recorded_count: data.session_progress?.recorded_count || prev.recorded_count + 1,
          } : null);
          toast.success("Recording saved!", { id: toastId });
        } else {
          toast.warning("Save failed, recording discarded", { id: toastId });
        }
      } catch (err) {
        console.error("Auto-save failed:", err);
        toast.warning("Save failed, recording discarded", { id: toastId });
      } finally {
        setIsProcessing(false);
      }
    }
    discardRecording();
    if (currentLineIndex > 0) {
      setCurrentLineIndex(currentLineIndex - 1);
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

  const exportSession = async () => {
    if (!session) return;

    const toastId = toast.loading("Exporting session...");
    try {
      const res = await fetch(
        `${API_BASE}/voice-recording/export/${session.session_id}`,
        { method: "POST" }
      );
      const data = await res.json();
      toast.success(
        `Exported ${data.recording_count} recordings (${data.total_duration_minutes} min)`,
        { id: toastId }
      );
    } catch (err) {
      console.error("Export failed:", err);
      toast.error("Export failed", { id: toastId });
    }
  };

  // Calculate progress
  const recordedLines = session?.recorded_count || 0;
  const progressPercent = totalLines > 0 ? (recordedLines / totalLines) * 100 : 0;

  // Get difficulty badge color
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "easy": return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "medium": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
      case "hard": return "bg-red-500/20 text-red-400 border-red-500/30";
      default: return "bg-slate-500/20 text-slate-400 border-slate-500/30";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Background pattern */}
      <div className="fixed inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiMyMDIwMjAiIGZpbGwtb3BhY2l0eT0iMC4xIj48cGF0aCBkPSJNMzYgMzRoLTJ2LTRoMnY0em0wLTZ2LTRoLTJ2NGgyem0tNiA2aC00djJoNHYtMnptLTYgMGgtNHYyaDR2LTJ6bTEyLTEydi00aC0ydjRoMnptLTYgMGgtNHYyaDR2LTJ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-50 pointer-events-none" />

      <div className="relative z-10 container mx-auto px-6 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-4">
            <div className="p-2 bg-gradient-to-br from-violet-500 to-purple-600 rounded-xl">
              <Scroll className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Script Recording Studio</h1>
          </div>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Record iconic movie quotes, Shakespeare lines, and famous speeches to train your voice model.
            Each line is auto-analyzed for emotional prosody.
          </p>
        </div>

        {/* Session not started */}
        {!session && (
          <div className="flex justify-center mb-8">
            <Button
              onClick={startSession}
              disabled={isStartingSession}
              className="h-14 px-8 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white font-semibold text-lg shadow-lg shadow-violet-500/20"
            >
              {isStartingSession ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Starting Session...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 mr-2" />
                  Start Recording Session
                </>
              )}
            </Button>
          </div>
        )}

        {/* Main Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: Emotion Selector */}
          <div className="lg:col-span-3 space-y-4">
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-violet-400" />
                  Emotions
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Select emotion to practice
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {emotions.map((emotion) => {
                  const config = EMOTION_CONFIG[emotion];
                  const lineCount = scriptLines[emotion]?.length || 0;
                  const isSelected = selectedEmotion === emotion;

                  return (
                    <button
                      key={emotion}
                      onClick={() => {
                        setSelectedEmotion(emotion);
                        setCurrentLineIndex(0);
                        discardRecording();
                      }}
                      className={`w-full text-left p-3 rounded-lg border transition-all ${
                        isSelected
                          ? `${config.bgColor} ${config.borderColor} ${config.color}`
                          : "bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-800 hover:border-slate-600"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium capitalize">{emotion}</span>
                        <Badge variant="outline" className={`text-xs ${isSelected ? config.borderColor : ""}`}>
                          {lineCount} lines
                        </Badge>
                      </div>
                    </button>
                  );
                })}
              </CardContent>
            </Card>

            {/* Progress Card */}
            {session && (
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      <Zap className="w-5 h-5 text-amber-400" />
                      Progress
                    </CardTitle>
                    <span className="text-2xl font-bold text-amber-400">
                      {progressPercent.toFixed(0)}%
                    </span>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Progress
                    value={progressPercent}
                    className="h-3 bg-slate-800"
                    indicatorClassName="bg-gradient-to-r from-amber-500 to-orange-500"
                  />
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-400">
                      {recordedLines} / {totalLines} lines
                    </span>
                    <Button
                      onClick={exportSession}
                      variant="outline"
                      size="sm"
                      disabled={recordedLines === 0}
                      className="border-slate-600 text-slate-300 hover:bg-slate-800"
                    >
                      <Download className="w-3 h-3 mr-1" />
                      Export
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Center: Recording Area */}
          <div className="lg:col-span-6 space-y-6">
            {/* Current Line Card */}
            {currentLine && (
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {currentLine.source.includes("Shakespeare") ? (
                        <Theater className="w-5 h-5 text-violet-400" />
                      ) : currentLine.source.includes("Speech") || currentLine.source.includes("MLK") || currentLine.source.includes("Churchill") || currentLine.source.includes("FDR") ? (
                        <Speech className="w-5 h-5 text-violet-400" />
                      ) : (
                        <Film className="w-5 h-5 text-violet-400" />
                      )}
                      <span className="text-sm text-slate-400">{currentLine.source}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="outline"
                        className={`${EMOTION_CONFIG[currentLine.emotion]?.borderColor} ${EMOTION_CONFIG[currentLine.emotion]?.color}`}
                      >
                        {currentLine.emotion}
                      </Badge>
                      <Badge
                        variant="outline"
                        className={getDifficultyColor(currentLine.difficulty)}
                      >
                        {currentLine.difficulty}
                      </Badge>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Script Text */}
                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6">
                    <p className="text-xl text-white leading-relaxed font-serif italic">
                      "{currentLine.text}"
                    </p>
                  </div>

                  {/* Line Navigation */}
                  <div className="flex items-center justify-between">
                    <Button
                      onClick={prevLine}
                      disabled={currentLineIndex === 0 || isProcessing}
                      variant="outline"
                      size="sm"
                      className="border-slate-600 text-slate-300 hover:bg-slate-800"
                    >
                      <ChevronLeft className="w-4 h-4 mr-1" />
                      Previous
                    </Button>
                    <span className="text-sm text-slate-400">
                      Line {currentLineIndex + 1} of {currentLines.length}
                    </span>
                    <Button
                      onClick={() => nextLine(false)}
                      disabled={isProcessing}
                      variant="outline"
                      size="sm"
                      className="border-slate-600 text-slate-300 hover:bg-slate-800"
                    >
                      Next
                      <ChevronRight className="w-4 h-4 ml-1" />
                    </Button>
                  </div>

                  {/* Waveform Visualization */}
                  {isRecording && (
                    <div className="h-16 bg-slate-800/50 rounded-xl flex items-center justify-center gap-1 px-4">
                      {waveformData.length > 0 ? (
                        waveformData.map((v, i) => (
                          <div
                            key={i}
                            className="w-2 bg-red-500 rounded-full transition-all duration-75"
                            style={{ height: `${Math.max(4, v * 56)}px` }}
                          />
                        ))
                      ) : (
                        Array.from({ length: 32 }).map((_, i) => (
                          <div
                            key={i}
                            className="w-2 h-1 bg-slate-600 rounded-full animate-pulse"
                            style={{ animationDelay: `${i * 50}ms` }}
                          />
                        ))
                      )}
                    </div>
                  )}

                  {/* Recording Controls */}
                  <div className="flex flex-col items-center gap-4">
                    {!isRecording && !audioUrl && (
                      <Button
                        onClick={startRecording}
                        disabled={!currentLine || isProcessing}
                        className="w-full h-14 bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600 text-white font-semibold text-lg shadow-lg shadow-red-500/20"
                      >
                        <Mic className="w-5 h-5 mr-2" />
                        Start Recording
                      </Button>
                    )}

                    {isRecording && (
                      <Button
                        onClick={stopRecording}
                        className="w-full h-14 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white font-semibold text-lg shadow-lg shadow-amber-500/20"
                      >
                        <Square className="w-5 h-5 mr-2" />
                        Stop Recording
                      </Button>
                    )}

                    {audioUrl && !isRecording && (
                      <div className="w-full space-y-4">
                        {/* Playback */}
                        <div className="bg-slate-800/50 rounded-xl p-4">
                          <audio
                            ref={audioRef}
                            src={audioUrl}
                            onEnded={() => setIsPlaying(false)}
                            className="hidden"
                          />
                          <div className="flex items-center justify-center gap-4">
                            <Button
                              onClick={togglePlayback}
                              size="lg"
                              className="w-14 h-14 rounded-full bg-violet-500 hover:bg-violet-600"
                            >
                              {isPlaying ? (
                                <Pause className="w-5 h-5" />
                              ) : (
                                <Play className="w-5 h-5 ml-0.5" />
                              )}
                            </Button>
                          </div>
                          <audio src={audioUrl} controls className="w-full mt-3 h-8" />
                        </div>

                        {/* Action Buttons */}
                        <div className="grid grid-cols-3 gap-3">
                          <Button
                            onClick={discardRecording}
                            variant="outline"
                            className="border-slate-600 text-slate-300 hover:bg-slate-800"
                          >
                            <RotateCcw className="w-4 h-4 mr-1" />
                            Re-record
                          </Button>
                          <Button
                            onClick={() => nextLine(false)}
                            disabled={isProcessing}
                            className="bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-600 hover:to-green-600 text-white"
                          >
                            {isProcessing ? (
                              <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                            ) : (
                              <Check className="w-4 h-4 mr-1" />
                            )}
                            Save & Next
                          </Button>
                          <Button
                            onClick={() => nextLine(true)}
                            variant="outline"
                            className="border-slate-600 text-slate-300 hover:bg-slate-800"
                          >
                            Skip
                            <ChevronRight className="w-4 h-4 ml-1" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* No Line Selected */}
            {!currentLine && selectedEmotion && (
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardContent className="py-16 text-center">
                  <AlertCircle className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">No lines available for this emotion</p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right: Analysis Results */}
          <div className="lg:col-span-3 space-y-4">
            {/* Last Recording Result */}
            {lastResult && (
              <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg text-white flex items-center gap-2">
                    {lastResult.emotion_match ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-amber-400" />
                    )}
                    Analysis Result
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Emotion Match */}
                  <div className={`p-3 rounded-lg border ${
                    lastResult.emotion_match
                      ? "bg-emerald-500/10 border-emerald-500/30"
                      : "bg-amber-500/10 border-amber-500/30"
                  }`}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-slate-400">Detected Emotion</span>
                      <Badge
                        variant="outline"
                        className={`${
                          EMOTION_CONFIG[lastResult.prosody_analysis?.predicted_emotion || "neutral"]?.borderColor
                        } ${
                          EMOTION_CONFIG[lastResult.prosody_analysis?.predicted_emotion || "neutral"]?.color
                        }`}
                      >
                        {lastResult.prosody_analysis?.predicted_emotion || "unknown"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">Expected</span>
                      <Badge
                        variant="outline"
                        className={`${EMOTION_CONFIG[lastResult.expected_emotion]?.borderColor} ${EMOTION_CONFIG[lastResult.expected_emotion]?.color}`}
                      >
                        {lastResult.expected_emotion}
                      </Badge>
                    </div>
                  </div>

                  {/* Confidence */}
                  {lastResult.prosody_analysis?.confidence && (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-slate-400">Confidence</span>
                        <span className="text-sm text-white font-medium">
                          {(lastResult.prosody_analysis.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      <Progress
                        value={lastResult.prosody_analysis.confidence * 100}
                        className="h-2 bg-slate-800"
                        indicatorClassName={
                          lastResult.prosody_analysis.confidence > 0.7
                            ? "bg-emerald-500"
                            : lastResult.prosody_analysis.confidence > 0.4
                            ? "bg-amber-500"
                            : "bg-red-500"
                        }
                      />
                    </div>
                  )}

                  {/* Emotion Scores */}
                  {lastResult.prosody_analysis?.emotion_scores && (
                    <div className="space-y-2">
                      <span className="text-sm text-slate-400">All Scores</span>
                      <div className="space-y-1">
                        {Object.entries(lastResult.prosody_analysis.emotion_scores)
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 4)
                          .map(([emotion, score]) => (
                            <div key={emotion} className="flex items-center gap-2">
                              <span className={`text-xs w-16 capitalize ${EMOTION_CONFIG[emotion]?.color}`}>
                                {emotion}
                              </span>
                              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                <div
                                  className={`h-full ${EMOTION_CONFIG[emotion]?.bgColor.replace('/20', '')}`}
                                  style={{ width: `${score * 100}%` }}
                                />
                              </div>
                              <span className="text-xs text-slate-500 w-8 text-right">
                                {(score * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Duration */}
                  <div className="flex items-center justify-between pt-2 border-t border-slate-800">
                    <span className="text-sm text-slate-400 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Duration
                    </span>
                    <span className="text-sm text-white">
                      {lastResult.duration?.toFixed(1)}s
                    </span>
                  </div>

                  {/* Model Used */}
                  {lastResult.prosody_analysis?.model_used && (
                    <div className="text-xs text-slate-500 text-center">
                      Analyzed with: {lastResult.prosody_analysis.model_used}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Tips Card */}
            <Card className="bg-slate-900/80 backdrop-blur-sm border-slate-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-lg text-white flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-violet-400" />
                  Recording Tips
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm text-slate-400">
                  <li className="flex items-start gap-2">
                    <span className="text-violet-400">*</span>
                    Get into character before recording
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-violet-400">*</span>
                    Speak clearly at a consistent volume
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-violet-400">*</span>
                    Really feel the emotion - it shows in prosody
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-violet-400">*</span>
                    Re-record if the analysis seems wrong
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-violet-400">*</span>
                    Aim for 3-5 recordings per emotion
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
