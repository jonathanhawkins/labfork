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
  Circle,
} from "lucide-react";

import { Button } from "@/components/ui/button";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8003";

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

// Collapsible Section Component
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

// Stat Row Component
function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
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

      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      waveformInterval.current = setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        const normalized = Array.from(dataArray.slice(0, 32)).map(v => v / 255);
        setWaveformData(normalized);
      }, 50);

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
      const currentEmotionIndex = emotions.indexOf(selectedEmotion || "");
      if (currentEmotionIndex < emotions.length - 1) {
        setSelectedEmotion(emotions[currentEmotionIndex + 1]);
        setCurrentLineIndex(0);
      }
    }
  };

  const prevLine = async () => {
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

  const recordedLines = session?.recorded_count || 0;
  const progressPercent = totalLines > 0 ? (recordedLines / totalLines) * 100 : 0;

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Emotions */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Emotions" defaultOpen>
          <div className="space-y-1">
            {emotions.map((emotion) => {
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
                  className={`w-full text-left p-3 rounded border transition-all flex items-center justify-between ${
                    isSelected
                      ? "bg-foreground/10 border-foreground/30 text-foreground-bright"
                      : "bg-background border-border text-foreground hover:bg-background-card hover:border-border"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Circle
                      className={`w-2 h-2 ${
                        isSelected ? "fill-foreground-bright text-foreground-bright" : "fill-transparent text-muted-foreground"
                      }`}
                    />
                    <span className="capitalize">{emotion}</span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {lineCount} lines
                  </span>
                </button>
              );
            })}
          </div>
        </Section>

        {/* Progress Section */}
        {session && (
          <Section title="Progress" defaultOpen>
            <div className="space-y-3">
              {/* Progress Bar */}
              <div>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Completion</span>
                  <span className="text-foreground-bright">
                    {progressPercent.toFixed(0)}%
                  </span>
                </div>
                <div className="h-1 bg-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-foreground transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>

              <StatRow label="Recorded" value={`${recordedLines} / ${totalLines}`} />

              <Button
                onClick={exportSession}
                variant="outline"
                size="sm"
                disabled={recordedLines === 0}
                className="w-full"
              >
                <Download className="w-3 h-3 mr-2" />
                Export Session
              </Button>
            </div>
          </Section>
        )}

        {/* Recording Tips */}
        <Section title="Tips">
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">*</span>
              Get into character before recording
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">*</span>
              Speak clearly at a consistent volume
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">*</span>
              Really feel the emotion - it shows in prosody
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground-bright">*</span>
              Re-record if the analysis seems wrong
            </li>
          </ul>
        </Section>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border mt-auto">
          <div className="text-xs text-muted-foreground">Script Recording</div>
          <div className="text-xxs text-foreground-subtle mt-0.5">
            Record iconic lines with emotion
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
        <div className="p-8 max-w-3xl mx-auto">
          {/* Session Start */}
          {!session && (
            <div className="text-center py-16">
              <div className="mb-6">
                <Scroll className="w-8 h-8 text-muted-foreground mx-auto mb-4" />
                <h2 className="text-lg text-foreground-bright mb-2">
                  Script Recording Studio
                </h2>
                <p className="text-sm text-muted-foreground max-w-md mx-auto">
                  Record iconic movie quotes, Shakespeare lines, and famous speeches
                  to train your voice model.
                </p>
              </div>
              <Button
                onClick={startSession}
                disabled={isStartingSession}
              >
                {isStartingSession ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Start Recording Session
                  </>
                )}
              </Button>
            </div>
          )}

          {/* Current Line */}
          {currentLine && session && (
            <div className="space-y-6">
              {/* Source & Tags */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  {currentLine.source.includes("Shakespeare") ? (
                    <Theater className="w-4 h-4" />
                  ) : currentLine.source.includes("Speech") ? (
                    <Speech className="w-4 h-4" />
                  ) : (
                    <Film className="w-4 h-4" />
                  )}
                  <span>{currentLine.source}</span>
                </div>
                <span className="text-xs text-muted-foreground px-2 py-0.5 border border-border rounded">
                  {currentLine.emotion}
                </span>
                <span className="text-xs text-foreground-subtle px-2 py-0.5 border border-border rounded">
                  {currentLine.difficulty}
                </span>
              </div>

              {/* Script Text */}
              <div className="bg-background-card border border-border rounded p-6">
                <p className="text-lg text-foreground-bright leading-relaxed font-serif italic">
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
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />
                  Previous
                </Button>
                <span className="text-sm text-muted-foreground">
                  Line {currentLineIndex + 1} of {currentLines.length}
                </span>
                <Button
                  onClick={() => nextLine(false)}
                  disabled={isProcessing}
                  variant="outline"
                  size="sm"
                >
                  Next
                  <ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>

              {/* Waveform Visualization */}
              {isRecording && (
                <div className="h-16 bg-background-card border border-border rounded flex items-center justify-center gap-1 px-4">
                  {waveformData.length > 0 ? (
                    waveformData.map((v, i) => (
                      <div
                        key={i}
                        className="w-1.5 bg-foreground rounded-full transition-all duration-75"
                        style={{ height: `${Math.max(4, v * 56)}px` }}
                      />
                    ))
                  ) : (
                    Array.from({ length: 32 }).map((_, i) => (
                      <div
                        key={i}
                        className="w-1.5 h-1 bg-border rounded-full animate-pulse"
                        style={{ animationDelay: `${i * 50}ms` }}
                      />
                    ))
                  )}
                </div>
              )}

              {/* Recording Controls */}
              <div className="space-y-4">
                {!isRecording && !audioUrl && (
                  <Button
                    onClick={startRecording}
                    disabled={!currentLine || isProcessing}
                    className="w-full h-14"
                  >
                    <Mic className="w-5 h-5 mr-2" />
                    Start Recording
                  </Button>
                )}

                {isRecording && (
                  <Button
                    onClick={stopRecording}
                    variant="outline"
                    className="w-full h-14 border-foreground/50"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    Stop Recording
                  </Button>
                )}

                {audioUrl && !isRecording && (
                  <div className="space-y-4">
                    {/* Playback */}
                    <div className="bg-background-card border border-border rounded p-4">
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
                          className="w-14 h-14 rounded-full"
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
                      >
                        <RotateCcw className="w-4 h-4 mr-1" />
                        Re-record
                      </Button>
                      <Button
                        onClick={() => nextLine(false)}
                        disabled={isProcessing}
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
                      >
                        Skip
                        <ChevronRight className="w-4 h-4 ml-1" />
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* No Lines */}
          {!currentLine && selectedEmotion && session && (
            <div className="text-center py-16">
              <AlertCircle className="w-8 h-8 text-muted-foreground mx-auto mb-4" />
              <p className="text-sm text-muted-foreground">
                No lines available for this emotion
              </p>
            </div>
          )}
        </div>
      </main>

      {/* Right Sidebar - Analysis Results */}
      <aside className="w-[280px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        {lastResult && (
          <Section title="Analysis Result" defaultOpen>
            <div className="space-y-4">
              {/* Emotion Match */}
              <div className={`p-3 rounded border ${
                lastResult.emotion_match
                  ? "border-foreground/30 bg-foreground/5"
                  : "border-border bg-background"
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-muted-foreground">Detected</span>
                  <span className="text-sm text-foreground-bright capitalize">
                    {lastResult.prosody_analysis?.predicted_emotion || "unknown"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Expected</span>
                  <span className="text-sm text-foreground capitalize">
                    {lastResult.expected_emotion}
                  </span>
                </div>
              </div>

              {/* Confidence */}
              {lastResult.prosody_analysis?.confidence && (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-muted-foreground">Confidence</span>
                    <span className="text-sm text-foreground-bright">
                      {(lastResult.prosody_analysis.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-1 bg-border rounded-full overflow-hidden">
                    <div
                      className="h-full bg-foreground transition-all"
                      style={{ width: `${lastResult.prosody_analysis.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Emotion Scores */}
              {lastResult.prosody_analysis?.emotion_scores && (
                <div className="space-y-2">
                  <span className="text-sm text-muted-foreground">All Scores</span>
                  <div className="space-y-1">
                    {Object.entries(lastResult.prosody_analysis.emotion_scores)
                      .sort(([,a], [,b]) => b - a)
                      .slice(0, 4)
                      .map(([emotion, score]) => (
                        <div key={emotion} className="flex items-center gap-2">
                          <span className="text-xs w-16 capitalize text-muted-foreground">
                            {emotion}
                          </span>
                          <div className="flex-1 h-1 bg-border rounded-full overflow-hidden">
                            <div
                              className="h-full bg-foreground/50"
                              style={{ width: `${score * 100}%` }}
                            />
                          </div>
                          <span className="text-xs text-foreground-subtle w-8 text-right">
                            {(score * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Duration */}
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <span className="text-sm text-muted-foreground flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  Duration
                </span>
                <span className="text-sm text-foreground">
                  {lastResult.duration?.toFixed(1)}s
                </span>
              </div>

              {/* Model Used */}
              {lastResult.prosody_analysis?.model_used && (
                <div className="text-xs text-foreground-subtle text-center">
                  Analyzed with: {lastResult.prosody_analysis.model_used}
                </div>
              )}
            </div>
          </Section>
        )}

        {!lastResult && session && (
          <Section title="Analysis" defaultOpen>
            <div className="text-center py-8">
              <Target className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">
                Record a line to see analysis
              </p>
            </div>
          </Section>
        )}

        {/* Session Info */}
        {session && (
          <Section title="Session">
            <div className="space-y-2">
              <StatRow label="Session ID" value={session.session_id.slice(0, 8)} />
              <StatRow label="Recorded" value={recordedLines} />
            </div>
          </Section>
        )}
      </aside>
    </div>
  );
}
