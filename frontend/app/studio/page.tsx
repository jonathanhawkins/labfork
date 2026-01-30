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
  Activity,
  Zap,
  Target,
  Volume2,
  Loader2,
  Circle,
} from "lucide-react";

import { Button } from "@/components/ui/button";

// Dynamic import for 3D visualizer (no SSR)
const ProsodyVisualizer = dynamic(
  () => import("../../components/ProsodyMatrixVisualizer"),
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
function StatRow({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  icon?: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <div className="flex items-center gap-2">
        {Icon && <Icon className="w-3.5 h-3.5 text-muted-foreground" />}
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
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
      toast.error(
        "Could not access microphone. Please allow microphone access."
      );
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
      toast.error("Processing failed. Check console for details.", {
        id: toastId,
      });
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
      toast.error("Export failed. Make sure you have approved samples.", {
        id: toastId,
      });
    }
  };

  const progressPercentage = stats
    ? Math.min(100, (stats.approved_duration_minutes / 60) * 100)
    : 0;

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Controls */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        {/* Recording Section */}
        <Section title="Recording" defaultOpen>
          <div className="space-y-3">
            {/* Status */}
            <div className="flex items-center justify-between py-1">
              <span className="text-xs text-muted-foreground">Status</span>
              <div className="flex items-center gap-2">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${isRecording ? "bg-red-500 animate-pulse" : "bg-foreground"}`}
                />
                <span
                  className={`text-xs ${isRecording ? "text-red-400" : "text-foreground"}`}
                >
                  {isRecording ? "Recording" : "Ready"}
                </span>
              </div>
            </div>

            {/* Record/Stop Button */}
            {!isRecording ? (
              <Button
                onClick={startRecording}
                disabled={isProcessing}
                className="w-full"
              >
                <Mic className="w-4 h-4 mr-2" />
                Start Recording
              </Button>
            ) : (
              <Button onClick={stopRecording} variant="outline" className="w-full">
                <Square className="w-4 h-4 mr-2" />
                Stop Recording
              </Button>
            )}

            {/* Audio Preview */}
            {audioUrl && !isRecording && (
              <div className="pt-2">
                <div className="text-xs text-muted-foreground mb-2">
                  Preview
                </div>
                <audio
                  src={audioUrl}
                  controls
                  className="w-full h-8 opacity-80"
                />
              </div>
            )}

            {/* Process Button */}
            {audioBlob && !isRecording && !currentSample && (
              <Button
                onClick={uploadAndProcess}
                disabled={isProcessing}
                variant="outline"
                className="w-full"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4 mr-2" />
                    Process Recording
                  </>
                )}
              </Button>
            )}
          </div>
        </Section>

        {/* Current Sample Review */}
        {currentSample && (
          <Section title="Review Sample" defaultOpen>
            <div className="space-y-4">
              {/* Transcript */}
              <div>
                <div className="text-xs text-muted-foreground mb-2">
                  Transcript
                </div>
                <div className="text-sm text-foreground bg-background border border-border rounded p-3 leading-relaxed">
                  {currentSample.transcript || "No transcript available"}
                </div>
              </div>

              {/* Prosody Labels */}
              {currentSample.prosody?.semantic && (
                <div className="space-y-2">
                  <StatRow
                    label="Emotion"
                    value={currentSample.prosody.semantic.emotion || "neutral"}
                  />
                  <StatRow
                    label="Tone"
                    value={
                      currentSample.prosody.semantic.tone || "conversational"
                    }
                  />
                  <StatRow
                    label="Energy"
                    value={
                      currentSample.prosody.semantic.energy_level || "medium"
                    }
                  />
                  <StatRow
                    label="Pace"
                    value={
                      currentSample.prosody.semantic.pace_category || "normal"
                    }
                  />
                </div>
              )}

              {/* Emphasis Words */}
              {currentSample.prosody?.semantic?.emphasis_words &&
                currentSample.prosody.semantic.emphasis_words.length > 0 && (
                  <div>
                    <div className="text-xs text-muted-foreground mb-2">
                      Emphasis Words
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {currentSample.prosody.semantic.emphasis_words.map(
                        (word, i) => (
                          <span
                            key={i}
                            className="text-xs text-foreground-bright bg-accent px-2 py-0.5 rounded"
                          >
                            {word}
                          </span>
                        )
                      )}
                    </div>
                  </div>
                )}

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <Button onClick={approveSample} className="flex-1">
                  <Check className="w-4 h-4 mr-1" />
                  Approve
                </Button>
                <Button
                  onClick={discardSample}
                  variant="outline"
                  className="flex-1"
                >
                  <X className="w-4 h-4 mr-1" />
                  Discard
                </Button>
              </div>
            </div>
          </Section>
        )}

        {/* Dataset Stats */}
        {stats && (
          <Section title="Dataset" defaultOpen>
            <div className="space-y-2">
              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="text-muted-foreground">Progress</span>
                  <span className="text-foreground-bright">
                    {progressPercentage.toFixed(0)}%
                  </span>
                </div>
                <div className="h-1 bg-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-foreground transition-all duration-300"
                    style={{ width: `${progressPercentage}%` }}
                  />
                </div>
              </div>

              <StatRow
                label="Approved"
                value={`${stats.approved_samples} samples`}
                icon={Check}
              />
              <StatRow
                label="Duration"
                value={`${stats.approved_duration_minutes.toFixed(1)} / 60 min`}
                icon={Clock}
              />
              <StatRow
                label="Total"
                value={`${stats.total_samples} samples`}
                icon={FileAudio}
              />

              <div className="pt-3">
                <Button
                  onClick={exportDataset}
                  disabled={stats.approved_samples === 0}
                  variant="outline"
                  className="w-full"
                >
                  <Package className="w-4 h-4 mr-2" />
                  Export Dataset
                </Button>
              </div>
            </div>
          </Section>
        )}
      </aside>

      {/* Main Content - Visualizer */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-hidden flex flex-col">
        <div className="flex-1 p-6">
          <div className="h-full bg-background-card border border-border rounded overflow-hidden">
            <ProsodyVisualizer
              analyserNode={analyserNode}
              isRecording={isRecording}
              isProcessing={isProcessing}
              prosodyData={prosodyData}
            />
          </div>
        </div>

        {/* Bottom Status Bar */}
        <div className="border-t border-border px-6 py-3 flex items-center justify-between text-xs text-muted-foreground">
          <span>Drag to rotate, scroll to zoom</span>
          <span>
            {isRecording
              ? "Recording in progress..."
              : isProcessing
                ? "Processing..."
                : "Ready to record"}
          </span>
        </div>
      </main>

      {/* Right Sidebar - Recent Samples */}
      <aside className="w-[300px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section
          title="Recent Samples"
          defaultOpen
          badge={
            samples.length > 0 ? (
              <span className="text-xs text-muted-foreground">
                {samples.length}
              </span>
            ) : null
          }
        >
          {samples.length === 0 ? (
            <div className="text-center py-8">
              <FileAudio className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
              <p className="text-sm text-muted-foreground">No samples yet</p>
              <p className="text-xs text-foreground-subtle mt-1">
                Start recording to build your dataset
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {samples.slice(0, 15).map((sample) => (
                <div
                  key={sample.id}
                  className={`p-3 rounded border transition-colors ${
                    sample.approved
                      ? "border-foreground/20 bg-foreground/5"
                      : "border-border bg-background"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm text-foreground leading-relaxed line-clamp-2 flex-1">
                      {sample.transcript || "No transcript"}
                    </p>
                    <div className="flex flex-col items-end gap-1 flex-shrink-0">
                      <span className="text-xs text-muted-foreground tabular-nums">
                        {sample.duration?.toFixed(1)}s
                      </span>
                      {sample.approved && (
                        <Circle className="w-2 h-2 fill-foreground-bright text-foreground-bright" />
                      )}
                    </div>
                  </div>
                  {sample.prosody?.semantic && (
                    <div className="flex gap-2 mt-2">
                      {sample.prosody.semantic.emotion && (
                        <span className="text-xs text-muted-foreground">
                          {sample.prosody.semantic.emotion}
                        </span>
                      )}
                      {sample.prosody.semantic.tone && (
                        <span className="text-xs text-foreground-subtle">
                          {sample.prosody.semantic.tone}
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </Section>
      </aside>
    </div>
  );
}
