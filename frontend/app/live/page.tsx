"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Radio,
  Wifi,
  WifiOff,
  Activity,
  Gauge,
  Server,
  Cpu,
  Circle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

// Backend configurations
const BACKENDS = {
  local: {
    id: "local" as const,
    name: "M4 Pro (Local)",
    api: "http://localhost:8003",
    ws: "ws://localhost:8003",
    description: "Local Mac - ~1.7s processing",
  },
  remote: {
    id: "remote" as const,
    name: "RTX 4090 (Remote)",
    api: "http://localhost:8004",
    ws: "ws://localhost:8004",
    description: "Remote GPU - ~0.4s processing",
  },
};

// Emotion configuration - design system compatible (grayscale with indicators)
const EMOTIONS = [
  { id: "neutral", label: "Neutral", description: "No transformation" },
  { id: "happy", label: "Happy", description: "Upbeat, cheerful" },
  { id: "sad", label: "Sad", description: "Melancholic, subdued" },
  { id: "angry", label: "Angry", description: "Intense, forceful" },
  { id: "calm", label: "Calm", description: "Relaxed, soothing" },
  { id: "fearful", label: "Fearful", description: "Anxious, worried" },
  { id: "excited", label: "Excited", description: "Energetic, enthusiastic" },
  { id: "surprised", label: "Surprised", description: "Astonished, amazed" },
];

interface TransformStatus {
  is_ready: boolean;
  device: string;
  current_emotion: string;
  current_intensity: number;
  available_emotions: string[];
  sample_rate: number;
  last_inference_time: number;
  total_processed: number;
}

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

export default function LiveTransformPage() {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [status, setStatus] = useState<TransformStatus | null>(null);
  const [selectedBackend, setSelectedBackend] = useState<"local" | "remote">("remote");

  // Get current backend config
  const currentBackend = BACKENDS[selectedBackend];

  // Audio state
  const [isListening, setIsListening] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [inputLevel, setInputLevel] = useState(0);
  const [outputLevel, setOutputLevel] = useState(0);

  // Emotion control
  const [selectedEmotion, setSelectedEmotion] = useState("happy");
  const [intensity, setIntensity] = useState(0.7);

  // Performance stats
  const [latency, setLatency] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Audio buffering for smoother playback
  const audioBufferRef = useRef<Float32Array[]>([]);
  const inputBufferRef = useRef<Float32Array[]>([]);
  const nextPlayTimeRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);
  const lastOutputChunkRef = useRef<Float32Array | null>(null);

  // Configurable chunk duration (user can adjust)
  const [chunkDurationMs, setChunkDurationMs] = useState(1500);
  const [bufferProgress, setBufferProgress] = useState(0);
  const [diffusionSteps, setDiffusionSteps] = useState(6);

  // Config for audio chunking
  const INPUT_SAMPLE_RATE = 24000;
  const CROSSFADE_SAMPLES = 512;

  // Fetch initial status when backend changes
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${currentBackend.api}/live-transform/status`);
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          if (data.available_emotions?.length > 0 && !data.available_emotions.includes(selectedEmotion)) {
            setSelectedEmotion(data.available_emotions[0]);
          }
        }
      } catch (error) {
        console.error("Failed to fetch status:", error);
        setStatus(null);
      }
    };
    fetchStatus();
  }, [selectedBackend, currentBackend.api]);

  // Track if component is mounted to avoid state updates after unmount
  const isMountedRef = useRef(true);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log("WebSocket already open or connecting");
      return;
    }

    setIsConnecting(true);
    console.log(`Connecting to ${currentBackend.name}...`);
    const ws = new WebSocket(`${currentBackend.ws}/ws/live-transform`);

    ws.onopen = () => {
      console.log(`WebSocket connected to ${currentBackend.name}`);
      if (!isMountedRef.current) {
        console.log("Component unmounted, closing WebSocket");
        ws.close();
        return;
      }
      setIsConnected(true);
      setIsConnecting(false);
      toast.success(`Connected to ${currentBackend.name}`);

      ws.send(JSON.stringify({ type: "get_status" }));
      ws.send(JSON.stringify({
        type: "set_emotion",
        emotion: selectedEmotion,
        intensity: intensity,
      }));
    };

    ws.onmessage = (event) => {
      if (!isMountedRef.current) return;
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === "audio_output") {
          playAudioChunk(msg.data, msg.sample_rate);
          setLatency(msg.latency_ms || 0);
          setProcessedCount((c) => c + 1);
        } else if (msg.type === "status") {
          setStatus(msg);
        } else if (msg.type === "error") {
          console.error("Server error:", msg.message);
          toast.error(msg.message);
        }
      } catch (e) {
        console.error("Failed to parse message:", e);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      if (isMountedRef.current) {
        setIsConnected(false);
        setIsConnecting(false);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      if (isMountedRef.current) {
        setIsConnecting(false);
        toast.error("Connection failed");
      }
    };

    wsRef.current = ws;
  }, [selectedEmotion, intensity, currentBackend]);

  // Disconnect WebSocket
  const disconnectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  // Apply crossfade between chunks for smooth transitions
  const applyCrossfade = useCallback((current: Float32Array, previous: Float32Array | null): Float32Array => {
    if (!previous || previous.length < CROSSFADE_SAMPLES || current.length < CROSSFADE_SAMPLES) {
      return current;
    }

    const output = new Float32Array(current.length);
    output.set(current);

    for (let i = 0; i < CROSSFADE_SAMPLES; i++) {
      const fadeIn = i / CROSSFADE_SAMPLES;
      const fadeOut = 1 - fadeIn;
      const prevIdx = previous.length - CROSSFADE_SAMPLES + i;
      output[i] = current[i] * fadeIn + previous[prevIdx] * fadeOut;
    }

    return output;
  }, []);

  // Play audio chunk with proper queuing and crossfade
  const playAudioChunk = useCallback((base64Data: string, sampleRate: number) => {
    if (isMuted) return;

    try {
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const int16Array = new Int16Array(bytes.buffer);
      let float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768;
      }

      const crossfadedArray = applyCrossfade(float32Array, lastOutputChunkRef.current);
      lastOutputChunkRef.current = new Float32Array(crossfadedArray);

      if (!outputAudioContextRef.current) {
        outputAudioContextRef.current = new AudioContext({ sampleRate });
      }

      const ctx = outputAudioContextRef.current;
      const buffer = ctx.createBuffer(1, crossfadedArray.length, sampleRate);
      buffer.getChannelData(0).set(crossfadedArray);

      const currentTime = ctx.currentTime;
      const startTime = Math.max(currentTime + 0.01, nextPlayTimeRef.current);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.start(startTime);

      const crossfadeDuration = CROSSFADE_SAMPLES / sampleRate;
      nextPlayTimeRef.current = startTime + buffer.duration - crossfadeDuration;

      const maxVal = Math.max(...Array.from(crossfadedArray).map(Math.abs));
      setOutputLevel(maxVal);
    } catch (e) {
      console.error("Failed to play audio:", e);
    }
  }, [isMuted, applyCrossfade]);

  const samplesPerChunk = Math.floor((chunkDurationMs / 1000) * INPUT_SAMPLE_RATE);

  // Send buffered audio to server
  const sendBufferedAudio = useCallback((forceSend = false) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (inputBufferRef.current.length === 0) return;

    const totalSamples = inputBufferRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
    const progress = Math.min(100, (totalSamples / samplesPerChunk) * 100);
    setBufferProgress(progress);

    if (!forceSend && totalSamples < samplesPerChunk) return;

    const mergedBuffer = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of inputBufferRef.current) {
      mergedBuffer.set(chunk, offset);
      offset += chunk.length;
    }

    inputBufferRef.current = [];
    setBufferProgress(0);

    const int16Array = new Int16Array(mergedBuffer.length);
    for (let i = 0; i < mergedBuffer.length; i++) {
      const s = Math.max(-1, Math.min(1, mergedBuffer[i]));
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    const bytes = new Uint8Array(int16Array.buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    wsRef.current.send(JSON.stringify({
      type: "audio_chunk",
      data: base64,
      sample_rate: INPUT_SAMPLE_RATE,
    }));

    console.log(`Sent ${totalSamples} samples (${(totalSamples / INPUT_SAMPLE_RATE * 1000).toFixed(0)}ms)`);
  }, [samplesPerChunk]);

  // Start listening (mic capture)
  const startListening = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: INPUT_SAMPLE_RATE,
        },
      });
      mediaStreamRef.current = stream;

      const audioContext = new AudioContext({ sampleRate: INPUT_SAMPLE_RATE });
      audioContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.3;
      analyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      inputBufferRef.current = [];
      nextPlayTimeRef.current = 0;

      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const copy = new Float32Array(inputData.length);
        copy.set(inputData);
        inputBufferRef.current.push(copy);
        sendBufferedAudio(false);
      };

      source.connect(processor);
      processor.connect(audioContext.destination);

      const updateLevels = () => {
        if (!analyserRef.current) return;

        const dataArray = new Float32Array(analyserRef.current.fftSize);
        analyserRef.current.getFloatTimeDomainData(dataArray);

        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sum / dataArray.length);
        const level = Math.min(1, rms * 2);
        setInputLevel(level);

        animationFrameRef.current = requestAnimationFrame(updateLevels);
      };
      updateLevels();

      setIsListening(true);
      toast.success("Microphone active - speak for 1.5s chunks");
    } catch (error) {
      console.error("Failed to start listening:", error);
      toast.error("Could not access microphone");
    }
  }, [sendBufferedAudio]);

  // Stop listening
  const stopListening = useCallback(() => {
    sendBufferedAudio(true);

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    inputBufferRef.current = [];
    nextPlayTimeRef.current = 0;
    lastOutputChunkRef.current = null;
    setBufferProgress(0);

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    setIsListening(false);
    setInputLevel(0);
    setOutputLevel(0);
  }, [sendBufferedAudio]);

  // Update emotion on server
  const updateEmotion = useCallback((emotion: string, newIntensity?: number) => {
    const intensityValue = newIntensity ?? intensity;
    setSelectedEmotion(emotion);
    if (newIntensity !== undefined) setIntensity(newIntensity);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "set_emotion",
        emotion,
        intensity: intensityValue,
      }));
    }
  }, [intensity]);

  // Update quality/speed on server
  const updateQuality = useCallback((steps: number) => {
    setDiffusionSteps(steps);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "set_quality",
        diffusion_steps: steps,
      }));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      stopListening();
      disconnectWebSocket();
    };
  }, [stopListening, disconnectWebSocket]);

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar - Controls */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Connection" defaultOpen>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              {isConnected ? (
                <Wifi className="w-4 h-4 text-foreground-bright" />
              ) : (
                <WifiOff className="w-4 h-4 text-muted-foreground" />
              )}
              <span className="text-sm text-foreground">
                {isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>

            <div className="text-xs text-muted-foreground">
              {currentBackend.name}
            </div>

            <Button
              onClick={isConnected ? disconnectWebSocket : connectWebSocket}
              variant={isConnected ? "outline" : "default"}
              size="sm"
              disabled={isConnecting}
              className="w-full"
            >
              {isConnecting ? "Connecting..." : isConnected ? "Disconnect" : "Connect"}
            </Button>
          </div>
        </Section>

        <Section title="Backend" defaultOpen>
          <div className="space-y-2">
            {Object.values(BACKENDS).map((backend) => (
              <button
                key={backend.id}
                onClick={() => {
                  if (isConnected) {
                    disconnectWebSocket();
                  }
                  setSelectedBackend(backend.id);
                }}
                className={`
                  w-full p-3 rounded text-left transition-colors border
                  ${selectedBackend === backend.id
                    ? "border-foreground-bright bg-background-card"
                    : "border-border hover:border-muted-foreground"
                  }
                `}
              >
                <div className="flex items-center gap-2">
                  {backend.id === "local" ? (
                    <Cpu className="w-4 h-4 text-muted-foreground" />
                  ) : (
                    <Server className="w-4 h-4 text-muted-foreground" />
                  )}
                  <span className="text-sm text-foreground">{backend.name}</span>
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {backend.description}
                </p>
              </button>
            ))}
          </div>
        </Section>

        <Section title="Emotion" defaultOpen>
          <div className="space-y-2">
            {EMOTIONS.map((emotion) => {
              const isAvailable = status?.available_emotions?.includes(emotion.id) ?? true;
              const isSelected = selectedEmotion === emotion.id;

              return (
                <button
                  key={emotion.id}
                  onClick={() => isAvailable && updateEmotion(emotion.id)}
                  disabled={!isAvailable}
                  className={`
                    w-full p-2.5 rounded text-left transition-colors border
                    ${isSelected
                      ? "border-foreground-bright bg-background-card"
                      : "border-border hover:border-muted-foreground"
                    }
                    ${!isAvailable && "opacity-50 cursor-not-allowed"}
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
                  </div>
                  <p className="text-xs text-muted-foreground mt-0.5 ml-4">
                    {emotion.description}
                  </p>
                </button>
              );
            })}
          </div>
        </Section>

        <Section title="Intensity">
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Level</span>
              <span className="text-sm text-foreground">{(intensity * 100).toFixed(0)}%</span>
            </div>
            <Slider
              value={[intensity * 100]}
              onValueChange={([value]) => {
                const newIntensity = value / 100;
                setIntensity(newIntensity);
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({
                    type: "set_emotion",
                    emotion: selectedEmotion,
                    intensity: newIntensity,
                  }));
                }
              }}
              min={0}
              max={100}
              step={5}
            />
            <p className="text-xs text-muted-foreground">
              Higher = stronger transformation
            </p>
          </div>
        </Section>

        <Section title="Settings">
          <div className="space-y-4">
            {/* Chunk Size */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Chunk Size</span>
                <span className="text-xs text-foreground">{chunkDurationMs}ms</span>
              </div>
              <Slider
                value={[chunkDurationMs]}
                onValueChange={([value]) => setChunkDurationMs(value)}
                min={750}
                max={2500}
                step={250}
                disabled={isListening}
              />
              <div className="flex justify-between text-xxs text-muted-foreground">
                <span>Fast</span>
                <span>Smooth</span>
              </div>
            </div>

            {/* Quality */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-muted-foreground">Quality</span>
                <span className="text-xs text-foreground">{diffusionSteps} steps</span>
              </div>
              <Slider
                value={[diffusionSteps]}
                onValueChange={([value]) => updateQuality(value)}
                min={4}
                max={10}
                step={1}
              />
              <div className="flex justify-between text-xxs text-muted-foreground">
                <span>Fast</span>
                <span>Quality</span>
              </div>
            </div>

            {/* Mute Toggle */}
            <div className="flex items-center justify-between py-2">
              <Label htmlFor="mute" className="flex items-center gap-2 text-sm text-muted-foreground">
                {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                Output Audio
              </Label>
              <Switch
                id="mute"
                checked={!isMuted}
                onCheckedChange={(checked) => setIsMuted(!checked)}
              />
            </div>
          </div>
        </Section>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border mt-auto">
          <div className="text-xs text-muted-foreground">
            Live Voice Transform
          </div>
          <div className="text-xxs text-foreground-subtle mt-0.5">
            Real-time voice conversion
          </div>
        </div>
      </aside>

      {/* Main Content - Microphone */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto flex flex-col items-center justify-center">
        <div className="text-center space-y-8 max-w-md">
          {/* Header */}
          <div>
            <div className="flex items-center justify-center gap-2 mb-2">
              <Radio className="w-5 h-5 text-foreground-bright" />
              <h1 className="text-lg text-foreground-bright">Live Voice Transform</h1>
            </div>
            <p className="text-sm text-muted-foreground">
              Speak normally, sound emotional - real-time voice conversion
            </p>
          </div>

          {/* Mic Button */}
          <div className="flex flex-col items-center gap-4">
            <Button
              size="lg"
              variant={isListening ? "destructive" : "default"}
              className="h-24 w-24 rounded-full"
              onClick={isListening ? stopListening : startListening}
              disabled={!isConnected}
            >
              {isListening ? (
                <MicOff className="h-10 w-10" />
              ) : (
                <Mic className="h-10 w-10" />
              )}
            </Button>

            <p className="text-sm text-muted-foreground">
              {!isConnected
                ? "Connect first to enable microphone"
                : isListening
                ? "Listening... Speak into your microphone"
                : "Click to start listening"}
            </p>
          </div>

          {/* Buffer Progress */}
          {isListening && (
            <div className="space-y-2 w-full max-w-xs mx-auto">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Buffer</span>
                <span>{bufferProgress.toFixed(0)}%</span>
              </div>
              <div className="h-1 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-foreground transition-all duration-100"
                  style={{ width: `${bufferProgress}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Collecting {chunkDurationMs}ms of audio...
              </p>
            </div>
          )}

          {/* Level Meters */}
          <div className="space-y-3 w-full max-w-xs mx-auto">
            <div className="space-y-1">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Input</span>
                <span>{(inputLevel * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-foreground transition-all duration-75"
                  style={{ width: `${inputLevel * 100}%` }}
                />
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Output</span>
                <span>{(outputLevel * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-border rounded-full overflow-hidden">
                <div
                  className="h-full bg-foreground-bright transition-all duration-75"
                  style={{ width: `${outputLevel * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="text-left border border-border rounded p-4 w-full max-w-xs mx-auto">
            <p className="text-sm text-foreground-bright mb-2">Tips</p>
            <ul className="text-xs text-muted-foreground space-y-1">
              <li>Speak clearly at a normal pace</li>
              <li>Use a good quality microphone</li>
              <li>Lower chunk size = faster but choppier</li>
              <li>"Neutral" passes through unchanged</li>
            </ul>
          </div>
        </div>
      </main>

      {/* Right Panel - Status */}
      <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Status" defaultOpen>
          <StatRow label="Connection" value={isConnected ? "Active" : "Inactive"} />
          <StatRow label="Backend" value={currentBackend.name.split(" ")[0]} />
          <StatRow label="Device" value={status?.device || (selectedBackend === "local" ? "MPS" : "CUDA")} />
          <StatRow label="Sample Rate" value={`${status?.sample_rate || 22050}Hz`} />
        </Section>

        <Section title="Performance" defaultOpen>
          <StatRow label="Latency" value={`${latency.toFixed(0)}ms`} />
          <StatRow label="Processed" value={`${processedCount} chunks`} />
          <StatRow label="Total" value={`${status?.total_processed || 0}`} />
        </Section>

        <Section title="Current">
          <StatRow label="Emotion" value={selectedEmotion} />
          <StatRow label="Intensity" value={`${(intensity * 100).toFixed(0)}%`} />
          <StatRow label="Chunk Size" value={`${chunkDurationMs}ms`} />
          <StatRow label="Quality" value={`${diffusionSteps} steps`} />
        </Section>

        <Section title="Available Emotions">
          <div className="text-xs text-muted-foreground">
            {status?.available_emotions?.join(", ") || "Loading..."}
          </div>
        </Section>
      </aside>
    </div>
  );
}
