"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";

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
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setIsRecording(true);
      
    } catch (err) {
      console.error("Failed to start recording:", err);
      alert("Could not access microphone. Please allow microphone access.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const uploadAndProcess = async () => {
    if (!audioBlob) return;
    
    setIsProcessing(true);
    
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
      
    } catch (err) {
      console.error("Processing failed:", err);
      alert("Processing failed. Check console for details.");
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
      
      // Reset for next recording
      setCurrentSample(null);
      setAudioBlob(null);
      setAudioUrl(null);
      setProsodyData(null);
      
      fetchSamples();
      fetchStats();
      
    } catch (err) {
      console.error("Approval failed:", err);
    }
  };

  const discardSample = async () => {
    if (!currentSample) return;
    
    try {
      await fetch(`${API_BASE}/sample/${currentSample.id}`, {
        method: "DELETE",
      });
      
      // Reset
      setCurrentSample(null);
      setAudioBlob(null);
      setAudioUrl(null);
      setProsodyData(null);
      
      fetchSamples();
      fetchStats();
      
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const exportDataset = async () => {
    try {
      const res = await fetch(`${API_BASE}/export`, { method: "POST" });
      const data = await res.json();
      alert(`Exported ${data.count} samples (${data.total_duration_minutes} minutes) to ${data.path}`);
    } catch (err) {
      console.error("Export failed:", err);
      alert("Export failed. Make sure you have approved samples.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-2">Voice Clone Pipeline</h1>
        <p className="text-gray-400 mb-8">Record, label, and train your voice model</p>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Recording & Controls */}
          <div className="space-y-6">
            {/* Stats */}
            {stats && (
              <div className="bg-gray-800 rounded-lg p-4">
                <h2 className="text-lg font-semibold mb-3">Dataset Progress</h2>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Approved:</span>{" "}
                    <span className="font-bold">{stats.approved_samples}</span> samples
                  </div>
                  <div>
                    <span className="text-gray-400">Duration:</span>{" "}
                    <span className="font-bold">{stats.approved_duration_minutes}</span> min
                  </div>
                  <div>
                    <span className="text-gray-400">Total:</span>{" "}
                    <span>{stats.total_samples}</span> samples
                  </div>
                  <div>
                    <span className="text-gray-400">Target:</span>{" "}
                    <span className="text-yellow-400">60 min</span>
                  </div>
                </div>
                <div className="mt-3 bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all"
                    style={{ width: `${Math.min(100, (stats.approved_duration_minutes / 60) * 100)}%` }}
                  />
                </div>
              </div>
            )}
            
            {/* Recording Controls */}
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4">Record</h2>
              
              <div className="flex gap-4 mb-4">
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={isProcessing}
                    className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 py-3 px-6 rounded-lg font-semibold transition"
                  >
                    🎤 Start Recording
                  </button>
                ) : (
                  <button
                    onClick={stopRecording}
                    className="flex-1 bg-yellow-600 hover:bg-yellow-700 py-3 px-6 rounded-lg font-semibold transition animate-pulse"
                  >
                    ⏹ Stop Recording
                  </button>
                )}
              </div>
              
              {/* Audio Playback */}
              {audioUrl && !isRecording && (
                <div className="mb-4">
                  <audio src={audioUrl} controls className="w-full" />
                </div>
              )}
              
              {/* Process Button */}
              {audioBlob && !isRecording && !currentSample && (
                <button
                  onClick={uploadAndProcess}
                  disabled={isProcessing}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 py-3 px-6 rounded-lg font-semibold transition"
                >
                  {isProcessing ? "Processing..." : "📤 Process Recording"}
                </button>
              )}
            </div>
            
            {/* Current Sample Review */}
            {currentSample && (
              <div className="bg-gray-800 rounded-lg p-6">
                <h2 className="text-lg font-semibold mb-4">Review Sample</h2>
                
                {/* Transcript */}
                <div className="mb-4">
                  <label className="text-sm text-gray-400">Transcript</label>
                  <p className="bg-gray-700 p-3 rounded mt-1">
                    {currentSample.transcript || "No transcript"}
                  </p>
                </div>
                
                {/* Prosody Labels */}
                {currentSample.prosody?.semantic && (
                  <div className="mb-4 grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-sm text-gray-400">Emotion</label>
                      <p className="bg-gray-700 p-2 rounded mt-1 capitalize">
                        {currentSample.prosody.semantic.emotion || "neutral"}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Tone</label>
                      <p className="bg-gray-700 p-2 rounded mt-1 capitalize">
                        {currentSample.prosody.semantic.tone || "conversational"}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Energy</label>
                      <p className="bg-gray-700 p-2 rounded mt-1 capitalize">
                        {currentSample.prosody.semantic.energy_level || "medium"}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Pace</label>
                      <p className="bg-gray-700 p-2 rounded mt-1 capitalize">
                        {currentSample.prosody.semantic.pace_category || "normal"}
                      </p>
                    </div>
                  </div>
                )}
                
                {/* Emphasis Words */}
                {currentSample.prosody?.semantic?.emphasis_words && (
                  <div className="mb-4">
                    <label className="text-sm text-gray-400">Emphasis Words</label>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {currentSample.prosody.semantic.emphasis_words.map((word, i) => (
                        <span key={i} className="bg-blue-600 px-2 py-1 rounded text-sm">
                          {word}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Actions */}
                <div className="flex gap-4">
                  <button
                    onClick={approveSample}
                    className="flex-1 bg-green-600 hover:bg-green-700 py-3 px-6 rounded-lg font-semibold transition"
                  >
                    ✓ Approve
                  </button>
                  <button
                    onClick={discardSample}
                    className="flex-1 bg-red-600 hover:bg-red-700 py-3 px-6 rounded-lg font-semibold transition"
                  >
                    ✗ Discard
                  </button>
                </div>
              </div>
            )}
            
            {/* Export Button */}
            <button
              onClick={exportDataset}
              disabled={!stats || stats.approved_samples === 0}
              className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 py-3 px-6 rounded-lg font-semibold transition"
            >
              📦 Export Dataset for Training
            </button>
          </div>
          
          {/* Right Column: Visualizer */}
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-4 h-[400px]">
              <h2 className="text-lg font-semibold mb-2">Prosody Cube</h2>
              <div className="h-[350px]">
                <ProsodyVisualizer
                  analyserNode={analyserNode}
                  isRecording={isRecording}
                  isProcessing={isProcessing}
                  prosodyData={prosodyData}
                />
              </div>
            </div>
            
            {/* Recent Samples */}
            <div className="bg-gray-800 rounded-lg p-4 max-h-[300px] overflow-y-auto">
              <h2 className="text-lg font-semibold mb-3">Recent Samples</h2>
              {samples.length === 0 ? (
                <p className="text-gray-400">No samples yet. Start recording!</p>
              ) : (
                <div className="space-y-2">
                  {samples.slice(0, 10).map((sample) => (
                    <div
                      key={sample.id}
                      className={`p-3 rounded ${
                        sample.approved ? "bg-green-900/30" : "bg-gray-700"
                      }`}
                    >
                      <div className="flex justify-between items-start">
                        <p className="text-sm truncate flex-1">
                          {sample.transcript || "No transcript"}
                        </p>
                        <span className="text-xs text-gray-400 ml-2">
                          {sample.duration?.toFixed(1)}s
                        </span>
                      </div>
                      {sample.prosody?.semantic && (
                        <div className="flex gap-2 mt-1">
                          <span className="text-xs bg-gray-600 px-2 py-0.5 rounded">
                            {sample.prosody.semantic.emotion}
                          </span>
                          <span className="text-xs bg-gray-600 px-2 py-0.5 rounded">
                            {sample.prosody.semantic.tone}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
