"use client";

import React, { useState, useRef } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Play,
  Loader2,
  Download,
  Volume2,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const EMOTIONS = [
  { id: "neutral", label: "Neutral" },
  { id: "happy", label: "Happy" },
  { id: "sad", label: "Sad" },
  { id: "angry", label: "Angry" },
  { id: "fearful", label: "Fearful" },
];

const API_BASE = "/api/demos/maskgct";

export default function MaskgctDemoPage() {
  const [text, setText] = useState("Hello, this is a test.");
  const [benchmark, setBenchmark] = useState("");
  const [emotion, setEmotion] = useState("happy");
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleGenerate = async () => {
    if (!text.trim()) return;
    setIsGenerating(true);
    setError(null);
    setAudioUrl(null);

    try {
      const response = await fetch(API_BASE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, benchmark, emotion }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Generation failed");
      }

      const data = await response.json();
      setAudioUrl(data.audioUrl);
      if (audioRef.current) {
        audioRef.current.load();
        audioRef.current.play();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-2xl mx-auto">
        <Link
          href="/demos"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-8"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Demos
        </Link>

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-foreground/10">
              <Zap className="w-5 h-5 text-foreground-bright" />
            </div>
            <h1 className="text-2xl font-bold text-foreground-bright">
              Maskgct
            </h1>
          </div>
          <p className="text-muted-foreground">
            MaskGCT Parallel Prosody Generation
          </p>
        </div>


        {/* Emotion Selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-3">
            Emotion
          </label>
          <div className="flex flex-wrap gap-2">
            {EMOTIONS.map((e) => (
              <button
                key={e.id}
                onClick={() => setEmotion(e.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  emotion === e.id
                    ? "bg-foreground text-background"
                    : "bg-background-card border border-border text-foreground hover:border-foreground-muted"
                }`}
              >
                {e.label}
              </button>
            ))}
          </div>
        </div>

        <Button
          onClick={handleGenerate}
          disabled={isGenerating || !text.trim()}
          className="w-full mb-6"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Generate Speech
            </>
          )}
        </Button>

        {error && (
          <div className="p-4 mb-6 bg-red-500/10 border border-red-500/20 rounded-lg text-red-500 text-sm">
            {error}
          </div>
        )}

        {audioUrl && (
          <div className="p-6 bg-background-card border border-border rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-foreground">
                <Volume2 className="w-5 h-5 text-foreground-bright" />
                <span className="font-medium">Generated Audio</span>
              </div>
              <a
                href={audioUrl}
                download="maskgct_output.wav"
                className="text-sm text-foreground-bright hover:text-foreground flex items-center gap-1"
              >
                <Download className="w-4 h-4" />
                Download
              </a>
            </div>
            <audio ref={audioRef} src={audioUrl} controls className="w-full" />
          </div>
        )}

        <div className="mt-8 text-xs text-muted-foreground">
          <p>Inference: <code className="text-foreground-bright">inference/generate_with_maskgct.py</code></p>
        </div>
      </div>
    </div>
  );
}
