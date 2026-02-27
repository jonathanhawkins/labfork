"use client";

import React, { useState, useRef } from "react";
import {
  Play,
  Loader2,
  Download,
  Volume2,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { DemoBreadcrumb } from "@/components/demos";

const API_BASE = "/api/demos/voxcpm";

export default function VoxcpmDemoPage() {
  const [text, setText] = useState("Hello, this is a test.");
  const [streaming, setStreaming] = useState("");
  const [compile, setCompile] = useState("");
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
        body: JSON.stringify({ text, streaming, compile }),
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
        <DemoBreadcrumb demoId="voxcpm" />

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-foreground/10">
              <Zap className="w-5 h-5 text-foreground-bright" />
            </div>
            <h1 className="text-2xl font-bold text-foreground-bright">
              Voxcpm
            </h1>
          </div>
          <p className="text-muted-foreground">
            VoxCPM Inference Script
          </p>
        </div>


        {/* Text Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            Text
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-24 px-4 py-3 bg-background-card border border-border rounded-lg text-foreground focus:border-foreground-muted focus:outline-none resize-none"
            placeholder="Enter text to synthesize..."
          />
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
                download="voxcpm_output.wav"
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
          <p>Inference: <code className="text-foreground-bright">inference/generate_with_voxcpm.py</code></p>
        </div>
      </div>
    </div>
  );
}
