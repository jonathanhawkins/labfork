"use client";

import React, { useState, useRef } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Play,
  Loader2,
  Download,
  Volume2,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";

// Emotions supported by EASV
const EMOTIONS = [
  { id: "neutral", label: "Neutral", color: "bg-gray-500" },
  { id: "happy", label: "Happy", color: "bg-yellow-500" },
  { id: "sad", label: "Sad", color: "bg-blue-500" },
  { id: "angry", label: "Angry", color: "bg-red-500" },
  { id: "fearful", label: "Fearful", color: "bg-purple-500" },
];

// API base - uses Next.js API route
const API_BASE = "/api/demos/easv";

export default function EASVDemoPage() {
  const [text, setText] = useState("Hello, this is a test of emotion control.");
  const [emotion, setEmotion] = useState("happy");
  const [intensity, setIntensity] = useState(0.7);
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
        body: JSON.stringify({ text, emotion, intensity }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "Generation failed");
      }

      const data = await response.json();
      setAudioUrl(data.audioUrl);

      // Auto-play
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
        {/* Back Link */}
        <Link
          href="/demos"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-8"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Demos
        </Link>

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-accent/10">
              <Sparkles className="w-5 h-5 text-accent" />
            </div>
            <h1 className="text-2xl font-bold text-foreground-bright">
              EASV Intensity Control
            </h1>
          </div>
          <p className="text-muted-foreground">
            Emotion-Adaptive Spherical Vectors allow smooth interpolation between
            neutral and emotional speech. Adjust the intensity slider to control
            how strongly the emotion is expressed.
          </p>
        </div>

        {/* Text Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            Text to Synthesize
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full h-24 px-4 py-3 bg-surface border border-border rounded-lg text-foreground focus:border-accent focus:outline-none resize-none"
            placeholder="Enter text to synthesize..."
          />
        </div>

        {/* Emotion Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-3">
            Target Emotion
          </label>
          <div className="flex flex-wrap gap-2">
            {EMOTIONS.map((e) => (
              <button
                key={e.id}
                onClick={() => setEmotion(e.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  emotion === e.id
                    ? "bg-accent text-background"
                    : "bg-surface border border-border text-foreground hover:border-accent"
                }`}
              >
                <span
                  className={`inline-block w-2 h-2 rounded-full mr-2 ${e.color}`}
                />
                {e.label}
              </button>
            ))}
          </div>
        </div>

        {/* Intensity Slider */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-foreground">
              Emotion Intensity
            </label>
            <span className="text-sm text-accent font-mono">
              {intensity.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={intensity}
            onChange={(e) => setIntensity(parseFloat(e.target.value))}
            className="w-full h-2 bg-surface rounded-lg appearance-none cursor-pointer accent-accent"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>Neutral</span>
            <span>Subtle</span>
            <span>Moderate</span>
            <span>Strong</span>
            <span>Full</span>
          </div>
        </div>

        {/* Generate Button */}
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

        {/* Error Display */}
        {error && (
          <div className="p-4 mb-6 bg-red-500/10 border border-red-500/20 rounded-lg text-red-500 text-sm">
            {error}
          </div>
        )}

        {/* Audio Player */}
        {audioUrl && (
          <div className="p-6 bg-surface border border-border rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-foreground">
                <Volume2 className="w-5 h-5 text-accent" />
                <span className="font-medium">Generated Audio</span>
              </div>
              <a
                href={audioUrl}
                download={`easv_${emotion}_${intensity}.wav`}
                className="text-sm text-accent hover:text-accent-bright flex items-center gap-1"
              >
                <Download className="w-4 h-4" />
                Download
              </a>
            </div>
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              className="w-full"
            />
            <p className="text-xs text-muted-foreground mt-3">
              Generated with {emotion} emotion at {(intensity * 100).toFixed(0)}% intensity
            </p>
          </div>
        )}

        {/* Info Box */}
        <div className="mt-8 p-4 bg-surface-raised border border-border rounded-lg">
          <h3 className="text-sm font-medium text-foreground mb-2">
            How EASV Works
          </h3>
          <p className="text-sm text-muted-foreground">
            EASV uses spherical emotion embeddings that allow smooth linear
            interpolation between neutral and target emotions. At intensity 0,
            speech is fully neutral. At intensity 1, the full emotional
            expression is applied. This enables fine-grained control over
            emotional expression without the abrupt transitions seen in discrete
            emotion systems.
          </p>
        </div>

        {/* Technical Details */}
        <div className="mt-4 text-xs text-muted-foreground">
          <p>
            Inference: <code className="text-accent">inference/generate_with_easv.py</code>
          </p>
          <p>
            Training: <code className="text-accent">training/easv_intensity.py</code>
          </p>
        </div>
      </div>
    </div>
  );
}
