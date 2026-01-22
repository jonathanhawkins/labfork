"use client";

import React, {
  useState,
  useRef,
  useCallback,
  useMemo,
  useEffect,
} from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Trash2, Plus, GripVertical, Type, Eye, EyeOff } from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface Keyframe {
  id: string;
  time: number; // 0.0 to 1.0
  emotion: string;
  intensity: number; // 0 to 100
}

export interface KeyframeTimelineProps {
  text: string;
  keyframes: Keyframe[];
  onChange: (keyframes: Keyframe[]) => void;
  duration?: number; // seconds, for display
  className?: string;
  // Playback state for text highlighting
  playbackTime?: number; // 0.0 to 1.0 normalized position
  isPlaying?: boolean;
}

// ============================================================================
// Constants
// ============================================================================

const EMOTIONS = [
  { value: "neutral", label: "Neutral" },
  { value: "happy", label: "Happy" },
  { value: "sad", label: "Sad" },
  { value: "angry", label: "Angry" },
  { value: "surprised", label: "Surprised" },
  { value: "calm", label: "Calm" },
  { value: "fearful", label: "Fearful" },
  { value: "excited", label: "Excited" },
] as const;

const EMOTION_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  neutral: {
    bg: "bg-zinc-500",
    border: "border-zinc-400",
    text: "text-zinc-300",
  },
  happy: {
    bg: "bg-yellow-500",
    border: "border-yellow-400",
    text: "text-yellow-300",
  },
  sad: {
    bg: "bg-blue-500",
    border: "border-blue-400",
    text: "text-blue-300",
  },
  angry: {
    bg: "bg-red-500",
    border: "border-red-400",
    text: "text-red-300",
  },
  surprised: {
    bg: "bg-orange-500",
    border: "border-orange-400",
    text: "text-orange-300",
  },
  calm: {
    bg: "bg-emerald-500",
    border: "border-emerald-400",
    text: "text-emerald-300",
  },
  fearful: {
    bg: "bg-purple-500",
    border: "border-purple-400",
    text: "text-purple-300",
  },
  excited: {
    bg: "bg-pink-500",
    border: "border-pink-400",
    text: "text-pink-300",
  },
};

// ============================================================================
// Utility Functions
// ============================================================================

function generateId(): string {
  return `kf-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
}

function getEmotionColors(emotion: string) {
  return EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;
}

// ============================================================================
// Keyframe Marker Component
// ============================================================================

interface KeyframeMarkerProps {
  keyframe: Keyframe;
  isSelected: boolean;
  onSelect: () => void;
  onDrag: (newTime: number) => void;
  containerRef: React.RefObject<HTMLDivElement>;
}

function KeyframeMarker({
  keyframe,
  isSelected,
  onSelect,
  onDrag,
  containerRef,
}: KeyframeMarkerProps) {
  const [isDragging, setIsDragging] = useState(false);
  const markerRef = useRef<HTMLDivElement>(null);

  const colors = getEmotionColors(keyframe.emotion);

  // Calculate marker size based on intensity (min 12px, max 24px)
  const markerSize = 12 + (keyframe.intensity / 100) * 12;

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
      onSelect();
    },
    [onSelect]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const newTime = Math.max(0, Math.min(1, x / rect.width));
      onDrag(newTime);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, onDrag, containerRef]);

  return (
    <div
      ref={markerRef}
      className={cn(
        "absolute top-1/2 -translate-y-1/2 cursor-grab active:cursor-grabbing",
        "transition-transform duration-100",
        isDragging && "z-20",
        isSelected && "z-10"
      )}
      style={{
        left: `${keyframe.time * 100}%`,
        transform: `translateX(-50%) translateY(-50%)`,
      }}
      onMouseDown={handleMouseDown}
    >
      {/* Diamond marker */}
      <div
        className={cn(
          "rotate-45 border-2 transition-all duration-150",
          colors.bg,
          colors.border,
          isSelected && "ring-2 ring-white ring-offset-1 ring-offset-background",
          isDragging && "scale-110"
        )}
        style={{
          width: markerSize,
          height: markerSize,
          opacity: 0.3 + (keyframe.intensity / 100) * 0.7,
        }}
      />

      {/* Vertical line extending from marker */}
      <div
        className={cn(
          "absolute left-1/2 -translate-x-1/2 w-px",
          colors.bg,
          "opacity-50"
        )}
        style={{
          top: markerSize / 2 + 4,
          height: 20,
        }}
      />

      {/* Tooltip on hover/select */}
      {isSelected && (
        <div
          className={cn(
            "absolute left-1/2 -translate-x-1/2 whitespace-nowrap",
            "px-2 py-1 rounded text-xs font-medium",
            "bg-popover border border-border shadow-md",
            "text-popover-foreground"
          )}
          style={{ top: -32 }}
        >
          <span className={colors.text}>{keyframe.emotion}</span>
          <span className="text-muted-foreground ml-1">
            {keyframe.intensity}%
          </span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Keyframe Editor Panel
// ============================================================================

interface KeyframeEditorProps {
  keyframe: Keyframe;
  onChange: (updated: Keyframe) => void;
  onDelete: () => void;
  duration?: number;
}

function KeyframeEditor({
  keyframe,
  onChange,
  onDelete,
  duration = 1,
}: KeyframeEditorProps) {
  const colors = getEmotionColors(keyframe.emotion);

  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-muted/30 border border-border">
      {/* Drag handle indicator */}
      <div className="flex items-center gap-2 text-muted-foreground">
        <GripVertical className="h-4 w-4" />
        <span className="text-xs font-mono w-16">
          {formatTime(keyframe.time * duration)}
        </span>
      </div>

      {/* Emotion selector */}
      <div className="flex flex-col gap-1">
        <Label className="text-xs text-muted-foreground">Emotion</Label>
        <Select
          value={keyframe.emotion}
          onValueChange={(value) => onChange({ ...keyframe, emotion: value })}
        >
          <SelectTrigger className="w-32 h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {EMOTIONS.map((emotion) => (
              <SelectItem key={emotion.value} value={emotion.value}>
                <div className="flex items-center gap-2">
                  <div
                    className={cn(
                      "w-2 h-2 rounded-full",
                      getEmotionColors(emotion.value).bg
                    )}
                  />
                  {emotion.label}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Intensity slider */}
      <div className="flex flex-col gap-1 flex-1 min-w-[120px] max-w-[200px]">
        <div className="flex justify-between items-center">
          <Label className="text-xs text-muted-foreground">Intensity</Label>
          <span className="text-xs font-mono text-muted-foreground">
            {keyframe.intensity}%
          </span>
        </div>
        <Slider
          value={[keyframe.intensity]}
          onValueChange={([value]) =>
            onChange({ ...keyframe, intensity: value })
          }
          min={0}
          max={100}
          step={1}
          className="w-full"
        />
      </div>

      {/* Delete button */}
      <Button
        variant="ghost"
        size="icon"
        onClick={onDelete}
        className="h-8 w-8 text-muted-foreground hover:text-destructive"
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  );
}

// ============================================================================
// Timeline Ruler Component
// ============================================================================

interface TimelineRulerProps {
  duration: number;
  divisions?: number;
}

function TimelineRuler({ duration, divisions = 10 }: TimelineRulerProps) {
  const ticks = useMemo(() => {
    const result = [];
    for (let i = 0; i <= divisions; i++) {
      const position = i / divisions;
      const time = position * duration;
      const isMain = i % 5 === 0;
      result.push({ position, time, isMain });
    }
    return result;
  }, [duration, divisions]);

  return (
    <div className="relative h-5 border-b border-border">
      {ticks.map(({ position, time, isMain }, index) => (
        <div
          key={index}
          className="absolute top-0 -translate-x-1/2"
          style={{ left: `${position * 100}%` }}
        >
          <div
            className={cn(
              "w-px bg-muted-foreground/50",
              isMain ? "h-3" : "h-2"
            )}
          />
          {isMain && (
            <span className="absolute top-3 left-1/2 -translate-x-1/2 text-[10px] font-mono text-muted-foreground whitespace-nowrap">
              {formatTime(time)}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Text Overlay Component - Shows words positioned on timeline
// ============================================================================

interface TextOverlayProps {
  text: string;
  playbackTime: number; // 0.0 to 1.0
  isPlaying: boolean;
}

function TextOverlay({ text, playbackTime, isPlaying }: TextOverlayProps) {
  // Split text into words and calculate positions
  const words = useMemo(() => {
    if (!text) return [];

    const wordList = text.split(/\s+/).filter(w => w.length > 0);
    const totalWords = wordList.length;

    // Distribute words evenly across the timeline
    // Add small padding at start/end
    const startPad = 0.02;
    const endPad = 0.02;
    const usableWidth = 1 - startPad - endPad;

    return wordList.map((word, index) => {
      // Position based on word index
      const position = startPad + (index / Math.max(1, totalWords - 1)) * usableWidth;
      // For single word, center it
      const finalPosition = totalWords === 1 ? 0.5 : position;

      return {
        word,
        position: finalPosition,
        index,
      };
    });
  }, [text]);

  // Find current word based on playback position
  const currentWordIndex = useMemo(() => {
    if (!isPlaying || words.length === 0) return -1;

    // Find which word the playhead is at or just passed
    for (let i = words.length - 1; i >= 0; i--) {
      if (playbackTime >= words[i].position - 0.05) {
        return i;
      }
    }
    return 0;
  }, [words, playbackTime, isPlaying]);

  if (words.length === 0) return null;

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {/* Playhead indicator */}
      {isPlaying && (
        <div
          className="absolute top-0 h-full w-0.5 bg-amber-500/70 z-10 transition-all duration-75"
          style={{ left: `${playbackTime * 100}%` }}
        />
      )}

      {/* Words */}
      <div className="absolute inset-x-0 top-1 h-6 flex items-center">
        {words.map(({ word, position, index }) => {
          const isCurrentWord = index === currentWordIndex;
          const isPastWord = index < currentWordIndex;

          return (
            <span
              key={index}
              className={cn(
                "absolute text-[10px] font-mono whitespace-nowrap transition-all duration-150",
                "-translate-x-1/2",
                isCurrentWord
                  ? "text-amber-400 font-bold scale-110"
                  : isPastWord
                    ? "text-slate-500"
                    : "text-slate-600"
              )}
              style={{
                left: `${position * 100}%`,
              }}
            >
              {word}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================================
// Main KeyframeTimeline Component
// ============================================================================

export default function KeyframeTimeline({
  text,
  keyframes,
  onChange,
  duration = 1,
  className,
  playbackTime = 0,
  isPlaying = false,
}: KeyframeTimelineProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showTextOverlay, setShowTextOverlay] = useState(true);
  const trackRef = useRef<HTMLDivElement>(null);

  // Sort keyframes by time for consistent rendering
  const sortedKeyframes = useMemo(
    () => [...keyframes].sort((a, b) => a.time - b.time),
    [keyframes]
  );

  const selectedKeyframe = useMemo(
    () => keyframes.find((kf) => kf.id === selectedId) || null,
    [keyframes, selectedId]
  );

  // Handle keyframe updates
  const handleKeyframeUpdate = useCallback(
    (id: string, updates: Partial<Keyframe>) => {
      const updated = keyframes.map((kf) =>
        kf.id === id ? { ...kf, ...updates } : kf
      );
      onChange(updated);
    },
    [keyframes, onChange]
  );

  // Handle keyframe deletion
  const handleKeyframeDelete = useCallback(
    (id: string) => {
      const filtered = keyframes.filter((kf) => kf.id !== id);
      onChange(filtered);
      if (selectedId === id) {
        setSelectedId(null);
      }
    },
    [keyframes, onChange, selectedId]
  );

  // Handle adding new keyframe via double-click
  const handleTrackDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!trackRef.current) return;

      const rect = trackRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const time = Math.max(0, Math.min(1, x / rect.width));

      const newKeyframe: Keyframe = {
        id: generateId(),
        time,
        emotion: "neutral",
        intensity: 75,
      };

      onChange([...keyframes, newKeyframe]);
      setSelectedId(newKeyframe.id);
    },
    [keyframes, onChange]
  );

  // Handle adding keyframe via button
  const handleAddKeyframe = useCallback(() => {
    // Find a good position for the new keyframe
    let time = 0.5;
    if (keyframes.length > 0) {
      // Place at the end if there's space, otherwise at 0.5
      const lastKeyframe = sortedKeyframes[sortedKeyframes.length - 1];
      if (lastKeyframe.time < 0.9) {
        time = Math.min(1, lastKeyframe.time + 0.1);
      }
    }

    const newKeyframe: Keyframe = {
      id: generateId(),
      time,
      emotion: "neutral",
      intensity: 75,
    };

    onChange([...keyframes, newKeyframe]);
    setSelectedId(newKeyframe.id);
  }, [keyframes, sortedKeyframes, onChange]);

  // Handle clicking outside to deselect
  const handleTrackClick = useCallback((e: React.MouseEvent) => {
    // Only deselect if clicking directly on the track, not on a marker
    if (e.target === e.currentTarget) {
      setSelectedId(null);
    }
  }, []);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {/* Header with text and buttons */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-muted-foreground mb-1">Utterance</p>
          <p className="text-sm font-medium leading-relaxed truncate">
            {text || "No text provided"}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {/* Toggle text overlay */}
          <Button
            variant={showTextOverlay ? "secondary" : "outline"}
            size="sm"
            onClick={() => setShowTextOverlay(!showTextOverlay)}
            title={showTextOverlay ? "Hide script overlay" : "Show script overlay"}
          >
            {showTextOverlay ? (
              <Eye className="h-4 w-4 mr-1" />
            ) : (
              <EyeOff className="h-4 w-4 mr-1" />
            )}
            Script
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleAddKeyframe}
          >
            <Plus className="h-4 w-4 mr-1" />
            Add Keyframe
          </Button>
        </div>
      </div>

      {/* Timeline container */}
      <div className="rounded-lg border border-border bg-card overflow-hidden">
        {/* Timeline ruler */}
        <TimelineRuler duration={duration} />

        {/* Main track area */}
        <div
          ref={trackRef}
          className={cn(
            "relative bg-muted/20 cursor-crosshair",
            "border-b border-border",
            showTextOverlay ? "h-20" : "h-16" // Taller when showing text
          )}
          onClick={handleTrackClick}
          onDoubleClick={handleTrackDoubleClick}
        >
          {/* Track background gradient */}
          <div className="absolute inset-0 bg-gradient-to-r from-muted/30 via-transparent to-muted/30 pointer-events-none" />

          {/* Text overlay - words positioned on timeline */}
          {showTextOverlay && (
            <TextOverlay
              text={text}
              playbackTime={playbackTime}
              isPlaying={isPlaying}
            />
          )}

          {/* Grid lines */}
          <div className="absolute inset-0 pointer-events-none">
            {[0.25, 0.5, 0.75].map((pos) => (
              <div
                key={pos}
                className="absolute top-0 h-full w-px bg-border/50"
                style={{ left: `${pos * 100}%` }}
              />
            ))}
          </div>

          {/* Keyframe markers - positioned lower when text overlay is shown */}
          <div
            className={cn(
              "absolute inset-x-0",
              showTextOverlay ? "top-8 bottom-0" : "inset-y-0"
            )}
          >
            {sortedKeyframes.map((keyframe) => (
              <KeyframeMarker
                key={keyframe.id}
                keyframe={keyframe}
                isSelected={keyframe.id === selectedId}
                onSelect={() => setSelectedId(keyframe.id)}
                onDrag={(newTime) =>
                  handleKeyframeUpdate(keyframe.id, { time: newTime })
                }
                containerRef={trackRef as React.RefObject<HTMLDivElement>}
              />
            ))}
          </div>

          {/* Hint text when empty */}
          {keyframes.length === 0 && !showTextOverlay && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <p className="text-sm text-muted-foreground">
                Double-click to add keyframes
              </p>
            </div>
          )}
        </div>

        {/* Percentage labels */}
        <div className="relative h-5 px-1">
          <span className="absolute left-1 text-[10px] text-muted-foreground">
            0%
          </span>
          <span className="absolute left-1/2 -translate-x-1/2 text-[10px] text-muted-foreground">
            50%
          </span>
          <span className="absolute right-1 text-[10px] text-muted-foreground">
            100%
          </span>
        </div>
      </div>

      {/* Keyframe editor panel */}
      {selectedKeyframe && (
        <KeyframeEditor
          keyframe={selectedKeyframe}
          onChange={(updated) =>
            handleKeyframeUpdate(updated.id, updated)
          }
          onDelete={() => handleKeyframeDelete(selectedKeyframe.id)}
          duration={duration}
        />
      )}

      {/* Keyframe list summary */}
      {keyframes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {sortedKeyframes.map((keyframe) => {
            const colors = getEmotionColors(keyframe.emotion);
            return (
              <button
                key={keyframe.id}
                onClick={() => setSelectedId(keyframe.id)}
                className={cn(
                  "inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs",
                  "border transition-all duration-150",
                  keyframe.id === selectedId
                    ? "bg-accent border-accent-foreground/20"
                    : "bg-muted/50 border-transparent hover:border-border"
                )}
              >
                <div
                  className={cn("w-2 h-2 rounded-full", colors.bg)}
                  style={{ opacity: 0.3 + (keyframe.intensity / 100) * 0.7 }}
                />
                <span className="font-mono text-muted-foreground">
                  {Math.round(keyframe.time * 100)}%
                </span>
                <span className="capitalize">{keyframe.emotion}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
