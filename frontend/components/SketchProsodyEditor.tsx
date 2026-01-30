"use client";

/**
 * SketchProsodyEditor: Interactive prosody curve drawing component
 *
 * Based on DrawSpeech (ICASSP 2025) - enables users to draw pitch/energy
 * curves that control speech prosody generation.
 *
 * Features:
 * - Draw pitch curves (blue) and energy curves (orange)
 * - Real-time curve smoothing
 * - Preset emotion patterns
 * - Integration with prosody generation API
 * - Keyframe markers with drag-to-edit
 */

import React, {
  useRef,
  useState,
  useCallback,
  useEffect,
  useMemo,
} from "react";

// Types
interface Point {
  x: number;
  y: number;
}

interface Keyframe {
  id: string;
  time: number; // 0-1 normalized
  pitch: number; // 0-1 normalized
  energy: number; // 0-1 normalized
}

interface SketchData {
  pitch: number[];
  energy: number[];
  keyframes: Keyframe[];
}

interface EmotionPreset {
  name: string;
  pitch: number[];
  energy: number[];
  color: string;
}

interface SketchProsodyEditorProps {
  width?: number;
  height?: number;
  resolution?: number; // Number of points in curves
  onSketchChange?: (sketch: SketchData) => void;
  onGenerate?: (sketch: SketchData) => void;
  text?: string; // Text to display with markers
  wordTimestamps?: { word: string; start: number; end: number }[];
  initialSketch?: SketchData;
  disabled?: boolean;
}

// Emotion presets with characteristic prosody patterns
const EMOTION_PRESETS: EmotionPreset[] = [
  {
    name: "Neutral",
    pitch: generateFlatCurve(0.5, 100),
    energy: generateFlatCurve(0.5, 100),
    color: "#6B7280",
  },
  {
    name: "Happy",
    pitch: generateSineCurve(0.6, 0.15, 4, 100),
    energy: generateFlatCurve(0.7, 100),
    color: "#F59E0B",
  },
  {
    name: "Sad",
    pitch: generateFallingCurve(0.5, 0.3, 100),
    energy: generateFlatCurve(0.3, 100),
    color: "#3B82F6",
  },
  {
    name: "Angry",
    pitch: generateSineCurve(0.55, 0.2, 6, 100),
    energy: generateFlatCurve(0.85, 100),
    color: "#EF4444",
  },
  {
    name: "Surprised",
    pitch: generateRisingCurve(0.4, 0.7, 100),
    energy: generateFlatCurve(0.6, 100),
    color: "#F97316",
  },
  {
    name: "Calm",
    pitch: generateFlatCurve(0.5, 100),
    energy: generateFlatCurve(0.4, 100),
    color: "#10B981",
  },
];

// Curve generation utilities
function generateFlatCurve(value: number, length: number): number[] {
  return Array(length).fill(value);
}

function generateSineCurve(
  base: number,
  amplitude: number,
  cycles: number,
  length: number
): number[] {
  return Array.from({ length }, (_, i) => {
    const t = i / (length - 1);
    return base + amplitude * Math.sin(t * cycles * Math.PI * 2);
  });
}

function generateFallingCurve(
  start: number,
  end: number,
  length: number
): number[] {
  return Array.from({ length }, (_, i) => {
    const t = i / (length - 1);
    return start + (end - start) * t;
  });
}

function generateRisingCurve(
  start: number,
  end: number,
  length: number
): number[] {
  return Array.from({ length }, (_, i) => {
    const t = i / (length - 1);
    return start + (end - start) * t;
  });
}

// Gaussian smoothing for drawn curves
function smoothCurve(curve: number[], sigma: number = 2): number[] {
  const kernelSize = Math.ceil(sigma * 4) | 1;
  const halfKernel = Math.floor(kernelSize / 2);

  // Generate Gaussian kernel
  const kernel: number[] = [];
  let sum = 0;
  for (let i = -halfKernel; i <= halfKernel; i++) {
    const value = Math.exp((-i * i) / (2 * sigma * sigma));
    kernel.push(value);
    sum += value;
  }
  // Normalize
  for (let i = 0; i < kernel.length; i++) {
    kernel[i] /= sum;
  }

  // Apply convolution
  const smoothed: number[] = [];
  for (let i = 0; i < curve.length; i++) {
    let value = 0;
    for (let j = 0; j < kernel.length; j++) {
      const idx = Math.max(0, Math.min(curve.length - 1, i + j - halfKernel));
      value += curve[idx] * kernel[j];
    }
    smoothed.push(value);
  }

  return smoothed;
}

// Convert points to curve array
function pointsToCurve(
  points: Point[],
  resolution: number,
  canvasWidth: number,
  canvasHeight: number
): number[] {
  if (points.length === 0) return generateFlatCurve(0.5, resolution);

  const curve: number[] = [];

  for (let i = 0; i < resolution; i++) {
    const targetX = (i / (resolution - 1)) * canvasWidth;

    // Find surrounding points
    let leftIdx = 0;
    let rightIdx = points.length - 1;

    for (let j = 0; j < points.length; j++) {
      if (points[j].x <= targetX) leftIdx = j;
      if (points[j].x >= targetX && rightIdx === points.length - 1)
        rightIdx = j;
    }

    // Interpolate
    if (leftIdx === rightIdx || points[leftIdx].x === points[rightIdx].x) {
      curve.push(1 - points[leftIdx].y / canvasHeight);
    } else {
      const t =
        (targetX - points[leftIdx].x) /
        (points[rightIdx].x - points[leftIdx].x);
      const y = points[leftIdx].y + t * (points[rightIdx].y - points[leftIdx].y);
      curve.push(Math.max(0, Math.min(1, 1 - y / canvasHeight)));
    }
  }

  return curve;
}

// Main component
export default function SketchProsodyEditor({
  width = 600,
  height = 300,
  resolution = 100,
  onSketchChange,
  onGenerate,
  text,
  wordTimestamps,
  initialSketch,
  disabled = false,
}: SketchProsodyEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [activeCurve, setActiveCurve] = useState<"pitch" | "energy">("pitch");
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPoints, setCurrentPoints] = useState<Point[]>([]);
  const [pitchCurve, setPitchCurve] = useState<number[]>(
    initialSketch?.pitch || generateFlatCurve(0.5, resolution)
  );
  const [energyCurve, setEnergyCurve] = useState<number[]>(
    initialSketch?.energy || generateFlatCurve(0.5, resolution)
  );
  const [keyframes, setKeyframes] = useState<Keyframe[]>(
    initialSketch?.keyframes || []
  );
  const [smoothingEnabled, setSmoothingEnabled] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [draggingKeyframe, setDraggingKeyframe] = useState<string | null>(null);

  // Canvas dimensions
  const padding = { top: 30, right: 20, bottom: 50, left: 50 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  // Memoized sketch data
  const sketchData = useMemo<SketchData>(
    () => ({
      pitch: pitchCurve,
      energy: energyCurve,
      keyframes,
    }),
    [pitchCurve, energyCurve, keyframes]
  );

  // Notify parent of changes
  useEffect(() => {
    onSketchChange?.(sketchData);
  }, [sketchData, onSketchChange]);

  // Draw canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = "#2a2a4e";
      ctx.lineWidth = 1;

      // Vertical lines
      for (let i = 0; i <= 10; i++) {
        const x = padding.left + (i / 10) * chartWidth;
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
      }

      // Horizontal lines
      for (let i = 0; i <= 5; i++) {
        const y = padding.top + (i / 5) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }
    }

    // Draw word markers if available
    if (wordTimestamps && wordTimestamps.length > 0) {
      ctx.font = "10px monospace";
      ctx.fillStyle = "#6B7280";
      ctx.textAlign = "center";

      wordTimestamps.forEach((word) => {
        const x = padding.left + word.start * chartWidth;
        ctx.fillText(word.word, x + 5, height - 10);

        ctx.strokeStyle = "#374151";
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, height - padding.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // Draw energy curve (orange, behind pitch)
    ctx.strokeStyle = "#F97316";
    ctx.lineWidth = 2;
    ctx.globalAlpha = activeCurve === "energy" ? 1 : 0.5;
    ctx.beginPath();
    energyCurve.forEach((value, i) => {
      const x = padding.left + (i / (resolution - 1)) * chartWidth;
      const y = padding.top + (1 - value) * chartHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw pitch curve (blue, in front)
    ctx.strokeStyle = "#3B82F6";
    ctx.lineWidth = 2;
    ctx.globalAlpha = activeCurve === "pitch" ? 1 : 0.5;
    ctx.beginPath();
    pitchCurve.forEach((value, i) => {
      const x = padding.left + (i / (resolution - 1)) * chartWidth;
      const y = padding.top + (1 - value) * chartHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.globalAlpha = 1;

    // Draw keyframes
    keyframes.forEach((kf) => {
      const x = padding.left + kf.time * chartWidth;
      const yPitch = padding.top + (1 - kf.pitch) * chartHeight;
      const yEnergy = padding.top + (1 - kf.energy) * chartHeight;

      // Pitch keyframe
      ctx.fillStyle = "#3B82F6";
      ctx.beginPath();
      ctx.arc(x, yPitch, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Energy keyframe
      ctx.fillStyle = "#F97316";
      ctx.beginPath();
      ctx.arc(x, yEnergy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.stroke();

      // Vertical line connecting them
      ctx.strokeStyle = "#ffffff40";
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(x, yPitch);
      ctx.lineTo(x, yEnergy);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Draw currently drawing points
    if (isDrawing && currentPoints.length > 1) {
      ctx.strokeStyle = activeCurve === "pitch" ? "#60A5FA" : "#FB923C";
      ctx.lineWidth = 3;
      ctx.beginPath();
      currentPoints.forEach((point, i) => {
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      });
      ctx.stroke();
    }

    // Draw axes labels
    ctx.fillStyle = "#9CA3AF";
    ctx.font = "12px sans-serif";

    // Y-axis labels
    ctx.textAlign = "right";
    ctx.fillText("High", padding.left - 5, padding.top + 10);
    ctx.fillText("Low", padding.left - 5, height - padding.bottom);

    // X-axis labels
    ctx.textAlign = "center";
    ctx.fillText("Start", padding.left, height - padding.bottom + 20);
    ctx.fillText("End", width - padding.right, height - padding.bottom + 20);

    // Legend
    ctx.textAlign = "left";
    ctx.fillStyle = "#3B82F6";
    ctx.fillText("Pitch", padding.left, 15);
    ctx.fillStyle = "#F97316";
    ctx.fillText("Energy", padding.left + 60, 15);
  }, [
    width,
    height,
    pitchCurve,
    energyCurve,
    keyframes,
    activeCurve,
    isDrawing,
    currentPoints,
    showGrid,
    wordTimestamps,
    resolution,
    chartWidth,
    chartHeight,
  ]);

  // Redraw on state change
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Mouse event handlers
  const getCanvasPoint = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): Point => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };

      const rect = canvas.getBoundingClientRect();
      return {
        x: Math.max(
          padding.left,
          Math.min(width - padding.right, e.clientX - rect.left)
        ),
        y: Math.max(
          padding.top,
          Math.min(height - padding.bottom, e.clientY - rect.top)
        ),
      };
    },
    [width, height]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (disabled) return;

      const point = getCanvasPoint(e);

      // Check if clicking a keyframe
      const clickedKeyframe = keyframes.find((kf) => {
        const x = padding.left + kf.time * chartWidth;
        const yPitch = padding.top + (1 - kf.pitch) * chartHeight;
        const yEnergy = padding.top + (1 - kf.energy) * chartHeight;

        const distPitch = Math.sqrt(
          (point.x - x) ** 2 + (point.y - yPitch) ** 2
        );
        const distEnergy = Math.sqrt(
          (point.x - x) ** 2 + (point.y - yEnergy) ** 2
        );

        return distPitch < 10 || distEnergy < 10;
      });

      if (clickedKeyframe) {
        setDraggingKeyframe(clickedKeyframe.id);
      } else {
        setIsDrawing(true);
        setCurrentPoints([point]);
      }
    },
    [disabled, keyframes, getCanvasPoint, chartWidth, chartHeight]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (disabled) return;

      const point = getCanvasPoint(e);

      if (draggingKeyframe) {
        // Update keyframe position
        setKeyframes((prev) =>
          prev.map((kf) => {
            if (kf.id !== draggingKeyframe) return kf;

            const time = (point.x - padding.left) / chartWidth;
            const value = 1 - (point.y - padding.top) / chartHeight;

            if (activeCurve === "pitch") {
              return {
                ...kf,
                time: Math.max(0, Math.min(1, time)),
                pitch: Math.max(0, Math.min(1, value)),
              };
            } else {
              return {
                ...kf,
                time: Math.max(0, Math.min(1, time)),
                energy: Math.max(0, Math.min(1, value)),
              };
            }
          })
        );
      } else if (isDrawing) {
        setCurrentPoints((prev) => [...prev, point]);
      }
    },
    [
      disabled,
      draggingKeyframe,
      isDrawing,
      getCanvasPoint,
      activeCurve,
      chartWidth,
      chartHeight,
    ]
  );

  const handleMouseUp = useCallback(() => {
    if (draggingKeyframe) {
      setDraggingKeyframe(null);
      return;
    }

    if (!isDrawing || currentPoints.length === 0) return;

    // Convert points to curve
    let curve = pointsToCurve(
      currentPoints,
      resolution,
      width - padding.left - padding.right,
      height - padding.top - padding.bottom
    );

    // Adjust x coordinates
    const adjustedPoints = currentPoints.map((p) => ({
      x: p.x - padding.left,
      y: p.y - padding.top,
    }));

    curve = pointsToCurve(adjustedPoints, resolution, chartWidth, chartHeight);

    // Apply smoothing if enabled
    if (smoothingEnabled) {
      curve = smoothCurve(curve, 2);
    }

    // Update the appropriate curve
    if (activeCurve === "pitch") {
      setPitchCurve(curve);
    } else {
      setEnergyCurve(curve);
    }

    setIsDrawing(false);
    setCurrentPoints([]);
  }, [
    draggingKeyframe,
    isDrawing,
    currentPoints,
    resolution,
    smoothingEnabled,
    activeCurve,
    chartWidth,
    chartHeight,
    width,
    height,
  ]);

  // Apply preset
  const applyPreset = useCallback((preset: EmotionPreset) => {
    setPitchCurve(preset.pitch);
    setEnergyCurve(preset.energy);
    setKeyframes([]);
  }, []);

  // Add keyframe
  const addKeyframe = useCallback(() => {
    const id = `kf_${Date.now()}`;
    const newKeyframe: Keyframe = {
      id,
      time: 0.5,
      pitch: 0.5,
      energy: 0.5,
    };
    setKeyframes((prev) => [...prev, newKeyframe]);
  }, []);

  // Remove keyframe
  const removeKeyframe = useCallback((id: string) => {
    setKeyframes((prev) => prev.filter((kf) => kf.id !== id));
  }, []);

  // Clear all
  const clearAll = useCallback(() => {
    setPitchCurve(generateFlatCurve(0.5, resolution));
    setEnergyCurve(generateFlatCurve(0.5, resolution));
    setKeyframes([]);
  }, [resolution]);

  // Handle generate button
  const handleGenerate = useCallback(() => {
    onGenerate?.(sketchData);
  }, [sketchData, onGenerate]);

  return (
    <div className="flex flex-col gap-4 p-4 bg-gray-900 rounded-lg">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Prosody Sketch</h3>
        <div className="flex gap-2">
          <button
            onClick={clearAll}
            disabled={disabled}
            className="px-3 py-1 text-sm text-gray-300 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50"
          >
            Clear
          </button>
          <button
            onClick={handleGenerate}
            disabled={disabled}
            className="px-4 py-1 text-sm text-white bg-blue-600 rounded hover:bg-blue-500 disabled:opacity-50"
          >
            Generate
          </button>
        </div>
      </div>

      {/* Curve selector */}
      <div className="flex items-center gap-4">
        <span className="text-sm text-gray-400">Drawing:</span>
        <div className="flex gap-2">
          <button
            onClick={() => setActiveCurve("pitch")}
            className={`px-3 py-1 text-sm rounded ${
              activeCurve === "pitch"
                ? "bg-blue-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            Pitch
          </button>
          <button
            onClick={() => setActiveCurve("energy")}
            className={`px-3 py-1 text-sm rounded ${
              activeCurve === "energy"
                ? "bg-orange-600 text-white"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
            }`}
          >
            Energy
          </button>
        </div>

        <div className="flex items-center gap-2 ml-auto">
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input
              type="checkbox"
              checked={smoothingEnabled}
              onChange={(e) => setSmoothingEnabled(e.target.checked)}
              className="rounded"
            />
            Smooth
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-400">
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
              className="rounded"
            />
            Grid
          </label>
        </div>
      </div>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`border border-gray-700 rounded cursor-crosshair ${
          disabled ? "opacity-50 cursor-not-allowed" : ""
        }`}
      />

      {/* Text display */}
      {text && (
        <div className="p-2 text-sm text-gray-300 bg-gray-800 rounded">
          <span className="text-gray-500">Text:</span> {text}
        </div>
      )}

      {/* Emotion presets */}
      <div className="flex flex-wrap gap-2">
        <span className="text-sm text-gray-400 w-full mb-1">Presets:</span>
        {EMOTION_PRESETS.map((preset) => (
          <button
            key={preset.name}
            onClick={() => applyPreset(preset)}
            disabled={disabled}
            className="px-3 py-1 text-xs rounded transition-colors disabled:opacity-50"
            style={{
              backgroundColor: `${preset.color}20`,
              color: preset.color,
              border: `1px solid ${preset.color}40`,
            }}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Keyframes */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Keyframes:</span>
          <button
            onClick={addKeyframe}
            disabled={disabled}
            className="px-2 py-1 text-xs text-gray-300 bg-gray-700 rounded hover:bg-gray-600 disabled:opacity-50"
          >
            + Add Keyframe
          </button>
        </div>
        {keyframes.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {keyframes.map((kf, idx) => (
              <div
                key={kf.id}
                className="flex items-center gap-2 px-2 py-1 text-xs bg-gray-800 rounded"
              >
                <span className="text-gray-400">#{idx + 1}</span>
                <span className="text-gray-300">
                  t={kf.time.toFixed(2)} p={kf.pitch.toFixed(2)} e=
                  {kf.energy.toFixed(2)}
                </span>
                <button
                  onClick={() => removeKeyframe(kf.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="text-xs text-gray-500">
        <p>Draw curves by clicking and dragging. Higher = more intense.</p>
        <p>Drag keyframe points to fine-tune positions.</p>
      </div>
    </div>
  );
}

// Export utility functions for use elsewhere
export {
  generateFlatCurve,
  generateSineCurve,
  generateFallingCurve,
  generateRisingCurve,
  smoothCurve,
};
export type { SketchData, Keyframe, EmotionPreset };
