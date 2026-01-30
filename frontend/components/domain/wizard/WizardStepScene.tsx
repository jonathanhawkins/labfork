"use client";

import { useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Box,
  Monitor,
  Mic,
  Speaker,
  Server,
  Cpu,
  Activity,
  Waves,
  Bot,
  TrendingUp,
  Beaker,
  Brain,
  Check,
  Sparkles,
} from "lucide-react";

/**
 * Scene configuration for a domain
 */
export interface SceneConfig {
  backgroundStyle: "sky" | "grid" | "gradient" | "particles" | "minimal";
  props: string[];
  cameraAngle: "isometric" | "front" | "top" | "orbit";
}

/**
 * Available 3D props for the lab scene
 */
const AVAILABLE_PROPS = [
  {
    id: "microphone",
    name: "Microphone",
    icon: Mic,
    description: "Recording device for audio capture",
    category: "audio",
  },
  {
    id: "speaker",
    name: "Speaker",
    icon: Speaker,
    description: "Audio output visualization",
    category: "audio",
  },
  {
    id: "waveform",
    name: "Waveform Display",
    icon: Waves,
    description: "Real-time audio waveform",
    category: "audio",
  },
  {
    id: "server-rack",
    name: "Server Rack",
    icon: Server,
    description: "Compute infrastructure",
    category: "compute",
  },
  {
    id: "gpu",
    name: "GPU Unit",
    icon: Cpu,
    description: "Graphics processing unit",
    category: "compute",
  },
  {
    id: "supercomputer",
    name: "Supercomputer",
    icon: Box,
    description: "High-performance computing",
    category: "compute",
  },
  {
    id: "monitor",
    name: "Monitor",
    icon: Monitor,
    description: "Display screen for metrics",
    category: "display",
  },
  {
    id: "data-hub",
    name: "Data Hub",
    icon: Activity,
    description: "Central data processor",
    category: "data",
  },
  {
    id: "robot-arm",
    name: "Robot Arm",
    icon: Bot,
    description: "Robotic manipulator",
    category: "robotics",
  },
  {
    id: "trading-terminal",
    name: "Trading Terminal",
    icon: TrendingUp,
    description: "Financial data display",
    category: "finance",
  },
  {
    id: "lab-equipment",
    name: "Lab Equipment",
    icon: Beaker,
    description: "Scientific apparatus",
    category: "science",
  },
  {
    id: "neural-network",
    name: "Neural Network",
    icon: Brain,
    description: "AI model visualization",
    category: "ai",
  },
];

/**
 * Background style options
 */
const BACKGROUND_STYLES = [
  {
    id: "sky",
    name: "Gradient Sky",
    description: "Soft gradient background",
  },
  {
    id: "grid",
    name: "Grid Floor",
    description: "Infinite grid pattern",
  },
  {
    id: "gradient",
    name: "Color Gradient",
    description: "Custom color blend",
  },
  {
    id: "particles",
    name: "Particle Field",
    description: "Floating particles",
  },
  {
    id: "minimal",
    name: "Minimal",
    description: "Plain dark background",
  },
];

/**
 * Camera angle options
 */
const CAMERA_ANGLES = [
  { id: "isometric", name: "Isometric", description: "3/4 view angle" },
  { id: "front", name: "Front", description: "Face-on view" },
  { id: "top", name: "Top Down", description: "Bird's eye view" },
  { id: "orbit", name: "Orbit", description: "Auto-rotating view" },
];

export interface WizardStepSceneProps {
  /** Current scene configuration */
  scene: SceneConfig;
  /** Called when scene changes */
  onSceneChange: (scene: SceneConfig) => void;
  /** Primary color for previews */
  primaryColor?: string;
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepScene - Step 4: Choose 3D props and scene configuration
 */
export function WizardStepScene({
  scene,
  onSceneChange,
  primaryColor = "#4ecdc4",
  className,
}: WizardStepSceneProps) {
  // Toggle prop selection
  const toggleProp = useCallback(
    (propId: string) => {
      const props = scene.props.includes(propId)
        ? scene.props.filter((p) => p !== propId)
        : [...scene.props, propId];
      onSceneChange({ ...scene, props });
    },
    [scene, onSceneChange]
  );

  // Get props by category
  const categories = [
    { id: "audio", name: "Audio" },
    { id: "compute", name: "Compute" },
    { id: "display", name: "Display" },
    { id: "data", name: "Data" },
    { id: "robotics", name: "Robotics" },
    { id: "finance", name: "Finance" },
    { id: "science", name: "Science" },
    { id: "ai", name: "AI" },
  ];

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-foreground-muted/10 mb-4">
          <Box className="w-6 h-6 text-foreground-muted" />
        </div>
        <h2 className="text-lg font-normal text-foreground-bright">
          Lab Scene
        </h2>
        <p className="text-sm text-foreground-muted mt-1">
          Customize your 3D lab environment
        </p>
      </div>

      {/* Background style */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted">Background Style</label>
        <div className="grid grid-cols-5 gap-2">
          {BACKGROUND_STYLES.map((style) => (
            <button
              key={style.id}
              onClick={() =>
                onSceneChange({
                  ...scene,
                  backgroundStyle: style.id as SceneConfig["backgroundStyle"],
                })
              }
              className={cn(
                "p-3 rounded-lg border transition-all text-center",
                scene.backgroundStyle === style.id
                  ? "border-foreground-bright bg-foreground-bright/10"
                  : "border-border bg-background-card hover:border-foreground-muted"
              )}
            >
              <div
                className={cn(
                  "w-full h-8 rounded mb-2",
                  style.id === "sky" && "bg-gradient-to-b from-blue-900 to-purple-900",
                  style.id === "grid" && "bg-background border-b border-foreground-muted/20",
                  style.id === "gradient" && "bg-gradient-to-br from-foreground-muted/20 to-transparent",
                  style.id === "particles" && "bg-background relative overflow-hidden",
                  style.id === "minimal" && "bg-background"
                )}
              >
                {style.id === "particles" && (
                  <Sparkles className="w-4 h-4 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-foreground-muted/50" />
                )}
              </div>
              <span className="text-[10px] text-foreground-muted">{style.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Camera angle */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted">Camera Angle</label>
        <div className="grid grid-cols-4 gap-2">
          {CAMERA_ANGLES.map((angle) => (
            <button
              key={angle.id}
              onClick={() =>
                onSceneChange({
                  ...scene,
                  cameraAngle: angle.id as SceneConfig["cameraAngle"],
                })
              }
              className={cn(
                "p-2.5 rounded-lg border transition-colors text-center",
                scene.cameraAngle === angle.id
                  ? "border-foreground-bright bg-foreground-bright/10"
                  : "border-border bg-background-card hover:border-foreground-muted"
              )}
            >
              <span className="text-xs text-foreground">{angle.name}</span>
              <p className="text-[10px] text-foreground-subtle mt-0.5">
                {angle.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* 3D Props selection */}
      <div className="space-y-3">
        <label className="text-sm text-foreground-muted flex items-center justify-between">
          <span>3D Props</span>
          <span className="text-xs text-foreground-subtle">
            {scene.props.length} selected
          </span>
        </label>

        {/* Props by category */}
        <div className="space-y-4">
          {categories
            .filter((cat) =>
              AVAILABLE_PROPS.some((p) => p.category === cat.id)
            )
            .map((category) => (
              <div key={category.id}>
                <h4 className="text-xs text-foreground-subtle mb-2">
                  {category.name}
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {AVAILABLE_PROPS.filter((p) => p.category === category.id).map(
                    (prop) => {
                      const Icon = prop.icon;
                      const isSelected = scene.props.includes(prop.id);

                      return (
                        <button
                          key={prop.id}
                          onClick={() => toggleProp(prop.id)}
                          className={cn(
                            "relative flex items-center gap-2 p-3 rounded-lg border transition-all text-left",
                            isSelected
                              ? "border-foreground-bright/50 bg-foreground-bright/5"
                              : "border-border bg-background-card hover:border-foreground-muted"
                          )}
                        >
                          {/* Selected indicator */}
                          {isSelected && (
                            <div
                              className="absolute top-2 right-2 w-4 h-4 rounded-full flex items-center justify-center"
                              style={{ backgroundColor: primaryColor }}
                            >
                              <Check className="w-2.5 h-2.5 text-white" />
                            </div>
                          )}

                          <div
                            className={cn(
                              "w-8 h-8 rounded-lg flex items-center justify-center",
                              isSelected
                                ? "bg-foreground-bright/20"
                                : "bg-foreground-muted/10"
                            )}
                          >
                            <Icon
                              className="w-4 h-4"
                              style={{
                                color: isSelected ? primaryColor : undefined,
                              }}
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p
                              className={cn(
                                "text-xs font-normal truncate",
                                isSelected
                                  ? "text-foreground-bright"
                                  : "text-foreground"
                              )}
                            >
                              {prop.name}
                            </p>
                            <p className="text-[10px] text-foreground-subtle truncate">
                              {prop.description}
                            </p>
                          </div>
                        </button>
                      );
                    }
                  )}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Scene preview placeholder */}
      <div className="space-y-2">
        <label className="text-sm text-foreground-muted">Scene Preview</label>
        <div
          className={cn(
            "h-32 rounded-lg border border-border overflow-hidden flex items-center justify-center",
            scene.backgroundStyle === "sky" &&
              "bg-gradient-to-b from-blue-900/50 to-purple-900/50",
            scene.backgroundStyle === "grid" && "bg-background",
            scene.backgroundStyle === "gradient" &&
              "bg-gradient-to-br from-foreground-muted/10 to-transparent",
            scene.backgroundStyle === "particles" && "bg-background",
            scene.backgroundStyle === "minimal" && "bg-background"
          )}
        >
          {scene.props.length > 0 ? (
            <div className="flex items-center gap-3">
              {scene.props.slice(0, 5).map((propId) => {
                const prop = AVAILABLE_PROPS.find((p) => p.id === propId);
                if (!prop) return null;
                const Icon = prop.icon;
                return (
                  <div
                    key={propId}
                    className="w-10 h-10 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${primaryColor}20` }}
                  >
                    <Icon className="w-5 h-5" style={{ color: primaryColor }} />
                  </div>
                );
              })}
              {scene.props.length > 5 && (
                <span className="text-xs text-foreground-muted">
                  +{scene.props.length - 5} more
                </span>
              )}
            </div>
          ) : (
            <p className="text-xs text-foreground-subtle">
              Select props to populate your scene
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export { AVAILABLE_PROPS, BACKGROUND_STYLES, CAMERA_ANGLES };
export default WizardStepScene;
