"use client";

import { useState, useEffect, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Cpu,
  Server,
  Cloud,
  Check,
  X,
  AlertTriangle,
  Loader2,
  RefreshCw,
  Terminal,
  Key,
  Info,
  Wifi,
  WifiOff,
  HardDrive,
  Zap,
} from "lucide-react";
import type { HardwareConfig, HardwareType, GpuInfo, SSHConfig, CloudConfig, LocalConfig } from "@/lib/lab-wizard/types";
import { CLOUD_PROVIDERS } from "@/lib/lab-wizard/types";

/**
 * Hardware option card data
 */
interface HardwareOption {
  id: HardwareType;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
}

const HARDWARE_OPTIONS: HardwareOption[] = [
  {
    id: "local",
    name: "Local Machine",
    description: "Use your current computer's GPU",
    icon: Cpu,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
  },
  {
    id: "remote-ssh",
    name: "Remote SSH",
    description: "Connect to a remote GPU server",
    icon: Server,
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
  },
  {
    id: "cloud",
    name: "Cloud Provider",
    description: "Use RunPod, AWS, or other cloud GPUs",
    icon: Cloud,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
  },
];

export interface WizardStepHardwareProps {
  /** Current hardware configuration */
  config: HardwareConfig;
  /** Called when hardware config changes */
  onConfigChange: (config: HardwareConfig) => void;
  /** Selected domain (for GPU requirements) */
  selectedDomain?: string;
  /** Custom class name */
  className?: string;
}

/**
 * WizardStepHardware - Hardware configuration step
 */
export function WizardStepHardware({
  config,
  onConfigChange,
  selectedDomain,
  className,
}: WizardStepHardwareProps) {
  const [isDetecting, setIsDetecting] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [localGpu, setLocalGpu] = useState<GpuInfo | null>(null);
  const [ollamaStatus, setOllamaStatus] = useState<"unknown" | "available" | "unavailable">("unknown");
  const [sshStatus, setSshStatus] = useState<"unknown" | "testing" | "connected" | "failed">("unknown");
  const [sshError, setSshError] = useState<string | null>(null);
  const [remoteGpu, setRemoteGpu] = useState<GpuInfo | null>(null);

  // Detect local hardware on mount and when switching to local
  useEffect(() => {
    if (config.type === "local" && !localGpu) {
      detectLocalHardware();
    }
  }, [config.type]);

  // Detect local GPU and Ollama
  const detectLocalHardware = useCallback(async () => {
    setIsDetecting(true);
    try {
      const response = await fetch("/api/lab/hardware");
      const data = await response.json();

      if (data.success) {
        if (data.gpu) {
          setLocalGpu(data.gpu);
          // Update config with detected GPU
          onConfigChange({
            ...config,
            type: "local",
            local: {
              ...config.local,
              gpu: data.gpu,
              systemInfo: data.systemInfo,
            },
          });
        }

        setOllamaStatus(data.ollama?.available ? "available" : "unavailable");
      }
    } catch (error) {
      console.error("Failed to detect local hardware:", error);
    } finally {
      setIsDetecting(false);
    }
  }, [config, onConfigChange]);

  // Test SSH connection
  const testSSHConnection = useCallback(async () => {
    if (!config.ssh?.host || !config.ssh?.user) {
      setSshError("Host and user are required");
      return;
    }

    setIsTesting(true);
    setSshStatus("testing");
    setSshError(null);
    setRemoteGpu(null);

    try {
      // Test connection
      const testResponse = await fetch("/api/lab/hardware/ssh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "test",
          host: config.ssh.host,
          port: config.ssh.port || 22,
          user: config.ssh.user,
          keyPath: config.ssh.keyPath,
        }),
      });

      const testData = await testResponse.json();

      if (!testData.success) {
        setSshStatus("failed");
        setSshError(testData.error || "Connection failed");
        return;
      }

      // Detect GPU on remote
      const gpuResponse = await fetch("/api/lab/hardware/ssh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "detect-gpu",
          host: config.ssh.host,
          port: config.ssh.port || 22,
          user: config.ssh.user,
          keyPath: config.ssh.keyPath,
        }),
      });

      const gpuData = await gpuResponse.json();

      setSshStatus("connected");

      if (gpuData.success && gpuData.gpu) {
        setRemoteGpu(gpuData.gpu);
        onConfigChange({
          ...config,
          ssh: {
            ...config.ssh,
            verified: true,
            remoteGpu: gpuData.gpu,
          },
        });
      } else {
        onConfigChange({
          ...config,
          ssh: {
            ...config.ssh,
            verified: true,
          },
        });
      }
    } catch (error) {
      setSshStatus("failed");
      setSshError(error instanceof Error ? error.message : "Connection failed");
    } finally {
      setIsTesting(false);
    }
  }, [config, onConfigChange]);

  // Handle hardware type change
  const handleTypeChange = (type: HardwareType) => {
    onConfigChange({
      ...config,
      type,
    });

    // Reset status when changing type
    setSshStatus("unknown");
    setSshError(null);
    setRemoteGpu(null);
  };

  // Handle SSH config change
  const handleSSHChange = (field: keyof SSHConfig, value: string | number) => {
    onConfigChange({
      ...config,
      ssh: {
        ...config.ssh,
        [field]: value,
        verified: false, // Reset verification on change
      },
    });
    setSshStatus("unknown");
    setSshError(null);
  };

  // Handle cloud config change
  const handleCloudChange = (field: keyof CloudConfig, value: string) => {
    onConfigChange({
      ...config,
      cloud: {
        ...config.cloud,
        [field]: value,
      },
    });
  };

  // Get GPU requirements for domain
  const getGpuRequirement = () => {
    switch (selectedDomain) {
      case "voice-clone":
        return "24GB VRAM recommended for training";
      case "quant-trading":
        return "8GB VRAM minimum";
      case "robotics":
        return "24GB VRAM recommended for simulation";
      case "biotech":
        return "48GB+ VRAM for molecular modeling";
      default:
        return "8GB VRAM minimum recommended";
    }
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Info banner */}
      <div className="flex items-start gap-3 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
        <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm text-foreground">
            Configure where your research will run. You can change this later.
          </p>
          {selectedDomain && (
            <p className="text-xs text-foreground-muted mt-1">
              {getGpuRequirement()}
            </p>
          )}
        </div>
      </div>

      {/* Hardware type selector */}
      <div className="space-y-3">
        <h3 className="text-sm text-foreground-muted">Select Hardware Type</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {HARDWARE_OPTIONS.map((option) => {
            const Icon = option.icon;
            const isSelected = config.type === option.id;

            return (
              <button
                key={option.id}
                onClick={() => handleTypeChange(option.id)}
                className={cn(
                  "flex flex-col items-center gap-3 p-4 rounded-lg border text-center transition-colors",
                  isSelected
                    ? "border-foreground-bright bg-foreground-bright/5"
                    : "border-border hover:border-foreground-muted bg-background-card"
                )}
              >
                <div
                  className={cn(
                    "w-12 h-12 rounded-lg flex items-center justify-center",
                    option.bgColor
                  )}
                >
                  <Icon className={cn("w-6 h-6", option.color)} />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-foreground">
                    {option.name}
                  </h4>
                  <p className="text-xs text-foreground-muted mt-1">
                    {option.description}
                  </p>
                </div>
                {isSelected && (
                  <Check className="w-5 h-5 text-foreground-bright" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Local hardware configuration */}
      {config.type === "local" && (
        <div className="space-y-4 p-4 rounded-lg bg-background border border-border">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-foreground">Local Hardware</h3>
            <button
              onClick={detectLocalHardware}
              disabled={isDetecting}
              className={cn(
                "flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg",
                "border border-border hover:bg-foreground-muted/10",
                "transition-colors disabled:opacity-50"
              )}
            >
              {isDetecting ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <RefreshCw className="w-3 h-3" />
              )}
              Detect
            </button>
          </div>

          {/* GPU Status */}
          <div className="space-y-3">
            <div className="flex items-center gap-3 p-3 rounded-lg bg-background-card">
              <div className={cn(
                "w-10 h-10 rounded-lg flex items-center justify-center",
                localGpu?.available ? "bg-green-500/10" : "bg-yellow-500/10"
              )}>
                <HardDrive className={cn(
                  "w-5 h-5",
                  localGpu?.available ? "text-green-400" : "text-yellow-400"
                )} />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium text-foreground">
                    {localGpu?.name || "No GPU Detected"}
                  </h4>
                  {localGpu?.available && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-green-500/20 text-green-400">
                      Available
                    </span>
                  )}
                </div>
                {localGpu?.vram && (
                  <p className="text-xs text-foreground-muted">
                    {localGpu.vram}GB VRAM
                    {localGpu.cudaVersion && ` • CUDA ${localGpu.cudaVersion}`}
                  </p>
                )}
                {!localGpu && !isDetecting && (
                  <p className="text-xs text-foreground-muted">
                    Click Detect to scan for GPU
                  </p>
                )}
              </div>
            </div>

            {/* Ollama Status */}
            <div className="flex items-center gap-3 p-3 rounded-lg bg-background-card">
              <div className={cn(
                "w-10 h-10 rounded-lg flex items-center justify-center",
                ollamaStatus === "available" ? "bg-green-500/10" : "bg-foreground-muted/10"
              )}>
                <Zap className={cn(
                  "w-5 h-5",
                  ollamaStatus === "available" ? "text-green-400" : "text-foreground-muted"
                )} />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium text-foreground">Ollama</h4>
                  {ollamaStatus === "available" && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-green-500/20 text-green-400">
                      Running
                    </span>
                  )}
                  {ollamaStatus === "unavailable" && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] bg-yellow-500/20 text-yellow-400">
                      Not Running
                    </span>
                  )}
                </div>
                <p className="text-xs text-foreground-muted">
                  {ollamaStatus === "available"
                    ? "Local LLM available for free usage"
                    : "Install Ollama for free local AI"}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Remote SSH configuration */}
      {config.type === "remote-ssh" && (
        <div className="space-y-4 p-4 rounded-lg bg-background border border-border">
          <h3 className="text-sm font-medium text-foreground">SSH Connection</h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                Host / IP Address
              </label>
              <input
                type="text"
                value={config.ssh?.host || ""}
                onChange={(e) => handleSSHChange("host", e.target.value)}
                placeholder="192.168.1.100 or hostname"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>

            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                Port
              </label>
              <input
                type="number"
                value={config.ssh?.port || 22}
                onChange={(e) => handleSSHChange("port", parseInt(e.target.value) || 22)}
                placeholder="22"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>

            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                Username
              </label>
              <input
                type="text"
                value={config.ssh?.user || ""}
                onChange={(e) => handleSSHChange("user", e.target.value)}
                placeholder="user"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>

            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                SSH Key Path (optional)
              </label>
              <input
                type="text"
                value={config.ssh?.keyPath || ""}
                onChange={(e) => handleSSHChange("keyPath", e.target.value)}
                placeholder="~/.ssh/id_rsa"
                className={cn(
                  "w-full px-3 py-2 rounded-lg text-sm",
                  "bg-background-card border border-border",
                  "text-foreground placeholder-foreground-subtle",
                  "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                )}
              />
            </div>
          </div>

          {/* Quick connect to known hosts */}
          <div className="pt-2 border-t border-border">
            <p className="text-xs text-foreground-muted mb-2">Quick Connect</p>
            <button
              onClick={() => {
                // Use environment variable or prompt user to configure
                const gpuHost = process.env.NEXT_PUBLIC_REMOTE_GPU_HOST || "";
                const gpuUser = process.env.NEXT_PUBLIC_REMOTE_GPU_USER || "doc";
                if (!gpuHost) {
                  alert("Set REMOTE_GPU_HOST in .env to enable quick connect");
                  return;
                }
                onConfigChange({
                  ...config,
                  ssh: {
                    host: gpuHost,
                    port: 22,
                    user: gpuUser,
                    keyPath: "~/.ssh/id_rsa",
                  },
                });
                setSshStatus("unknown");
              }}
              className={cn(
                "flex items-center gap-2 px-3 py-2 text-xs rounded-lg",
                "border border-border hover:bg-foreground-muted/10",
                "transition-colors"
              )}
            >
              <Server className="w-4 h-4 text-purple-400" />
              RTX 4090 (Tailscale)
            </button>
          </div>

          {/* Test connection button */}
          <div className="flex items-center gap-3">
            <button
              onClick={testSSHConnection}
              disabled={isTesting || !config.ssh?.host || !config.ssh?.user}
              className={cn(
                "flex items-center gap-2 px-4 py-2 text-sm rounded-lg",
                "bg-foreground-bright text-background hover:bg-white",
                "transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              )}
            >
              {isTesting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Terminal className="w-4 h-4" />
              )}
              Test Connection
            </button>

            {/* Connection status */}
            {sshStatus === "connected" && (
              <div className="flex items-center gap-2 text-green-400 text-sm">
                <Wifi className="w-4 h-4" />
                Connected
              </div>
            )}
            {sshStatus === "failed" && (
              <div className="flex items-center gap-2 text-red-400 text-sm">
                <WifiOff className="w-4 h-4" />
                {sshError || "Connection failed"}
              </div>
            )}
          </div>

          {/* Remote GPU info */}
          {sshStatus === "connected" && remoteGpu && (
            <div className="flex items-center gap-3 p-3 rounded-lg bg-green-500/10 border border-green-500/20">
              <HardDrive className="w-5 h-5 text-green-400" />
              <div>
                <h4 className="text-sm font-medium text-foreground">
                  {remoteGpu.name}
                </h4>
                <p className="text-xs text-foreground-muted">
                  {remoteGpu.vram}GB VRAM
                  {remoteGpu.cudaVersion && ` • CUDA ${remoteGpu.cudaVersion}`}
                </p>
              </div>
              <Check className="w-5 h-5 text-green-400 ml-auto" />
            </div>
          )}
        </div>
      )}

      {/* Cloud provider configuration */}
      {config.type === "cloud" && (
        <div className="space-y-4 p-4 rounded-lg bg-background border border-border">
          <h3 className="text-sm font-medium text-foreground">Cloud Provider</h3>

          {/* Provider selector */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {CLOUD_PROVIDERS.map((provider) => {
              const isSelected = config.cloud?.provider === provider.id;
              return (
                <button
                  key={provider.id}
                  onClick={() => handleCloudChange("provider", provider.id)}
                  className={cn(
                    "flex flex-col items-center gap-2 p-3 rounded-lg border text-center transition-colors",
                    isSelected
                      ? "border-foreground-bright bg-foreground-bright/5"
                      : "border-border hover:border-foreground-muted bg-background-card"
                  )}
                >
                  <span className="text-lg">{provider.icon}</span>
                  <span className="text-xs text-foreground">{provider.name}</span>
                  {isSelected && (
                    <Check className="w-4 h-4 text-foreground-bright" />
                  )}
                </button>
              );
            })}
          </div>

          {/* API Key input */}
          {config.cloud?.provider && (
            <div>
              <label className="block text-xs text-foreground-muted mb-1">
                API Key
              </label>
              <div className="relative">
                <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-muted" />
                <input
                  type="password"
                  value={config.cloud?.apiKey || ""}
                  onChange={(e) => handleCloudChange("apiKey", e.target.value)}
                  placeholder={`${config.cloud.provider} API key`}
                  className={cn(
                    "w-full pl-10 pr-3 py-2 rounded-lg text-sm",
                    "bg-background-card border border-border",
                    "text-foreground placeholder-foreground-subtle",
                    "focus:outline-none focus:ring-2 focus:ring-foreground-bright/50"
                  )}
                />
              </div>
              <p className="text-xs text-foreground-subtle mt-1">
                Your API key is stored locally and never sent to our servers
              </p>
            </div>
          )}

          {/* Provider-specific info */}
          {config.cloud?.provider && (
            <div className="p-3 rounded-lg bg-foreground-muted/5 border border-border">
              {config.cloud.provider === "runpod" && (
                <div className="space-y-2">
                  <p className="text-sm text-foreground">RunPod - Serverless GPUs</p>
                  <p className="text-xs text-foreground-muted">
                    Starting at $0.20/hr for RTX 3090, $0.74/hr for A100
                  </p>
                </div>
              )}
              {config.cloud.provider === "aws" && (
                <div className="space-y-2">
                  <p className="text-sm text-foreground">AWS SageMaker</p>
                  <p className="text-xs text-foreground-muted">
                    Enterprise-grade ML infrastructure with auto-scaling
                  </p>
                </div>
              )}
              {config.cloud.provider === "gcp" && (
                <div className="space-y-2">
                  <p className="text-sm text-foreground">Google Cloud AI Platform</p>
                  <p className="text-xs text-foreground-muted">
                    TPU support and Vertex AI integration
                  </p>
                </div>
              )}
              {config.cloud.provider === "lambda-labs" && (
                <div className="space-y-2">
                  <p className="text-sm text-foreground">Lambda Labs</p>
                  <p className="text-xs text-foreground-muted">
                    ML-focused cloud with A100 and H100 GPUs
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Validation warning */}
      {config.type === "remote-ssh" && !config.ssh?.verified && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
          <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-foreground">
              Please test your SSH connection before continuing
            </p>
            <p className="text-xs text-foreground-muted mt-1">
              This ensures your remote machine is accessible
            </p>
          </div>
        </div>
      )}

      {config.type === "cloud" && !config.cloud?.apiKey && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
          <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-foreground">
              API key required for cloud provider
            </p>
            <p className="text-xs text-foreground-muted mt-1">
              Enter your API key to enable cloud GPU access
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default WizardStepHardware;
