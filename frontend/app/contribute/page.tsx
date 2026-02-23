"use client";

/**
 * Contribute Page - Let users contribute GPU to distributed compute network
 *
 * Features:
 * - WebGPU detection and capability analysis
 * - Performance benchmarking
 * - Device tier classification
 * - Contribution potential estimates
 * - Mobile-first responsive design
 * - PWA installation prompt
 * - Battery status monitoring
 * - Real distributed compute execution
 */

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  Cpu,
  Zap,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  Play,
  Loader2,
  HelpCircle,
  Award,
  BarChart3,
  Wifi,
  ChevronDown,
  Battery,
  BatteryCharging,
  Settings,
  Pause,
  Square,
  Activity,
  Trophy,
  Download,
} from "lucide-react";
import {
  detectGPU,
  runBenchmark,
  classifyTier,
  getTierInfo,
  formatTFLOPS,
  formatMemory,
  type GPUInfo,
  type BenchmarkResult,
  type DeviceTier,
} from "@/lib/compute/gpu-detect";
import { LiveDashboard } from "@/components/compute/LiveDashboard";
import { InstallPrompt } from "@/components/compute/InstallPrompt";
import { OnboardingWizard } from "@/components/compute/OnboardingWizard";
import Leaderboard from "@/components/compute/Leaderboard";
import {
  getBatteryStatus,
  monitorBattery,
  registerServiceWorker,
  type BatteryStatus,
} from "@/lib/compute/pwa-utils";
import {
  hasCompletedOnboarding,
  resetOnboarding,
} from "@/lib/compute/onboarding";
import {
  useDeviceAgent,
  formatUptime,
  formatCredits,
  getStatusColor,
  getStatusLabel,
} from "@/lib/compute/useDeviceAgent";

type BenchmarkState = 'idle' | 'running' | 'complete' | 'error';

export default function ContributePage() {
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [benchmarkState, setBenchmarkState] = useState<BenchmarkState>('idle');
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const [tier, setTier] = useState<DeviceTier | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [faqOpen, setFaqOpen] = useState<number | null>(null);
  const [batteryStatus, setBatteryStatus] = useState<BatteryStatus | null>(null);

  // Device Agent Hook - manages connection to compute network
  const {
    status: agentStatus,
    device,
    currentTask,
    taskProgress,
    stats,
    isReady,
    isBusy,
    isPaused,
    start: startAgent,
    pause: pauseAgent,
    resume: resumeAgent,
    stop: stopAgent,
    error: agentError,
    clearError,
    isModelLoaded,
    loadModel,
    modelLoadProgress,
  } = useDeviceAgent({
    autoStart: false, // Don't auto-start, let user click button
    autoStop: true,
  });

  // Derive isContributing from agent status
  const isContributing = agentStatus === 'online' || agentStatus === 'paused' || isBusy;

  // Check onboarding status on mount
  useEffect(() => {
    setShowOnboarding(!hasCompletedOnboarding());
  }, []);

  // Detect GPU on mount
  useEffect(() => {
    async function detect() {
      try {
        const info = await detectGPU();
        setGpuInfo(info);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to detect GPU');
      }
    }
    detect();
  }, []);

  // Register service worker and monitor battery
  useEffect(() => {
    // Register service worker for PWA features
    registerServiceWorker().catch((err) => {
      console.error('[PWA] Service worker registration failed:', err);
    });

    // Get initial battery status
    getBatteryStatus().then((status) => {
      if (status) {
        setBatteryStatus(status);
      }
    });

    // Monitor battery changes
    const cleanup = monitorBattery(
      (status) => {
        setBatteryStatus(status);
      },
      () => {
        // Low battery - pause contribution via agent
        if (agentStatus === 'online' || isBusy) {
          console.log('[Battery] Low battery detected, pausing contribution');
          pauseAgent();
        }
      },
      0.2 // 20% threshold
    );

    return cleanup;
  }, [agentStatus, isBusy, pauseAgent]);

  // Run benchmark
  async function handleRunBenchmark() {
    setBenchmarkState('running');
    setError(null);

    try {
      const result = await runBenchmark();
      setBenchmarkResult(result);
      const deviceTier = classifyTier(result);
      setTier(deviceTier);
      setBenchmarkState('complete');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Benchmark failed');
      setBenchmarkState('error');
    }
  }

  // Start contributing - connects to the compute network
  const handleStartContributing = useCallback(async () => {
    // Check battery level
    if (batteryStatus && !batteryStatus.charging && batteryStatus.level < 0.3) {
      setError('Battery level too low. Please connect to power or wait until battery is charged.');
      return;
    }

    setError(null);
    clearError();

    try {
      // Start the device agent - this will:
      // 1. Run GPU benchmark
      // 2. Register with orchestrator
      // 3. Start heartbeat loop
      // 4. Begin polling for tasks
      await startAgent();
      console.log('[Contribute] Agent started, connected to compute network');
    } catch (err) {
      console.error('[Contribute] Failed to start agent:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect to compute network');
    }
  }, [batteryStatus, startAgent, clearError]);

  // Pause contributing - stop accepting new tasks
  const handlePauseContributing = useCallback(() => {
    pauseAgent();
    console.log('[Contribute] Paused contribution');
  }, [pauseAgent]);

  // Resume contributing
  const handleResumeContributing = useCallback(() => {
    resumeAgent();
    console.log('[Contribute] Resumed contribution');
  }, [resumeAgent]);

  // Stop contributing completely
  const handleStopContributing = useCallback(() => {
    stopAgent();
    console.log('[Contribute] Stopped contribution');
  }, [stopAgent]);

  const tierInfo = tier ? getTierInfo(tier) : null;

  // Handler for replaying onboarding
  function handleReplayOnboarding() {
    resetOnboarding();
    setShowOnboarding(true);
  }

  return (
    <main className="min-h-screen bg-background">
      {/* Onboarding Wizard */}
      {showOnboarding && (
        <OnboardingWizard
          onComplete={() => setShowOnboarding(false)}
          onSkip={() => setShowOnboarding(false)}
        />
      )}

      {/* Hero Section */}
      <section className="relative pt-24 pb-16 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-foreground-muted/10 border border-border mb-6">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span className="text-xs text-foreground-muted">
                  Distributed Compute Network
                </span>
              </div>

              <h1 className="text-4xl md:text-5xl font-medium text-foreground-bright mb-4">
                Power AI Research
                <br />
                <span className="text-foreground-muted">with Your Device</span>
              </h1>

              <p className="text-base text-foreground-muted max-w-2xl mx-auto mb-8">
                Join the distributed compute network and earn credits by
                contributing your GPU to train AI models and run inference for
                researchers worldwide.
              </p>
            </motion.div>
          </div>

          {/* Settings button to replay onboarding */}
          <div className="flex justify-center mt-4">
            <button
              onClick={handleReplayOnboarding}
              className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-foreground-muted/10 active:bg-foreground-muted/20 transition-colors text-sm text-foreground-muted min-h-[44px]"
            >
              <Settings className="w-4 h-4" />
              <span>Replay Tutorial</span>
            </button>
          </div>
        </div>
      </section>

      {/* Main Content */}
      <section className="px-4 pb-24">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Live Network Dashboard */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <LiveDashboard showPersonalStats={tier !== null} />
          </motion.div>

          {/* GPU Detection Card */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
            className="bg-background-card border border-border rounded-xl p-6"
          >
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-lg bg-foreground-muted/10 flex items-center justify-center flex-shrink-0">
                <Cpu className="w-6 h-6 text-foreground-muted" />
              </div>

              <div className="flex-1 min-w-0">
                <h2 className="text-lg font-medium text-foreground-bright mb-1">
                  Device Detection
                </h2>
                <p className="text-sm text-foreground-muted mb-4">
                  {gpuInfo?.available
                    ? 'WebGPU detected - your device is compatible'
                    : 'Checking GPU capabilities...'}
                </p>

                {gpuInfo?.available && gpuInfo.adapterInfo && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span className="text-foreground">
                        {gpuInfo.adapterInfo.description || gpuInfo.adapterInfo.device ||
                          `${gpuInfo.adapterInfo.vendor}${gpuInfo.adapterInfo.architecture ? ` (${gpuInfo.adapterInfo.architecture})` : ''}`}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
                      <div className="bg-background/50 rounded-lg p-3 border border-border">
                        <p className="text-xs text-foreground-muted mb-1">Vendor</p>
                        <p className="text-sm text-foreground font-medium">
                          {gpuInfo.adapterInfo.vendor}
                        </p>
                      </div>

                      {gpuInfo.estimatedMemoryMB && (
                        <div className="bg-background/50 rounded-lg p-3 border border-border">
                          <p className="text-xs text-foreground-muted mb-1">
                            Estimated Memory
                          </p>
                          <p className="text-sm text-foreground font-medium">
                            {formatMemory(gpuInfo.estimatedMemoryMB)}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {gpuInfo && !gpuInfo.available && (
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                    <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-yellow-400">
                      <p className="font-medium mb-1">WebGPU not available</p>
                      <p className="text-xs text-yellow-400/80">
                        Your browser doesn't support WebGPU. Try Chrome or Edge (version 113+)
                        with GPU acceleration enabled.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>

          {/* Benchmark Card */}
          {gpuInfo?.available && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.3 }}
              className="bg-background-card border border-border rounded-xl p-6"
            >
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-lg bg-foreground-muted/10 flex items-center justify-center flex-shrink-0">
                  <BarChart3 className="w-6 h-6 text-foreground-muted" />
                </div>

                <div className="flex-1 min-w-0">
                  <h2 className="text-lg font-medium text-foreground-bright mb-1">
                    Performance Benchmark
                  </h2>
                  <p className="text-sm text-foreground-muted mb-4">
                    Run a quick test to measure your device's compute power
                  </p>

                  {benchmarkState === 'idle' && (
                    <button
                      onClick={handleRunBenchmark}
                      className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-foreground-bright text-background hover:bg-white active:bg-foreground-muted transition-colors min-h-[44px]"
                    >
                      <Play className="w-4 h-4" />
                      Run Benchmark (~5 seconds)
                    </button>
                  )}

                  {benchmarkState === 'running' && (
                    <div className="flex items-center gap-3 text-sm text-foreground-muted">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Running compute test...</span>
                    </div>
                  )}

                  {benchmarkState === 'complete' && benchmarkResult && (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 text-sm text-green-400">
                        <CheckCircle2 className="w-4 h-4" />
                        Benchmark complete
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <div className="bg-background/50 rounded-lg p-3 border border-border">
                          <p className="text-xs text-foreground-muted mb-1">Performance</p>
                          <p className="text-sm text-foreground font-medium">
                            {formatTFLOPS(benchmarkResult.tflops)}
                          </p>
                        </div>

                        <div className="bg-background/50 rounded-lg p-3 border border-border">
                          <p className="text-xs text-foreground-muted mb-1">Duration</p>
                          <p className="text-sm text-foreground font-medium">
                            {benchmarkResult.duration.toFixed(0)}ms
                          </p>
                        </div>

                        <div className="bg-background/50 rounded-lg p-3 border border-border">
                          <p className="text-xs text-foreground-muted mb-1">Operations</p>
                          <p className="text-sm text-foreground font-medium">
                            {(benchmarkResult.operations / 1e6).toFixed(1)}M
                          </p>
                        </div>
                      </div>

                      <button
                        onClick={handleRunBenchmark}
                        className="text-sm text-foreground-muted hover:text-foreground active:text-foreground-bright transition-colors min-h-[44px] px-3 py-2 rounded-lg hover:bg-foreground-muted/10 active:bg-foreground-muted/20"
                      >
                        Run again
                      </button>
                    </div>
                  )}

                  {benchmarkState === 'error' && error && (
                    <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                      <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                      <div className="text-sm text-red-400">
                        <p className="font-medium mb-1">Benchmark failed</p>
                        <p className="text-xs text-red-400/80">{error}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* Tier Badge Card */}
          {tier && tierInfo && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.4 }}
              className={cn(
                "bg-background-card border rounded-xl p-6",
                tierInfo.bgColor,
                `border-${tierInfo.color.replace('text-', '')}/20`
              )}
            >
              <div className="flex items-start gap-4">
                <div className={cn(
                  "w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0",
                  tierInfo.bgColor
                )}>
                  <Award className={cn("w-6 h-6", tierInfo.color)} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className={cn("text-lg font-medium", tierInfo.color)}>
                      {tierInfo.name}
                    </h2>
                  </div>
                  <p className="text-sm text-foreground-muted mb-4">
                    {tierInfo.description}
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="bg-background/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-foreground-muted mb-1">
                        Estimated Contribution
                      </p>
                      <p className="text-sm text-foreground font-medium">
                        {tierInfo.tokensPerHour.toLocaleString()} tokens/hour
                      </p>
                    </div>

                    <div className="bg-background/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-foreground-muted mb-1">
                        Estimated Earnings
                      </p>
                      <p className="text-sm text-foreground font-medium">
                        {tierInfo.creditsPerHour} credits/hour
                      </p>
                    </div>
                  </div>

                  {/* Battery Status */}
                  {batteryStatus && (
                    <div className="mt-3 p-3 rounded-lg bg-foreground-muted/10 border border-border">
                      <div className="flex items-center gap-2">
                        {batteryStatus.charging ? (
                          <BatteryCharging className="w-4 h-4 text-green-400" />
                        ) : (
                          <Battery className={cn(
                            "w-4 h-4",
                            batteryStatus.level > 0.5 ? "text-green-400" :
                            batteryStatus.level > 0.2 ? "text-yellow-400" :
                            "text-red-400"
                          )} />
                        )}
                        <p className="text-xs text-foreground-muted">
                          Battery: {Math.round(batteryStatus.level * 100)}%
                          {batteryStatus.charging && " (Charging)"}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* Start Contributing CTA - Always visible */}
          {gpuInfo?.available && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.5 }}
              className="bg-background-card border border-border rounded-xl p-6"
            >
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <div className={cn(
                  "w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0",
                  isContributing ? "bg-green-500/10" : "bg-foreground-muted/10"
                )}>
                  {agentStatus === 'connecting' ? (
                    <Loader2 className="w-6 h-6 text-blue-400 animate-spin" />
                  ) : isContributing ? (
                    <Activity className="w-6 h-6 text-green-400" />
                  ) : (
                    <Wifi className="w-6 h-6 text-foreground-muted" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h2 className="text-lg font-medium text-foreground-bright">
                      {isContributing ? 'Contributing to Network' : 'Ready to Contribute'}
                    </h2>
                    {isContributing && (
                      <>
                        <span className={cn(
                          "px-2 py-0.5 rounded-full text-xs font-medium",
                          agentStatus === 'online' ? "bg-green-500/20 text-green-400" :
                          agentStatus === 'paused' ? "bg-yellow-500/20 text-yellow-400" :
                          "bg-blue-500/20 text-blue-400"
                        )}>
                          {getStatusLabel(agentStatus)}
                        </span>
                        <span className={cn(
                          "px-2 py-0.5 rounded-full text-xs font-medium",
                          isModelLoaded
                            ? "bg-green-500/20 text-green-400"
                            : "bg-amber-500/20 text-amber-400"
                        )}>
                          {isModelLoaded ? "Real Compute" : "Simulated"}
                        </span>
                      </>
                    )}
                  </div>
                  <p className="text-sm text-foreground-muted">
                    {agentStatus === 'connecting' ? 'Connecting to orchestrator...' :
                     isContributing ? `Device ID: ${device?.id.slice(0, 12)}...` :
                     !tier ? 'Run the benchmark above to measure your device\'s compute power' :
                     'Connect to the orchestrator and start earning credits'}
                  </p>
                </div>

                <div className="flex gap-2 w-full sm:w-auto">
                  {!isContributing ? (
                    <button
                      onClick={handleStartContributing}
                      disabled={agentStatus === 'connecting' || !tier}
                      className={cn(
                        "w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium min-h-[44px] transition-colors",
                        tier
                          ? "bg-foreground-bright text-background hover:bg-white active:bg-foreground-muted"
                          : "bg-foreground-muted/20 text-foreground-muted cursor-not-allowed"
                      )}
                    >
                      {agentStatus === 'connecting' ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Connecting...
                        </>
                      ) : (
                        <>
                          <TrendingUp className="w-4 h-4" />
                          Start Contributing
                        </>
                      )}
                    </button>
                  ) : (
                    <>
                      {isPaused ? (
                        <button
                          onClick={handleResumeContributing}
                          className="flex-1 sm:flex-initial flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-green-500/10 text-green-400 border border-green-500/20 hover:bg-green-500/20 active:bg-green-500/30 transition-colors font-medium min-h-[44px]"
                        >
                          <Play className="w-4 h-4" />
                          Resume
                        </button>
                      ) : (
                        <button
                          onClick={handlePauseContributing}
                          className="flex-1 sm:flex-initial flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-yellow-500/10 text-yellow-400 border border-yellow-500/20 hover:bg-yellow-500/20 active:bg-yellow-500/30 transition-colors font-medium min-h-[44px]"
                        >
                          <Pause className="w-4 h-4" />
                          Pause
                        </button>
                      )}
                      <button
                        onClick={handleStopContributing}
                        className="flex-1 sm:flex-initial flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 active:bg-red-500/30 transition-colors font-medium min-h-[44px]"
                      >
                        <Square className="w-4 h-4" />
                        Stop
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Live Stats when contributing */}
              {isContributing && (
                <div className="mt-4 space-y-3">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-background/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-foreground-muted mb-1">Tasks Completed</p>
                      <p className="text-lg font-medium text-foreground">{stats.tasksCompleted}</p>
                    </div>
                    <div className="bg-background/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-foreground-muted mb-1">Credits Earned</p>
                      <p className="text-lg font-medium text-foreground">{formatCredits(stats.creditsEarned)}</p>
                    </div>
                    <div className="bg-background/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-foreground-muted mb-1">Uptime</p>
                      <p className="text-lg font-medium text-foreground">{formatUptime(stats.uptimeSeconds)}</p>
                    </div>
                    <div className="bg-background/50 rounded-lg p-3 border border-border">
                      <p className="text-xs text-foreground-muted mb-1">Compute Time</p>
                      <p className="text-lg font-medium text-foreground">{formatUptime(Math.round(stats.totalComputeTime))}</p>
                    </div>
                  </div>

                  {/* Credit rate and model loading */}
                  <div className={cn(
                    "p-3 rounded-lg border",
                    isModelLoaded ? "bg-green-500/5 border-green-500/20" : "bg-amber-500/5 border-amber-500/20"
                  )}>
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 text-sm">
                        <Zap className={cn("w-4 h-4", isModelLoaded ? "text-green-400" : "text-amber-400")} />
                        <span className={isModelLoaded ? "text-green-400" : "text-amber-400"}>
                          {isModelLoaded
                            ? "1.0 credits/task (real compute)"
                            : "0.1 credits/task (simulated)"
                          }
                        </span>
                      </div>
                      {!isModelLoaded && (
                        <button
                          onClick={loadModel}
                          disabled={modelLoadProgress > 0 && modelLoadProgress < 100}
                          className="flex items-center gap-2 px-3 py-2.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20 active:bg-amber-500/30 transition-colors text-xs font-medium min-h-[44px]"
                        >
                          {modelLoadProgress > 0 && modelLoadProgress < 100 ? (
                            <>
                              <Loader2 className="w-3 h-3 animate-spin" />
                              {modelLoadProgress}%
                            </>
                          ) : (
                            <>
                              <Download className="w-3 h-3" />
                              Load AI Model
                            </>
                          )}
                        </button>
                      )}
                    </div>
                    {!isModelLoaded && (
                      <p className="text-xs text-foreground-muted mt-2">
                        Load a WebLLM model to earn 10x more credits with real AI inference.
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* Current Task Progress */}
              {currentTask && (
                <div className="mt-4 p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-blue-400 font-medium">
                      Processing Task: {currentTask.type}
                    </span>
                    <span className="text-xs text-blue-400/80">
                      {Math.round(taskProgress)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-blue-900/30 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-400 rounded-full transition-all duration-300"
                      style={{ width: `${taskProgress}%` }}
                    />
                  </div>
                  <p className="text-xs text-blue-400/60 mt-2">
                    Task ID: {currentTask.id} | Reward: {currentTask.reward} credits
                  </p>
                </div>
              )}

              {/* Status message */}
              <div className={cn(
                "mt-4 p-3 rounded-lg border",
                !tier ? "bg-blue-500/5 border-blue-500/20" : "bg-foreground-muted/5 border-border"
              )}>
                <div className="flex items-start gap-2 text-sm">
                  {!tier ? (
                    <>
                      <Play className="w-4 h-4 mt-0.5 flex-shrink-0 text-blue-400" />
                      <p className="text-xs text-blue-400">
                        <span className="font-medium">Step 1:</span> Run the Performance Benchmark above to measure your device's compute power. This takes about 5 seconds and determines your contribution tier.
                      </p>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-foreground-muted" />
                      <p className="text-xs text-foreground-muted">
                        {isContributing
                          ? isPaused
                            ? "Contribution paused. Click Resume to continue processing tasks."
                            : isBusy
                              ? "Processing a compute task. Your device is actively contributing to AI research."
                              : "Connected and waiting for tasks. The orchestrator will assign work when available."
                          : "Your device will only contribute when idle and connected to power. You can pause or stop at any time."
                        }
                      </p>
                    </>
                  )}
                </div>
              </div>

              {/* Error display */}
              {(error || agentError) && (
                <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                  <div className="flex items-start gap-2 text-sm text-red-400">
                    <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                    <p className="text-xs">{error || agentError?.message}</p>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* PWA Install Prompt */}
          <InstallPrompt variant="card" />

          {/* Top Contributors Leaderboard */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.55 }}
            className="bg-background-card border border-border rounded-xl p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-lg bg-yellow-500/10 flex items-center justify-center">
                <Trophy className="w-6 h-6 text-yellow-400" />
              </div>
              <div>
                <h2 className="text-lg font-medium text-foreground-bright">
                  Top Contributors
                </h2>
                <p className="text-sm text-foreground-muted">
                  Leaderboard of top compute contributors
                </p>
              </div>
            </div>

            <Leaderboard limit={10} />
          </motion.div>

          {/* FAQ Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.6 }}
            className="bg-background-card border border-border rounded-xl p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-lg bg-foreground-muted/10 flex items-center justify-center">
                <HelpCircle className="w-6 h-6 text-foreground-muted" />
              </div>
              <div>
                <h2 className="text-lg font-medium text-foreground-bright">
                  Frequently Asked Questions
                </h2>
                <p className="text-sm text-foreground-muted">
                  Learn how distributed compute works
                </p>
              </div>
            </div>

            <div className="space-y-3">
              {faqs.map((faq, index) => (
                <div
                  key={index}
                  className="border border-border rounded-lg overflow-hidden"
                >
                  <button
                    onClick={() => setFaqOpen(faqOpen === index ? null : index)}
                    className="w-full flex items-center justify-between gap-4 p-4 text-left hover:bg-foreground-muted/5 active:bg-foreground-muted/10 transition-colors min-h-[44px]"
                  >
                    <span className="text-sm font-medium text-foreground">
                      {faq.question}
                    </span>
                    <ChevronDown
                      className={cn(
                        "w-4 h-4 text-foreground-muted transition-transform flex-shrink-0",
                        faqOpen === index && "rotate-180"
                      )}
                    />
                  </button>

                  <AnimatePresence>
                    {faqOpen === index && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2, ease: "easeOut" }}
                      >
                        <div className="px-4 pb-4 text-sm text-foreground-muted border-t border-border pt-4">
                          {faq.answer}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Powered By / Trust Badges */}
      <section className="py-8 border-t border-border">
        <div className="max-w-4xl mx-auto px-4">
          <p className="text-xs text-foreground-muted text-center mb-4 uppercase tracking-wider">
            Powered by trusted open-source technology
          </p>
          <div className="flex flex-wrap items-center justify-center gap-6">
            {/* Next.js */}
            <a
              href="https://nextjs.org"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-foreground-muted hover:text-foreground transition-colors"
              aria-label="Next.js"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M11.572 0c-.176 0-.31.001-.358.007a19.76 19.76 0 0 1-.364.033C7.443.346 4.25 2.185 2.228 5.012a11.875 11.875 0 0 0-2.119 5.243c-.096.659-.108.854-.108 1.747s.012 1.089.108 1.748c.652 4.506 3.86 8.292 8.209 9.695.779.251 1.6.422 2.534.525.363.04 1.935.04 2.299 0 1.611-.178 2.977-.577 4.323-1.264.207-.106.247-.134.219-.158-.02-.013-.9-1.193-1.955-2.62l-1.919-2.592-2.404-3.558a338.739 338.739 0 0 0-2.422-3.556c-.009-.002-.018 1.579-.023 3.51-.007 3.38-.01 3.515-.052 3.595a.426.426 0 0 1-.206.214c-.075.037-.14.044-.495.044H7.81l-.108-.068a.438.438 0 0 1-.157-.171l-.05-.106.006-4.703.007-4.705.072-.092a.645.645 0 0 1 .174-.143c.096-.047.134-.051.54-.051.478 0 .558.018.682.154.035.038 1.337 1.999 2.895 4.361a10760.433 10760.433 0 0 0 4.735 7.17l1.9 2.879.096-.063a12.317 12.317 0 0 0 2.466-2.163 11.944 11.944 0 0 0 2.824-6.134c.096-.66.108-.854.108-1.748 0-.893-.012-1.088-.108-1.747-.652-4.506-3.859-8.292-8.208-9.695a12.597 12.597 0 0 0-2.499-.523A33.119 33.119 0 0 0 11.572 0zm4.069 7.217c.347 0 .408.005.486.047a.473.473 0 0 1 .237.277c.018.06.023 1.365.018 4.304l-.006 4.218-.744-1.14-.746-1.14v-3.066c0-1.982.01-3.097.023-3.15a.478.478 0 0 1 .233-.296c.096-.05.13-.054.5-.054z" />
              </svg>
              <span className="text-xs">Next.js</span>
            </a>

            {/* React */}
            <a
              href="https://react.dev"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-foreground-muted hover:text-[#61dafb] transition-colors"
              aria-label="React"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M14.23 12.004a2.236 2.236 0 0 1-2.235 2.236 2.236 2.236 0 0 1-2.236-2.236 2.236 2.236 0 0 1 2.235-2.236 2.236 2.236 0 0 1 2.236 2.236zm2.648-10.69c-1.346 0-3.107.96-4.888 2.622-1.78-1.653-3.542-2.602-4.887-2.602-.41 0-.783.093-1.106.278-1.375.793-1.683 3.264-.973 6.365C1.98 8.917 0 10.42 0 12.004c0 1.59 1.99 3.097 5.043 4.03-.704 3.113-.39 5.588.988 6.38.32.187.69.275 1.102.275 1.345 0 3.107-.96 4.888-2.624 1.78 1.654 3.542 2.603 4.887 2.603.41 0 .783-.09 1.106-.275 1.374-.792 1.683-3.263.973-6.365C22.02 15.096 24 13.59 24 12.004c0-1.59-1.99-3.097-5.043-4.032.704-3.11.39-5.587-.988-6.38a2.167 2.167 0 0 0-1.092-.278z" />
              </svg>
              <span className="text-xs">React</span>
            </a>

            {/* WebGPU */}
            <div className="flex items-center gap-2 text-foreground-muted">
              <Cpu className="w-4 h-4" />
              <span className="text-xs">WebGPU</span>
            </div>

            {/* TypeScript */}
            <a
              href="https://www.typescriptlang.org"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-foreground-muted hover:text-[#3178c6] transition-colors"
              aria-label="TypeScript"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                <path d="M1.125 0C.502 0 0 .502 0 1.125v21.75C0 23.498.502 24 1.125 24h21.75c.623 0 1.125-.502 1.125-1.125V1.125C24 .502 23.498 0 22.875 0zm17.363 9.75c.612 0 1.154.037 1.627.111a6.38 6.38 0 0 1 1.306.34v2.458a3.95 3.95 0 0 0-.643-.361 5.093 5.093 0 0 0-.717-.26 5.453 5.453 0 0 0-1.426-.2c-.3 0-.573.028-.819.086a2.1 2.1 0 0 0-.623.242c-.17.104-.3.229-.393.374a.888.888 0 0 0-.14.49c0 .196.053.373.156.529.104.156.252.304.443.444s.423.276.696.41c.273.135.582.274.926.416.47.197.892.407 1.266.628.374.222.695.473.963.753.268.279.472.598.614.957.142.359.214.776.214 1.253 0 .657-.125 1.21-.373 1.656a3.033 3.033 0 0 1-1.012 1.085 4.38 4.38 0 0 1-1.487.596c-.566.12-1.163.18-1.79.18a9.916 9.916 0 0 1-1.84-.164 5.544 5.544 0 0 1-1.512-.493v-2.63a5.033 5.033 0 0 0 3.237 1.2c.333 0 .624-.03.872-.09.249-.06.456-.144.623-.25.166-.108.29-.234.373-.38a1.023 1.023 0 0 0-.074-1.089 2.12 2.12 0 0 0-.537-.5 5.597 5.597 0 0 0-.807-.444 27.72 27.72 0 0 0-1.007-.436c-.918-.383-1.602-.852-2.053-1.405-.45-.553-.676-1.222-.676-2.005 0-.614.123-1.141.369-1.582.246-.441.58-.804 1.004-1.089a4.494 4.494 0 0 1 1.47-.629 7.536 7.536 0 0 1 1.77-.201zm-15.113.188h9.563v2.166H9.506v9.646H6.789v-9.646H3.375z" />
              </svg>
              <span className="text-xs">TypeScript</span>
            </a>

            {/* Open Source */}
            <a
              href="https://github.com/jonathanhawkins/labfork"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-foreground-muted hover:text-foreground transition-colors"
              aria-label="Open Source on GitHub"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
              </svg>
              <span className="text-xs">Open Source</span>
            </a>
          </div>
          <p className="text-xs text-foreground-muted/60 text-center mt-4">
            All compute is processed securely using standard web technologies
          </p>
        </div>
      </section>
    </main>
  );
}

const faqs = [
  {
    question: "How does distributed compute work?",
    answer:
      "When you contribute your GPU, it joins a network of devices that collectively train AI models and run inference. The orchestrator distributes work based on your device tier and availability. You earn credits for completed compute tasks that can be used for your own research.",
  },
  {
    question: "What are device tiers?",
    answer:
      "Devices are classified into Power (5+ TFLOPS), Standard (1-5 TFLOPS), and Crowd (<1 TFLOPS) tiers based on their compute performance. Higher tiers can handle more demanding tasks and earn more credits per hour.",
  },
  {
    question: "Will this slow down my device?",
    answer:
      "No. Compute tasks only run when your device is idle and optionally when connected to power. You can configure these settings and pause contribution at any time. Background compute uses low-priority scheduling to ensure your active work is never impacted.",
  },
  {
    question: "Is my data safe?",
    answer:
      "Yes. Your device only processes encrypted compute tasks - never your personal data. All communication with the orchestrator is encrypted, and you can audit exactly what tasks your device is processing.",
  },
  {
    question: "How are credits calculated?",
    answer:
      "Credits are awarded based on completed compute work, measured in token equivalents. The rate depends on your device tier and the complexity of tasks. Credits can be used to run your own research tasks on the network.",
  },
  {
    question: "Can I use this on mobile?",
    answer:
      "Yes! Mobile devices with WebGPU support can contribute. However, contribution is limited to when the device is charging and on WiFi to preserve battery life and avoid data charges.",
  },
];
