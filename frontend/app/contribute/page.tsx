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
                      <span className={cn(
                        "px-2 py-0.5 rounded-full text-xs font-medium",
                        agentStatus === 'online' ? "bg-green-500/20 text-green-400" :
                        agentStatus === 'paused' ? "bg-yellow-500/20 text-yellow-400" :
                        "bg-blue-500/20 text-blue-400"
                      )}>
                        {getStatusLabel(agentStatus)}
                      </span>
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
                <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
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
