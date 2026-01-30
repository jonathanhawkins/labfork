"use client";

/**
 * OnboardingWizard - Interactive 4-step wizard for new contributors
 *
 * Guides new users through:
 * 1. Introduction to distributed compute network
 * 2. Understanding the credit system
 * 3. Device detection and tier assignment
 * 4. Ready to start contributing
 *
 * Features:
 * - Mobile-first responsive design
 * - Framer Motion transitions
 * - Progress indicators
 * - Back navigation
 * - Skip option
 * - Touch-friendly (44px minimum targets)
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import {
  Network,
  Coins,
  Cpu,
  Rocket,
  ArrowRight,
  ArrowLeft,
  X,
  CheckCircle2,
  Loader2,
  Smartphone,
  Laptop,
  Server,
  Zap,
  TrendingUp,
  Award,
} from 'lucide-react';
import {
  detectGPU,
  runBenchmark,
  classifyTier,
  getTierInfo,
  formatTFLOPS,
  type GPUInfo,
  type BenchmarkResult,
  type DeviceTier,
} from '@/lib/compute/gpu-detect';
import { markOnboardingComplete } from '@/lib/compute/onboarding';

interface OnboardingWizardProps {
  onComplete: () => void;
  onSkip: () => void;
}

type Step = 1 | 2 | 3 | 4;

export function OnboardingWizard({ onComplete, onSkip }: OnboardingWizardProps) {
  const [currentStep, setCurrentStep] = useState<Step>(1);
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const [tier, setTier] = useState<DeviceTier | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-detect GPU when reaching step 3
  useEffect(() => {
    if (currentStep === 3 && !gpuInfo && !isDetecting) {
      handleDetectDevice();
    }
  }, [currentStep]);

  async function handleDetectDevice() {
    setIsDetecting(true);
    setError(null);

    try {
      // Detect GPU
      const info = await detectGPU();
      setGpuInfo(info);

      if (!info.available) {
        setError('WebGPU not available. Your device may still contribute via CPU.');
        setIsDetecting(false);
        return;
      }

      // Run benchmark
      const result = await runBenchmark();
      setBenchmarkResult(result);

      // Classify tier
      const deviceTier = classifyTier(result);
      setTier(deviceTier);

      setIsDetecting(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Detection failed');
      setIsDetecting(false);
    }
  }

  function handleNext() {
    if (currentStep < 4) {
      setCurrentStep((prev) => (prev + 1) as Step);
    }
  }

  function handleBack() {
    if (currentStep > 1) {
      setCurrentStep((prev) => (prev - 1) as Step);
    }
  }

  function handleComplete() {
    markOnboardingComplete();
    onComplete();
  }

  function handleSkip() {
    markOnboardingComplete();
    onSkip();
  }

  const canProceed = currentStep === 3 ? !isDetecting && (gpuInfo !== null || error !== null) : true;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/95 backdrop-blur-sm">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="relative w-full max-w-2xl bg-background-card border border-border rounded-xl shadow-2xl overflow-hidden"
      >
        {/* Header */}
        <div className="relative p-6 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center">
                <Network className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <h2 className="text-lg font-medium text-foreground-bright">
                  Welcome to LabFork Compute
                </h2>
                <p className="text-xs text-foreground-muted">
                  Step {currentStep} of 4
                </p>
              </div>
            </div>

            <button
              onClick={handleSkip}
              className="p-2 rounded-lg hover:bg-foreground-muted/10 transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
              aria-label="Skip onboarding"
            >
              <X className="w-5 h-5 text-foreground-muted" />
            </button>
          </div>

          {/* Progress dots */}
          <div className="flex items-center gap-2 mt-4">
            {[1, 2, 3, 4].map((step) => (
              <div
                key={step}
                className={cn(
                  "h-1.5 flex-1 rounded-full transition-colors duration-300",
                  step <= currentStep ? "bg-blue-400" : "bg-border"
                )}
              />
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="relative min-h-[400px]">
          <AnimatePresence mode="wait">
            {currentStep === 1 && (
              <StepWelcome key="step1" onNext={handleNext} />
            )}
            {currentStep === 2 && (
              <StepCredits key="step2" onNext={handleNext} onBack={handleBack} />
            )}
            {currentStep === 3 && (
              <StepDetection
                key="step3"
                gpuInfo={gpuInfo}
                benchmarkResult={benchmarkResult}
                tier={tier}
                isDetecting={isDetecting}
                error={error}
                onNext={handleNext}
                onBack={handleBack}
                onRetry={handleDetectDevice}
                canProceed={canProceed}
              />
            )}
            {currentStep === 4 && (
              <StepReady
                key="step4"
                tier={tier}
                onComplete={handleComplete}
                onBack={handleBack}
              />
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}

// Step 1: Welcome
function StepWelcome({ onNext }: { onNext: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="p-6 space-y-6"
    >
      <div className="text-center space-y-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 mb-2"
        >
          <Network className="w-10 h-10 text-blue-400" />
        </motion.div>

        <h3 className="text-xl font-medium text-foreground-bright">
          What is LabFork's Distributed Compute Network?
        </h3>

        <p className="text-sm text-foreground-muted max-w-md mx-auto leading-relaxed">
          Join thousands of devices worldwide that help power AI research. Your phone,
          tablet, or computer processes research tasks when idle, advancing science
          while you earn credits.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.3 }}
          className="p-4 rounded-lg bg-background/50 border border-border text-center"
        >
          <Smartphone className="w-6 h-6 text-green-400 mx-auto mb-2" />
          <p className="text-xs font-medium text-foreground mb-1">Mobile Friendly</p>
          <p className="text-xxs text-foreground-muted">
            Works on phones and tablets
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.3 }}
          className="p-4 rounded-lg bg-background/50 border border-border text-center"
        >
          <Laptop className="w-6 h-6 text-blue-400 mx-auto mb-2" />
          <p className="text-xs font-medium text-foreground mb-1">Desktop Power</p>
          <p className="text-xxs text-foreground-muted">
            Maximize your contribution
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.3 }}
          className="p-4 rounded-lg bg-background/50 border border-border text-center"
        >
          <Server className="w-6 h-6 text-purple-400 mx-auto mb-2" />
          <p className="text-xs font-medium text-foreground mb-1">Global Network</p>
          <p className="text-xxs text-foreground-muted">
            Distributed worldwide
          </p>
        </motion.div>
      </div>

      <div className="flex justify-end pt-4">
        <button
          onClick={onNext}
          className="flex items-center gap-2 px-6 py-3 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors font-medium min-h-[44px]"
        >
          Next
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
}

// Step 2: Credits
function StepCredits({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  const tiers = [
    { name: 'Crowd Tier', icon: Smartphone, credits: 10, tokens: '5K', color: 'text-green-400', bg: 'bg-green-500/10' },
    { name: 'Standard Tier', icon: Laptop, credits: 30, tokens: '15K', color: 'text-blue-400', bg: 'bg-blue-500/10' },
    { name: 'Power Tier', icon: Server, credits: 100, tokens: '50K', color: 'text-purple-400', bg: 'bg-purple-500/10' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="p-6 space-y-6"
    >
      <div className="text-center space-y-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-yellow-500/20 to-orange-500/20 mb-2"
        >
          <Coins className="w-10 h-10 text-yellow-400" />
        </motion.div>

        <h3 className="text-xl font-medium text-foreground-bright">
          How Credits Work
        </h3>

        <p className="text-sm text-foreground-muted max-w-md mx-auto leading-relaxed">
          Earn credits by contributing compute power. Use credits to run your own
          research, train models, or access premium features. The more you contribute,
          the more you earn.
        </p>
      </div>

      <div className="space-y-3 mt-8">
        {tiers.map((tier, index) => (
          <motion.div
            key={tier.name}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 + index * 0.1, duration: 0.3 }}
            className={cn(
              "p-4 rounded-lg border border-border",
              tier.bg
            )}
          >
            <div className="flex items-center gap-4">
              <div className={cn("w-12 h-12 rounded-lg flex items-center justify-center", tier.bg)}>
                <tier.icon className={cn("w-6 h-6", tier.color)} />
              </div>
              <div className="flex-1 min-w-0">
                <p className={cn("text-sm font-medium", tier.color)}>{tier.name}</p>
                <p className="text-xs text-foreground-muted">
                  {tier.tokens} tokens/hour
                </p>
              </div>
              <div className="text-right">
                <p className="text-lg font-medium text-foreground-bright">
                  {tier.credits}
                </p>
                <p className="text-xxs text-foreground-muted">credits/hour</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
        <div className="flex items-start gap-2">
          <Zap className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-blue-400">
            Credits never expire and can be used for training, inference, or transferred to other users
          </p>
        </div>
      </div>

      <div className="flex items-center justify-between pt-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-6 py-3 rounded-lg hover:bg-foreground-muted/10 transition-colors font-medium min-h-[44px]"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onNext}
          className="flex items-center gap-2 px-6 py-3 rounded-lg bg-foreground-bright text-background hover:bg-white transition-colors font-medium min-h-[44px]"
        >
          Next
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
}

// Step 3: Device Detection
interface StepDetectionProps {
  gpuInfo: GPUInfo | null;
  benchmarkResult: BenchmarkResult | null;
  tier: DeviceTier | null;
  isDetecting: boolean;
  error: string | null;
  onNext: () => void;
  onBack: () => void;
  onRetry: () => void;
  canProceed: boolean;
}

function StepDetection({
  gpuInfo,
  benchmarkResult,
  tier,
  isDetecting,
  error,
  onNext,
  onBack,
  onRetry,
  canProceed,
}: StepDetectionProps) {
  const tierInfo = tier ? getTierInfo(tier) : null;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="p-6 space-y-6"
    >
      <div className="text-center space-y-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-green-500/20 to-blue-500/20 mb-2"
        >
          <Cpu className="w-10 h-10 text-green-400" />
        </motion.div>

        <h3 className="text-xl font-medium text-foreground-bright">
          Device Detection
        </h3>

        <p className="text-sm text-foreground-muted max-w-md mx-auto leading-relaxed">
          Analyzing your device capabilities to determine your compute tier
        </p>
      </div>

      <div className="space-y-4 mt-8">
        {isDetecting && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-6 rounded-lg bg-background/50 border border-border text-center"
          >
            <Loader2 className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-3" />
            <p className="text-sm text-foreground-muted">
              Detecting GPU and running performance benchmark...
            </p>
          </motion.div>
        )}

        {!isDetecting && gpuInfo && !error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
            className="space-y-4"
          >
            {/* GPU Info */}
            <div className="p-4 rounded-lg bg-background/50 border border-border">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                <p className="text-sm font-medium text-foreground-bright">
                  GPU Detected
                </p>
              </div>
              {gpuInfo.adapterInfo && (
                <div className="space-y-2">
                  <p className="text-sm text-foreground">
                    {gpuInfo.adapterInfo.description || gpuInfo.adapterInfo.device ||
                      `${gpuInfo.adapterInfo.vendor}${gpuInfo.adapterInfo.architecture ? ` (${gpuInfo.adapterInfo.architecture})` : ''}`}
                  </p>
                  {(gpuInfo.adapterInfo.description || gpuInfo.adapterInfo.device) && (
                    <div className="flex items-center gap-2 text-xs text-foreground-muted">
                      <span>{gpuInfo.adapterInfo.vendor}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Benchmark Results */}
            {benchmarkResult && (
              <div className="p-4 rounded-lg bg-background/50 border border-border">
                <p className="text-sm font-medium text-foreground-bright mb-3">
                  Performance Benchmark
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs text-foreground-muted mb-1">Performance</p>
                    <p className="text-sm text-foreground font-medium">
                      {formatTFLOPS(benchmarkResult.tflops)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-foreground-muted mb-1">Duration</p>
                    <p className="text-sm text-foreground font-medium">
                      {benchmarkResult.duration.toFixed(0)}ms
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Tier Badge */}
            {tierInfo && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.3 }}
                className={cn(
                  "p-4 rounded-lg border",
                  tierInfo.bgColor,
                  `border-${tierInfo.color.replace('text-', '')}/20`
                )}
              >
                <div className="flex items-center gap-3">
                  <div className={cn("w-12 h-12 rounded-lg flex items-center justify-center", tierInfo.bgColor)}>
                    <Award className={cn("w-6 h-6", tierInfo.color)} />
                  </div>
                  <div className="flex-1">
                    <p className={cn("text-sm font-medium", tierInfo.color)}>
                      {tierInfo.name}
                    </p>
                    <p className="text-xs text-foreground-muted">
                      {tierInfo.creditsPerHour} credits/hour
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {!isDetecting && error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20"
          >
            <div className="flex items-start gap-2">
              <Zap className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <p className="text-sm text-yellow-400 mb-2">{error}</p>
                <button
                  onClick={onRetry}
                  className="text-xs text-yellow-400 hover:text-yellow-300 underline min-h-[44px]"
                >
                  Try again
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      <div className="flex items-center justify-between pt-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-6 py-3 rounded-lg hover:bg-foreground-muted/10 transition-colors font-medium min-h-[44px]"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onNext}
          disabled={!canProceed}
          className={cn(
            "flex items-center gap-2 px-6 py-3 rounded-lg font-medium min-h-[44px] transition-colors",
            canProceed
              ? "bg-foreground-bright text-background hover:bg-white"
              : "bg-foreground-muted/20 text-foreground-muted cursor-not-allowed"
          )}
        >
          Next
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </motion.div>
  );
}

// Step 4: Ready to Start
interface StepReadyProps {
  tier: DeviceTier | null;
  onComplete: () => void;
  onBack: () => void;
}

function StepReady({ tier, onComplete, onBack }: StepReadyProps) {
  const tierInfo = tier ? getTierInfo(tier) : null;

  const benefits = [
    { icon: Coins, text: 'Earn credits automatically while idle' },
    { icon: TrendingUp, text: 'Contribute to cutting-edge AI research' },
    { icon: Zap, text: 'Use credits for your own research projects' },
    { icon: Award, text: 'Join a global network of contributors' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="p-6 space-y-6"
    >
      <div className="text-center space-y-4">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 mb-2"
        >
          <Rocket className="w-10 h-10 text-purple-400" />
        </motion.div>

        <h3 className="text-xl font-medium text-foreground-bright">
          Ready to Start Contributing!
        </h3>

        {tierInfo && (
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-foreground-muted/10 border border-border">
            <Award className={cn("w-4 h-4", tierInfo.color)} />
            <span className={cn("text-xs font-medium", tierInfo.color)}>
              {tierInfo.name}
            </span>
          </div>
        )}
      </div>

      <div className="space-y-3 mt-8">
        {benefits.map((benefit, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 + index * 0.1, duration: 0.3 }}
            className="flex items-start gap-3 p-3 rounded-lg bg-background/50 border border-border"
          >
            <div className="w-8 h-8 rounded-lg bg-foreground-muted/10 flex items-center justify-center flex-shrink-0">
              <benefit.icon className="w-4 h-4 text-foreground-muted" />
            </div>
            <p className="text-sm text-foreground pt-1">{benefit.text}</p>
          </motion.div>
        ))}
      </div>

      <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
        <div className="flex items-start gap-2">
          <CheckCircle2 className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
          <div className="text-xs text-blue-400 space-y-1">
            <p className="font-medium">Privacy & Control</p>
            <p className="text-blue-400/80">
              You control when to contribute. Your device only processes encrypted
              tasks - never your personal data. Pause or stop anytime.
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between pt-4">
        <button
          onClick={onBack}
          className="flex items-center gap-2 px-6 py-3 rounded-lg hover:bg-foreground-muted/10 transition-colors font-medium min-h-[44px]"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        <button
          onClick={onComplete}
          className="flex items-center gap-2 px-6 py-3 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:from-blue-600 hover:to-purple-600 transition-all font-medium min-h-[44px]"
        >
          <Rocket className="w-4 h-4" />
          Start Contributing
        </button>
      </div>
    </motion.div>
  );
}
