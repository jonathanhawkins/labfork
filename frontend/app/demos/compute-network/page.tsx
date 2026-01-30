"use client";

/**
 * Distributed Compute Network Demo
 *
 * Interactive showcase of the full compute network workflow:
 * - GPU detection and benchmarking
 * - Device registration and tier classification
 * - Speculative decoding visualization
 * - Credit system and rewards
 * - Live network analytics
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu,
  Zap,
  CheckCircle2,
  XCircle,
  Loader2,
  Award,
  TrendingUp,
  ArrowRight,
  Play,
  RotateCcw,
  Sparkles,
  Activity,
  Users,
  Coins,
  Target,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { LiveDashboard } from "@/components/compute/LiveDashboard";

import type {
  GPUInfo,
  BenchmarkResult,
  DeviceTier,
  TierInfo,
} from "@/lib/compute/gpu-detect";
import {
  formatTFLOPS,
  formatMemory,
  getTierInfo,
  classifyTier,
} from "@/lib/compute/gpu-detect";
import type { DraftToken } from "@/lib/compute/speculative-decoding";
import { calculateSpeedupFactor } from "@/lib/compute/speculative-decoding";

// Mock data for demonstrations
const MOCK_GPU_INFO: GPUInfo = {
  available: true,
  adapterInfo: {
    vendor: "Apple",
    architecture: "Apple Silicon",
    device: "Apple M2 Max",
    description: "High-performance GPU",
  },
  limits: {
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeInvocationsPerWorkgroup: 1024,
  },
  features: ["shader-f16", "timestamp-query"],
  estimatedMemoryMB: 32768,
};

const MOCK_BENCHMARK: BenchmarkResult = {
  tflops: 6.4,
  duration: 45.2,
  operations: 33554432,
  timestamp: Date.now(),
};

const EXAMPLE_DRAFT_TOKENS: DraftToken[] = [
  { tokenId: 128, text: "The", logProb: -0.8, confidence: 0.92, position: 0 },
  { tokenId: 256, text: " weather", logProb: -1.2, confidence: 0.88, position: 1 },
  { tokenId: 512, text: " today", logProb: -0.9, confidence: 0.91, position: 2 },
  { tokenId: 1024, text: " is", logProb: -0.7, confidence: 0.94, position: 3 },
  { tokenId: 2048, text: " beautiful", logProb: -1.5, confidence: 0.85, position: 4 },
  { tokenId: 4096, text: " and", logProb: -2.1, confidence: 0.72, position: 5 },
  { tokenId: 8192, text: " sunny", logProb: -1.8, confidence: 0.78, position: 6 },
  { tokenId: 16384, text: ".", logProb: -0.6, confidence: 0.96, position: 7 },
];

export default function ComputeNetworkDemo() {
  const [activeSection, setActiveSection] = useState<string>("gpu");

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4 space-y-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="text-center space-y-3"
        >
          <div className="flex items-center justify-center gap-2 mb-2">
            <Cpu className="w-8 h-8 text-purple-400" />
            <h1 className="text-4xl font-bold text-foreground">
              Distributed Compute Network
            </h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Interactive demo showcasing GPU detection, device registration, speculative decoding,
            and the credit reward system that powers LabFork's distributed AI network.
          </p>
        </motion.div>

        {/* Section Navigation */}
        <Tabs value={activeSection} onValueChange={setActiveSection} className="w-full">
          <TabsList className="grid w-full grid-cols-5 gap-2">
            <TabsTrigger value="gpu" className="gap-2">
              <Cpu className="w-4 h-4" />
              GPU Detection
            </TabsTrigger>
            <TabsTrigger value="registration" className="gap-2">
              <Users className="w-4 h-4" />
              Registration
            </TabsTrigger>
            <TabsTrigger value="speculative" className="gap-2">
              <Zap className="w-4 h-4" />
              Speculative Decoding
            </TabsTrigger>
            <TabsTrigger value="credits" className="gap-2">
              <Coins className="w-4 h-4" />
              Credit System
            </TabsTrigger>
            <TabsTrigger value="network" className="gap-2">
              <Activity className="w-4 h-4" />
              Network Stats
            </TabsTrigger>
          </TabsList>

          {/* Section 1: GPU Detection */}
          <TabsContent value="gpu" className="mt-6">
            <GPUDetectionDemo />
          </TabsContent>

          {/* Section 2: Device Registration */}
          <TabsContent value="registration" className="mt-6">
            <DeviceRegistrationDemo />
          </TabsContent>

          {/* Section 3: Speculative Decoding */}
          <TabsContent value="speculative" className="mt-6">
            <SpeculativeDecodingDemo />
          </TabsContent>

          {/* Section 4: Credit System */}
          <TabsContent value="credits" className="mt-6">
            <CreditSystemDemo />
          </TabsContent>

          {/* Section 5: Network Stats */}
          <TabsContent value="network" className="mt-6">
            <NetworkStatsDemo />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

/**
 * GPU Detection Demo Section
 */
function GPUDetectionDemo() {
  const [isRunning, setIsRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [progress, setProgress] = useState(0);
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [tier, setTier] = useState<DeviceTier | null>(null);

  const runBenchmark = async () => {
    setIsRunning(true);
    setProgress(0);
    setGpuInfo(null);
    setBenchmark(null);
    setTier(null);

    // Simulate detection phase
    await new Promise((resolve) => setTimeout(resolve, 500));
    setProgress(30);
    setGpuInfo(MOCK_GPU_INFO);

    // Simulate benchmark phase
    await new Promise((resolve) => setTimeout(resolve, 800));
    setProgress(70);

    // Complete benchmark
    await new Promise((resolve) => setTimeout(resolve, 600));
    setBenchmark(MOCK_BENCHMARK);
    const detectedTier = classifyTier(MOCK_BENCHMARK);
    setTier(detectedTier);
    setProgress(100);
    setIsRunning(false);
    setHasRun(true);
  };

  const tierInfo = tier ? getTierInfo(tier) : null;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            GPU Detection & Benchmarking
          </CardTitle>
          <CardDescription>
            Detect WebGPU capabilities and run performance benchmarks to classify your device tier
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Run Button */}
          <div className="flex items-center gap-3">
            <Button
              onClick={runBenchmark}
              disabled={isRunning}
              size="lg"
              className="gap-2"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running Benchmark...
                </>
              ) : hasRun ? (
                <>
                  <RotateCcw className="w-4 h-4" />
                  Run Again
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run GPU Benchmark
                </>
              )}
            </Button>
            {isRunning && (
              <div className="flex-1 max-w-md">
                <Progress value={progress} className="h-2" />
              </div>
            )}
          </div>

          {/* GPU Info */}
          <AnimatePresence>
            {gpuInfo && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="bg-muted/50">
                  <CardHeader>
                    <CardTitle className="text-lg">Detected GPU</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label className="text-xs text-muted-foreground">Vendor</Label>
                        <p className="text-sm font-medium">{gpuInfo.adapterInfo?.vendor}</p>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Architecture</Label>
                        <p className="text-sm font-medium">{gpuInfo.adapterInfo?.architecture}</p>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Device</Label>
                        <p className="text-sm font-medium">{gpuInfo.adapterInfo?.device}</p>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Memory</Label>
                        <p className="text-sm font-medium">
                          {gpuInfo.estimatedMemoryMB ? formatMemory(gpuInfo.estimatedMemoryMB) : "N/A"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Benchmark Results */}
          <AnimatePresence>
            {benchmark && tierInfo && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: 0.1 }}
                className="space-y-4"
              >
                <Card className="bg-gradient-to-br from-purple-500/10 to-blue-500/10 border-purple-500/20">
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Zap className="w-5 h-5 text-yellow-400" />
                      Benchmark Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Performance</span>
                      <span className="text-2xl font-bold text-purple-400">
                        {formatTFLOPS(benchmark.tflops)}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Benchmark Time</span>
                      <span className="text-sm font-medium">{benchmark.duration.toFixed(1)}ms</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Tier Classification */}
                <Card className={cn("border-2", tierInfo.bgColor)}>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Target className={tierInfo.color} />
                      Device Tier: {tierInfo.name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-muted-foreground">{tierInfo.description}</p>
                    <div className="grid grid-cols-2 gap-4 pt-2">
                      <div className="space-y-1">
                        <Label className="text-xs text-muted-foreground">Tokens/Hour</Label>
                        <p className={cn("text-lg font-bold", tierInfo.color)}>
                          {tierInfo.tokensPerHour.toLocaleString()}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs text-muted-foreground">Credits/Hour</Label>
                        <p className={cn("text-lg font-bold", tierInfo.color)}>
                          {tierInfo.creditsPerHour}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Device Registration Demo Section
 */
function DeviceRegistrationDemo() {
  const [deviceName, setDeviceName] = useState("My MacBook Pro");
  const [isRegistering, setIsRegistering] = useState(false);
  const [isRegistered, setIsRegistered] = useState(false);
  const [deviceId, setDeviceId] = useState<string>("");
  const [assignedTier, setAssignedTier] = useState<TierInfo | null>(null);

  const handleRegister = async () => {
    setIsRegistering(true);

    // Simulate registration process
    await new Promise((resolve) => setTimeout(resolve, 1500));

    const id = `dev_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    setDeviceId(id);
    setAssignedTier(getTierInfo("standard"));
    setIsRegistered(true);
    setIsRegistering(false);
  };

  const handleReset = () => {
    setIsRegistered(false);
    setDeviceId("");
    setAssignedTier(null);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="w-5 h-5 text-blue-400" />
            Device Registration Simulation
          </CardTitle>
          <CardDescription>
            Register your device to join the distributed compute network
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {!isRegistered ? (
            <>
              {/* Registration Form */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="deviceName">Device Name</Label>
                  <Input
                    id="deviceName"
                    value={deviceName}
                    onChange={(e) => setDeviceName(e.target.value)}
                    placeholder="Enter device name..."
                    disabled={isRegistering}
                  />
                </div>

                <Button
                  onClick={handleRegister}
                  disabled={isRegistering || !deviceName.trim()}
                  size="lg"
                  className="w-full gap-2"
                >
                  {isRegistering ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Registering Device...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Register Device
                    </>
                  )}
                </Button>
              </div>

              {/* Preview Info */}
              <div className="rounded-lg border border-border p-4 space-y-2">
                <p className="text-sm font-medium text-muted-foreground">
                  Registration will assign:
                </p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="w-3 h-3 text-green-400" />
                    Unique device ID
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="w-3 h-3 text-green-400" />
                    Tier based on capabilities
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle2 className="w-3 h-3 text-green-400" />
                    Credit earning potential
                  </li>
                </ul>
              </div>
            </>
          ) : (
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
                className="space-y-4"
              >
                {/* Success Message */}
                <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                      <CheckCircle2 className="w-5 h-5 text-green-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-green-400">
                        Device Successfully Registered!
                      </p>
                      <p className="text-xs text-green-400/80 mt-0.5">
                        You're now part of the distributed compute network
                      </p>
                    </div>
                  </div>
                </div>

                {/* Device Details */}
                <Card className="bg-muted/50">
                  <CardContent className="pt-6 space-y-4">
                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">Device ID</Label>
                      <div className="flex items-center gap-2">
                        <code className="text-sm font-mono bg-background px-2 py-1 rounded flex-1">
                          {deviceId}
                        </code>
                        <Badge variant="outline" className="text-xs">Active</Badge>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-xs text-muted-foreground">Device Name</Label>
                      <p className="text-sm font-medium">{deviceName}</p>
                    </div>

                    {assignedTier && (
                      <div className="space-y-2">
                        <Label className="text-xs text-muted-foreground">Assigned Tier</Label>
                        <div className={cn("rounded-lg p-3", assignedTier.bgColor)}>
                          <p className={cn("text-sm font-bold mb-1", assignedTier.color)}>
                            {assignedTier.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {assignedTier.description}
                          </p>
                          <div className="grid grid-cols-2 gap-3 mt-3">
                            <div>
                              <p className="text-xs text-muted-foreground">Earning Potential</p>
                              <p className={cn("text-sm font-bold", assignedTier.color)}>
                                {assignedTier.creditsPerHour} credits/hr
                              </p>
                            </div>
                            <div>
                              <p className="text-xs text-muted-foreground">Token Output</p>
                              <p className={cn("text-sm font-bold", assignedTier.color)}>
                                {assignedTier.tokensPerHour.toLocaleString()} tok/hr
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Button onClick={handleReset} variant="outline" className="w-full gap-2">
                  <RotateCcw className="w-4 h-4" />
                  Reset Demo
                </Button>
              </motion.div>
            </AnimatePresence>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Speculative Decoding Demo Section
 */
function SpeculativeDecodingDemo() {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState<"idle" | "draft" | "verify" | "complete">("idle");
  const [verifiedTokens, setVerifiedTokens] = useState<boolean[]>([]);
  const [acceptedCount, setAcceptedCount] = useState(0);
  const [speedupFactor, setSpeedupFactor] = useState(0);

  const runDemo = async () => {
    setIsRunning(true);
    setCurrentStep("draft");
    setVerifiedTokens([]);
    setAcceptedCount(0);
    setSpeedupFactor(0);

    // Simulate draft generation
    await new Promise((resolve) => setTimeout(resolve, 1200));

    // Simulate verification
    setCurrentStep("verify");
    const results: boolean[] = [];
    let accepted = 0;

    for (let i = 0; i < EXAMPLE_DRAFT_TOKENS.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 200));
      // Mock: Accept tokens with confidence > 0.8
      const isAccepted = EXAMPLE_DRAFT_TOKENS[i].confidence > 0.8;
      results.push(isAccepted);
      if (isAccepted) accepted++;
      setVerifiedTokens([...results]);
      setAcceptedCount(accepted);
    }

    // Calculate speedup
    const speedup = calculateSpeedupFactor(EXAMPLE_DRAFT_TOKENS.length, accepted);
    setSpeedupFactor(speedup);
    setCurrentStep("complete");
    setIsRunning(false);
  };

  const reset = () => {
    setCurrentStep("idle");
    setVerifiedTokens([]);
    setAcceptedCount(0);
    setSpeedupFactor(0);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-yellow-400" />
            Speculative Decoding Visualization
          </CardTitle>
          <CardDescription>
            Watch how crowd devices generate draft tokens and power devices verify them in parallel
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Control Buttons */}
          <div className="flex items-center gap-3">
            <Button
              onClick={runDemo}
              disabled={isRunning}
              size="lg"
              className="gap-2"
            >
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running Demo...
                </>
              ) : currentStep === "complete" ? (
                <>
                  <RotateCcw className="w-4 h-4" />
                  Run Again
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Run Speculative Decoding
                </>
              )}
            </Button>
            {currentStep === "complete" && (
              <Button onClick={reset} variant="outline" className="gap-2">
                <RotateCcw className="w-4 h-4" />
                Reset
              </Button>
            )}
          </div>

          {/* Process Visualization */}
          <div className="space-y-4">
            {/* Step 1: Draft Generation */}
            <motion.div
              animate={{
                opacity: currentStep !== "idle" ? 1 : 0.5,
                scale: currentStep === "draft" ? 1.02 : 1,
              }}
              className="rounded-lg border border-border p-4 space-y-3"
            >
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center",
                    currentStep === "draft"
                      ? "bg-green-500/20 animate-pulse"
                      : currentStep === "idle"
                      ? "bg-muted"
                      : "bg-green-500/20"
                  )}
                >
                  {currentStep === "draft" ? (
                    <Loader2 className="w-4 h-4 text-green-400 animate-spin" />
                  ) : currentStep !== "idle" ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  ) : (
                    <span className="text-sm font-medium text-muted-foreground">1</span>
                  )}
                </div>
                <div>
                  <p className="text-sm font-medium">Draft Token Generation</p>
                  <p className="text-xs text-muted-foreground">
                    Crowd devices (phones/browsers) generate 8 draft tokens
                  </p>
                </div>
              </div>
              {currentStep !== "idle" && (
                <div className="pl-10 space-y-1">
                  <div className="flex flex-wrap gap-2">
                    {EXAMPLE_DRAFT_TOKENS.map((token) => (
                      <Badge key={token.position} variant="outline" className="text-xs">
                        {token.text}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Small model: ~1s on phone
                  </p>
                </div>
              )}
            </motion.div>

            {/* Arrow */}
            {currentStep !== "idle" && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-center"
              >
                <ArrowRight className="w-5 h-5 text-muted-foreground" />
              </motion.div>
            )}

            {/* Step 2: Verification */}
            <motion.div
              animate={{
                opacity: currentStep === "verify" || currentStep === "complete" ? 1 : 0.5,
                scale: currentStep === "verify" ? 1.02 : 1,
              }}
              className="rounded-lg border border-border p-4 space-y-3"
            >
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center",
                    currentStep === "verify"
                      ? "bg-purple-500/20 animate-pulse"
                      : currentStep === "complete"
                      ? "bg-purple-500/20"
                      : "bg-muted"
                  )}
                >
                  {currentStep === "verify" ? (
                    <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
                  ) : currentStep === "complete" ? (
                    <CheckCircle2 className="w-4 h-4 text-purple-400" />
                  ) : (
                    <span className="text-sm font-medium text-muted-foreground">2</span>
                  )}
                </div>
                <div>
                  <p className="text-sm font-medium">Draft Verification</p>
                  <p className="text-xs text-muted-foreground">
                    Power devices (GPUs) verify all 8 tokens in parallel
                  </p>
                </div>
              </div>
              {(currentStep === "verify" || currentStep === "complete") && (
                <div className="pl-10 space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {EXAMPLE_DRAFT_TOKENS.map((token, idx) => (
                      <motion.div
                        key={token.position}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: idx * 0.2 }}
                      >
                        <Badge
                          variant="outline"
                          className={cn(
                            "text-xs",
                            verifiedTokens[idx] !== undefined &&
                              (verifiedTokens[idx]
                                ? "bg-green-500/20 border-green-500/50 text-green-400"
                                : "bg-red-500/20 border-red-500/50 text-red-400")
                          )}
                        >
                          {token.text}
                          {verifiedTokens[idx] !== undefined && (
                            <span className="ml-1">
                              {verifiedTokens[idx] ? (
                                <CheckCircle2 className="w-3 h-3 inline" />
                              ) : (
                                <XCircle className="w-3 h-3 inline" />
                              )}
                            </span>
                          )}
                        </Badge>
                      </motion.div>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Large model: ~0.5s on RTX 4090 (parallel verification)
                  </p>
                </div>
              )}
            </motion.div>

            {/* Results */}
            <AnimatePresence>
              {currentStep === "complete" && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Card className="bg-gradient-to-br from-green-500/10 to-purple-500/10 border-green-500/20">
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-green-400" />
                        Speedup Achieved
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <Label className="text-xs text-muted-foreground">Tokens Accepted</Label>
                          <p className="text-2xl font-bold text-green-400">
                            {acceptedCount}/{EXAMPLE_DRAFT_TOKENS.length}
                          </p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Acceptance Rate</Label>
                          <p className="text-2xl font-bold text-purple-400">
                            {((acceptedCount / EXAMPLE_DRAFT_TOKENS.length) * 100).toFixed(0)}%
                          </p>
                        </div>
                        <div>
                          <Label className="text-xs text-muted-foreground">Speedup Factor</Label>
                          <p className="text-2xl font-bold text-yellow-400">
                            {speedupFactor.toFixed(1)}x
                          </p>
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground pt-2 border-t border-border">
                        <p>
                          <strong>Sequential time:</strong> 8 tokens × 0.5s = 4.0s
                        </p>
                        <p>
                          <strong>Speculative time:</strong> 1.0s (draft) + 0.5s (verify) = 1.5s
                        </p>
                        <p className="text-green-400 font-medium mt-1">
                          Time saved: 2.5s (~2.7x faster)
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Credit System Demo Section
 */
function CreditSystemDemo() {
  const [creditsEarned, setCreditsEarned] = useState(0);
  const [tasksCompleted, setTasksCompleted] = useState(0);
  const [rank, setRank] = useState<"bronze" | "silver" | "gold" | "platinum">("bronze");
  const [isSimulating, setIsSimulating] = useState(false);

  const simulateEarning = async () => {
    setIsSimulating(true);

    // Simulate earning credits over time
    for (let i = 0; i < 10; i++) {
      await new Promise((resolve) => setTimeout(resolve, 300));
      setCreditsEarned((prev) => prev + Math.floor(Math.random() * 5) + 3);
      setTasksCompleted((prev) => prev + 1);
    }

    // Update rank based on credits
    const newCredits = creditsEarned + 40;
    if (newCredits >= 1000) setRank("platinum");
    else if (newCredits >= 500) setRank("gold");
    else if (newCredits >= 100) setRank("silver");
    else setRank("bronze");

    setIsSimulating(false);
  };

  const reset = () => {
    setCreditsEarned(0);
    setTasksCompleted(0);
    setRank("bronze");
  };

  const rankColors = {
    bronze: "text-orange-400",
    silver: "text-gray-300",
    gold: "text-yellow-400",
    platinum: "text-purple-400",
  };

  const rankProgress = {
    bronze: { min: 0, max: 100 },
    silver: { min: 100, max: 500 },
    gold: { min: 500, max: 1000 },
    platinum: { min: 1000, max: 5000 },
  };

  const currentRankInfo = rankProgress[rank];
  const progressPercent = Math.min(
    ((creditsEarned - currentRankInfo.min) / (currentRankInfo.max - currentRankInfo.min)) * 100,
    100
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Coins className="w-5 h-5 text-yellow-400" />
            Credit System & Rewards
          </CardTitle>
          <CardDescription>
            Earn credits by contributing compute power and watch your rank progress
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Simulate Button */}
          <Button
            onClick={simulateEarning}
            disabled={isSimulating}
            size="lg"
            className="w-full gap-2"
          >
            {isSimulating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Simulating Contributions...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Simulate Credit Earning
              </>
            )}
          </Button>

          {/* Stats Display */}
          <div className="grid grid-cols-2 gap-4">
            <Card className="bg-muted/50">
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 mb-2">
                  <Award className="w-4 h-4 text-yellow-400" />
                  <Label className="text-xs text-muted-foreground">Total Credits</Label>
                </div>
                <motion.p
                  key={creditsEarned}
                  initial={{ scale: 1.2 }}
                  animate={{ scale: 1 }}
                  className="text-3xl font-bold text-yellow-400"
                >
                  {creditsEarned}
                </motion.p>
              </CardContent>
            </Card>

            <Card className="bg-muted/50">
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 mb-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  <Label className="text-xs text-muted-foreground">Tasks Completed</Label>
                </div>
                <motion.p
                  key={tasksCompleted}
                  initial={{ scale: 1.2 }}
                  animate={{ scale: 1 }}
                  className="text-3xl font-bold text-green-400"
                >
                  {tasksCompleted}
                </motion.p>
              </CardContent>
            </Card>
          </div>

          {/* Rank Progress */}
          <Card className="bg-gradient-to-br from-purple-500/10 to-yellow-500/10 border-purple-500/20">
            <CardContent className="pt-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-xs text-muted-foreground">Current Rank</Label>
                  <p className={cn("text-2xl font-bold capitalize", rankColors[rank])}>
                    {rank}
                  </p>
                </div>
                <Award className={cn("w-12 h-12", rankColors[rank])} />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{currentRankInfo.min} credits</span>
                  <span>{currentRankInfo.max} credits</span>
                </div>
                <Progress value={progressPercent} className="h-2" />
                <p className="text-xs text-muted-foreground text-center">
                  {creditsEarned >= currentRankInfo.max
                    ? "Rank maxed out!"
                    : `${currentRankInfo.max - creditsEarned} credits until next rank`}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Transaction History Preview */}
          {tasksCompleted > 0 && (
            <Card className="bg-muted/50">
              <CardHeader>
                <CardTitle className="text-sm">Recent Transactions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {Array.from({ length: Math.min(tasksCompleted, 3) }).map((_, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.1 }}
                    className="flex items-center justify-between text-sm py-2 border-b border-border last:border-0"
                  >
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-green-400" />
                      <span className="text-muted-foreground">
                        Task completed
                      </span>
                    </div>
                    <span className="font-medium text-green-400">
                      +{Math.floor(Math.random() * 5) + 3} credits
                    </span>
                  </motion.div>
                ))}
              </CardContent>
            </Card>
          )}

          {/* Reset Button */}
          {(creditsEarned > 0 || tasksCompleted > 0) && (
            <Button onClick={reset} variant="outline" className="w-full gap-2">
              <RotateCcw className="w-4 h-4" />
              Reset Demo
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/**
 * Network Stats Demo Section
 */
function NetworkStatsDemo() {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            Live Network Analytics
          </CardTitle>
          <CardDescription>
            Real-time visualization of the distributed compute network activity
          </CardDescription>
        </CardHeader>
        <CardContent>
          <LiveDashboard showPersonalStats={false} />
        </CardContent>
      </Card>

      {/* Additional Info */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-sm">About the Network</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            The LabFork distributed compute network enables anyone with a device to contribute
            to AI research and training. By pooling compute power from phones, laptops, and GPUs,
            we democratize access to computational resources.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
            <div className="space-y-1">
              <p className="font-medium text-purple-400">Power Tier</p>
              <p className="text-xs">High-end GPUs handle full inference and verification</p>
            </div>
            <div className="space-y-1">
              <p className="font-medium text-blue-400">Standard Tier</p>
              <p className="text-xs">Mid-range devices run smaller models efficiently</p>
            </div>
            <div className="space-y-1">
              <p className="font-medium text-green-400">Crowd Tier</p>
              <p className="text-xs">Phones and browsers contribute through draft generation</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
