"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Cpu, Zap, Pause, Play, Settings, Clock } from "lucide-react";
import { TaskProgress } from "./TaskProgress";
import { NetworkStats } from "./NetworkStats";
import { ContributionChart } from "./ContributionChart";

type ContributionStatus = "active" | "paused" | "idle";

interface ContributorDashboardProps {
  onSettingsClick?: () => void;
}

// Animated counter component
function AnimatedCounter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const duration = 1000; // 1 second animation
    const steps = 60;
    const increment = (value - displayValue) / steps;
    let currentStep = 0;

    const timer = setInterval(() => {
      currentStep++;
      if (currentStep >= steps) {
        setDisplayValue(value);
        clearInterval(timer);
      } else {
        setDisplayValue((prev) => Math.min(value, prev + increment));
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [value, displayValue]);

  return (
    <span className="font-mono font-semibold">
      {Math.floor(displayValue).toLocaleString()}
      {suffix}
    </span>
  );
}

// Mock data generator
function getMockData() {
  return {
    status: "active" as ContributionStatus,
    tasksCompleted: 147,
    creditsEarned: 2941,
    currentTask: {
      name: "Voice Model Training - Batch 47",
      progress: 67,
      timeRemaining: "3m 24s",
      isActive: true,
    },
    sessionDuration: "2h 14m",
    networkStats: {
      activeContributors: 1247,
      networkComputePower: 847.3,
      yourRank: 23,
      contributionPercent: 1.8,
    },
    history: [
      { day: "Mon", tasksCompleted: 18, creditsEarned: 360 },
      { day: "Tue", tasksCompleted: 23, creditsEarned: 460 },
      { day: "Wed", tasksCompleted: 21, creditsEarned: 420 },
      { day: "Thu", tasksCompleted: 25, creditsEarned: 500 },
      { day: "Fri", tasksCompleted: 19, creditsEarned: 380 },
      { day: "Sat", tasksCompleted: 22, creditsEarned: 440 },
      { day: "Sun", tasksCompleted: 19, creditsEarned: 381 },
    ],
  };
}

export function ContributorDashboard({ onSettingsClick }: ContributorDashboardProps) {
  const [status, setStatus] = useState<ContributionStatus>("active");
  const [data, setData] = useState(getMockData());

  // Simulate real-time updates
  useEffect(() => {
    if (status !== "active") return;

    const interval = setInterval(() => {
      setData((prev) => ({
        ...prev,
        currentTask: {
          ...prev.currentTask,
          progress: Math.min(100, prev.currentTask.progress + 1),
        },
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, [status]);

  const handleTogglePause = useCallback(() => {
    setStatus((prev) => (prev === "active" ? "paused" : "active"));
  }, []);

  const statusConfig = {
    active: {
      color: "bg-green-500",
      textColor: "text-green-400",
      label: "Contributing",
      icon: Zap,
    },
    paused: {
      color: "bg-amber-500",
      textColor: "text-amber-400",
      label: "Paused",
      icon: Pause,
    },
    idle: {
      color: "bg-foreground-subtle",
      textColor: "text-muted-foreground",
      label: "Idle",
      icon: Cpu,
    },
  };

  const currentStatus = statusConfig[status];
  const StatusIcon = currentStatus.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="w-full max-w-4xl mx-auto p-4 space-y-6"
    >
      {/* Header with Status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Cpu className="w-6 h-6 text-foreground-bright" />
            <div className={`absolute -bottom-1 -right-1 w-3 h-3 ${currentStatus.color} rounded-full ${status === "active" ? "animate-pulse" : ""}`} />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground-bright">Compute Contribution</h2>
            <div className="flex items-center gap-2">
              <StatusIcon className="w-3 h-3" />
              <span className={`text-sm ${currentStatus.textColor}`}>{currentStatus.label}</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleTogglePause}
            className="min-h-[44px] min-w-[44px] flex items-center justify-center px-4 py-2 bg-background-card border border-border rounded-lg hover:bg-background-elevated transition-colors"
            aria-label={status === "active" ? "Pause contribution" : "Resume contribution"}
          >
            {status === "active" ? (
              <Pause className="w-4 h-4 text-foreground-bright" />
            ) : (
              <Play className="w-4 h-4 text-foreground-bright" />
            )}
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={onSettingsClick}
            className="min-h-[44px] min-w-[44px] flex items-center justify-center px-4 py-2 bg-background-card border border-border rounded-lg hover:bg-background-elevated transition-colors"
            aria-label="Open settings"
          >
            <Settings className="w-4 h-4 text-foreground-bright" />
          </motion.button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, ease: "easeOut", delay: 0.1 }}
          className="p-4 bg-background-card border border-border rounded-lg"
        >
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Tasks Completed</span>
          </div>
          <div className="text-2xl text-foreground-bright">
            <AnimatedCounter value={data.tasksCompleted} />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, ease: "easeOut", delay: 0.15 }}
          className="p-4 bg-background-card border border-border rounded-lg"
        >
          <div className="flex items-center gap-2 mb-1">
            <Zap className="w-3.5 h-3.5 text-green-500" />
            <span className="text-xs text-muted-foreground">Credits Earned</span>
          </div>
          <div className="text-2xl text-green-400">
            <AnimatedCounter value={data.creditsEarned} />
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, ease: "easeOut", delay: 0.2 }}
          className="p-4 bg-background-card border border-border rounded-lg"
        >
          <div className="flex items-center gap-2 mb-1">
            <Clock className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Session Duration</span>
          </div>
          <div className="text-2xl text-foreground-bright font-mono font-semibold">
            {data.sessionDuration}
          </div>
        </motion.div>
      </div>

      {/* Current Task Progress */}
      <AnimatePresence mode="wait">
        {status === "active" && data.currentTask.isActive && (
          <TaskProgress
            taskName={data.currentTask.name}
            progress={data.currentTask.progress}
            timeRemaining={data.currentTask.timeRemaining}
            isActive={true}
          />
        )}
      </AnimatePresence>

      {/* Paused Message */}
      {status === "paused" && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className="p-4 bg-background-card border border-amber-500/20 rounded-lg text-center"
        >
          <Pause className="w-5 h-5 text-amber-400 mx-auto mb-2" />
          <p className="text-sm text-amber-400 font-medium">Contribution Paused</p>
          <p className="text-xs text-muted-foreground mt-1">
            Click the play button to resume contributing compute
          </p>
        </motion.div>
      )}

      {/* Network Stats */}
      <NetworkStats
        activeContributors={data.networkStats.activeContributors}
        networkComputePower={data.networkStats.networkComputePower}
        yourRank={data.networkStats.yourRank}
        contributionPercent={data.networkStats.contributionPercent}
      />

      {/* Contribution History Chart */}
      <ContributionChart data={data.history} />

      {/* Footer Info */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3, ease: "easeOut", delay: 0.5 }}
        className="p-3 bg-background-card/50 border border-border-subtle rounded-lg"
      >
        <p className="text-xs text-muted-foreground text-center">
          Your compute helps train AI models and advance research.{" "}
          <button className="text-foreground-bright hover:text-foreground transition-colors underline">
            Learn more
          </button>
        </p>
      </motion.div>
    </motion.div>
  );
}
