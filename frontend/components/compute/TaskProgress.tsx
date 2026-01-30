"use client";

import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

export interface TaskProgressProps {
  taskName: string;
  progress: number;
  timeRemaining?: string;
  isActive: boolean;
}

export function TaskProgress({ taskName, progress, timeRemaining, isActive }: TaskProgressProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="p-4 bg-background-card border border-border rounded-lg"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {isActive && <Loader2 className="w-3.5 h-3.5 text-green-500 animate-spin" />}
          <span className="text-sm text-foreground-bright font-medium">{taskName}</span>
        </div>
        <span className="text-xs text-muted-foreground font-mono">{progress}%</span>
      </div>

      {/* Progress Bar */}
      <div className="h-1.5 bg-border rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, progress)}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className="h-full bg-gradient-to-r from-green-600 to-green-500"
        />
      </div>

      {timeRemaining && (
        <div className="mt-2 text-xs text-muted-foreground">
          Est. {timeRemaining} remaining
        </div>
      )}
    </motion.div>
  );
}
