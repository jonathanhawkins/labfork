"use client";

import { motion } from "framer-motion";
import { useMemo } from "react";

export interface ContributionChartProps {
  data: {
    day: string;
    tasksCompleted: number;
    creditsEarned: number;
  }[];
}

export function ContributionChart({ data }: ContributionChartProps) {
  const maxTasks = useMemo(() => Math.max(...data.map((d) => d.tasksCompleted), 1), [data]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground-bright">7-Day History</h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 bg-green-500 rounded-sm" />
            <span className="text-muted-foreground">Tasks</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 bg-foreground-muted rounded-sm" />
            <span className="text-muted-foreground">Credits</span>
          </div>
        </div>
      </div>

      <div className="p-4 bg-background-card border border-border rounded-lg">
        <div className="flex items-end justify-between gap-2 h-32">
          {data.map((item, index) => {
            const heightPercent = (item.tasksCompleted / maxTasks) * 100;
            return (
              <div key={item.day} className="flex-1 flex flex-col items-center gap-2">
                {/* Bar */}
                <div className="w-full flex flex-col gap-0.5">
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${heightPercent}%` }}
                    transition={{ duration: 0.5, ease: "easeOut", delay: index * 0.05 }}
                    className="w-full bg-green-500/80 rounded-t min-h-[2px] hover:bg-green-500 transition-colors cursor-pointer group relative"
                  >
                    {/* Tooltip on hover */}
                    <div className="absolute -top-12 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                      <div className="bg-background-elevated border border-border px-2 py-1 rounded text-xs whitespace-nowrap">
                        <div className="text-foreground-bright font-mono">
                          {item.tasksCompleted} tasks
                        </div>
                        <div className="text-green-400 font-mono">
                          {item.creditsEarned} credits
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>

                {/* Day Label */}
                <span className="text-[10px] text-muted-foreground font-mono">{item.day}</span>
              </div>
            );
          })}
        </div>

        {/* Summary */}
        <div className="mt-4 pt-3 border-t border-border-subtle flex items-center justify-between text-xs">
          <span className="text-muted-foreground">
            Total: {data.reduce((sum, d) => sum + d.tasksCompleted, 0)} tasks
          </span>
          <span className="text-green-400 font-mono">
            {data.reduce((sum, d) => sum + d.creditsEarned, 0)} credits earned
          </span>
        </div>
      </div>
    </div>
  );
}
