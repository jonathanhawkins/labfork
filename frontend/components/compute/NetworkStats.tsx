"use client";

import { motion } from "framer-motion";
import { Users, Cpu, TrendingUp } from "lucide-react";

export interface NetworkStatsProps {
  activeContributors: number;
  networkComputePower: number;
  yourRank: number;
  contributionPercent: number;
}

function StatCard({
  icon: Icon,
  label,
  value,
  subValue,
  index,
}: {
  icon: typeof Users;
  label: string;
  value: string;
  subValue?: string;
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut", delay: index * 0.1 }}
      className="p-3 bg-background-card border border-border rounded-lg"
    >
      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <div className="text-lg text-foreground-bright font-mono font-semibold">{value}</div>
      {subValue && <div className="text-xs text-muted-foreground mt-0.5">{subValue}</div>}
    </motion.div>
  );
}

export function NetworkStats({
  activeContributors,
  networkComputePower,
  yourRank,
  contributionPercent,
}: NetworkStatsProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground-bright">Network Statistics</h3>
        <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <StatCard
          icon={Users}
          label="Active Contributors"
          value={activeContributors.toLocaleString()}
          index={0}
        />
        <StatCard
          icon={Cpu}
          label="Network Power"
          value={`${networkComputePower.toFixed(1)} TFLOPS`}
          index={1}
        />
        <StatCard
          icon={TrendingUp}
          label="Your Rank"
          value={`#${yourRank}`}
          subValue={`Top ${contributionPercent.toFixed(1)}%`}
          index={2}
        />
      </div>
    </div>
  );
}
