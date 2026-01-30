"use client";

import { useState, useMemo } from "react";

interface UsageConfig {
  hoursPerWeek: number;
  gpuType: "none" | "rtx4090" | "a100" | "apple-silicon";
  deploymentMethod: "local" | "vercel-free" | "vercel-pro" | "railway" | "runpod" | "aws";
  storageGB: number;
  teamSize: number;
}

interface CostBreakdown {
  compute: number;
  storage: number;
  bandwidth: number;
  platform: number;
  total: number;
  perUser: number;
}

const PRICING = {
  gpu: {
    none: 0,
    rtx4090: 0.44, // RunPod RTX 4090 per hour
    a100: 1.89, // RunPod A100 per hour
    "apple-silicon": 0, // Local - no cloud cost
  },
  storage: {
    local: 0,
    cloud: 0.023, // per GB per month (S3 pricing)
  },
  bandwidth: {
    free: 0,
    paid: 0.09, // per GB
  },
  platform: {
    local: 0,
    "vercel-free": 0,
    "vercel-pro": 20,
    railway: 5,
    runpod: 0, // Pay-as-you-go
    aws: 0, // Pay-as-you-go
  },
};

export function CostCalculator() {
  const [config, setConfig] = useState<UsageConfig>({
    hoursPerWeek: 10,
    gpuType: "none",
    deploymentMethod: "local",
    storageGB: 20,
    teamSize: 1,
  });

  const costs = useMemo((): CostBreakdown => {
    const weeksPerMonth = 4.33;
    const hoursPerMonth = config.hoursPerWeek * weeksPerMonth;

    // Compute costs
    let compute = 0;
    if (config.deploymentMethod === "runpod" || config.deploymentMethod === "aws") {
      compute = hoursPerMonth * PRICING.gpu[config.gpuType];
    }

    // Storage costs (only for cloud deployments)
    let storage = 0;
    if (config.deploymentMethod !== "local") {
      storage = config.storageGB * PRICING.storage.cloud;
    }

    // Bandwidth (estimate 10GB per user per month for cloud)
    let bandwidth = 0;
    if (config.deploymentMethod === "vercel-pro" || config.deploymentMethod === "aws") {
      bandwidth = 10 * config.teamSize * PRICING.bandwidth.paid;
    }

    // Platform costs
    const platform = PRICING.platform[config.deploymentMethod] || 0;

    const total = compute + storage + bandwidth + platform;
    const perUser = config.teamSize > 0 ? total / config.teamSize : 0;

    return { compute, storage, bandwidth, platform, total, perUser };
  }, [config]);

  const updateConfig = (key: keyof UsageConfig, value: number | string) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
        Deployment Cost Calculator
      </h2>

      <div className="space-y-6">
        {/* Deployment Method */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Deployment Method
          </label>
          <select
            value={config.deploymentMethod}
            onChange={(e) => updateConfig("deploymentMethod", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value="local">Local (Docker/Native)</option>
            <option value="vercel-free">Vercel Free Tier</option>
            <option value="vercel-pro">Vercel Pro ($20/mo)</option>
            <option value="railway">Railway</option>
            <option value="runpod">RunPod (GPU Cloud)</option>
            <option value="aws">AWS/GCP</option>
          </select>
        </div>

        {/* GPU Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            GPU Type
          </label>
          <select
            value={config.gpuType}
            onChange={(e) => updateConfig("gpuType", e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value="none">CPU Only (Free)</option>
            <option value="apple-silicon">Apple Silicon (Local)</option>
            <option value="rtx4090">RTX 4090 ($0.44/hr)</option>
            <option value="a100">A100 ($1.89/hr)</option>
          </select>
        </div>

        {/* Hours per Week */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            GPU Hours per Week: {config.hoursPerWeek}
          </label>
          <input
            type="range"
            min="0"
            max="168"
            value={config.hoursPerWeek}
            onChange={(e) => updateConfig("hoursPerWeek", parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0 hrs</span>
            <span>84 hrs</span>
            <span>168 hrs (24/7)</span>
          </div>
        </div>

        {/* Storage */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Storage (GB): {config.storageGB}
          </label>
          <input
            type="range"
            min="10"
            max="500"
            step="10"
            value={config.storageGB}
            onChange={(e) => updateConfig("storageGB", parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>10 GB</span>
            <span>250 GB</span>
            <span>500 GB</span>
          </div>
        </div>

        {/* Team Size */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Team Size: {config.teamSize}
          </label>
          <input
            type="range"
            min="1"
            max="20"
            value={config.teamSize}
            onChange={(e) => updateConfig("teamSize", parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1 user</span>
            <span>10 users</span>
            <span>20 users</span>
          </div>
        </div>
      </div>

      {/* Cost Breakdown */}
      <div className="mt-8 pt-6 border-t border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Monthly Cost Estimate
        </h3>

        <div className="space-y-3">
          <CostRow label="Compute (GPU)" value={costs.compute} />
          <CostRow label="Storage" value={costs.storage} />
          <CostRow label="Bandwidth" value={costs.bandwidth} />
          <CostRow label="Platform Fee" value={costs.platform} />

          <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center">
              <span className="text-lg font-bold text-gray-900 dark:text-white">Total</span>
              <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                ${costs.total.toFixed(2)}/mo
              </span>
            </div>
            {config.teamSize > 1 && (
              <div className="flex justify-between items-center mt-1 text-sm text-gray-500">
                <span>Per user</span>
                <span>${costs.perUser.toFixed(2)}/mo</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">
          Recommendation
        </h4>
        <p className="text-sm text-blue-700 dark:text-blue-400">
          {getRecommendation(config, costs)}
        </p>
      </div>
    </div>
  );
}

function CostRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-gray-600 dark:text-gray-400">{label}</span>
      <span className="font-medium text-gray-900 dark:text-white">
        ${value.toFixed(2)}
      </span>
    </div>
  );
}

function getRecommendation(config: UsageConfig, costs: CostBreakdown): string {
  if (costs.total === 0) {
    return "You are using a free deployment option. This is great for development and small projects.";
  }

  if (config.hoursPerWeek > 100 && config.gpuType !== "none") {
    return "For heavy GPU usage, consider reserved instances on AWS/GCP for better rates, or purchasing local hardware like an RTX 4090.";
  }

  if (config.deploymentMethod === "runpod" && config.hoursPerWeek < 20) {
    return "RunPod is cost-effective for occasional GPU usage. For heavier usage, consider dedicated cloud instances.";
  }

  if (config.teamSize > 5 && config.deploymentMethod === "vercel-free") {
    return "For larger teams, consider upgrading to Vercel Pro for better performance and higher limits.";
  }

  if (costs.total > 100) {
    return "Your estimated costs are significant. Consider optimizing GPU usage with spot instances or scheduling non-urgent tasks during off-peak hours.";
  }

  return "Your deployment configuration looks balanced. Monitor usage and adjust as needed.";
}

export default CostCalculator;
