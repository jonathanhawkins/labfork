import { CostCalculator } from "@/components/CostCalculator";

export const metadata = {
  title: "Deploy - AI Research Lab",
  description: "Deploy your AI Research Lab and estimate costs",
};

export default function DeployPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Deploy Your AI Research Lab
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Choose from multiple deployment options and estimate your monthly costs.
            Get started in as little as 5 minutes.
          </p>
        </div>

        {/* Quick Deploy Options */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <DeployCard
            title="Docker (Local)"
            time="5 min"
            cost="Free"
            difficulty="Easy"
            command="./setup.sh && docker compose up -d"
            href="/DEPLOYMENT.md#docker-compose-local"
          />
          <DeployCard
            title="Vercel + RunPod"
            time="10 min"
            cost="$20-50/mo"
            difficulty="Medium"
            command="vercel deploy"
            href="/DEPLOYMENT.md#vercel--cloud-gpu"
            featured
          />
          <DeployCard
            title="Railway"
            time="3 min"
            cost="$5-20/mo"
            difficulty="Easy"
            command="./deploy/deploy-railway.sh"
            href="/DEPLOYMENT.md#railway"
          />
        </div>

        {/* Cost Calculator */}
        <CostCalculator />

        {/* Hardware Requirements */}
        <div className="mt-12 bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Hardware Requirements
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    Component
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    Minimum
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    Recommended
                  </th>
                </tr>
              </thead>
              <tbody className="text-gray-600 dark:text-gray-400">
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">RAM</td>
                  <td className="py-3 px-4">8 GB</td>
                  <td className="py-3 px-4">16+ GB</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">Storage</td>
                  <td className="py-3 px-4">20 GB</td>
                  <td className="py-3 px-4">50+ GB</td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">CPU</td>
                  <td className="py-3 px-4">4 cores</td>
                  <td className="py-3 px-4">8+ cores</td>
                </tr>
                <tr>
                  <td className="py-3 px-4">GPU</td>
                  <td className="py-3 px-4">None (CPU-only)</td>
                  <td className="py-3 px-4">8+ GB VRAM</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* GPU Recommendations */}
        <div className="mt-8 bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            GPU Recommendations
          </h2>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    GPU
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    VRAM
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    Best Models
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300">
                    Performance
                  </th>
                </tr>
              </thead>
              <tbody className="text-gray-600 dark:text-gray-400">
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">CPU Only</td>
                  <td className="py-3 px-4">-</td>
                  <td className="py-3 px-4">qwen3-coder:7b</td>
                  <td className="py-3 px-4">
                    <span className="text-yellow-600 dark:text-yellow-400">Slow</span>
                  </td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">RTX 3060/4070</td>
                  <td className="py-3 px-4">12 GB</td>
                  <td className="py-3 px-4">qwen3-coder:14b</td>
                  <td className="py-3 px-4">
                    <span className="text-green-600 dark:text-green-400">Good</span>
                  </td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">RTX 4090</td>
                  <td className="py-3 px-4">24 GB</td>
                  <td className="py-3 px-4">qwen3-coder:30b</td>
                  <td className="py-3 px-4">
                    <span className="text-blue-600 dark:text-blue-400">Excellent</span>
                  </td>
                </tr>
                <tr className="border-b dark:border-gray-700">
                  <td className="py-3 px-4">A100</td>
                  <td className="py-3 px-4">40+ GB</td>
                  <td className="py-3 px-4">qwen3-coder:30b</td>
                  <td className="py-3 px-4">
                    <span className="text-purple-600 dark:text-purple-400">Best</span>
                  </td>
                </tr>
                <tr>
                  <td className="py-3 px-4">Apple M3/M4</td>
                  <td className="py-3 px-4">18-128 GB</td>
                  <td className="py-3 px-4">qwen3-coder:14b-30b</td>
                  <td className="py-3 px-4">
                    <span className="text-green-600 dark:text-green-400">Good</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DeployCard({
  title,
  time,
  cost,
  difficulty,
  command,
  href,
  featured,
}: {
  title: string;
  time: string;
  cost: string;
  difficulty: string;
  command: string;
  href: string;
  featured?: boolean;
}) {
  return (
    <div
      className={`relative bg-white dark:bg-gray-900 rounded-lg shadow-lg p-6 ${
        featured ? "ring-2 ring-blue-500" : ""
      }`}
    >
      {featured && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-blue-500 text-white text-xs font-medium rounded-full">
          Recommended
        </div>
      )}

      <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        {title}
      </h3>

      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Setup Time</span>
          <span className="font-medium text-gray-900 dark:text-white">{time}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Cost</span>
          <span className="font-medium text-gray-900 dark:text-white">{cost}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-500">Difficulty</span>
          <span className="font-medium text-gray-900 dark:text-white">{difficulty}</span>
        </div>
      </div>

      <div className="bg-gray-100 dark:bg-gray-800 rounded p-2 mb-4">
        <code className="text-xs text-gray-700 dark:text-gray-300 break-all">
          {command}
        </code>
      </div>

      <a
        href={href}
        className="block w-full text-center py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors"
      >
        View Instructions
      </a>
    </div>
  );
}
