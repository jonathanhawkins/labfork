/**
 * Example Component: Device Agent Integration
 *
 * This example demonstrates how to use the useDeviceAgent hook
 * to integrate distributed compute into a React component.
 *
 * NOTE: This is an example file for reference - not used in production.
 */

"use client";

import { useDeviceAgent, formatUptime, formatComputeTime, formatCredits, getStatusColor, getStatusLabel } from "./useDeviceAgent";

export default function DeviceAgentExample() {
  const {
    status,
    device,
    currentTask,
    taskProgress,
    stats,
    error,
    isReady,
    isBusy,
    isPaused,
    start,
    pause,
    resume,
    stop,
    clearError,
  } = useDeviceAgent({
    deviceName: "My Device",
    autoStart: true,
    availability: {
      wifiOnly: true,
      chargingOnly: false,
      minBattery: 20,
      maxUtilization: 80,
    },
  });

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h2 className="text-2xl font-bold mb-4">Device Agent Status</h2>

        {/* Status */}
        <div className="mb-4">
          <span className="text-gray-400">Status: </span>
          <span className={`font-semibold ${getStatusColor(status)}`}>
            {getStatusLabel(status)}
          </span>
        </div>

        {/* Device Info */}
        {device && (
          <div className="space-y-2 mb-4">
            <div>
              <span className="text-gray-400">Device ID: </span>
              <span className="font-mono text-sm">{device.id}</span>
            </div>
            <div>
              <span className="text-gray-400">Tier: </span>
              <span className="font-semibold capitalize">{device.tier}</span>
            </div>
            <div>
              <span className="text-gray-400">Compute: </span>
              <span>{device.capabilities.compute.toFixed(2)} TFLOPS</span>
            </div>
            <div>
              <span className="text-gray-400">Memory: </span>
              <span>{device.capabilities.memory.toFixed(1)} GB</span>
            </div>
            <div>
              <span className="text-gray-400">GPU: </span>
              <span className="text-sm">{device.capabilities.gpuName}</span>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded p-3 mb-4">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-red-400 font-semibold">Error</p>
                <p className="text-sm text-gray-300">{error.message}</p>
              </div>
              <button
                onClick={clearError}
                className="text-red-400 hover:text-red-300"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Current Task */}
        {currentTask && (
          <div className="bg-blue-500/10 border border-blue-500/20 rounded p-4 mb-4">
            <p className="text-blue-400 font-semibold mb-2">Current Task</p>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-400">ID: </span>
                <span className="font-mono">{currentTask.id}</span>
              </div>
              <div>
                <span className="text-gray-400">Type: </span>
                <span className="capitalize">{currentTask.type.replace("_", " ")}</span>
              </div>
              <div>
                <span className="text-gray-400">Reward: </span>
                <span>{currentTask.reward} credits</span>
              </div>
              <div className="mt-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-400">Progress</span>
                  <span>{Math.round(taskProgress)}%</span>
                </div>
                <div className="bg-gray-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-blue-500 h-full transition-all duration-300"
                    style={{ width: `${taskProgress}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-gray-800 rounded p-3">
            <p className="text-gray-400 text-xs mb-1">Tasks Completed</p>
            <p className="text-2xl font-bold">{stats.tasksCompleted}</p>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <p className="text-gray-400 text-xs mb-1">Credits Earned</p>
            <p className="text-2xl font-bold">{formatCredits(stats.creditsEarned)}</p>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <p className="text-gray-400 text-xs mb-1">Compute Time</p>
            <p className="text-xl font-bold">{formatComputeTime(stats.totalComputeTime)}</p>
          </div>
          <div className="bg-gray-800 rounded p-3">
            <p className="text-gray-400 text-xs mb-1">Uptime</p>
            <p className="text-xl font-bold">{formatUptime(stats.uptimeSeconds)}</p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex gap-2">
          {!isReady && status !== "connecting" && status !== "initializing" && (
            <button
              onClick={start}
              className="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded transition-colors"
            >
              Start Agent
            </button>
          )}

          {isReady && !isBusy && (
            <button
              onClick={pause}
              className="flex-1 bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-2 px-4 rounded transition-colors"
            >
              Pause
            </button>
          )}

          {isPaused && (
            <button
              onClick={resume}
              className="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded transition-colors"
            >
              Resume
            </button>
          )}

          {(isReady || isPaused) && (
            <button
              onClick={stop}
              className="flex-1 bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded transition-colors"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      {/* Info Card */}
      <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h3 className="text-lg font-semibold mb-3">How It Works</h3>
        <ul className="space-y-2 text-sm text-gray-300">
          <li className="flex items-start">
            <span className="text-green-400 mr-2">✓</span>
            <span>Your browser connects to the LabFork compute network</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-400 mr-2">✓</span>
            <span>GPU is benchmarked and classified into a tier</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-400 mr-2">✓</span>
            <span>Agent receives and executes compute tasks</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-400 mr-2">✓</span>
            <span>Earn credits for completed tasks</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-400 mr-2">✓</span>
            <span>Use credits to run your own experiments</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
