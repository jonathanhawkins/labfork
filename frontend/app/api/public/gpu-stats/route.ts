import { NextResponse } from 'next/server';

/**
 * Public GPU Stats API
 * Returns sanitized GPU information without sensitive details
 * - No IP addresses
 * - No file paths
 * - No process details
 * - Just utilization metrics
 */

// Backend API URL
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8003';

interface RawGpuStats {
  connected: boolean;
  error?: string;
  timestamp: string;
  gpu: {
    name: string;
    driverVersion: string;
    cudaVersion: string;
    utilization: number;
    memoryUsed: number;
    memoryTotal: number;
    memoryPercent: number;
    temperature: number;
    powerDraw: number;
    powerLimit: number;
    fanSpeed?: number;
  } | null;
  processes: Array<{
    pid: string;
    name: string;
    memoryUsed: string;
    script?: string;
    session?: string;
    config?: string;
    progress?: string;
  }>;
}

interface SanitizedGpuStats {
  connected: boolean;
  clusterName: string;
  timestamp: string;
  gpu: {
    name: string;
    utilization: number;
    memoryPercent: number;
    temperature: number;
    powerPercent: number;
  } | null;
  hasActiveTraining: boolean;
  trainingStatus?: string;
}

/**
 * Sanitize progress string to remove sensitive paths
 */
function sanitizeProgress(progress: string): string {
  // Keep general progress info, remove paths
  let result = progress;

  // Remove file paths
  result = result.replace(/\/[^\s]+\//g, '');
  result = result.replace(/config\/[^\s]+\.yaml/g, '');
  result = result.replace(/checkpoints?\/[^\s]+/g, '');

  // Keep epoch/loss info
  const epochMatch = result.match(/Epoch\s+\d+/i);
  const lossMatch = result.match(/loss[:\s]+[\d.]+/i);

  if (epochMatch || lossMatch) {
    const parts = [];
    if (epochMatch) parts.push(epochMatch[0]);
    if (lossMatch) parts.push(lossMatch[0]);
    return parts.join(' - ');
  }

  return 'Training in progress';
}

/**
 * Generate demo GPU stats with realistic variations
 */
function getDemoStats(): SanitizedGpuStats {
  // Simulate realistic GPU utilization patterns
  const baseUtil = 65;
  const variation = Math.sin(Date.now() / 10000) * 15 + Math.random() * 10;
  const utilization = Math.round(Math.min(95, Math.max(30, baseUtil + variation)));

  const memVariation = Math.sin(Date.now() / 15000) * 5 + Math.random() * 3;
  const memoryPercent = Math.round(Math.min(90, Math.max(60, 75 + memVariation)));

  const tempVariation = Math.sin(Date.now() / 20000) * 8 + Math.random() * 5;
  const temperature = Math.round(Math.min(78, Math.max(55, 65 + tempVariation)));

  const trainingStatuses = [
    'Epoch 147 - loss: 0.0234',
    'Epoch 148 - loss: 0.0221',
    'Processing batch 2847/5000',
    'Validating checkpoint',
    'Epoch 149 - loss: 0.0198',
  ];

  return {
    connected: true,
    clusterName: 'Training Cluster',
    timestamp: new Date().toISOString(),
    gpu: {
      name: 'RTX 4090',
      utilization,
      memoryPercent,
      temperature,
      powerPercent: Math.round(utilization * 0.85 + 15),
    },
    hasActiveTraining: utilization > 50,
    trainingStatus: utilization > 50
      ? trainingStatuses[Math.floor(Date.now() / 30000) % trainingStatuses.length]
      : undefined,
  };
}

export async function GET() {
  // Check if we have a real backend URL configured (ngrok/Tailscale tunnel)
  const hasRealBackend = process.env.BACKEND_URL && !process.env.BACKEND_URL.includes('localhost');

  // On Vercel without a real backend, return demo data
  const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;
  if (isVercel && !hasRealBackend) {
    return NextResponse.json(getDemoStats());
  }

  // Local development: try to fetch from real backend
  try {
    const response = await fetch(`${BACKEND_URL}/api/lab/gpu-stats`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const rawData: RawGpuStats = await response.json();

    // Return sanitized data
    const sanitized: SanitizedGpuStats = {
      connected: rawData.connected,
      clusterName: 'Training Cluster',
      timestamp: rawData.timestamp,
      gpu: null,
      hasActiveTraining: false,
    };

    if (rawData.connected && rawData.gpu) {
      const gpuName = rawData.gpu.name
        .replace(/NVIDIA\s*/i, '')
        .replace(/GeForce\s*/i, '')
        .trim();

      sanitized.gpu = {
        name: gpuName,
        utilization: rawData.gpu.utilization,
        memoryPercent: rawData.gpu.memoryPercent,
        temperature: rawData.gpu.temperature,
        powerPercent: Math.round((rawData.gpu.powerDraw / rawData.gpu.powerLimit) * 100),
      };

      if (rawData.processes && rawData.processes.length > 0) {
        sanitized.hasActiveTraining = true;
        const trainingProc = rawData.processes.find(p => p.progress);
        if (trainingProc?.progress) {
          sanitized.trainingStatus = sanitizeProgress(trainingProc.progress);
        } else {
          sanitized.trainingStatus = 'Training in progress';
        }
      }
    }

    return NextResponse.json(sanitized);
  } catch (error: any) {
    // Fallback to demo stats if backend unavailable
    return NextResponse.json(getDemoStats());
  }
}
