import { NextResponse } from 'next/server';
import { execSync } from 'child_process';

export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8003';

const REMOTE_HOST = 'doc@100.83.78.111';

/**
 * Get real GPU stats from 4090 via SSH + nvidia-smi
 */
function getRemoteGpuStats() {
  try {
    const raw = execSync(
      `ssh -o ConnectTimeout=3 ${REMOTE_HOST} "/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null"`,
      { encoding: 'utf-8', timeout: 8000 }
    ).trim();

    if (!raw) return null;

    const parts = raw.split(',').map(s => s.trim());
    if (parts.length < 8) return null;

    const memoryUsed = parseInt(parts[3]);
    const memoryTotal = parseInt(parts[4]);

    // Get running processes
    let processes: any[] = [];
    try {
      const procRaw = execSync(
        `ssh -o ConnectTimeout=3 ${REMOTE_HOST} "/usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits 2>/dev/null"`,
        { encoding: 'utf-8', timeout: 5000 }
      ).trim();
      if (procRaw) {
        processes = procRaw.split('\n').filter(Boolean).map(line => {
          const p = line.split(',').map(s => s.trim());
          return { pid: p[0], name: p[1], memoryUsed: `${p[2]} MiB` };
        });
      }
    } catch { /* no processes */ }

    return {
      connected: true,
      timestamp: new Date().toISOString(),
      gpu: {
        name: parts[0],
        driverVersion: parts[1],
        utilization: parseInt(parts[2]),
        memoryUsed,
        memoryTotal,
        memoryPercent: Math.round((memoryUsed / memoryTotal) * 100),
        temperature: parseInt(parts[5]),
        powerDraw: Math.round(parseFloat(parts[6])),
        powerLimit: Math.round(parseFloat(parts[7])),
      },
      processes,
    };
  } catch {
    return null;
  }
}

export async function GET() {
  // Try real backend API first
  const hasRealBackend = BACKEND_URL && !BACKEND_URL.includes('localhost');
  if (hasRealBackend) {
    try {
      const response = await fetch(`${BACKEND_URL}/api/lab/gpu-stats`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        cache: 'no-store',
      });
      if (response.ok) {
        return NextResponse.json(await response.json());
      }
    } catch { /* fall through */ }
  }

  // Try SSH to 4090 directly
  const stats = getRemoteGpuStats();
  if (stats) {
    return NextResponse.json(stats);
  }

  // No connection available
  return NextResponse.json({
    connected: false,
    timestamp: new Date().toISOString(),
    gpu: null,
    processes: [],
  });
}
