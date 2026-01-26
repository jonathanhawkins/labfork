import { NextResponse } from 'next/server';

// Backend API URL - server-side routes use localhost since backend runs on same machine
const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8003';

/**
 * Generate demo agent status for public Vercel deployment
 */
function getDemoAgentStatus() {
  const tasks = [
    'Analyzing prosody patterns',
    'Training emotion encoder',
    'Evaluating voice quality',
    'Processing audio samples',
    'Optimizing model weights',
    'Reviewing code changes',
    'Testing API endpoints',
    'Generating test cases',
  ];

  const outputs = [
    'Epoch 147 complete - loss: 0.0234',
    'Found 3 relevant papers',
    'Implementing feature request #42',
    'Running unit tests...',
    'Refactoring emotion module',
    'Validating audio quality metrics',
  ];

  // Simulate realistic agent states
  const now = Date.now();
  const cyclePosition = (now / 30000) % 8; // 30 second cycles

  const agents = [
    {
      id: 'lab-manager',
      name: 'Lab-Manager',
      status: cyclePosition >= 2 && cyclePosition < 6 ? 'working' : 'idle',
      task: cyclePosition >= 2 && cyclePosition < 6 ? tasks[Math.floor(now / 45000) % tasks.length] : undefined,
      lastOutput: outputs[Math.floor(cyclePosition + 1) % outputs.length],
    },
  ];

  return {
    connected: true,
    timestamp: new Date().toISOString(),
    agents,
  };
}

export async function GET() {
  // Check if we have a real backend URL configured (ngrok/Tailscale tunnel)
  const hasRealBackend = BACKEND_URL && !BACKEND_URL.includes('localhost');

  // On Vercel without a real backend, return demo data
  const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;
  if (isVercel && !hasRealBackend) {
    return NextResponse.json(getDemoAgentStatus());
  }

  // Local development: try to fetch from real backend
  try {
    const response = await fetch(`${BACKEND_URL}/api/lab/agent-status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    // Fallback to demo status if backend unavailable
    return NextResponse.json(getDemoAgentStatus());
  }
}
