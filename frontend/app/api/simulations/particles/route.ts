import { NextResponse } from "next/server";

// Proxy particle positions from the MOF Water Harvester simulation
// Port 8006 = new physics-accurate MOF simulation
// Port 8005 = old generic particle sim (fallback)
const WARP_SERVER_URL = process.env.WARP_SERVER_URL || "http://100.100.219.33:8006";

export async function GET() {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000); // 5s timeout

    const response = await fetch(`${WARP_SERVER_URL}/particles`, {
      cache: "no-store",
      headers: { "Accept": "application/json" },
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`Warp server returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      engine: "nvidia_warp_gpu",
      particles: data.positions || [],
      states: data.states || [],
      temperatures: data.temperatures || [],
      count: data.positions?.length || 0,

      // MOF simulation specific data
      phase: data.phase,
      timeOfDay: data.time_of_day,
      timeLabel: data.time_label,
      sorbentTemp: data.sorbent_temp,
      stateCounts: data.state_counts,
    });
  } catch (error) {
    // Return mock data if GPU server unavailable
    console.warn("Warp server unavailable, using mock particles");
    return NextResponse.json({
      success: true,
      engine: "mock_fallback",
      particles: generateMockParticles(500),
      count: 500,
    });
  }
}

function generateMockParticles(count: number): number[][] {
  const particles: number[][] = [];
  for (let i = 0; i < count; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * 0.4;
    particles.push([
      Math.cos(angle) * radius,
      0.3 + Math.random() * 0.6,
      Math.sin(angle) * radius,
    ]);
  }
  return particles;
}
