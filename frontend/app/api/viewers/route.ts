import { NextResponse } from 'next/server';

/**
 * Viewers API - Track live viewers on the /watch page
 *
 * Current implementation: Simulates realistic viewer counts
 * Future: Can be upgraded to use Vercel KV or Upstash Redis for real tracking
 *
 * How real tracking would work:
 * 1. POST /api/viewers - Client sends heartbeat every 30s
 * 2. Store viewer ID + timestamp in Redis
 * 3. GET /api/viewers - Count entries within last 60s
 */

// Simulated base viewer count (would come from Redis in production)
function getSimulatedCount(): number {
  // Time-based variation to make it feel alive
  const hour = new Date().getHours();

  // Higher activity during peak hours (9am-9pm)
  const isPeakHour = hour >= 9 && hour <= 21;
  const baseCount = isPeakHour ? 3 : 1;

  // Add some randomness (±2)
  const variation = Math.floor(Math.random() * 5) - 2;

  return Math.max(1, baseCount + variation);
}

export async function GET() {
  // In production with real tracking:
  // const count = await redis.zcount('viewers', Date.now() - 60000, Date.now());

  const count = getSimulatedCount();

  return NextResponse.json({
    count,
    timestamp: new Date().toISOString(),
  });
}

export async function POST() {
  // In production: Store viewer heartbeat
  // const viewerId = request.cookies.get('viewer_id')?.value || generateId();
  // await redis.zadd('viewers', Date.now(), viewerId);

  return NextResponse.json({
    success: true,
    timestamp: new Date().toISOString(),
  });
}
