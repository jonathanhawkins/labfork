/**
 * Compute Network Events API (Server-Sent Events)
 *
 * GET /api/compute/events - Stream real-time network updates
 *
 * Streams:
 * - Network stats every 5 seconds
 * - Task completions as they happen
 * - Device activity updates
 */

import { NextRequest } from "next/server";
import { getOrchestrator } from "@/lib/compute/orchestrator";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET /api/compute/events
 * Stream real-time network events via SSE
 */
export async function GET(req: NextRequest) {
  const encoder = new TextEncoder();
  let intervalId: NodeJS.Timeout | null = null;

  // Create a TransformStream for SSE
  const stream = new TransformStream();
  const writer = stream.writable.getWriter();

  // Helper to send SSE messages
  const sendEvent = async (event: string, data: unknown) => {
    try {
      const message = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
      await writer.write(encoder.encode(message));
    } catch (error) {
      console.error("Error sending SSE event:", error);
    }
  };

  // Helper to send heartbeat (keeps connection alive)
  const sendHeartbeat = async () => {
    try {
      await writer.write(encoder.encode(": heartbeat\n\n"));
    } catch (error) {
      console.error("Error sending heartbeat:", error);
    }
  };

  // Start streaming
  (async () => {
    try {
      const orchestrator = getOrchestrator();

      // Send initial stats immediately
      const initialStats = orchestrator.getNetworkStats();
      await sendEvent("stats", initialStats);

      // Track last task count to detect new completions
      let lastCompletedCount = initialStats.completedToday;

      // Send stats every 5 seconds
      intervalId = setInterval(async () => {
        try {
          const stats = orchestrator.getNetworkStats();

          // Include tier breakdown in stats
          const enrichedStats = {
            ...stats,
            tierBreakdown: {
              power: stats.devicesByTier.power || 0,
              standard: stats.devicesByTier.standard || 0,
              crowd: stats.devicesByTier.crowd || 0,
            },
          };

          await sendEvent("stats", enrichedStats);

          // Check if tasks were completed since last update
          if (stats.completedToday > lastCompletedCount) {
            const newCompletions = stats.completedToday - lastCompletedCount;
            lastCompletedCount = stats.completedToday;

            // Send completion event
            await sendEvent("completion", {
              count: newCompletions,
              totalToday: stats.completedToday,
              creditsAwarded: stats.creditsToday,
            });
          }

          // Send heartbeat
          await sendHeartbeat();
        } catch (error) {
          console.error("Error in stats interval:", error);
        }
      }, 5000);

      // Keep connection alive with heartbeats every 30 seconds
      const heartbeatId = setInterval(async () => {
        await sendHeartbeat();
      }, 30000);

      // Wait for client disconnect
      await new Promise<void>((resolve) => {
        req.signal.addEventListener("abort", () => {
          if (intervalId) clearInterval(intervalId);
          clearInterval(heartbeatId);
          resolve();
        });
      });
    } catch (error) {
      console.error("SSE stream error:", error);
    } finally {
      if (intervalId) clearInterval(intervalId);
      try {
        await writer.close();
      } catch (e) {
        // Ignore close errors
      }
    }
  })();

  // Return SSE response
  return new Response(stream.readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no", // Disable nginx buffering
    },
  });
}
