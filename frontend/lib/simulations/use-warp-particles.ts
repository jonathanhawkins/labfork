import { useState, useEffect, useRef, useCallback } from "react";

export interface WarpParticleData {
  positions: number[][];
  states?: number[];
  temperatures?: number[];
  count: number;
  engine: string;

  // MOF simulation specific
  phase?: string;           // adsorbing, heating, releasing, condensing, dripping
  timeOfDay?: number;       // 0-1
  timeLabel?: string;       // "12:00"
  sorbentTemp?: number;     // celsius
  stateCounts?: Record<string, number>;
}

interface UseWarpParticlesOptions {
  enabled?: boolean;
  pollInterval?: number; // ms
}

export function useWarpParticles(options: UseWarpParticlesOptions = {}) {
  const { enabled = true, pollInterval = 50 } = options; // 20 FPS default

  const [particles, setParticles] = useState<WarpParticleData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchParticles = useCallback(async () => {
    try {
      const response = await fetch("/api/simulations/particles");
      if (!response.ok) throw new Error("Failed to fetch particles");

      const data = await response.json();
      if (data.success) {
        setParticles({
          positions: data.particles,
          states: data.states,
          temperatures: data.temperatures,
          count: data.count,
          engine: data.engine,

          // MOF simulation specific
          phase: data.phase,
          timeOfDay: data.timeOfDay,
          timeLabel: data.timeLabel,
          sorbentTemp: data.sorbentTemp,
          stateCounts: data.stateCounts,
        });
        setIsConnected(data.engine === "nvidia_warp_gpu");
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    // Initial fetch
    fetchParticles();

    // Start polling
    intervalRef.current = setInterval(fetchParticles, pollInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled, pollInterval, fetchParticles]);

  return {
    particles,
    isConnected,
    error,
    refetch: fetchParticles,
  };
}
