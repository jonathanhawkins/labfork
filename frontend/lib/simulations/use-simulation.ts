'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  SimulationParams,
  SimulationResult,
  SimulationResponse,
  UseSimulationState,
} from './types';

const POLL_INTERVAL = 2000; // 2 seconds

export function useSimulation(): UseSimulationState {
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Poll for simulation status
  const pollStatus = useCallback(async (id: string) => {
    try {
      const response = await fetch(`/api/simulations?id=${id}`);
      const data: SimulationResponse = await response.json();

      if (data.success && data.simulation) {
        setSimulation(data.simulation);

        // Stop polling if completed or failed
        if (data.simulation.status === 'completed' || data.simulation.status === 'failed') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
          }
          setIsPolling(false);

          if (data.simulation.status === 'failed') {
            setError(data.simulation.error || 'Simulation failed');
          }
        }
      }
    } catch (err) {
      console.error('Poll error:', err);
      // Don't set error for poll failures, just retry
    }
  }, []);

  // Run a simulation
  const runSimulation = useCallback(async (params: SimulationParams) => {
    // Cancel any existing polling
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    setIsLoading(true);
    setError(null);
    setSimulation(null);

    try {
      abortControllerRef.current = new AbortController();

      const response = await fetch('/api/simulations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
        signal: abortControllerRef.current.signal,
      });

      const data: SimulationResponse = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to start simulation');
      }

      if (data.simulation) {
        setSimulation(data.simulation);

        // If quick mode, we're done
        if (params.mode === 'quick') {
          setIsLoading(false);
          return;
        }

        // For full mode, start polling
        if (data.simulation.status === 'pending' || data.simulation.status === 'running') {
          setIsPolling(true);
          pollIntervalRef.current = setInterval(() => {
            pollStatus(data.simulation!.id);
          }, POLL_INTERVAL);
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // Cancelled, don't set error
        return;
      }
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, [pollStatus]);

  // Cancel current simulation
  const cancelSimulation = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsPolling(false);
    setIsLoading(false);
  }, []);

  return {
    simulation,
    isLoading,
    isPolling,
    error,
    runSimulation,
    cancelSimulation,
  };
}

// Helper to format simulation results for display
export function formatSimulationResults(results: SimulationResult['results']) {
  if (!results) return null;

  const waterResults = results as {
    daily_yield_liters?: number;
    collection_rate_ml_per_hour?: number;
    efficiency_percent?: number;
    peak_temperature_c?: number;
    condensation_rate_g_per_m2_hour?: number;
  };

  return {
    dailyYield: waterResults.daily_yield_liters?.toFixed(2) + ' L/day',
    collectionRate: waterResults.collection_rate_ml_per_hour?.toFixed(1) + ' mL/hr',
    efficiency: waterResults.efficiency_percent + '%',
    peakTemp: waterResults.peak_temperature_c + '°C',
    condensationRate: waterResults.condensation_rate_g_per_m2_hour?.toFixed(1) + ' g/m²/hr',
  };
}
