'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

// Types for CFD data
interface TemperatureData {
  min_K: number;
  max_K: number;
  avg_K: number;
  min_C: number;
  max_C: number;
  avg_C: number;
}

interface VelocityData {
  max_m_s: number;
  max_cm_s: number;
}

interface HeatTransferData {
  heat_flux_W_m2: number;
  estimated_yield_ml_hr: number;
  estimated_yield_L_day: number;
}

interface ValidationData {
  mit_paper_yield_L_day: number;
  our_estimate_L_day: number;
  ratio: number;
}

interface ProbeTimeData {
  time: number[];
  values: number[][] | number[][][];
}

interface CFDResults {
  converged: boolean;
  latest_time: number;
  iterations: number;
  temperature: TemperatureData;
  velocity: VelocityData;
  heat_transfer: HeatTransferData;
  validation: ValidationData;
  probes: {
    temperature: ProbeTimeData;
    velocity: ProbeTimeData;
    pressure: ProbeTimeData;
  };
}

// Mock data for development when API is unavailable
const generateMockCFDResults = (): CFDResults => {
  // Generate realistic probe data (temperature over iterations)
  const timeSteps = Array.from({ length: 50 }, (_, i) => (i + 1) * 10);

  // Temperature probes converging to steady state
  const tempValues = timeSteps.map((t) => {
    const progress = Math.min(t / 300, 1);
    return [
      340 - 5 * progress + Math.random() * 0.5, // Near sorbent (hot)
      323 - 3 * progress + Math.random() * 0.3, // Mid-height
      308 + 2 * progress + Math.random() * 0.2, // Near dome (cool)
    ];
  });

  return {
    converged: true,
    latest_time: 500,
    iterations: 500,
    temperature: {
      min_K: 303.15,
      max_K: 343.0,
      avg_K: 318.5,
      min_C: 30.0,
      max_C: 69.9,
      avg_C: 45.4,
    },
    velocity: {
      max_m_s: 0.0285,
      max_cm_s: 2.85,
    },
    heat_transfer: {
      heat_flux_W_m2: 3.47,
      estimated_yield_ml_hr: 0.55,
      estimated_yield_L_day: 0.013,
    },
    validation: {
      mit_paper_yield_L_day: 2.8,
      our_estimate_L_day: 0.013,
      ratio: 0.0047,
    },
    probes: {
      temperature: { time: timeSteps, values: tempValues },
      velocity: { time: timeSteps, values: [] },
      pressure: { time: timeSteps, values: [] },
    },
  };
};

// Gradient bar for temperature visualization
function TemperatureGradient({ minC, maxC }: { minC: number; maxC: number }) {
  return (
    <div className="relative h-8 rounded-lg overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          background: 'linear-gradient(to right, #3b82f6, #22c55e, #eab308, #f97316, #ef4444)',
        }}
      />
      <div className="absolute inset-0 flex justify-between items-center px-2 text-xs font-mono text-white drop-shadow-md">
        <span>{minC.toFixed(1)}°C</span>
        <span>{maxC.toFixed(1)}°C</span>
      </div>
    </div>
  );
}

// 2D heatmap visualization of the cavity
function CavityHeatmap({ temperature }: { temperature: TemperatureData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Create temperature gradient (bottom hot, top cool)
    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, '#ef4444'); // Hot (sorbent)
    gradient.addColorStop(0.3, '#f97316');
    gradient.addColorStop(0.5, '#eab308');
    gradient.addColorStop(0.7, '#22c55e');
    gradient.addColorStop(1, '#3b82f6'); // Cool (dome)

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    // Add convection flow arrows
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);

    // Rising hot air (center)
    drawArrow(ctx, width / 2, height - 20, width / 2, 30);

    // Descending cool air (sides)
    drawArrow(ctx, 30, 30, 30, height - 20);
    drawArrow(ctx, width - 30, 30, width - 30, height - 20);

    // Horizontal flow at top
    drawArrow(ctx, width / 2 + 20, 40, width - 40, 40);
    drawArrow(ctx, width / 2 - 20, 40, 40, 40);

    ctx.setLineDash([]);
  }, [temperature]);

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={200}
        height={200}
        className="rounded-lg border border-gray-600"
      />
      <div className="absolute top-2 left-1/2 -translate-x-1/2 text-xs bg-blue-500/80 px-2 py-0.5 rounded text-white">
        Dome (30°C)
      </div>
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-xs bg-red-500/80 px-2 py-0.5 rounded text-white">
        Sorbent (70°C)
      </div>
    </div>
  );
}

function drawArrow(ctx: CanvasRenderingContext2D, fromX: number, fromY: number, toX: number, toY: number) {
  const headlen = 8;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const angle = Math.atan2(dy, dx);

  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.lineTo(toX - headlen * Math.cos(angle - Math.PI / 6), toY - headlen * Math.sin(angle - Math.PI / 6));
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headlen * Math.cos(angle + Math.PI / 6), toY - headlen * Math.sin(angle + Math.PI / 6));
  ctx.stroke();
}

// Stat card component
function StatCard({ label, value, unit, subtext }: { label: string; value: string | number; unit: string; subtext?: string }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="text-gray-400 text-sm mb-1">{label}</div>
      <div className="text-2xl font-bold text-white">
        {value}
        <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>
      </div>
      {subtext && <div className="text-xs text-gray-500 mt-1">{subtext}</div>}
    </div>
  );
}

// Validation comparison component
function ValidationComparison({ validation }: { validation: ValidationData }) {
  const ratio = validation.ratio;
  const percentOfTarget = (ratio * 100).toFixed(1);

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-lg font-semibold text-white mb-4">Validation vs MIT Paper</h3>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">Our CFD Estimate</span>
            <span className="text-yellow-400 font-mono">{validation.our_estimate_L_day.toFixed(3)} L/day</span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-yellow-500 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(ratio * 100, 100)}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-400">MIT Paper Result</span>
            <span className="text-green-400 font-mono">{validation.mit_paper_yield_L_day} L/day</span>
          </div>
          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
            <div className="h-full bg-green-500 rounded-full" style={{ width: '100%' }} />
          </div>
        </div>

        <div className="pt-2 border-t border-gray-700">
          <div className="text-sm text-gray-400">
            Current model achieves <span className="text-yellow-400 font-bold">{percentOfTarget}%</span> of MIT&apos;s reported yield
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Note: MIT paper uses MOF-801 with optimized geometry and full humidity/desorption cycle.
            Our simplified 2D CFD focuses on heat transfer only.
          </div>
        </div>
      </div>
    </div>
  );
}

// Main component
export default function CFDResultsVisualization() {
  const [results, setResults] = useState<CFDResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);

  const fetchResults = useCallback(async () => {
    try {
      const response = await fetch('/api/cfd/results');
      if (!response.ok) {
        throw new Error('CFD API unavailable');
      }
      const data = await response.json();
      setResults(data);
      setIsLive(true);
      setError(null);
    } catch {
      // Use mock data when API is unavailable
      setResults(generateMockCFDResults());
      setIsLive(false);
      setError('Using simulated data (CFD API unavailable)');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchResults();
    // Poll every 5 seconds if live
    const interval = setInterval(fetchResults, 5000);
    return () => clearInterval(interval);
  }, [fetchResults]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-xl">
        <div className="text-gray-400">Loading CFD results...</div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-900 rounded-xl">
        <div className="text-red-400">No CFD data available</div>
      </div>
    );
  }

  // Prepare chart data for temperature probes
  const probeChartData = results.probes.temperature.time.map((t, i) => {
    const values = results.probes.temperature.values[i] as number[];
    return {
      iteration: t,
      nearSorbent: values?.[0] ? values[0] - 273.15 : null, // Convert K to C
      midHeight: values?.[1] ? values[1] - 273.15 : null,
      nearDome: values?.[2] ? values[2] - 273.15 : null,
    };
  });

  return (
    <div className="bg-gray-900 rounded-xl p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">OpenFOAM CFD Results</h2>
          <p className="text-gray-400 text-sm">MOF Water Harvester Heat Transfer Simulation</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
            isLive ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
          }`}>
            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'}`} />
            {isLive ? 'Live Data' : 'Simulated'}
          </div>
          <div className={`px-3 py-1 rounded-full text-sm ${
            results.converged ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
          }`}>
            {results.converged ? '✓ Converged' : '⟳ Running'}
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-2 text-yellow-400 text-sm">
          {error}
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Max Temperature"
          value={results.temperature.max_C.toFixed(1)}
          unit="°C"
          subtext="Sorbent bed"
        />
        <StatCard
          label="Min Temperature"
          value={results.temperature.min_C.toFixed(1)}
          unit="°C"
          subtext="Dome surface"
        />
        <StatCard
          label="Max Velocity"
          value={results.velocity.max_cm_s.toFixed(2)}
          unit="cm/s"
          subtext="Convection flow"
        />
        <StatCard
          label="Heat Flux"
          value={results.heat_transfer.heat_flux_W_m2.toFixed(2)}
          unit="W/m²"
          subtext="Through air gap"
        />
      </div>

      {/* Temperature Gradient */}
      <div>
        <div className="text-sm text-gray-400 mb-2">Temperature Distribution</div>
        <TemperatureGradient minC={results.temperature.min_C} maxC={results.temperature.max_C} />
      </div>

      {/* Main visualization area */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Cavity heatmap */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Cavity Heat Map</h3>
          <div className="flex justify-center">
            <CavityHeatmap temperature={results.temperature} />
          </div>
          <div className="mt-4 text-xs text-gray-500 text-center">
            Arrows show natural convection flow pattern
          </div>
        </div>

        {/* Validation comparison */}
        <ValidationComparison validation={results.validation} />
      </div>

      {/* Temperature probe chart */}
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4">Probe Temperature History</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={probeChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="iteration"
                stroke="#9ca3af"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                label={{ value: 'Iteration', position: 'bottom', fill: '#9ca3af' }}
              />
              <YAxis
                stroke="#9ca3af"
                tick={{ fill: '#9ca3af', fontSize: 12 }}
                label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="nearSorbent"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.3}
                name="Near Sorbent"
              />
              <Area
                type="monotone"
                dataKey="midHeight"
                stroke="#eab308"
                fill="#eab308"
                fillOpacity={0.3}
                name="Mid Height"
              />
              <Area
                type="monotone"
                dataKey="nearDome"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.3}
                name="Near Dome"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Yield estimation */}
      <div className="bg-gradient-to-r from-blue-500/10 to-green-500/10 rounded-lg p-4 border border-blue-500/30">
        <h3 className="text-lg font-semibold text-white mb-2">Estimated Water Yield</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400">{results.heat_transfer.estimated_yield_ml_hr.toFixed(2)}</div>
            <div className="text-sm text-gray-400">mL/hour</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400">{(results.heat_transfer.estimated_yield_ml_hr * 24).toFixed(1)}</div>
            <div className="text-sm text-gray-400">mL/day</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-yellow-400">{results.heat_transfer.estimated_yield_L_day.toFixed(3)}</div>
            <div className="text-sm text-gray-400">L/day</div>
          </div>
        </div>
        <div className="mt-4 text-xs text-gray-500">
          Based on Fourier&apos;s law heat transfer through 30cm air gap with k=0.026 W/(m·K).
          Assumes 0.09 m² active area and complete condensation.
        </div>
      </div>

      {/* Simulation details */}
      <div className="text-xs text-gray-500 flex flex-wrap gap-4">
        <span>Solver: buoyantSimpleFoam</span>
        <span>Mesh: 3600 cells (2D)</span>
        <span>Iterations: {results.iterations}</span>
        <span>Time: {results.latest_time}</span>
      </div>
    </div>
  );
}
