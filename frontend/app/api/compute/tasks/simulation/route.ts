/**
 * Simulation Task API
 *
 * POST /api/compute/tasks/simulation - Create a simulation task for distributed compute
 *
 * This endpoint creates simulation tasks that can be picked up by power-tier devices
 * (like the RTX 4090) for running physics/CFD simulations.
 */

import { NextResponse } from 'next/server';
import type { SimulationParams } from '@/lib/simulations/types';

interface CreateSimulationTaskRequest {
  labSlug: string;
  params: SimulationParams;
  priority?: number;
}

interface SimulationTask {
  id: string;
  type: 'simulation';
  input: {
    simulationParams: {
      type: string;
      labSlug: string;
      parameters: Record<string, unknown>;
      mode: 'quick' | 'full';
    };
  };
  config: {
    modelId: string;
    maxTokens: number;
    temperature: number;
    minTier: 'power';
  };
  status: 'pending';
  priority: number;
  reward: number;
  createdAt: string;
}

// In-memory task store (would use D1 in production via workers)
const simulationTasks = new Map<string, SimulationTask>();

function generateTaskId(): string {
  return `sim_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * POST - Create a new simulation task
 */
export async function POST(request: Request) {
  try {
    const body: CreateSimulationTaskRequest = await request.json();
    const { labSlug, params, priority = 5 } = body;

    if (!labSlug || !params) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields: labSlug, params' },
        { status: 400 }
      );
    }

    // Validate simulation type
    const validTypes = ['water_harvester', 'droplet_dynamics', 'heat_transfer', 'solar_concentration', 'sorbent_desorption'];
    if (!validTypes.includes(params.type)) {
      return NextResponse.json(
        { success: false, error: `Invalid simulation type. Must be one of: ${validTypes.join(', ')}` },
        { status: 400 }
      );
    }

    // Calculate reward based on mode
    // Full simulations get more reward since they're GPU-intensive
    const baseReward = params.mode === 'full' ? 50 : 10;

    // Create the task
    const task: SimulationTask = {
      id: generateTaskId(),
      type: 'simulation',
      input: {
        simulationParams: {
          type: params.type,
          labSlug,
          parameters: params.parameters,
          mode: params.mode,
        },
      },
      config: {
        modelId: `simulation_${params.type}`,
        maxTokens: 0, // Not applicable for simulations
        temperature: 0,
        minTier: 'power', // Simulations require power tier (4090)
      },
      status: 'pending',
      priority,
      reward: baseReward,
      createdAt: new Date().toISOString(),
    };

    simulationTasks.set(task.id, task);

    // In production, this would be pushed to D1 via Cloudflare Workers
    // For now, we'll also try to dispatch to the local simulations API
    if (params.mode === 'quick') {
      // Quick mode can run locally - dispatch immediately
      try {
        const simResponse = await fetch(`${process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3003'}/api/simulations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params),
        });

        if (simResponse.ok) {
          const simResult = await simResponse.json();
          return NextResponse.json({
            success: true,
            task,
            simulation: simResult.simulation,
            message: 'Quick simulation completed immediately',
          });
        }
      } catch {
        // Fall through to queue the task for pickup
      }
    }

    return NextResponse.json({
      success: true,
      task,
      message: params.mode === 'full'
        ? 'Simulation task queued for power-tier device pickup (RTX 4090)'
        : 'Simulation task created',
    });

  } catch (error) {
    console.error('Failed to create simulation task:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to create simulation task' },
      { status: 500 }
    );
  }
}

/**
 * GET - List pending simulation tasks (for device polling)
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status') || 'pending';
    const labSlug = searchParams.get('lab');
    const limit = parseInt(searchParams.get('limit') || '10');

    // Filter tasks
    let tasks = Array.from(simulationTasks.values())
      .filter(t => t.status === status);

    if (labSlug) {
      tasks = tasks.filter(t => t.input.simulationParams.labSlug === labSlug);
    }

    // Sort by priority (highest first)
    tasks.sort((a, b) => b.priority - a.priority);

    // Limit
    tasks = tasks.slice(0, limit);

    return NextResponse.json({
      success: true,
      tasks,
      total: tasks.length,
    });

  } catch (error) {
    console.error('Failed to list simulation tasks:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to list simulation tasks' },
      { status: 500 }
    );
  }
}
