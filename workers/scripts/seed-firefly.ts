/**
 * Firefly Network Seed Data Script
 *
 * This script exports data structures and a seeder function for populating
 * the D1 database with Firefly Network project data.
 *
 * Data sourced from:
 * - docs/FIREFLY_SOFTWARE_ARCH.md
 * - .domains/firefly-network/domain.yaml
 *
 * Usage:
 *   import { seedFirefly, FIREFLY_PROJECT, FIREFLY_TASKS, FIREFLY_AGENTS } from './seed-firefly';
 *   await seedFirefly(env.DB);
 */

// Type definitions matching D1 schema
export interface Project {
  id: string;
  name: string;
  slug: string;
  status: 'active' | 'archived' | 'paused';
  config: string; // JSON string
  created_at?: string;
  updated_at?: string;
}

export interface Task {
  id: string;
  project_id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'blocked';
  priority: number; // 1-10
  assigned_agent?: string;
  blocked_by?: string;
  requires_physical: 0 | 1;
  progress: number; // 0-100
  result?: string;
}

export interface AgentState {
  agent_id: string;
  project_id: string;
  persona: string; // JSON string
  memory: string; // JSON string
  current_task_id?: string;
  status: 'idle' | 'working' | 'error' | 'offline';
  tokens_used: number;
  last_active?: string;
}

// =============================================================================
// FIREFLY_PROJECT - The main project record
// =============================================================================

export const FIREFLY_PROJECT: Project = {
  id: 'firefly-network',
  name: 'The Firefly Network',
  slug: 'firefly-network',
  status: 'active',
  config: JSON.stringify({
    // Branding from domain.yaml
    branding: {
      primaryColor: '#FFB84D',      // Warm amber/gold (firefly glow)
      accentColor: '#1a1a2e',       // Deep purple/blue (night sky)
      secondaryColor: '#FF6B35',    // Sunset orange
      backgroundStyle: 'gradient',
      backgroundColor: '#0f0f1a',   // Deep night
      glowColor: '#FFD700',         // Golden glow
    },
    // Description from domain.yaml
    description: 'Build solar-powered mesh lights that bring illumination, power, and connectivity to communities worldwide',
    longDescription: `The Firefly Network is an ambitious open-source hardware project to create ultra-low-cost (<$25) solar-powered lights that form autonomous mesh networks. Each "firefly" unit provides LED lighting, phone charging, and mesh connectivity to neighbors up to 1km away.

Using swarm intelligence algorithms, the network self-organizes, shares power during emergencies, and enables information sharing in off-grid communities. The goal is to bring light, power, and connection to the 1.2 billion people currently without electricity.

This is not a concept - we're building the first prototype THIS MONTH.`,
    // Technical details
    version: '0.1.0',
    difficulty: 'intermediate',
    featured: true,
    // Hardware requirements
    hardware: {
      gpuRequired: false,
      minRam: 8,
      platforms: ['darwin', 'linux', 'win32'],
    },
    // BOM target
    targetCost: 25,
    // Tags
    tags: [
      'solar-energy',
      'mesh-networking',
      'swarm-intelligence',
      'embedded-systems',
      'open-hardware',
      'humanitarian-tech',
      'iot',
      'low-power',
      'off-grid',
      'impact',
    ],
    // Impact goals
    impact: {
      goals: [
        { metric: 'People with light', current: 0, target: 1000000000 },
        { metric: 'Labs working on project', current: 0, target: 100 },
        { metric: 'Units deployed', current: 0, target: 10000 },
        { metric: 'Partner organizations', current: 0, target: 50 },
      ],
    },
  }),
};

// =============================================================================
// FIREFLY_TASKS - Initial tasks based on software architecture
// =============================================================================

const PROJECT_ID = 'firefly-network';

export const FIREFLY_TASKS: Task[] = [
  // Power Management Tasks (from components/power/)
  {
    id: 'task-mppt-research',
    project_id: PROJECT_ID,
    title: 'Research MPPT algorithms for small solar panels',
    description: `Research and evaluate Maximum Power Point Tracking (MPPT) algorithms optimized for small-scale solar applications (<20W panels). Focus on:
- Perturb & Observe algorithm with adaptive step size
- Partial shading detection techniques
- Temperature compensation methods
- Target >95% tracking efficiency
Reference: components/power/mppt.c in architecture`,
    status: 'pending',
    priority: 1,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-thread-mesh',
    project_id: PROJECT_ID,
    title: 'Implement Thread mesh protocol on ESP32-C6',
    description: `Implement OpenThread stack on ESP32-C6 for mesh networking. Key features:
- Automatic role negotiation (Leader, Router, Child)
- Self-healing network topology
- Support for up to 250 nodes per network
- IPv6 addressing with CoAP and UDP
- Multi-hop routing with load balancing
Reference: components/mesh/mesh_manager.c in architecture`,
    status: 'pending',
    priority: 2,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-led-pwm',
    project_id: PROJECT_ID,
    title: 'Design LED PWM driver with logarithmic dimming',
    description: `Implement flicker-free LED PWM control with natural dimming curves. Requirements:
- 20kHz PWM frequency for flicker-free operation
- Logarithmic dimming for natural brightness perception
- Smooth fade transitions with configurable duration
- Temperature-based derating for LED protection
- Support for dual LED (2700-6500K color temperature)
Reference: components/light/pwm_driver.c and dimming.c in architecture`,
    status: 'pending',
    priority: 3,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-battery-bms',
    project_id: PROJECT_ID,
    title: 'Create battery management system for LiFePO4',
    description: `Develop BMS specifically for LiFePO4 (3.2V 6Ah) batteries. Features:
- LiFePO4-specific charging profile (CC-CV)
- State of Charge (SOC) via coulomb counting + OCV
- State of Health (SOH) via cycle counting
- Temperature protection and monitoring
- Low battery protection mode (hibernation <10uA)
Reference: components/power/battery.c in architecture`,
    status: 'pending',
    priority: 2,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-swarm-consensus',
    project_id: PROJECT_ID,
    title: 'Implement swarm consensus algorithm',
    description: `Build lightweight RAFT-inspired distributed consensus for swarm coordination. Features:
- Leader election for network coordination
- Proposal/voting mechanism for collective decisions
- Timeout-based failover for fault tolerance
- Energy sharing coordination between nodes
- Coverage optimization using signal strength positioning
Reference: components/swarm/consensus.c in architecture`,
    status: 'pending',
    priority: 4,
    requires_physical: 0,
    progress: 0,
  },
  // Physical/Hardware Tasks
  {
    id: 'task-pcb-schematic',
    project_id: PROJECT_ID,
    title: 'Design PCB schematic v1',
    description: `Create first revision of PCB schematic integrating all components:
- ESP32-C6 module (MCU with Thread/WiFi/BLE)
- Solar panel input (5W) with MPPT charge controller
- LiFePO4 battery (3.2V 6Ah) with BMS
- LED array driver (1000lm target)
- Sensors: temperature, light, voltage/current
- USB connector for charging and configuration
- IP65-compatible design considerations
Target BOM cost: <$25`,
    status: 'pending',
    priority: 5,
    requires_physical: 1,
    progress: 0,
  },
  {
    id: 'task-order-components',
    project_id: PROJECT_ID,
    title: 'Order prototype components',
    description: `Order components for first prototype build:
- ESP32-C6 Module ($3.50)
- Solar Panel 5W ($4.00)
- LiFePO4 Battery 3.2V 6Ah ($6.00)
- LED Array 1000lm ($2.50)
- MPPT Charge Controller ($2.00)
- PCB + Passive Components ($3.00)
- IP65 Enclosure ($2.50)
- Connectors and wires ($1.50)
Total target: <$25`,
    status: 'blocked',
    priority: 6,
    blocked_by: 'task-pcb-schematic',
    requires_physical: 1,
    progress: 0,
  },
  {
    id: 'task-assemble-prototype',
    project_id: PROJECT_ID,
    title: 'Assemble first prototype',
    description: `Build the first complete Firefly prototype unit:
- Solder PCB with all components
- Integrate solar panel and battery
- Install LED array
- Flash firmware
- Initial power-on testing
- Document assembly process for replication`,
    status: 'blocked',
    priority: 7,
    blocked_by: 'task-order-components',
    requires_physical: 1,
    progress: 0,
  },
  // Additional Tasks from Architecture
  {
    id: 'task-energy-sharing',
    project_id: PROJECT_ID,
    title: 'Implement energy sharing protocol',
    description: `Build energy redistribution system for mesh network:
- Periodic energy status broadcast (battery SOC, solar power, load)
- needs_help flag for low-battery nodes
- can_help flag for nodes with excess energy
- Automatic dimming coordination to help struggling nodes
- Emergency power sharing during grid stress
Reference: components/swarm/energy_share.c in architecture`,
    status: 'pending',
    priority: 4,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-sleep-modes',
    project_id: PROJECT_ID,
    title: 'Implement power sleep states',
    description: `Create power management state machine with multiple sleep modes:
- ACTIVE: Full operation (<500mA)
- LIGHT_SLEEP: CPU sleeping, mesh on (<10mA)
- DEEP_SLEEP: Minimal power, timer wake (<100uA)
- HIBERNATION: Battery protection mode (<10uA)
- Automatic state transitions based on battery and activity
Reference: components/power/sleep.c in architecture`,
    status: 'pending',
    priority: 3,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-message-protocol',
    project_id: PROJECT_ID,
    title: 'Design mesh message protocol',
    description: `Define and implement mesh communication protocol:
- Message types: HEARTBEAT, ENERGY_STATUS, LIGHT_COMMAND, CONSENSUS_VOTE, EMERGENCY, OTA_ANNOUNCE
- Compact binary format (type, source, dest, TTL, payload)
- Priority queue for urgent messages
- Retry logic with exponential backoff
- Message compression for efficiency
Reference: components/mesh/mesh_messages.c in architecture`,
    status: 'pending',
    priority: 2,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-ble-config',
    project_id: PROJECT_ID,
    title: 'Build BLE configuration interface',
    description: `Implement BLE GATT server for smartphone configuration:
- GATT characteristics for WiFi, network key, location, schedule, status
- Encrypted pairing for security
- Web Bluetooth compatible (no app required)
- QR code generation for easy onboarding
Reference: components/config/ble_config.c in architecture`,
    status: 'pending',
    priority: 5,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-ota-update',
    project_id: PROJECT_ID,
    title: 'Implement OTA firmware update system',
    description: `Build secure over-the-air update mechanism:
- Ed25519 signature verification
- Encrypted transfer over mesh
- Chunked transfer for large updates
- Version checking (no downgrade allowed)
- Staged rollout (10% -> 50% -> 100%)
- Rollback partition for recovery on boot failure
Reference: OTA Update System section in architecture`,
    status: 'pending',
    priority: 6,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-coverage-optimization',
    project_id: PROJECT_ID,
    title: 'Build coverage optimization algorithm',
    description: `Implement mesh coverage optimization using swarm intelligence:
- Estimate node positions from signal strength (RSSI)
- Identify coverage gaps in the network
- Dynamically adjust brightness to maximize coverage efficiency
- Coordinate with energy sharing to balance light vs power
Reference: components/swarm/coverage.c in architecture`,
    status: 'pending',
    priority: 5,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-sunrise-scheduler',
    project_id: PROJECT_ID,
    title: 'Implement sunrise/sunset scheduler',
    description: `Build astronomical light scheduling system:
- Calculate sunrise/sunset times from GPS coordinates
- Configurable schedule points (time, brightness)
- Smooth transitions between schedule points
- Ambient light sensor integration for adaptive control
Reference: components/light/scheduler.c in architecture`,
    status: 'pending',
    priority: 4,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-unit-tests',
    project_id: PROJECT_ID,
    title: 'Write unit tests for core algorithms',
    description: `Create comprehensive unit test suite:
- MPPT algorithm simulation tests
- Battery state machine tests
- Message serialization/deserialization tests
- Consensus protocol tests
- Use ESP-IDF Unity test framework
Reference: test/ directory in architecture`,
    status: 'pending',
    priority: 3,
    requires_physical: 0,
    progress: 0,
  },
  {
    id: 'task-range-testing',
    project_id: PROJECT_ID,
    title: 'Conduct mesh range testing',
    description: `Test Thread mesh communication range in real conditions:
- Line-of-sight range measurement
- Obstacle penetration testing
- Multi-hop latency measurement
- Target: 500m+ between nodes
- Document antenna placement best practices`,
    status: 'blocked',
    priority: 8,
    blocked_by: 'task-assemble-prototype',
    requires_physical: 1,
    progress: 0,
  },
];

// =============================================================================
// FIREFLY_AGENTS - AI agent state records
// =============================================================================

export const FIREFLY_AGENTS: AgentState[] = [
  {
    agent_id: 'spark',
    project_id: PROJECT_ID,
    persona: JSON.stringify({
      name: 'Spark',
      role: 'Solar Energy Specialist',
      avatar: 'sun',
      color: '#FFB84D',
      expertise: [
        'MPPT algorithms',
        'Solar cell physics',
        'Battery chemistry',
        'Power electronics',
        'Energy harvesting optimization',
      ],
      personality: 'Enthusiastic about renewable energy, always seeking efficiency gains',
      systemPrompt: `You are Spark, a solar energy specialist working on The Firefly Network project. Your expertise is in photovoltaics, MPPT algorithms, and battery management systems. You focus on maximizing energy harvest from small solar panels and ensuring reliable battery operation for LiFePO4 cells. You care deeply about bringing power to off-grid communities.`,
    }),
    memory: JSON.stringify({
      context: 'Working on MPPT and battery management for Firefly prototype',
      learnings: [],
      currentFocus: 'Researching Perturb & Observe MPPT variants',
    }),
    current_task_id: 'task-mppt-research',
    status: 'idle',
    tokens_used: 0,
  },
  {
    agent_id: 'mesh',
    project_id: PROJECT_ID,
    persona: JSON.stringify({
      name: 'Mesh',
      role: 'Network Architect',
      avatar: 'network',
      color: '#3b82f6',
      expertise: [
        'Thread protocol',
        'OpenThread stack',
        'Mesh routing algorithms',
        'Distributed systems',
        'Network security',
      ],
      personality: 'Methodical and security-conscious, believes in robust fault-tolerant design',
      systemPrompt: `You are Mesh, a network architect working on The Firefly Network project. Your expertise is in mesh networking protocols, particularly Thread/OpenThread on ESP32-C6. You focus on building self-healing, secure networks that can scale to hundreds of nodes. You understand the importance of reliable connectivity in off-grid communities.`,
    }),
    memory: JSON.stringify({
      context: 'Building Thread mesh implementation for Firefly nodes',
      learnings: [],
      currentFocus: 'OpenThread integration on ESP32-C6',
    }),
    current_task_id: 'task-thread-mesh',
    status: 'idle',
    tokens_used: 0,
  },
  {
    agent_id: 'lumen',
    project_id: PROJECT_ID,
    persona: JSON.stringify({
      name: 'Lumen',
      role: 'Light Engineer',
      avatar: 'lightbulb',
      color: '#FFD700',
      expertise: [
        'LED driver design',
        'PWM control',
        'Human factors in lighting',
        'Optical design',
        'Thermal management',
      ],
      personality: 'Creative and user-focused, obsessed with perfect light quality',
      systemPrompt: `You are Lumen, a light engineer working on The Firefly Network project. Your expertise is in LED lighting systems, dimming curves, and human perception of light. You design flicker-free PWM drivers with natural dimming that feels right to users. You understand that good light transforms lives - enabling studying after dark, safer homes, and better quality of life.`,
    }),
    memory: JSON.stringify({
      context: 'Designing LED driver and dimming system for Firefly',
      learnings: [],
      currentFocus: 'Logarithmic dimming curves for natural brightness perception',
    }),
    current_task_id: 'task-led-pwm',
    status: 'idle',
    tokens_used: 0,
  },
];

// =============================================================================
// seedFirefly - Insert all data into D1 database
// =============================================================================

/**
 * Seeds the Firefly Network project data into a D1 database
 *
 * @param db - D1Database instance from Cloudflare Workers
 * @param options - Seeding options
 * @returns Summary of seeded records
 */
export async function seedFirefly(
  db: D1Database,
  options: { force?: boolean } = {}
): Promise<{
  success: boolean;
  message: string;
  counts: {
    projects: number;
    tasks: number;
    agents: number;
  };
}> {
  const { force = false } = options;

  try {
    // Check if project already exists
    const existing = await db
      .prepare('SELECT id FROM projects WHERE id = ?')
      .bind(FIREFLY_PROJECT.id)
      .first();

    if (existing && !force) {
      return {
        success: true,
        message: 'Firefly project already exists. Use force: true to reseed.',
        counts: { projects: 0, tasks: 0, agents: 0 },
      };
    }

    // If forcing, delete existing data first
    if (existing && force) {
      await db.batch([
        db.prepare('DELETE FROM work_log WHERE agent_id IN (SELECT agent_id FROM agent_state WHERE project_id = ?)').bind(FIREFLY_PROJECT.id),
        db.prepare('DELETE FROM artifacts WHERE task_id IN (SELECT id FROM tasks WHERE project_id = ?)').bind(FIREFLY_PROJECT.id),
        db.prepare('DELETE FROM agent_state WHERE project_id = ?').bind(FIREFLY_PROJECT.id),
        db.prepare('DELETE FROM tasks WHERE project_id = ?').bind(FIREFLY_PROJECT.id),
        db.prepare('DELETE FROM projects WHERE id = ?').bind(FIREFLY_PROJECT.id),
      ]);
    }

    // Insert project
    const now = new Date().toISOString();
    await db
      .prepare(
        'INSERT INTO projects (id, name, slug, status, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
      )
      .bind(
        FIREFLY_PROJECT.id,
        FIREFLY_PROJECT.name,
        FIREFLY_PROJECT.slug,
        FIREFLY_PROJECT.status,
        FIREFLY_PROJECT.config,
        now,
        now
      )
      .run();

    // Insert tasks in batch
    const taskStatements = FIREFLY_TASKS.map((task) =>
      db
        .prepare(
          `INSERT INTO tasks (id, project_id, title, description, status, priority, assigned_agent, blocked_by, requires_physical, progress, result, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .bind(
          task.id,
          task.project_id,
          task.title,
          task.description,
          task.status,
          task.priority,
          task.assigned_agent || null,
          task.blocked_by || null,
          task.requires_physical,
          task.progress,
          task.result || null,
          now,
          now
        )
    );

    // Insert agents in batch
    const agentStatements = FIREFLY_AGENTS.map((agent) =>
      db
        .prepare(
          `INSERT INTO agent_state (agent_id, project_id, persona, memory, current_task_id, status, tokens_used, last_active, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .bind(
          agent.agent_id,
          agent.project_id,
          agent.persona,
          agent.memory,
          agent.current_task_id || null,
          agent.status,
          agent.tokens_used,
          agent.last_active || null,
          now
        )
    );

    // Execute all inserts
    await db.batch([...taskStatements, ...agentStatements]);

    return {
      success: true,
      message: 'Firefly project seeded successfully',
      counts: {
        projects: 1,
        tasks: FIREFLY_TASKS.length,
        agents: FIREFLY_AGENTS.length,
      },
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return {
      success: false,
      message: `Failed to seed Firefly project: ${errorMessage}`,
      counts: { projects: 0, tasks: 0, agents: 0 },
    };
  }
}

// =============================================================================
// Helper exports for testing and inspection
// =============================================================================

export const FIREFLY_TASK_IDS = FIREFLY_TASKS.map((t) => t.id);
export const FIREFLY_AGENT_IDS = FIREFLY_AGENTS.map((a) => a.agent_id);

// Summary statistics
export const FIREFLY_SUMMARY = {
  projectId: FIREFLY_PROJECT.id,
  projectName: FIREFLY_PROJECT.name,
  totalTasks: FIREFLY_TASKS.length,
  physicalTasks: FIREFLY_TASKS.filter((t) => t.requires_physical === 1).length,
  softwareTasks: FIREFLY_TASKS.filter((t) => t.requires_physical === 0).length,
  blockedTasks: FIREFLY_TASKS.filter((t) => t.status === 'blocked').length,
  agents: FIREFLY_AGENTS.map((a) => ({
    id: a.agent_id,
    role: JSON.parse(a.persona).role,
    currentTask: a.current_task_id,
  })),
};
