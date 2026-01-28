"use client";

import React, { useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { CSS2DRenderer, CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";

// Import activity system and props
import { ActivityWithConfig, PropType } from "./lab/activities";
import {
  createProp3D,
  animateProp3D,
  disposeProp3D,
  getProp3DGroup,
  Prop3DRefs,
} from "./lab/props";
import {
  animateEmotionVerify3D,
  EmotionVerify3DRefs,
} from "./lab/props/EmotionVerify3D";
import {
  animateSupercomputer3D,
  Supercomputer3DRefs,
  GpuStatsData,
  TrainingData,
} from "./lab/props/Supercomputer3D";

// Agent definitions
interface Agent {
  id: string;
  name: string;
  color: number;
  position: [number, number, number];
  task?: string;
  status: "idle" | "working" | "thinking";
}

interface Lab3DProps {
  agents?: Agent[];
  activities?: ActivityWithConfig[];
  onAgentClick?: (agent: Agent) => void;
  onComputerClick?: () => void;  // Called when supercomputer is clicked
  showDemoProps?: boolean;  // Show demo props for testing
}

// Pastel colors - Katamari Damacy style
const COLORS = {
  sky: 0xffeef5,
  ground: 0xb8e6c1,
  groundAccent: 0x9dd4a8,
  codex: 0xffb3ba,    // Soft pink
  opus: 0xbae1ff,     // Soft blue
  explorer: 0xffffba, // Soft yellow
  planner: 0xbaffc9,  // Soft green
  labManager: 0x4ecdc4, // Teal/cyan - 4090 lab-manager
  desk: 0xffe4b8,     // Soft orange/wood
  screen: 0x2a2a3a,   // Dark screen
  screenGlow: 0x66ffaa,
  particles: 0xffccee,
};

const DEFAULT_AGENTS: Agent[] = [
  { id: "codex", name: "Codex", color: COLORS.codex, position: [-3, 0, -2], task: "Writing code...", status: "working" },
  { id: "opus", name: "Opus", color: COLORS.opus, position: [3, 0, -2], task: "Analyzing data...", status: "working" },
  { id: "explorer", name: "Explorer", color: COLORS.explorer, position: [-3, 0, 3], task: "Searching files...", status: "thinking" },
  { id: "planner", name: "Planner", color: COLORS.planner, position: [3, 0, 3], task: "Planning tasks...", status: "idle" },
  // 4090 Agents (FREE Ollama)
  { id: "lab-manager", name: "Lab-Manager", color: COLORS.labManager, position: [0, 0, 4], task: "Executing tasks...", status: "working" },
];

// Manager agent colors
const MANAGER_COLOR = 0xf472b6;  // Pink/magenta for manager

// Patrol points for manager to visit agents
const MANAGER_PATROL_POINTS: [number, number, number][] = [
  [-5, 0, -3.5],  // In front of supercomputer (not through it)
  [-2, 0, -1],    // Near Codex
  [2, 0, -1],     // Near Opus
  [-2, 0, 2],     // Near Explorer
  [2, 0, 2],      // Near Planner
];

// Work slots around the lab - agents claim these when assigned tasks
// Each slot is positioned near equipment/props where agents can work
interface WorkSlot {
  id: string;
  position: [number, number, number];
  nearProp: string;  // Which equipment this slot is near
  facing: number;    // Y rotation to face the equipment (radians)
}

const WORK_SLOTS: WorkSlot[] = [
  // Slots near supercomputer (back-left corner)
  { id: 'supercomputer-1', position: [-4.5, 0, -4], nearProp: 'supercomputer', facing: Math.PI * 0.75 },
  { id: 'supercomputer-2', position: [-5.5, 0, -3], nearProp: 'supercomputer', facing: Math.PI * 0.5 },

  // Slots near server rack (back-right corner)
  { id: 'server-1', position: [4.5, 0, -4], nearProp: 'server', facing: Math.PI * 0.25 },
  { id: 'server-2', position: [5.5, 0, -3], nearProp: 'server', facing: -Math.PI * 0.5 },

  // Slots near microphone (front-left corner)
  { id: 'mic-1', position: [-4.5, 0, 4], nearProp: 'microphone', facing: -Math.PI * 0.75 },
  { id: 'mic-2', position: [-5.5, 0, 3], nearProp: 'microphone', facing: Math.PI * 0.5 },

  // Slots near speaker (front-right corner)
  { id: 'speaker-1', position: [4.5, 0, 4], nearProp: 'speaker', facing: -Math.PI * 0.25 },
  { id: 'speaker-2', position: [5.5, 0, 3], nearProp: 'speaker', facing: -Math.PI * 0.5 },

  // Central work area slots (for general tasks)
  { id: 'center-1', position: [-1.5, 0, 0], nearProp: 'hub', facing: 0 },
  { id: 'center-2', position: [1.5, 0, 0], nearProp: 'hub', facing: Math.PI },
  { id: 'center-3', position: [0, 0, 1.5], nearProp: 'hub', facing: -Math.PI * 0.5 },
  { id: 'center-4', position: [0, 0, -1.5], nearProp: 'hub', facing: Math.PI * 0.5 },
];

// Icon SVGs for status indicators (Sims-style)
const TASK_ICONS: Record<string, string> = {
  // Code/Writing
  code: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  // Search/Exploring
  search: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>`,
  // Brain/Thinking/Analyzing
  brain: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>`,
  // List/Planning
  list: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="6" height="6" x="3" y="3" rx="1"/><path d="M3 13h18"/><path d="M3 17h18"/><path d="M3 21h18"/><path d="M13 3h6"/><path d="M13 7h6"/></svg>`,
  // File/Document
  file: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/></svg>`,
  // Zap/Processing
  zap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/></svg>`,
  // Pause/Idle
  pause: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>`,
  // Sparkles
  sparkles: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg>`,
};

// Get icon based on task text
const getTaskIcon = (task: string, status: string): string => {
  const taskLower = task.toLowerCase();

  if (status === "idle") return TASK_ICONS.pause;

  if (taskLower.includes("code") || taskLower.includes("writing") || taskLower.includes("implement")) {
    return TASK_ICONS.code;
  }
  if (taskLower.includes("search") || taskLower.includes("explor") || taskLower.includes("find") || taskLower.includes("locat")) {
    return TASK_ICONS.search;
  }
  if (taskLower.includes("think") || taskLower.includes("analyz") || taskLower.includes("review")) {
    return TASK_ICONS.brain;
  }
  if (taskLower.includes("plan") || taskLower.includes("task") || taskLower.includes("schedul") || taskLower.includes("creat")) {
    return TASK_ICONS.list;
  }
  if (taskLower.includes("file") || taskLower.includes("document") || taskLower.includes("read")) {
    return TASK_ICONS.file;
  }
  if (taskLower.includes("optim") || taskLower.includes("process") || taskLower.includes("render")) {
    return TASK_ICONS.zap;
  }
  if (taskLower.includes("synth") || taskLower.includes("generat")) {
    return TASK_ICONS.sparkles;
  }

  return TASK_ICONS.zap; // Default
};

// Get status color
const getStatusColor = (status: string): string => {
  switch (status) {
    case "working": return "#4ade80"; // Green
    case "thinking": return "#facc15"; // Yellow
    default: return "#94a3b8"; // Gray
  }
};

export default function Lab3D({ agents = DEFAULT_AGENTS, activities = [], onAgentClick, onComputerClick, showDemoProps = true }: Lab3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const labelRendererRef = useRef<CSS2DRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animationRef = useRef<number>(0);
  const timeRef = useRef(0);

  // Store initial agents for scene setup (avoid recreating scene on every agent update)
  const initialAgentsRef = useRef(agents);
  const showManagerRef = useRef(
    agents.some((agent) =>
      agent.id === "manager" || agent.name.toLowerCase().includes("manager")
    )
  );
  const onAgentClickRef = useRef(onAgentClick);
  const agentsRef = useRef(agents);

  // Callback refs
  const onComputerClickRef = useRef(onComputerClick);

  // Keep refs updated without triggering re-render
  useEffect(() => {
    onAgentClickRef.current = onAgentClick;
  }, [onAgentClick]);

  useEffect(() => {
    onComputerClickRef.current = onComputerClick;
  }, [onComputerClick]);

  useEffect(() => {
    agentsRef.current = agents;
  }, [agents]);

  // Agent meshes for animation
  const agentMeshesRef = useRef<Map<string, THREE.Group>>(new Map());
  const agentLabelsRef = useRef<Map<string, CSS2DObject>>(new Map());
  const particlesRef = useRef<THREE.Points[]>([]);
  const screenMeshesRef = useRef<THREE.Mesh[]>([]);

  // Activity props storage
  const propsRef = useRef<Map<string, { type: PropType; refs: Prop3DRefs }>>(new Map());
  const activitiesRef = useRef<ActivityWithConfig[]>([]);

  // Agent target positions (for moving toward props)
  const agentTargetsRef = useRef<Map<string, THREE.Vector3>>(new Map());

  // Slot occupancy tracking: slotId -> agentId
  const slotOccupancyRef = useRef<Map<string, string>>(new Map());
  // Reverse mapping: agentId -> slotId (for quick lookups)
  const agentSlotRef = useRef<Map<string, string>>(new Map());
  // Target rotations for agents (to face equipment)
  const agentTargetRotationRef = useRef<Map<string, number>>(new Map());

  // Manager agent state
  const managerRef = useRef<THREE.Group | null>(null);
  const managerPatrolIndexRef = useRef(0);
  const managerWaitTimeRef = useRef(0);
  const managerTargetRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));

  // Demo props storage (always visible equipment)
  const demoPropsRef = useRef<Map<string, { type: PropType; refs: Prop3DRefs }>>(new Map());

  // GPU stats for diegetic display
  const gpuStatsRef = useRef<GpuStatsData | null>(null);

  // Raycaster for click detection
  const raycasterRef = useRef<THREE.Raycaster>(new THREE.Raycaster());
  const mouseRef = useRef<THREE.Vector2>(new THREE.Vector2());

  // Helper: Find the best available slot for an agent near a target position
  const findBestSlot = useCallback((targetPosition: [number, number, number], preferredProp?: string): WorkSlot | null => {
    const targetVec = new THREE.Vector3(...targetPosition);
    let bestSlot: WorkSlot | null = null;
    let bestScore = Infinity;

    for (const slot of WORK_SLOTS) {
      // Skip occupied slots
      if (slotOccupancyRef.current.has(slot.id)) continue;

      const slotVec = new THREE.Vector3(...slot.position);
      const distance = targetVec.distanceTo(slotVec);

      // Calculate score (lower is better)
      // Prefer slots near the target, with bonus for matching prop type
      let score = distance;
      if (preferredProp && slot.nearProp === preferredProp) {
        score -= 5; // Strong preference for matching prop
      }

      if (score < bestScore) {
        bestScore = score;
        bestSlot = slot;
      }
    }

    return bestSlot;
  }, []);

  // Helper: Claim a slot for an agent
  const claimSlot = useCallback((slotId: string, agentId: string) => {
    // Release any existing slot for this agent
    const existingSlot = agentSlotRef.current.get(agentId);
    if (existingSlot) {
      slotOccupancyRef.current.delete(existingSlot);
    }

    // Claim the new slot
    slotOccupancyRef.current.set(slotId, agentId);
    agentSlotRef.current.set(agentId, slotId);
  }, []);

  // Helper: Release a slot
  const releaseSlot = useCallback((agentId: string) => {
    const slotId = agentSlotRef.current.get(agentId);
    if (slotId) {
      slotOccupancyRef.current.delete(slotId);
      agentSlotRef.current.delete(agentId);
    }
  }, []);

  // Create status indicator label for agent
  const createStatusLabel = useCallback((agent: Agent): CSS2DObject => {
    const container = document.createElement("div");
    container.className = "agent-status-label";
    container.style.cssText = `
      display: flex;
      flex-direction: column;
      align-items: center;
      pointer-events: none;
      transform: translateY(-20px);
    `;

    // Thought bubble with icon
    const bubble = document.createElement("div");
    bubble.className = "status-bubble";
    const glowColor = agent.status === "working" ? "0, 255, 100" : agent.status === "thinking" ? "255, 200, 50" : "150, 150, 150";
    bubble.style.cssText = `
      background: rgba(255, 255, 255, 0.95);
      border-radius: 12px;
      padding: 8px 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.2), 0 0 0 2px ${getStatusColor(agent.status)}, 0 0 ${agent.status === "working" ? "15px" : "8px"} rgba(${glowColor}, 0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      min-width: 36px;
      min-height: 36px;
      animation: float 2s ease-in-out infinite${agent.status === "working" ? ", glow 1.5s ease-in-out infinite" : ""};
      backdrop-filter: blur(4px);
    `;

    // Icon
    const iconDiv = document.createElement("div");
    iconDiv.className = "status-icon";
    iconDiv.style.cssText = `
      width: 20px;
      height: 20px;
      color: ${getStatusColor(agent.status)};
    `;
    iconDiv.innerHTML = getTaskIcon(agent.task || "", agent.status);
    bubble.appendChild(iconDiv);

    // Task text (short)
    const taskText = document.createElement("div");
    taskText.className = "task-text";
    taskText.style.cssText = `
      font-size: 9px;
      font-weight: 600;
      color: #475569;
      max-width: 60px;
      text-align: center;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      margin-left: 4px;
    `;
    const shortTask = (agent.task || "Idle").split(" ").slice(0, 2).join(" ");
    taskText.textContent = shortTask.replace("...", "");
    bubble.appendChild(taskText);

    container.appendChild(bubble);

    // Progress bar (hidden by default, shown when progress > 0)
    const progressContainer = document.createElement("div");
    progressContainer.className = "progress-container";
    progressContainer.style.cssText = `
      width: 60px;
      height: 4px;
      background: rgba(0,0,0,0.2);
      border-radius: 2px;
      margin-top: 4px;
      overflow: hidden;
      display: none;
    `;
    const progressBar = document.createElement("div");
    progressBar.className = "progress-bar";
    progressBar.style.cssText = `
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, ${getStatusColor(agent.status)}, #fff);
      border-radius: 2px;
      transition: width 0.3s ease;
    `;
    progressContainer.appendChild(progressBar);
    container.appendChild(progressContainer);

    // Bubble tail (triangle pointing down)
    const tail = document.createElement("div");
    tail.style.cssText = `
      width: 0;
      height: 0;
      border-left: 6px solid transparent;
      border-right: 6px solid transparent;
      border-top: 8px solid rgba(255, 255, 255, 0.95);
      margin-top: -1px;
    `;
    container.appendChild(tail);

    // Name tag
    const nameTag = document.createElement("div");
    nameTag.style.cssText = `
      background: ${getStatusColor(agent.status)};
      color: white;
      font-size: 10px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 10px;
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    `;
    nameTag.textContent = agent.name;
    container.appendChild(nameTag);

    const label = new CSS2DObject(container);
    label.position.set(0, 2.5, 0);
    label.userData = { agent };

    return label;
  }, []);

  // Update status label content
  const updateStatusLabel = useCallback((agentId: string, agent: Agent, progress?: number) => {
    const label = agentLabelsRef.current.get(agentId);
    if (!label) return;

    const container = label.element;
    const bubble = container.querySelector(".status-bubble") as HTMLElement;
    const iconDiv = container.querySelector(".status-icon") as HTMLElement;
    const taskText = container.querySelector(".task-text") as HTMLElement;
    const nameTag = container.querySelector("div:last-child") as HTMLElement;
    const progressContainer = container.querySelector(".progress-container") as HTMLElement;
    const progressBar = container.querySelector(".progress-bar") as HTMLElement;

    const glowColor = agent.status === "working" ? "0, 255, 100" : agent.status === "thinking" ? "255, 200, 50" : "150, 150, 150";

    if (bubble) {
      bubble.style.boxShadow = `0 4px 12px rgba(0,0,0,0.2), 0 0 0 2px ${getStatusColor(agent.status)}, 0 0 ${agent.status === "working" ? "15px" : "8px"} rgba(${glowColor}, 0.4)`;
      bubble.style.animation = `float 2s ease-in-out infinite${agent.status === "working" ? ", glow 1.5s ease-in-out infinite" : ""}`;
    }
    if (iconDiv) {
      iconDiv.style.color = getStatusColor(agent.status);
      iconDiv.innerHTML = getTaskIcon(agent.task || "", agent.status);
    }
    if (taskText) {
      const shortTask = (agent.task || "Idle").split(" ").slice(0, 2).join(" ");
      taskText.textContent = shortTask.replace("...", "");
    }
    if (nameTag) {
      nameTag.style.background = getStatusColor(agent.status);
    }
    // Update progress bar
    if (progressContainer && progressBar) {
      if (progress !== undefined && progress > 0) {
        progressContainer.style.display = "block";
        progressBar.style.width = `${Math.min(100, progress)}%`;
        progressBar.style.background = `linear-gradient(90deg, ${getStatusColor(agent.status)}, #fff)`;
      } else {
        progressContainer.style.display = "none";
      }
    }
  }, []);

  // Create a cute robot/agent character from primitives
  const createAgent = useCallback((agent: Agent, scene: THREE.Scene) => {
    const group = new THREE.Group();
    group.userData = { agent };

    // Body (rounded cylinder/capsule-like)
    const bodyGeometry = new THREE.CapsuleGeometry(0.35, 0.5, 8, 16);
    const bodyMaterial = new THREE.MeshToonMaterial({
      color: agent.color,
    });
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
    body.position.y = 0.6;
    body.castShadow = true;
    group.add(body);

    // Head (sphere)
    const headGeometry = new THREE.SphereGeometry(0.3, 16, 16);
    const headMaterial = new THREE.MeshToonMaterial({
      color: agent.color,
    });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    head.position.y = 1.3;
    head.castShadow = true;
    group.add(head);

    // Eyes (small dark spheres)
    const eyeGeometry = new THREE.SphereGeometry(0.06, 8, 8);
    const eyeMaterial = new THREE.MeshBasicMaterial({ color: 0x2a2a2a });

    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.1, 1.35, 0.25);
    group.add(leftEye);

    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.1, 1.35, 0.25);
    group.add(rightEye);

    // Eye highlights (tiny white spheres)
    const highlightGeometry = new THREE.SphereGeometry(0.02, 4, 4);
    const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });

    const leftHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
    leftHighlight.position.set(-0.08, 1.37, 0.29);
    group.add(leftHighlight);

    const rightHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
    rightHighlight.position.set(0.12, 1.37, 0.29);
    group.add(rightHighlight);

    // Antenna (for working agents)
    if (agent.status === "working" || agent.status === "thinking") {
      const antennaGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.2, 8);
      const antennaMaterial = new THREE.MeshToonMaterial({ color: 0x666666 });
      const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
      antenna.position.y = 1.7;
      group.add(antenna);

      const antennaBallGeometry = new THREE.SphereGeometry(0.05, 8, 8);
      const antennaBallMaterial = new THREE.MeshBasicMaterial({
        color: agent.status === "working" ? 0x44ff44 : 0xffff44,
      });
      const antennaBall = new THREE.Mesh(antennaBallGeometry, antennaBallMaterial);
      antennaBall.position.y = 1.85;
      antennaBall.name = "antennaBall";
      group.add(antennaBall);
    }

    // Arms (small capsules)
    const armGeometry = new THREE.CapsuleGeometry(0.08, 0.25, 4, 8);
    const armMaterial = new THREE.MeshToonMaterial({ color: agent.color });

    const leftArm = new THREE.Mesh(armGeometry, armMaterial);
    leftArm.position.set(-0.45, 0.6, 0);
    leftArm.rotation.z = Math.PI / 6;
    leftArm.name = "leftArm";
    group.add(leftArm);

    const rightArm = new THREE.Mesh(armGeometry, armMaterial);
    rightArm.position.set(0.45, 0.6, 0);
    rightArm.rotation.z = -Math.PI / 6;
    rightArm.name = "rightArm";
    group.add(rightArm);

    // Position the agent
    group.position.set(...agent.position);
    group.position.y = 0;

    // Create and attach status label
    const label = createStatusLabel(agent);
    group.add(label);
    agentLabelsRef.current.set(agent.id, label);

    scene.add(group);
    agentMeshesRef.current.set(agent.id, group);

    return group;
  }, [createStatusLabel]);

  // Create a desk with computer
  const createDesk = useCallback((position: [number, number, number], rotation: number, scene: THREE.Scene) => {
    const group = new THREE.Group();

    // Desk top
    const deskGeometry = new THREE.BoxGeometry(1.5, 0.1, 0.8);
    const deskMaterial = new THREE.MeshToonMaterial({ color: COLORS.desk });
    const desk = new THREE.Mesh(deskGeometry, deskMaterial);
    desk.position.y = 0.7;
    desk.castShadow = true;
    desk.receiveShadow = true;
    group.add(desk);

    // Desk legs
    const legGeometry = new THREE.BoxGeometry(0.1, 0.7, 0.1);
    const legMaterial = new THREE.MeshToonMaterial({ color: 0xccb088 });

    const legPositions = [
      [-0.6, 0.35, 0.3],
      [0.6, 0.35, 0.3],
      [-0.6, 0.35, -0.3],
      [0.6, 0.35, -0.3],
    ];

    legPositions.forEach(([x, y, z]) => {
      const leg = new THREE.Mesh(legGeometry, legMaterial);
      leg.position.set(x, y, z);
      leg.castShadow = true;
      group.add(leg);
    });

    // Monitor
    const monitorGeometry = new THREE.BoxGeometry(0.8, 0.5, 0.05);
    const monitorMaterial = new THREE.MeshToonMaterial({ color: 0x333333 });
    const monitor = new THREE.Mesh(monitorGeometry, monitorMaterial);
    monitor.position.set(0, 1.1, -0.2);
    monitor.castShadow = true;
    group.add(monitor);

    // Monitor screen (glowing)
    const screenGeometry = new THREE.BoxGeometry(0.7, 0.4, 0.01);
    const screenMaterial = new THREE.MeshBasicMaterial({
      color: COLORS.screenGlow,
      transparent: true,
      opacity: 0.9,
    });
    const screen = new THREE.Mesh(screenGeometry, screenMaterial);
    screen.position.set(0, 1.1, -0.17);
    screen.name = "screen";
    group.add(screen);
    screenMeshesRef.current.push(screen);

    // Monitor stand
    const standGeometry = new THREE.BoxGeometry(0.1, 0.25, 0.1);
    const stand = new THREE.Mesh(standGeometry, monitorMaterial);
    stand.position.set(0, 0.87, -0.2);
    group.add(stand);

    // Keyboard
    const keyboardGeometry = new THREE.BoxGeometry(0.5, 0.03, 0.2);
    const keyboardMaterial = new THREE.MeshToonMaterial({ color: 0x444444 });
    const keyboard = new THREE.Mesh(keyboardGeometry, keyboardMaterial);
    keyboard.position.set(0, 0.77, 0.15);
    group.add(keyboard);

    group.position.set(...position);
    group.rotation.y = rotation;

    scene.add(group);
    return group;
  }, []);

  // Create floating data particles between agents
  const createDataParticles = useCallback((scene: THREE.Scene) => {
    const particleCount = 100;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    const sizes = new Float32Array(particleCount);

    const color1 = new THREE.Color(COLORS.codex);
    const color2 = new THREE.Color(COLORS.opus);
    const color3 = new THREE.Color(COLORS.particles);

    for (let i = 0; i < particleCount; i++) {
      // Random positions in the lab area
      const angle = Math.random() * Math.PI * 2;
      const radius = 1 + Math.random() * 4;
      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = 1 + Math.random() * 3;
      positions[i * 3 + 2] = Math.sin(angle) * radius;

      // Random colors from palette
      const colorChoice = Math.random();
      const chosenColor = colorChoice < 0.33 ? color1 : colorChoice < 0.66 ? color2 : color3;
      colors[i * 3] = chosenColor.r;
      colors[i * 3 + 1] = chosenColor.g;
      colors[i * 3 + 2] = chosenColor.b;

      sizes[i] = 0.05 + Math.random() * 0.1;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    const material = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);
    particlesRef.current.push(particles);

    return particles;
  }, []);

  // Create decorative elements
  const createDecorations = useCallback((scene: THREE.Scene) => {
    // Potted plants
    const createPlant = (x: number, z: number) => {
      const group = new THREE.Group();

      // Pot
      const potGeometry = new THREE.CylinderGeometry(0.15, 0.12, 0.2, 12);
      const potMaterial = new THREE.MeshToonMaterial({ color: 0xcc8866 });
      const pot = new THREE.Mesh(potGeometry, potMaterial);
      pot.position.y = 0.1;
      group.add(pot);

      // Plant (stacked spheres)
      const plantMaterial = new THREE.MeshToonMaterial({ color: 0x66bb66 });
      for (let i = 0; i < 3; i++) {
        const leafGeometry = new THREE.SphereGeometry(0.12 - i * 0.02, 8, 8);
        const leaf = new THREE.Mesh(leafGeometry, plantMaterial);
        leaf.position.y = 0.25 + i * 0.12;
        leaf.position.x = (Math.random() - 0.5) * 0.1;
        leaf.position.z = (Math.random() - 0.5) * 0.1;
        group.add(leaf);
      }

      group.position.set(x, 0, z);
      scene.add(group);
    };

    createPlant(-5, 0);
    createPlant(5, 0);
    createPlant(0, -5);
    createPlant(0, 5);

    // Floating cubes (data visualization style)
    const cubeGeometry = new THREE.BoxGeometry(0.2, 0.2, 0.2);
    for (let i = 0; i < 8; i++) {
      const hue = i / 8;
      const color = new THREE.Color().setHSL(hue, 0.6, 0.7);
      const cubeMaterial = new THREE.MeshToonMaterial({
        color,
        transparent: true,
        opacity: 0.7,
      });
      const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
      const angle = (i / 8) * Math.PI * 2;
      cube.position.set(
        Math.cos(angle) * 6,
        2 + Math.sin(i * 1.5) * 0.5,
        Math.sin(angle) * 6
      );
      cube.userData = { floatPhase: i * 0.5 };
      cube.name = `floatingCube${i}`;
      scene.add(cube);
    }
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.sky);
    scene.fog = new THREE.Fog(COLORS.sky, 10, 30);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      50,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      100
    );
    camera.position.set(8, 6, 8);
    camera.lookAt(0, 1, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // CSS2D Renderer for labels
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.top = "0";
    labelRenderer.domElement.style.left = "0";
    labelRenderer.domElement.style.pointerEvents = "none";
    containerRef.current.appendChild(labelRenderer.domElement);
    labelRendererRef.current = labelRenderer;

    // Add floating animation CSS
    const styleSheet = document.createElement("style");
    styleSheet.id = "lab3d-animations";
    // Remove existing if present
    const existingStyle = document.getElementById("lab3d-animations");
    if (existingStyle) existingStyle.remove();

    styleSheet.textContent = `
      @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
      }
      @keyframes glow {
        0%, 100% {
          box-shadow: 0 4px 12px rgba(0,0,0,0.2), 0 0 0 2px #4ade80, 0 0 15px rgba(0, 255, 100, 0.4);
        }
        50% {
          box-shadow: 0 4px 12px rgba(0,0,0,0.2), 0 0 0 2px #4ade80, 0 0 25px rgba(0, 255, 100, 0.6);
        }
      }
      @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
      }
      .agent-status-label {
        transition: opacity 0.3s ease;
      }
      .status-bubble {
        transition: box-shadow 0.3s ease;
      }
      .status-icon svg {
        transition: transform 0.2s ease;
      }
      .status-bubble:hover .status-icon svg {
        transform: scale(1.1);
      }
      .progress-bar {
        animation: progress-shimmer 1.5s ease-in-out infinite;
      }
      @keyframes progress-shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
      }
      .progress-container {
        box-shadow: 0 0 8px rgba(74, 222, 128, 0.3);
      }
    `;
    document.head.appendChild(styleSheet);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.1;
    controls.minDistance = 5;
    controls.maxDistance = 20;
    controls.target.set(0, 1, 0);

    // Lighting - soft and playful
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xfff5ee, 0.8);
    mainLight.position.set(10, 15, 10);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 50;
    mainLight.shadow.camera.left = -15;
    mainLight.shadow.camera.right = 15;
    mainLight.shadow.camera.top = 15;
    mainLight.shadow.camera.bottom = -15;
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xffeeff, 0.3);
    fillLight.position.set(-5, 5, -5);
    scene.add(fillLight);

    // Ground - checkered grass pattern
    const groundSize = 20;
    const groundGeometry = new THREE.PlaneGeometry(groundSize, groundSize, 20, 20);
    const groundMaterial = new THREE.MeshToonMaterial({
      color: COLORS.ground,
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    // Ground pattern (simple circles)
    for (let x = -8; x <= 8; x += 4) {
      for (let z = -8; z <= 8; z += 4) {
        const circleGeometry = new THREE.CircleGeometry(0.5, 16);
        const circleMaterial = new THREE.MeshToonMaterial({
          color: COLORS.groundAccent,
        });
        const circle = new THREE.Mesh(circleGeometry, circleMaterial);
        circle.rotation.x = -Math.PI / 2;
        circle.position.set(x, 0.01, z);
        scene.add(circle);
      }
    }

    // Create agents (use initial ref to avoid recreating scene on updates)
    initialAgentsRef.current.forEach((agent) => {
      createAgent(agent, scene);
    });

    // Create Manager agent (special - taller with clipboard)
    const createManager = () => {
      const group = new THREE.Group();
      group.userData = { agent: { id: "manager", name: "Manager", status: "working" } };

      // Body (taller than regular agents)
      const bodyGeometry = new THREE.CapsuleGeometry(0.35, 0.7, 8, 16);
      const bodyMaterial = new THREE.MeshToonMaterial({ color: MANAGER_COLOR });
      const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
      body.position.y = 0.7;
      body.castShadow = true;
      group.add(body);

      // Head
      const headGeometry = new THREE.SphereGeometry(0.32, 16, 16);
      const headMaterial = new THREE.MeshToonMaterial({ color: MANAGER_COLOR });
      const head = new THREE.Mesh(headGeometry, headMaterial);
      head.position.y = 1.5;
      head.castShadow = true;
      group.add(head);

      // Boss hat/crown
      const hatGeometry = new THREE.CylinderGeometry(0.15, 0.25, 0.15, 8);
      const hatMaterial = new THREE.MeshToonMaterial({ color: 0xffd700 });  // Gold
      const hat = new THREE.Mesh(hatGeometry, hatMaterial);
      hat.position.y = 1.85;
      group.add(hat);

      // Eyes
      const eyeGeometry = new THREE.SphereGeometry(0.06, 8, 8);
      const eyeMaterial = new THREE.MeshBasicMaterial({ color: 0x2a2a2a });
      const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
      leftEye.position.set(-0.1, 1.55, 0.27);
      group.add(leftEye);
      const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
      rightEye.position.set(0.1, 1.55, 0.27);
      group.add(rightEye);

      // Eye highlights
      const highlightGeometry = new THREE.SphereGeometry(0.02, 4, 4);
      const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const leftHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
      leftHighlight.position.set(-0.08, 1.57, 0.31);
      group.add(leftHighlight);
      const rightHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
      rightHighlight.position.set(0.12, 1.57, 0.31);
      group.add(rightHighlight);

      // Clipboard in hand
      const clipboardGeometry = new THREE.BoxGeometry(0.25, 0.35, 0.03);
      const clipboardMaterial = new THREE.MeshToonMaterial({ color: 0x8b4513 });
      const clipboard = new THREE.Mesh(clipboardGeometry, clipboardMaterial);
      clipboard.position.set(0.4, 0.8, 0.2);
      clipboard.rotation.z = -0.3;
      clipboard.rotation.y = -0.2;
      group.add(clipboard);

      // Paper on clipboard
      const paperGeometry = new THREE.BoxGeometry(0.2, 0.3, 0.01);
      const paperMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const paper = new THREE.Mesh(paperGeometry, paperMaterial);
      paper.position.set(0.4, 0.8, 0.23);
      paper.rotation.z = -0.3;
      paper.rotation.y = -0.2;
      group.add(paper);

      // Arms
      const armGeometry = new THREE.CapsuleGeometry(0.08, 0.3, 4, 8);
      const armMaterial = new THREE.MeshToonMaterial({ color: MANAGER_COLOR });
      const leftArm = new THREE.Mesh(armGeometry, armMaterial);
      leftArm.position.set(-0.5, 0.7, 0);
      leftArm.rotation.z = Math.PI / 6;
      leftArm.name = "leftArm";
      group.add(leftArm);
      const rightArm = new THREE.Mesh(armGeometry, armMaterial);
      rightArm.position.set(0.5, 0.7, 0);
      rightArm.rotation.z = -Math.PI / 4;
      rightArm.rotation.x = -0.3;  // Holding clipboard
      rightArm.name = "rightArm";
      group.add(rightArm);

      // Start at center
      group.position.set(0, 0, 0);

      // Create manager's speech bubble label
      const managerLabelContainer = document.createElement("div");
      managerLabelContainer.className = "manager-status-label";
      managerLabelContainer.style.cssText = `
        display: flex;
        flex-direction: column;
        align-items: center;
        pointer-events: none;
        transform: translateY(-20px);
      `;

      const managerBubble = document.createElement("div");
      managerBubble.className = "manager-bubble";
      managerBubble.style.cssText = `
        background: linear-gradient(135deg, rgba(244, 114, 182, 0.95), rgba(236, 72, 153, 0.95));
        border-radius: 12px;
        padding: 8px 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3), 0 0 15px rgba(244, 114, 182, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 80px;
        animation: float 2s ease-in-out infinite, glow 1.5s ease-in-out infinite;
      `;

      const managerIcon = document.createElement("div");
      managerIcon.style.cssText = `
        width: 16px;
        height: 16px;
        color: white;
        margin-right: 6px;
      `;
      managerIcon.innerHTML = TASK_ICONS.list;
      managerBubble.appendChild(managerIcon);

      const managerText = document.createElement("div");
      managerText.className = "manager-task-text";
      managerText.style.cssText = `
        font-size: 10px;
        font-weight: 600;
        color: white;
        max-width: 100px;
        text-align: center;
      `;
      managerText.textContent = "Supervising";
      managerBubble.appendChild(managerText);

      managerLabelContainer.appendChild(managerBubble);

      // Bubble tail
      const managerTail = document.createElement("div");
      managerTail.style.cssText = `
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 8px solid rgba(244, 114, 182, 0.95);
        margin-top: -1px;
      `;
      managerLabelContainer.appendChild(managerTail);

      // Manager name tag
      const managerNameTag = document.createElement("div");
      managerNameTag.style.cssText = `
        background: linear-gradient(135deg, #ffd700, #ffaa00);
        color: #333;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 10px;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      `;
      managerNameTag.textContent = "MANAGER";
      managerLabelContainer.appendChild(managerNameTag);

      const managerLabel = new CSS2DObject(managerLabelContainer);
      managerLabel.position.set(0, 2.8, 0);
      group.add(managerLabel);

      scene.add(group);
      managerRef.current = group;

      // Initialize patrol
      managerTargetRef.current.set(...MANAGER_PATROL_POINTS[0]);
    };

    if (showManagerRef.current) {
      createManager();
    }

    // Create desks for each agent
    createDesk([-3, 0, -3.5], 0, scene);
    createDesk([3, 0, -3.5], 0, scene);
    createDesk([-3, 0, 4.5], Math.PI, scene);
    createDesk([3, 0, 4.5], Math.PI, scene);

    // Create data particles
    createDataParticles(scene);

    // Create decorations
    createDecorations(scene);

    // Create demo props (equipment around the lab)
    if (showDemoProps) {
      // Supercomputer - towering server rack in back-left corner
      const supercomputerRefs = createProp3D('supercomputer', {
        position: [-6, 0, -5],
        scale: 1.3,
        accentColor: 0x00ffaa,
      });
      if (supercomputerRefs) {
        scene.add(supercomputerRefs.group);
        demoPropsRef.current.set('demo-supercomputer', { type: 'supercomputer', refs: supercomputerRefs });
      }

      // Microphone - front-left corner
      const micRefs = createProp3D('microphone', {
        position: [-6, 0, 5],
        scale: 2.0,
        accentColor: 0x4ecdc4,
      });
      if (micRefs) {
        scene.add(micRefs.group);
        demoPropsRef.current.set('demo-mic', { type: 'microphone', refs: micRefs });
      }

      // Speaker - front-right corner
      const speakerRefs = createProp3D('speaker', {
        position: [6, 0, 5],
        scale: 2.0,
        accentColor: 0xffe66d,
      });
      if (speakerRefs) {
        scene.add(speakerRefs.group);
        demoPropsRef.current.set('demo-speaker', { type: 'speaker', refs: speakerRefs });
      }

      // Server rack - back-right corner
      const serverRefs = createProp3D('server', {
        position: [6, 0, -5],
        scale: 2.0,
        accentColor: 0x3b82f6,
      });
      if (serverRefs) {
        scene.add(serverRefs.group);
        demoPropsRef.current.set('demo-server', { type: 'server', refs: serverRefs });
      }

      // Emotion Verify - V7 verification display (center-back, visible from default view)
      // Raised off ground (y=0.3) to prevent z-fighting with floor
      const emotionVerifyRefs = createProp3D('emotion-verify', {
        position: [0, 0.3, -6],
        scale: 1.5,
        accentColor: 0x4ade80,
      });
      if (emotionVerifyRefs) {
        scene.add(emotionVerifyRefs.group);
        demoPropsRef.current.set('demo-emotion-verify', { type: 'emotion-verify', refs: emotionVerifyRefs });
      }
    }

    // Central hub visualization (like a data nexus)
    const hubGeometry = new THREE.TorusGeometry(1, 0.1, 8, 32);
    const hubMaterial = new THREE.MeshToonMaterial({
      color: 0xffaacc,
      transparent: true,
      opacity: 0.6,
    });
    const hub = new THREE.Mesh(hubGeometry, hubMaterial);
    hub.rotation.x = Math.PI / 2;
    hub.position.y = 2;
    hub.name = "hub";
    scene.add(hub);

    const hub2 = new THREE.Mesh(hubGeometry, hubMaterial.clone());
    hub2.rotation.x = Math.PI / 2;
    hub2.rotation.z = Math.PI / 4;
    hub2.position.y = 2;
    hub2.name = "hub2";
    scene.add(hub2);

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      timeRef.current += 0.016;
      const time = timeRef.current;

      controls.update();

      // Animate agents (bobbing and arm movement)
      agentMeshesRef.current.forEach((group) => {
        const storedAgent = group.userData.agent as Agent;
        // Get current agent state from ref (for up-to-date status)
        const currentAgent = agentsRef.current.find(a => a.id === storedAgent.id) || storedAgent;

        // Move agent toward target if assigned
        const target = agentTargetsRef.current.get(storedAgent.id);
        const targetRotation = agentTargetRotationRef.current.get(storedAgent.id);
        if (target) {
          const currentPos = new THREE.Vector3(group.position.x, 0, group.position.z);
          const targetPos = new THREE.Vector3(target.x, 0, target.z);
          const distance = currentPos.distanceTo(targetPos);

          if (distance > 0.1) {
            // Smoothly move toward target
            const direction = targetPos.clone().sub(currentPos).normalize();
            const moveSpeed = 0.03;
            group.position.x += direction.x * moveSpeed;
            group.position.z += direction.z * moveSpeed;

            // Face the direction of movement while walking
            const angle = Math.atan2(direction.x, direction.z);
            group.rotation.y = THREE.MathUtils.lerp(group.rotation.y, angle, 0.1);
          } else if (targetRotation !== undefined) {
            // At destination - rotate to face the equipment
            group.rotation.y = THREE.MathUtils.lerp(group.rotation.y, targetRotation, 0.05);
          }
        }

        // Gentle bobbing
        group.position.y = Math.sin(time * 2 + group.position.x) * 0.05;

        // Arm animation for working agents (use current status)
        if (currentAgent.status === "working") {
          const leftArm = group.getObjectByName("leftArm") as THREE.Mesh;
          const rightArm = group.getObjectByName("rightArm") as THREE.Mesh;
          if (leftArm && rightArm) {
            leftArm.rotation.x = Math.sin(time * 8) * 0.3;
            rightArm.rotation.x = Math.sin(time * 8 + Math.PI) * 0.3;
          }
        }

        // Antenna glow pulse
        const antennaBall = group.getObjectByName("antennaBall") as THREE.Mesh;
        if (antennaBall) {
          const material = antennaBall.material as THREE.MeshBasicMaterial;
          material.opacity = 0.5 + Math.sin(time * 4) * 0.5;
        }
      });

      // Animate Manager patrol
      if (showManagerRef.current && managerRef.current) {
        const manager = managerRef.current;
        const target = managerTargetRef.current;
        const currentPos = new THREE.Vector3(manager.position.x, 0, manager.position.z);
        const targetPos = new THREE.Vector3(target.x, 0, target.z);
        const distance = currentPos.distanceTo(targetPos);

        // Agent names at each patrol point
        const patrolAgentNames = ["Supercomputer", "Codex", "Opus", "Explorer", "Planner"];
        const patrolActions = ["Monitoring GPU", "Reviewing code", "Analyzing data", "Checking search", "Assigning tasks"];

        // Update manager's speech bubble
        const managerLabel = manager.children.find(c => c instanceof CSS2DObject) as CSS2DObject | undefined;
        if (managerLabel) {
          const textEl = managerLabel.element.querySelector(".manager-task-text") as HTMLElement;
          if (textEl) {
            // Check if there's a real manager activity running
            const managerActivity = activitiesRef.current.find(
              a => a.active && a.assignedAgent === 'manager'
            );

            if (managerActivity) {
              // Show real activity message when orchestrating
              textEl.textContent = managerActivity.message || 'Orchestrating...';
            } else {
              // Default patrol messages when idle
              const patrolIdx = managerPatrolIndexRef.current;
              if (distance > 0.5) {
                textEl.textContent = `Going to ${patrolAgentNames[patrolIdx]}`;
              } else {
                textEl.textContent = patrolActions[patrolIdx];
              }
            }
          }
        }

        if (distance > 0.2) {
          // Move toward target
          const direction = targetPos.clone().sub(currentPos).normalize();
          const moveSpeed = 0.025;
          manager.position.x += direction.x * moveSpeed;
          manager.position.z += direction.z * moveSpeed;

          // Face movement direction
          const angle = Math.atan2(direction.x, direction.z);
          manager.rotation.y = THREE.MathUtils.lerp(manager.rotation.y, angle, 0.1);

          // Walking bob
          manager.position.y = Math.abs(Math.sin(time * 8)) * 0.08;
        } else {
          // At destination - wait and then pick next point
          managerWaitTimeRef.current += 0.016;
          manager.position.y = Math.sin(time * 2) * 0.03;  // Idle bob

          if (managerWaitTimeRef.current > 3) {  // Wait 3 seconds
            managerWaitTimeRef.current = 0;
            managerPatrolIndexRef.current = (managerPatrolIndexRef.current + 1) % MANAGER_PATROL_POINTS.length;
            const nextPoint = MANAGER_PATROL_POINTS[managerPatrolIndexRef.current];
            managerTargetRef.current.set(...nextPoint);
          }
        }

        // Clipboard wiggle when near an agent
        if (distance < 0.5 && managerPatrolIndexRef.current > 0) {
          // "Giving orders" animation - clipboard wiggle
          const clipboard = manager.children.find(c =>
            c instanceof THREE.Mesh && c.position.x > 0.3 && c.position.y > 0.7
          );
          if (clipboard) {
            clipboard.rotation.x = Math.sin(time * 6) * 0.15;
          }
        }
      }

      // Animate activity props
      propsRef.current.forEach(({ type, refs }, activityId) => {
        const activity = activitiesRef.current.find((a) => a.id === activityId);
        animateProp3D(type, refs, time, {
          isActive: activity?.active ?? false,
          progress: activity?.progress,
          loadLevel: (activity?.progress ?? 50) / 100,
          audioLevel: activity?.active ? 0.5 + Math.sin(time * 5) * 0.3 : 0,
        });
      });

      // Animate demo props (always active for visual interest)
      demoPropsRef.current.forEach(({ type, refs }, demoId) => {
        // Simulate activity based on prop type
        const isSupercomputer = demoId.includes('supercomputer');
        const isGpu = demoId.includes('gpu');
        const isMic = demoId.includes('mic');
        const isSpeaker = demoId.includes('speaker');
        const isEmotionVerify = demoId.includes('emotion-verify');

        // Special handling for emotion-verify prop
        if (isEmotionVerify && type === 'emotion-verify') {
          const emotionRefs = refs as EmotionVerify3DRefs;

          // Simulate V7 verification - Happy F0 should be higher than Sad F0
          const happyF0 = 0.65 + Math.sin(time * 0.8) * 0.1;  // Varies around 0.65
          const sadF0 = 0.35 + Math.sin(time * 0.6) * 0.08;   // Varies around 0.35

          // Verification passes when happy > sad (which should be true)
          const verified = happyF0 > sadF0;

          animateEmotionVerify3D(emotionRefs, time, {
            happyF0,
            sadF0,
            isVerifying: !verified,
            verified,
          });
        } else if (isSupercomputer && type === 'supercomputer') {
          // Use real GPU stats for supercomputer diegetic display
          const supercomputerRefs = refs as Supercomputer3DRefs;
          const gpuStats = gpuStatsRef.current;

          animateSupercomputer3D(supercomputerRefs, time, {
            isProcessing: true,
            progress: gpuStats?.connected ? gpuStats.utilization : 50 + Math.sin(time * 0.5) * 30,
            loadLevel: gpuStats?.connected ? gpuStats.utilization / 100 : 0.5 + Math.sin(time * 0.3) * 0.3,
            gpuStats: gpuStats || undefined,
            camera: camera,  // Pass camera for billboard effect
          });
        } else {
          animateProp3D(type, refs, time, {
            isActive: true,
            progress: isGpu ? 50 + Math.sin(time * 0.5) * 30 : undefined,
            loadLevel: isGpu ? 0.5 + Math.sin(time * 0.3) * 0.3 : 0.5,
            audioLevel: (isMic || isSpeaker) ? 0.4 + Math.sin(time * 3) * 0.3 : 0,
          });
        }
      });

      // Animate particles (flowing/orbiting)
      particlesRef.current.forEach((particles) => {
        const positions = particles.geometry.attributes.position.array as Float32Array;
        for (let i = 0; i < positions.length / 3; i++) {
          const idx = i * 3;
          const x = positions[idx];
          const z = positions[idx + 2];
          const angle = Math.atan2(z, x) + 0.01;
          const radius = Math.sqrt(x * x + z * z);
          positions[idx] = Math.cos(angle) * radius;
          positions[idx + 2] = Math.sin(angle) * radius;
          positions[idx + 1] += Math.sin(time + i) * 0.005;

          // Keep particles in bounds
          if (positions[idx + 1] > 4) positions[idx + 1] = 1;
          if (positions[idx + 1] < 1) positions[idx + 1] = 4;
        }
        particles.geometry.attributes.position.needsUpdate = true;
      });

      // Animate screens (flickering glow)
      screenMeshesRef.current.forEach((screen, idx) => {
        const material = screen.material as THREE.MeshBasicMaterial;
        const brightness = 0.7 + Math.sin(time * 3 + idx) * 0.3;
        material.color.setHSL(0.4 + Math.sin(time * 0.5 + idx) * 0.1, 0.8, brightness * 0.5);
      });

      // Animate hub rings
      const hub = scene.getObjectByName("hub") as THREE.Mesh;
      const hub2 = scene.getObjectByName("hub2") as THREE.Mesh;
      if (hub) {
        hub.rotation.z = time * 0.5;
        hub.scale.setScalar(1 + Math.sin(time * 2) * 0.1);
      }
      if (hub2) {
        hub2.rotation.z = -time * 0.3;
        hub2.scale.setScalar(1 + Math.cos(time * 2) * 0.1);
      }

      // Animate floating cubes
      for (let i = 0; i < 8; i++) {
        const cube = scene.getObjectByName(`floatingCube${i}`) as THREE.Mesh;
        if (cube) {
          const phase = cube.userData.floatPhase;
          cube.rotation.x = time * 0.5 + phase;
          cube.rotation.y = time * 0.7 + phase;
          cube.position.y = 2 + Math.sin(time + phase) * 0.3;
        }
      }

      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
      labelRenderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    };

    window.addEventListener("resize", handleResize);

    // Handle click (use ref to avoid stale closure)
    const handleClick = (event: MouseEvent) => {
      if (!containerRef.current) return;

      const rect = containerRef.current.getBoundingClientRect();
      mouseRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouseRef.current.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycasterRef.current.setFromCamera(mouseRef.current, camera);

      // First check for supercomputer click
      const supercomputerProp = demoPropsRef.current.get('demo-supercomputer');
      if (supercomputerProp && onComputerClickRef.current) {
        const supercomputerIntersects = raycasterRef.current.intersectObjects(
          [supercomputerProp.refs.group],
          true
        );
        if (supercomputerIntersects.length > 0) {
          onComputerClickRef.current();
          return;
        }
      }

      // Then check for agent clicks
      if (onAgentClickRef.current) {
        const agentGroups = Array.from(agentMeshesRef.current.values());
        const intersects = raycasterRef.current.intersectObjects(
          agentGroups.flatMap((g) => g.children),
          true
        );

        if (intersects.length > 0) {
          let parent = intersects[0].object.parent;
          while (parent && !parent.userData?.agent) {
            parent = parent.parent;
          }
          if (parent?.userData?.agent) {
            onAgentClickRef.current(parent.userData.agent);
          }
        }
      }
    };

    containerRef.current.addEventListener("click", handleClick);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      containerRef.current?.removeEventListener("click", handleClick);
      cancelAnimationFrame(animationRef.current);
      renderer.dispose();
      agentMeshesRef.current.clear();
      agentLabelsRef.current.clear();
      particlesRef.current = [];
      screenMeshesRef.current = [];
      if (containerRef.current) {
        if (renderer.domElement) {
          containerRef.current.removeChild(renderer.domElement);
        }
        if (labelRenderer.domElement) {
          containerRef.current.removeChild(labelRenderer.domElement);
        }
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createAgent, createDesk, createDataParticles, createDecorations]);

  // Dynamically create agent meshes when new agents arrive (e.g. from API after initial render)
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    agents.forEach((agent) => {
      if (!agentMeshesRef.current.has(agent.id)) {
        createAgent(agent, scene);
      }
    });
    // Remove agents that are no longer in the list
    const currentIds = new Set(agents.map(a => a.id));
    agentMeshesRef.current.forEach((group, id) => {
      if (!currentIds.has(id)) {
        scene.remove(group);
        agentMeshesRef.current.delete(id);
        const label = agentLabelsRef.current.get(id);
        if (label) {
          scene.remove(label);
          agentLabelsRef.current.delete(id);
        }
      }
    });
  }, [agents, createAgent]);

  // Update labels when agents change (include progress from activities)
  useEffect(() => {
    agents.forEach((agent) => {
      // Find the activity for this agent to get progress
      const activity = activities.find(a => a.active && a.assignedAgent === agent.id);
      updateStatusLabel(agent.id, agent, activity?.progress);
    });
  }, [agents, activities, updateStatusLabel]);

  // Manage activity props
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Store current activities for animation loop
    activitiesRef.current = activities;

    // Track which activities are currently active
    const activeActivityIds = new Set(activities.filter((a) => a.active).map((a) => a.id));

    // Remove props for activities that are no longer active
    propsRef.current.forEach((propData, activityId) => {
      if (!activeActivityIds.has(activityId)) {
        scene.remove(propData.refs.group);
        disposeProp3D(propData.type, propData.refs);
        propsRef.current.delete(activityId);
      }
    });

    // Clear agent targets and release slots for agents whose activities are no longer active
    // This makes them return to their default positions
    const activeAgentIds = new Set(
      activities.filter(a => a.active).map(a => a.assignedAgent).filter(Boolean)
    );
    agentTargetsRef.current.forEach((_, agentId) => {
      if (!activeAgentIds.has(agentId)) {
        agentTargetsRef.current.delete(agentId);
        agentTargetRotationRef.current.delete(agentId);
        releaseSlot(agentId);
      }
    });

    // Add props for new active activities AND direct agents to slots
    activities.forEach((activity) => {
      if (!activity.active) return;

      // Assign agent to a work slot near the prop
      if (activity.assignedAgent) {
        // Check if agent already has a slot assigned
        const existingSlot = agentSlotRef.current.get(activity.assignedAgent);
        if (!existingSlot) {
          // Determine which prop type this activity is related to
          const propType = activity.config.prop !== 'none' ? activity.config.prop : undefined;

          // Find the best available slot
          const slot = findBestSlot(activity.config.propPosition, propType);
          if (slot) {
            // Claim the slot
            claimSlot(slot.id, activity.assignedAgent);

            // Set agent target to slot position
            agentTargetsRef.current.set(
              activity.assignedAgent,
              new THREE.Vector3(...slot.position)
            );

            // Set target rotation to face the equipment
            agentTargetRotationRef.current.set(activity.assignedAgent, slot.facing);
          } else {
            // Fallback: no slots available, use basic offset positioning
            const [px, , pz] = activity.config.propPosition;
            const offsetX = px > 0 ? -1.5 : 1.5;
            const offsetZ = pz > 0 ? -1.5 : 1.5;
            agentTargetsRef.current.set(
              activity.assignedAgent,
              new THREE.Vector3(px + offsetX, 0, pz + offsetZ)
            );
          }
        }
      }

      // Skip prop creation if already exists or prop type is 'none'
      if (propsRef.current.has(activity.id)) return;
      if (activity.config.prop === 'none') return;

      // Create the prop (only for activities that define their own props)
      const propRefs = createProp3D(activity.config.prop, {
        position: activity.config.propPosition,
        scale: activity.config.propScale,
        accentColor: activity.config.color,
      });

      if (propRefs) {
        scene.add(propRefs.group);
        propsRef.current.set(activity.id, {
          type: activity.config.prop,
          refs: propRefs,
        });
      }
    });
  }, [activities, findBestSlot, claimSlot, releaseSlot]);

  // Cleanup props on unmount
  useEffect(() => {
    return () => {
      propsRef.current.forEach(({ type, refs }) => {
        disposeProp3D(type, refs);
      });
      propsRef.current.clear();
    };
  }, []);

  // Fetch GPU stats periodically for diegetic display
  useEffect(() => {
    const fetchGpuStats = async () => {
      try {
        // Use Next.js API route (handles Vercel demo mode automatically)
        const response = await fetch("/api/lab/gpu-stats");
        const data = await response.json();

        if (data.connected && data.gpu) {
          // Parse training info from processes
          let trainingData = undefined;
          if (data.processes && data.processes.length > 0) {
            // Find a training process with progress info
            const trainingProcess = data.processes.find(
              (p: { progress?: string; script?: string }) => p.progress || p.script
            );
            if (trainingProcess?.progress) {
              // Parse "Epoch 22 • loss: 0.5734" format
              const epochMatch = trainingProcess.progress.match(/Epoch\s+(\d+)/i);
              const lossMatch = trainingProcess.progress.match(/loss[:\s]+(\d+\.?\d*)/i);
              trainingData = {
                isTraining: true,
                epoch: epochMatch ? parseInt(epochMatch[1]) : undefined,
                loss: lossMatch ? parseFloat(lossMatch[1]) : undefined,
                script: trainingProcess.script,
              };
            }
          }

          gpuStatsRef.current = {
            connected: true,
            utilization: data.gpu.utilization || 0,
            temperature: data.gpu.temperature || 0,
            memoryUsed: data.gpu.memoryUsed || 0,
            memoryTotal: data.gpu.memoryTotal || 24564,
            powerDraw: data.gpu.powerDraw || 0,
            powerLimit: data.gpu.powerLimit || 450,
            training: trainingData,
          };
        } else {
          gpuStatsRef.current = {
            connected: false,
            utilization: 0,
            temperature: 0,
            memoryUsed: 0,
            memoryTotal: 0,
            powerDraw: 0,
            powerLimit: 0,
          };
        }
      } catch (error) {
        gpuStatsRef.current = {
          connected: false,
          utilization: 0,
          temperature: 0,
          memoryUsed: 0,
          memoryTotal: 0,
          powerDraw: 0,
          powerLimit: 0,
        };
      }
    };

    // Fetch immediately and then every 5 seconds
    fetchGpuStats();
    const interval = setInterval(fetchGpuStats, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-full h-full cursor-pointer relative"
      style={{ minHeight: "500px" }}
    />
  );
}
