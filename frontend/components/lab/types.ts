// Shared types for Lab 3D visualization system

import * as THREE from "three";

export interface Agent {
  id: string;
  name: string;
  color: number;
  position: [number, number, number];
  task?: string;
  status: "idle" | "working" | "thinking";
}

export interface AgentMeshRefs {
  group: THREE.Group;
  leftArm?: THREE.Mesh;
  rightArm?: THREE.Mesh;
  antennaBall?: THREE.Mesh;
}

export interface Lab3DSceneProps {
  agents?: Agent[];
  onAgentClick?: (agent: Agent) => void;
  showParticles?: boolean;
  showHub?: boolean;
  showDecorations?: boolean;
  cameraPosition?: [number, number, number];
  cameraTarget?: [number, number, number];
}

export interface Workstation {
  position: [number, number, number];
  rotation: number;
  screenColor?: number;
}

export interface Decoration {
  type: "plant" | "cube" | "custom";
  position: [number, number, number];
  color?: number;
  scale?: number;
}
