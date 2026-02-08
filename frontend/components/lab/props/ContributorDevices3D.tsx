/**
 * ContributorDevices3D - Visualizes distributed compute network contributors
 *
 * Renders a swarm of small computer/laptop meshes representing devices
 * that are contributing compute power to the network. Uses instanced
 * meshes for performance when displaying 50+ devices.
 */

import * as THREE from 'three';
import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';
import type { DeviceTier, DeviceStatus } from '@/lib/compute/types';

/** Escape HTML special characters to prevent XSS when using innerHTML */
function escapeHtml(str: string): string {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Tier colors matching the design system
const TIER_COLORS = {
  power: 0xa855f7,    // Purple - high-end GPUs
  standard: 0x3b82f6, // Blue - mid-range
  crowd: 0x22c55e,    // Green - browsers/phones
};

// Status colors for glow effects
const STATUS_COLORS = {
  online: 0x22c55e,   // Green - ready
  busy: 0xf59e0b,     // Amber - processing
  offline: 0x6b7280,  // Gray - disconnected
  paused: 0x94a3b8,   // Slate - paused
};

export interface ContributorDevice {
  id: string;
  name: string;
  tier: DeviceTier;
  status: DeviceStatus;
  compute?: number;
  platform?: string;
  stats?: {
    tasksCompleted: number;
    creditsEarned: number;
    totalComputeTime: number;
  };
}

export interface ContributorDevices3DRefs {
  group: THREE.Group;
  instancedMesh: THREE.InstancedMesh | null;
  glowMesh: THREE.InstancedMesh | null;
  // Individual meshes for each device (more reliable than instanced colors)
  deviceMeshes: Map<string, {
    mesh: THREE.Mesh;
    glowMesh: THREE.Mesh;
    device: ContributorDevice;
    baseY: number;
    phase: number;
  }>;
  devices: Map<string, {
    index: number;
    device: ContributorDevice;
    baseY: number;
    phase: number;
  }>;
  tooltip: CSS2DObject | null;
  tooltipElement: HTMLElement | null;
}

export interface ContributorDevices3DOptions {
  /** Center position of the device cluster */
  position?: [number, number, number];
  /** Radius of the circular arrangement */
  radius?: number;
  /** Maximum devices to display (for performance) */
  maxDevices?: number;
}

// Device mesh geometry - simple laptop/computer shape
function createDeviceGeometry(): THREE.BufferGeometry {
  const group = new THREE.Group();

  // Base (laptop bottom)
  const baseGeometry = new THREE.BoxGeometry(0.3, 0.04, 0.2);

  // Screen (angled up)
  const screenGeometry = new THREE.BoxGeometry(0.28, 0.18, 0.02);

  // Combine into single geometry
  const mergedGeometry = new THREE.BufferGeometry();

  // Create base vertices
  const basePositions = baseGeometry.attributes.position.array;
  const screenPositions = screenGeometry.attributes.position.array;

  // We'll create a simple box for now - the visual is small enough
  return new THREE.BoxGeometry(0.25, 0.15, 0.2);
}

/**
 * Create the contributor devices visualization
 */
export function createContributorDevices3D(
  options: ContributorDevices3DOptions = {}
): ContributorDevices3DRefs {
  const {
    position = [0, 0, 0],
    radius = 3,
    maxDevices = 100,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);

  // Create device geometry - small computer/laptop shape
  const deviceGeometry = new THREE.BoxGeometry(0.2, 0.12, 0.15);

  // Create base material - MeshBasicMaterial for reliable instance colors
  // (doesn't require lighting, color comes purely from instance colors)
  const deviceMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
  });

  // Create instanced mesh for main devices
  const instancedMesh = new THREE.InstancedMesh(
    deviceGeometry,
    deviceMaterial,
    maxDevices
  );
  instancedMesh.count = 0; // Start with no visible instances
  instancedMesh.castShadow = true;
  instancedMesh.receiveShadow = true;

  // Initialize instance colors with a default (required for setColorAt to work properly)
  const defaultColor = new THREE.Color(TIER_COLORS.crowd);
  for (let i = 0; i < maxDevices; i++) {
    instancedMesh.setColorAt(i, defaultColor);
  }
  if (instancedMesh.instanceColor) {
    instancedMesh.instanceColor.needsUpdate = true;
  }

  group.add(instancedMesh);

  // Create glow effect mesh (slightly larger, additive blending)
  const glowGeometry = new THREE.BoxGeometry(0.24, 0.16, 0.19);
  const glowMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.3,
    blending: THREE.AdditiveBlending,
  });
  const glowMesh = new THREE.InstancedMesh(glowGeometry, glowMaterial, maxDevices);
  glowMesh.count = 0;

  // Initialize glow instance colors
  const defaultGlowColor = new THREE.Color(STATUS_COLORS.online);
  for (let i = 0; i < maxDevices; i++) {
    glowMesh.setColorAt(i, defaultGlowColor);
  }
  if (glowMesh.instanceColor) {
    glowMesh.instanceColor.needsUpdate = true;
  }

  group.add(glowMesh);

  // Create tooltip element (hidden by default)
  const tooltipElement = document.createElement('div');
  tooltipElement.className = 'contributor-tooltip';
  tooltipElement.style.cssText = `
    display: none;
    background: rgba(0, 0, 0, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    color: white;
    font-size: 11px;
    pointer-events: none;
    white-space: nowrap;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  `;

  const tooltip = new CSS2DObject(tooltipElement);
  tooltip.position.set(0, 0.5, 0);
  group.add(tooltip);

  return {
    group,
    instancedMesh,
    glowMesh,
    deviceMeshes: new Map(),
    devices: new Map(),
    tooltip,
    tooltipElement,
  };
}

/**
 * Update devices in the visualization
 * Uses individual meshes for reliable coloring (instanced colors are unreliable)
 */
export function updateContributorDevices3D(
  refs: ContributorDevices3DRefs,
  devices: ContributorDevice[],
  options: ContributorDevices3DOptions = {}
): void {
  const { radius = 3, maxDevices = 100 } = options;

  // Remove old device meshes
  refs.deviceMeshes.forEach(({ mesh, glowMesh }) => {
    refs.group.remove(mesh);
    refs.group.remove(glowMesh);
    mesh.geometry.dispose();
    (mesh.material as THREE.Material).dispose();
    glowMesh.geometry.dispose();
    (glowMesh.material as THREE.Material).dispose();
  });
  refs.deviceMeshes.clear();
  refs.devices.clear();

  // Hide instanced meshes completely (not using them anymore)
  if (refs.instancedMesh) {
    refs.instancedMesh.count = 0;
    refs.instancedMesh.visible = false;
  }
  if (refs.glowMesh) {
    refs.glowMesh.count = 0;
    refs.glowMesh.visible = false;
  }

  // Limit devices for performance
  const visibleDevices = devices.slice(0, maxDevices);
  const deviceCount = visibleDevices.length;

  if (deviceCount === 0) return;

  // Group devices by tier for better visual organization
  const powerDevices = visibleDevices.filter(d => d.tier === 'power');
  const standardDevices = visibleDevices.filter(d => d.tier === 'standard');
  const crowdDevices = visibleDevices.filter(d => d.tier === 'crowd');

  // Reorder: power in center ring, standard middle, crowd outer
  const orderedDevices = [...powerDevices, ...standardDevices, ...crowdDevices];

  // Create geometry - like little monitors/laptops
  // Wider than tall, thin depth (like a screen)
  const deviceGeometry = new THREE.BoxGeometry(0.45, 0.35, 0.08);
  const glowGeometry = new THREE.BoxGeometry(0.5, 0.4, 0.12);

  orderedDevices.forEach((device, index) => {
    // Grid layout for multiple devices
    // Arrange in a 2D grid pattern (like a wall of computers)
    const gridSpacing = 0.7;  // Space between devices
    const maxPerRow = 4;      // Max devices per row

    const row = Math.floor(index / maxPerRow);
    const col = index % maxPerRow;

    // Center the grid
    const rowCount = Math.ceil(deviceCount / maxPerRow);
    const colsInThisRow = row === rowCount - 1 ? ((deviceCount - 1) % maxPerRow) + 1 : maxPerRow;
    const rowOffset = (colsInThisRow - 1) * gridSpacing / 2;

    const posX = col * gridSpacing - rowOffset;
    const baseY = 0.3 + row * gridSpacing;  // Stack rows vertically
    const posZ = 0;  // All on same plane


    // Face forward (toward viewer)
    const facingAngle = 0;

    // Scale based on tier (power devices slightly larger) - increased for visibility
    const tierScale = device.tier === 'power' ? 1.5 : device.tier === 'standard' ? 1.3 : 1.1;

    // Create device mesh with tier-colored material
    const tierColorValue = TIER_COLORS[device.tier as keyof typeof TIER_COLORS] || TIER_COLORS.crowd;

    const deviceMaterial = new THREE.MeshToonMaterial({
      color: tierColorValue,
    });
    const mesh = new THREE.Mesh(deviceGeometry.clone(), deviceMaterial);
    mesh.position.set(posX, baseY, posZ);
    mesh.rotation.y = facingAngle;
    mesh.scale.setScalar(tierScale);
    mesh.castShadow = true;
    refs.group.add(mesh);

    // Create glow mesh with status-colored material
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: STATUS_COLORS[device.status] || STATUS_COLORS.online,
      transparent: true,
      opacity: 0.3,
      blending: THREE.AdditiveBlending,
    });
    const glow = new THREE.Mesh(glowGeometry.clone(), glowMaterial);
    glow.position.set(posX, baseY, posZ);
    glow.rotation.y = facingAngle;
    glow.scale.setScalar(tierScale);
    refs.group.add(glow);

    // Store device info for animation and interaction
    const phase = Math.random() * Math.PI * 2;
    refs.deviceMeshes.set(device.id, {
      mesh,
      glowMesh: glow,
      device,
      baseY,
      phase,
    });

    // Also store in devices map for compatibility
    refs.devices.set(device.id, {
      index,
      device,
      baseY,
      phase,
    });
  });
}

/**
 * Animate the contributor devices
 */
export function animateContributorDevices3D(
  refs: ContributorDevices3DRefs,
  time: number,
  options?: {
    hoveredDeviceId?: string | null;
  }
): void {
  const { deviceMeshes, tooltip, tooltipElement } = refs;

  // Animate each device using individual meshes
  deviceMeshes.forEach(({ mesh, glowMesh, device, baseY, phase }) => {
    // Bobbing animation for all devices
    const bobAmount = device.status === 'busy' ? 0.08 : 0.03;
    const bobSpeed = device.status === 'busy' ? 4 : 2;
    mesh.position.y = baseY + Math.sin(time * bobSpeed + phase) * bobAmount;
    glowMesh.position.y = mesh.position.y;

    // Busy devices also rotate slightly
    if (device.status === 'busy') {
      mesh.rotation.y += Math.sin(time * 3 + phase) * 0.002;
      glowMesh.rotation.y = mesh.rotation.y;
    }

    // Pulse scale for busy devices
    if (device.status === 'busy') {
      const pulseScale = 1 + Math.sin(time * 5 + phase) * 0.05;
      const tierScale = device.tier === 'power' ? 1.2 : device.tier === 'standard' ? 1.0 : 0.85;
      mesh.scale.setScalar(tierScale * pulseScale);
      glowMesh.scale.setScalar(tierScale * pulseScale);
    }

    // Animate glow opacity based on status
    const glowIntensity = device.status === 'busy'
      ? 0.4 + Math.sin(time * 6 + phase) * 0.2
      : device.status === 'online'
      ? 0.25 + Math.sin(time * 2 + phase) * 0.1
      : 0.1;

    const glowMaterial = glowMesh.material as THREE.MeshBasicMaterial;
    glowMaterial.opacity = glowIntensity;
  });

  // Handle tooltip for hovered device
  if (options?.hoveredDeviceId && tooltipElement) {
    const deviceInfo = deviceMeshes.get(options.hoveredDeviceId);
    if (deviceInfo) {
      const { mesh, device } = deviceInfo;
      tooltipElement.style.display = 'block';
      const safeName = escapeHtml(device.name);
      const safeTier = escapeHtml(device.tier);
      const safeStatus = escapeHtml(device.status);
      tooltipElement.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 4px;">${safeName}</div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <span style="
            background: ${device.tier === 'power' ? '#a855f7' : device.tier === 'standard' ? '#3b82f6' : '#22c55e'};
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 9px;
            text-transform: uppercase;
          ">${safeTier}</span>
          <span style="color: ${device.status === 'busy' ? '#f59e0b' : device.status === 'online' ? '#22c55e' : '#6b7280'}">
            ${device.status === 'busy' ? 'Processing' : safeStatus}
          </span>
        </div>
        ${device.compute ? `<div style="margin-top: 4px; color: #9ca3af;">${Number(device.compute).toFixed(1)} TFLOPS</div>` : ''}
        ${device.stats ? `<div style="color: #9ca3af;">${Number(device.stats.tasksCompleted)} tasks completed</div>` : ''}
      `;

      // Position tooltip near hovered device
      if (tooltip) {
        tooltip.position.copy(mesh.position);
        tooltip.position.y += 0.4;
      }
    }
  } else if (tooltipElement) {
    tooltipElement.style.display = 'none';
  }
}

/**
 * Get device at raycaster intersection
 */
export function getDeviceAtIntersection(
  refs: ContributorDevices3DRefs,
  raycaster: THREE.Raycaster
): ContributorDevice | null {
  const { deviceMeshes } = refs;

  // Check each individual device mesh
  for (const [deviceId, { mesh, device }] of deviceMeshes) {
    const intersects = raycaster.intersectObject(mesh);
    if (intersects.length > 0) {
      return device;
    }
  }

  return null;
}

/**
 * Dispose of all resources
 */
export function disposeContributorDevices3D(refs: ContributorDevices3DRefs): void {
  const { instancedMesh, glowMesh, deviceMeshes, tooltip, tooltipElement } = refs;

  // Dispose individual device meshes
  deviceMeshes.forEach(({ mesh, glowMesh }) => {
    mesh.geometry.dispose();
    (mesh.material as THREE.Material).dispose();
    glowMesh.geometry.dispose();
    (glowMesh.material as THREE.Material).dispose();
  });
  deviceMeshes.clear();

  if (instancedMesh) {
    instancedMesh.geometry.dispose();
    if (instancedMesh.material instanceof THREE.Material) {
      instancedMesh.material.dispose();
    }
  }

  if (glowMesh) {
    glowMesh.geometry.dispose();
    if (glowMesh.material instanceof THREE.Material) {
      glowMesh.material.dispose();
    }
  }

  if (tooltip) {
    tooltip.removeFromParent();
  }

  if (tooltipElement) {
    tooltipElement.remove();
  }

  refs.devices.clear();
}

/**
 * Create a summary stats label for the device cluster
 */
export function createDeviceStatsLabel(deviceCount: number, busyCount: number): CSS2DObject {
  const container = document.createElement('div');
  container.className = 'device-stats-label';
  container.style.cssText = `
    display: flex;
    flex-direction: column;
    align-items: center;
    pointer-events: none;
    transform: translateY(-10px);
  `;

  const bubble = document.createElement('div');
  bubble.style.cssText = `
    background: rgba(0, 0, 0, 0.8);
    border: 1px solid rgba(34, 197, 94, 0.5);
    border-radius: 8px;
    padding: 6px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.3);
  `;

  const deviceIcon = document.createElement('div');
  deviceIcon.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="2">
      <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
      <line x1="8" y1="21" x2="16" y2="21"/>
      <line x1="12" y1="17" x2="12" y2="21"/>
    </svg>
  `;
  bubble.appendChild(deviceIcon);

  const statsText = document.createElement('div');
  statsText.style.cssText = `
    font-size: 11px;
    color: white;
    font-weight: 500;
  `;
  statsText.innerHTML = `
    <span style="color: #22c55e; font-weight: 600;">${deviceCount}</span> devices
    <span style="color: #6b7280; margin: 0 4px;">|</span>
    <span style="color: #f59e0b;">${busyCount}</span> active
  `;
  bubble.appendChild(statsText);

  container.appendChild(bubble);

  const label = new CSS2DObject(container);
  label.position.set(0, 1.5, 0);

  return label;
}

/**
 * Update the stats label with current counts
 */
export function updateDeviceStatsLabel(
  label: CSS2DObject,
  deviceCount: number,
  busyCount: number,
  tierCounts: { power: number; standard: number; crowd: number }
): void {
  const statsText = label.element.querySelector('div > div:last-child');
  if (statsText) {
    statsText.innerHTML = `
      <span style="color: #22c55e; font-weight: 600;">${deviceCount}</span> devices
      <span style="color: #6b7280; margin: 0 4px;">|</span>
      <span style="color: #f59e0b;">${busyCount}</span> active
    `;
  }
}
