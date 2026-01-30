"use client";

import * as THREE from "three";
import { LAB_COLORS } from "./colors";

export interface DataHubRefs {
  group: THREE.Group;
  rings: THREE.Mesh[];
}

/**
 * Create a central data hub visualization with rotating rings
 * Represents the nexus where agents share information
 */
export function createDataHub3D(
  position: THREE.Vector3 = new THREE.Vector3(0, 2, 0),
  ringCount: number = 2,
  radius: number = 1,
  color: number = LAB_COLORS.hub
): DataHubRefs {
  const group = new THREE.Group();
  const rings: THREE.Mesh[] = [];

  const hubGeometry = new THREE.TorusGeometry(radius, 0.1, 8, 32);
  const hubMaterial = new THREE.MeshToonMaterial({
    color,
    transparent: true,
    opacity: 0.6,
  });

  for (let i = 0; i < ringCount; i++) {
    const ring = new THREE.Mesh(hubGeometry, hubMaterial.clone());
    ring.rotation.x = Math.PI / 2;
    ring.rotation.z = (i * Math.PI) / ringCount;
    ring.name = `hubRing${i}`;
    group.add(ring);
    rings.push(ring);
  }

  // Central core sphere
  const coreGeometry = new THREE.SphereGeometry(0.15, 16, 16);
  const coreMaterial = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.8,
  });
  const core = new THREE.Mesh(coreGeometry, coreMaterial);
  core.name = "hubCore";
  group.add(core);

  group.position.copy(position);

  return { group, rings };
}

/**
 * Animate the data hub with rotating rings and pulsing core
 */
export function animateDataHub(refs: DataHubRefs, time: number) {
  refs.rings.forEach((ring, i) => {
    const direction = i % 2 === 0 ? 1 : -1;
    const speed = 0.5 - i * 0.2;
    ring.rotation.z = time * speed * direction;
    ring.scale.setScalar(1 + Math.sin(time * 2 + i) * 0.1);
  });

  // Pulse the core
  const core = refs.group.getObjectByName("hubCore") as THREE.Mesh;
  if (core) {
    const scale = 1 + Math.sin(time * 3) * 0.2;
    core.scale.setScalar(scale);
  }
}

/**
 * Create connection lines from hub to agents
 */
export function createHubConnections(
  hubPosition: THREE.Vector3,
  agentPositions: THREE.Vector3[],
  color: number = LAB_COLORS.hub
): THREE.Line[] {
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0.3,
  });

  return agentPositions.map((agentPos) => {
    const points = [hubPosition.clone(), agentPos.clone().setY(1.5)];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    return new THREE.Line(geometry, material);
  });
}
