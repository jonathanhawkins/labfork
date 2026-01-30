"use client";

import * as THREE from "three";
import { LAB_COLORS } from "./colors";

export interface ParticleSystemRefs {
  particles: THREE.Points;
  positions: Float32Array;
}

/**
 * Create floating data particles that orbit around the scene
 * Represents data flowing between agents
 */
export function createDataParticles3D(
  particleCount: number = 100,
  radius: number = 5,
  colors?: number[]
): ParticleSystemRefs {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const particleColors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);

  // Default color palette
  const colorPalette = colors || [LAB_COLORS.codex, LAB_COLORS.opus, LAB_COLORS.particles];
  const threeColors = colorPalette.map((c) => new THREE.Color(c));

  for (let i = 0; i < particleCount; i++) {
    // Random positions in a ring around the scene
    const angle = Math.random() * Math.PI * 2;
    const r = 1 + Math.random() * radius;
    positions[i * 3] = Math.cos(angle) * r;
    positions[i * 3 + 1] = 1 + Math.random() * 3;
    positions[i * 3 + 2] = Math.sin(angle) * r;

    // Random colors from palette
    const chosenColor = threeColors[Math.floor(Math.random() * threeColors.length)];
    particleColors[i * 3] = chosenColor.r;
    particleColors[i * 3 + 1] = chosenColor.g;
    particleColors[i * 3 + 2] = chosenColor.b;

    sizes[i] = 0.05 + Math.random() * 0.1;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(particleColors, 3));
  geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

  const material = new THREE.PointsMaterial({
    size: 0.1,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    blending: THREE.AdditiveBlending,
  });

  const particles = new THREE.Points(geometry, material);

  return { particles, positions };
}

/**
 * Animate particles - orbital motion with vertical oscillation
 */
export function animateDataParticles(refs: ParticleSystemRefs, time: number) {
  const positions = refs.particles.geometry.attributes.position.array as Float32Array;

  for (let i = 0; i < positions.length / 3; i++) {
    const idx = i * 3;
    const x = positions[idx];
    const z = positions[idx + 2];

    // Orbital motion
    const angle = Math.atan2(z, x) + 0.01;
    const radius = Math.sqrt(x * x + z * z);
    positions[idx] = Math.cos(angle) * radius;
    positions[idx + 2] = Math.sin(angle) * radius;

    // Vertical oscillation
    positions[idx + 1] += Math.sin(time + i) * 0.005;

    // Keep particles in bounds
    if (positions[idx + 1] > 4) positions[idx + 1] = 1;
    if (positions[idx + 1] < 1) positions[idx + 1] = 4;
  }

  refs.particles.geometry.attributes.position.needsUpdate = true;
}

/**
 * Create particle stream between two points (for data transfer visualization)
 */
export function createParticleStream(
  from: THREE.Vector3,
  to: THREE.Vector3,
  count: number = 20,
  color: number = LAB_COLORS.particles
): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const t = i / count;
    positions[i * 3] = from.x + (to.x - from.x) * t;
    positions[i * 3 + 1] = from.y + (to.y - from.y) * t + Math.sin(t * Math.PI) * 0.5;
    positions[i * 3 + 2] = from.z + (to.z - from.z) * t;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    size: 0.08,
    color,
    transparent: true,
    opacity: 0.7,
    blending: THREE.AdditiveBlending,
  });

  return new THREE.Points(geometry, material);
}
