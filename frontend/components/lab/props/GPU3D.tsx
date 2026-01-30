// GPU3D - Chunky stylized graphics card for training activities
// Features: Spinning fans, LED strips, heat wave particles

import * as THREE from 'three';
import { LAB_COLORS } from '../colors';

export interface GPU3DRefs {
  group: THREE.Group;
  fans: THREE.Mesh[];
  leds: THREE.Mesh[];
  heatParticles?: THREE.Points;
}

export interface GPU3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  gpuUtilization?: number;  // 0-100, affects fan speed
  progress?: number;        // 0-100, for LED progress
}

/**
 * Create a cute chunky GPU with fans and LEDs
 */
export function createGPU3D(options: GPU3DOptions): GPU3DRefs {
  const {
    position,
    scale = 1,
    accentColor = 0xff6b6b,
    gpuUtilization = 50,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const fans: THREE.Mesh[] = [];
  const leds: THREE.Mesh[] = [];

  // GPU body (main card)
  const bodyGeometry = new THREE.BoxGeometry(1.2, 0.15, 0.8);
  const bodyMaterial = new THREE.MeshToonMaterial({
    color: 0x2d2d2d,
  });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.position.y = 0.5;
  body.castShadow = true;
  group.add(body);

  // GPU shroud/cover (rounded top)
  const shroudGeometry = new THREE.BoxGeometry(1.15, 0.12, 0.75);
  const shroudMaterial = new THREE.MeshToonMaterial({
    color: 0x1a1a1a,
  });
  const shroud = new THREE.Mesh(shroudGeometry, shroudMaterial);
  shroud.position.y = 0.62;
  group.add(shroud);

  // Accent stripe on top
  const stripeGeometry = new THREE.BoxGeometry(1.1, 0.02, 0.1);
  const stripeMaterial = new THREE.MeshToonMaterial({
    color: accentColor,
  });
  const stripe = new THREE.Mesh(stripeGeometry, stripeMaterial);
  stripe.position.set(0, 0.69, -0.25);
  group.add(stripe);

  // Create two fans
  const fanPositions = [-0.35, 0.35];
  fanPositions.forEach((xPos, idx) => {
    // Fan housing (circular indent)
    const housingGeometry = new THREE.CylinderGeometry(0.22, 0.22, 0.05, 24);
    const housingMaterial = new THREE.MeshToonMaterial({
      color: 0x1a1a1a,
    });
    const housing = new THREE.Mesh(housingGeometry, housingMaterial);
    housing.rotation.x = Math.PI / 2;
    housing.position.set(xPos, 0.52, 0.38);
    group.add(housing);

    // Fan blades (using merged geometry for better performance)
    const fanGroup = new THREE.Group();
    const bladeGeometry = new THREE.BoxGeometry(0.35, 0.02, 0.08);
    const bladeMaterial = new THREE.MeshToonMaterial({
      color: 0x444444,
    });

    for (let i = 0; i < 7; i++) {
      const blade = new THREE.Mesh(bladeGeometry, bladeMaterial);
      blade.rotation.y = (i / 7) * Math.PI * 2;
      fanGroup.add(blade);
    }

    // Fan center hub
    const hubGeometry = new THREE.CylinderGeometry(0.05, 0.05, 0.03, 12);
    const hubMaterial = new THREE.MeshToonMaterial({
      color: accentColor,
    });
    const hub = new THREE.Mesh(hubGeometry, hubMaterial);
    hub.rotation.x = Math.PI / 2;
    fanGroup.add(hub);

    fanGroup.position.set(xPos, 0.52, 0.39);
    fanGroup.rotation.x = Math.PI / 2;
    fanGroup.userData = { fanIndex: idx, baseSpeed: 2 + idx * 0.5 };
    group.add(fanGroup);
    fans.push(fanGroup as unknown as THREE.Mesh);
  });

  // LED strip on side
  const ledCount = 8;
  for (let i = 0; i < ledCount; i++) {
    const ledGeometry = new THREE.BoxGeometry(0.03, 0.04, 0.03);
    const ledMaterial = new THREE.MeshBasicMaterial({
      color: accentColor,
      transparent: true,
      opacity: 0.3,
    });
    const led = new THREE.Mesh(ledGeometry, ledMaterial);
    led.position.set(-0.5 + (i / (ledCount - 1)) * 1, 0.52, -0.36);
    led.userData = { ledIndex: i };
    group.add(led);
    leds.push(led);
  }

  // PCIe connector (back)
  const pcieGeometry = new THREE.BoxGeometry(0.8, 0.08, 0.05);
  const pcieMaterial = new THREE.MeshToonMaterial({
    color: 0xc4a24b,  // Gold pins
  });
  const pcie = new THREE.Mesh(pcieGeometry, pcieMaterial);
  pcie.position.set(0, 0.44, -0.42);
  group.add(pcie);

  // Backplate
  const backplateGeometry = new THREE.BoxGeometry(1.1, 0.02, 0.7);
  const backplateMaterial = new THREE.MeshToonMaterial({
    color: 0x333333,
  });
  const backplate = new THREE.Mesh(backplateGeometry, backplateMaterial);
  backplate.position.y = 0.42;
  group.add(backplate);

  // Heat sink fins (visible from back)
  const finCount = 15;
  for (let i = 0; i < finCount; i++) {
    const finGeometry = new THREE.BoxGeometry(0.02, 0.08, 0.5);
    const finMaterial = new THREE.MeshToonMaterial({
      color: 0x666666,
    });
    const fin = new THREE.Mesh(finGeometry, finMaterial);
    fin.position.set(-0.5 + (i / (finCount - 1)) * 1, 0.46, 0);
    group.add(fin);
  }

  // Create heat particle system
  const particleCount = 50;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);

  const heatColor1 = new THREE.Color(0xff6b6b);
  const heatColor2 = new THREE.Color(0xffa500);

  for (let i = 0; i < particleCount; i++) {
    // Start around the GPU top
    positions[i * 3] = (Math.random() - 0.5) * 1;
    positions[i * 3 + 1] = 0.7 + Math.random() * 0.3;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 0.6;

    const mixFactor = Math.random();
    const color = heatColor1.clone().lerp(heatColor2, mixFactor);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;

    sizes[i] = 0.03 + Math.random() * 0.04;
  }

  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const particleMaterial = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
  });

  const heatParticles = new THREE.Points(particleGeometry, particleMaterial);
  heatParticles.visible = gpuUtilization > 30;
  group.add(heatParticles);

  // Store utilization for animation
  group.userData = { gpuUtilization, accentColor };

  return {
    group,
    fans,
    leds,
    heatParticles,
  };
}

/**
 * Animate the GPU (fans, LEDs, particles)
 */
export function animateGPU3D(
  refs: GPU3DRefs,
  time: number,
  options?: {
    gpuUtilization?: number;
    progress?: number;
  }
): void {
  const gpuUtilization = options?.gpuUtilization ?? refs.group.userData.gpuUtilization ?? 50;
  const progress = options?.progress ?? 0;

  // Animate fans - speed based on utilization
  const fanSpeedMultiplier = 0.5 + (gpuUtilization / 100) * 2.5;
  refs.fans.forEach((fan, idx) => {
    const baseSpeed = fan.userData?.baseSpeed || 2;
    (fan as unknown as THREE.Group).rotation.y += baseSpeed * fanSpeedMultiplier * 0.1;
  });

  // Animate LEDs - show progress as lit LEDs
  refs.leds.forEach((led, idx) => {
    const material = led.material as THREE.MeshBasicMaterial;
    const ledProgress = (idx / (refs.leds.length - 1)) * 100;

    if (ledProgress <= progress) {
      // Lit LED with pulse
      const pulse = Math.sin(time * 4 + idx * 0.5) * 0.3 + 0.7;
      material.opacity = pulse;
    } else {
      // Dim LED
      material.opacity = 0.15;
    }
  });

  // Animate heat particles
  if (refs.heatParticles) {
    refs.heatParticles.visible = gpuUtilization > 30;

    if (refs.heatParticles.visible) {
      const positions = refs.heatParticles.geometry.attributes.position.array as Float32Array;

      for (let i = 0; i < positions.length / 3; i++) {
        // Rise and spread
        positions[i * 3 + 1] += (gpuUtilization / 100) * 0.02;
        positions[i * 3] += (Math.random() - 0.5) * 0.01;

        // Reset when too high
        if (positions[i * 3 + 1] > 2) {
          positions[i * 3] = (Math.random() - 0.5) * 1;
          positions[i * 3 + 1] = 0.7;
          positions[i * 3 + 2] = (Math.random() - 0.5) * 0.6;
        }
      }

      refs.heatParticles.geometry.attributes.position.needsUpdate = true;

      // Adjust opacity based on utilization
      const material = refs.heatParticles.material as THREE.PointsMaterial;
      material.opacity = 0.3 + (gpuUtilization / 100) * 0.5;
    }
  }
}

/**
 * Update GPU accent color (for different training types)
 */
export function setGPU3DColor(refs: GPU3DRefs, color: number): void {
  refs.group.userData.accentColor = color;

  // Update LEDs
  refs.leds.forEach((led) => {
    const material = led.material as THREE.MeshBasicMaterial;
    material.color.setHex(color);
  });
}

/**
 * Dispose GPU resources
 */
export function disposeGPU3D(refs: GPU3DRefs): void {
  refs.group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });

  if (refs.heatParticles) {
    refs.heatParticles.geometry.dispose();
    (refs.heatParticles.material as THREE.Material).dispose();
  }
}
