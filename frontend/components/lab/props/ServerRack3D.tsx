// ServerRack3D - Mini server rack for inference activities
// Features: Blinking status LEDs, data flow particles

import * as THREE from 'three';

export interface ServerRack3DRefs {
  group: THREE.Group;
  leds: THREE.Mesh[];
  particles?: THREE.Points;
}

export interface ServerRack3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  isProcessing?: boolean;
  status?: 'idle' | 'processing' | 'ready' | 'error';
}

/**
 * Create a cute mini server rack
 */
export function createServerRack3D(options: ServerRack3DOptions): ServerRack3DRefs {
  const {
    position,
    scale = 1,
    accentColor = 0x3b82f6,
    status = 'idle',
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const leds: THREE.Mesh[] = [];

  // Server rack cabinet
  const cabinetGeometry = new THREE.BoxGeometry(0.5, 0.8, 0.4);
  const cabinetMaterial = new THREE.MeshToonMaterial({
    color: 0x1a1a1a,
  });
  const cabinet = new THREE.Mesh(cabinetGeometry, cabinetMaterial);
  cabinet.position.y = 0.4;
  cabinet.castShadow = true;
  group.add(cabinet);

  // Server units (3 stacked)
  const serverColors = [0x2d2d2d, 0x333333, 0x2d2d2d];
  for (let i = 0; i < 3; i++) {
    const serverGeometry = new THREE.BoxGeometry(0.45, 0.18, 0.35);
    const serverMaterial = new THREE.MeshToonMaterial({
      color: serverColors[i],
    });
    const server = new THREE.Mesh(serverGeometry, serverMaterial);
    server.position.set(0, 0.2 + i * 0.22, 0.02);
    group.add(server);

    // Front panel details (ventilation)
    const ventGeometry = new THREE.PlaneGeometry(0.35, 0.12);
    const ventMaterial = new THREE.MeshBasicMaterial({
      color: 0x1a1a1a,
      transparent: true,
      opacity: 0.8,
    });
    const vent = new THREE.Mesh(ventGeometry, ventMaterial);
    vent.position.set(-0.03, 0.2 + i * 0.22, 0.2);
    group.add(vent);

    // Status LEDs for each server
    const ledGeometry = new THREE.CircleGeometry(0.015, 8);
    for (let j = 0; j < 3; j++) {
      const ledColor = j === 0 ? 0x00ff00 : j === 1 ? 0xffff00 : 0x0066ff;
      const ledMaterial = new THREE.MeshBasicMaterial({
        color: ledColor,
        transparent: true,
        opacity: 0.3,
      });
      const led = new THREE.Mesh(ledGeometry, ledMaterial);
      led.position.set(0.15 + j * 0.03, 0.2 + i * 0.22, 0.2);
      led.userData = { serverIndex: i, ledIndex: j };
      group.add(led);
      leds.push(led);
    }
  }

  // Ethernet ports on side
  for (let i = 0; i < 4; i++) {
    const portGeometry = new THREE.BoxGeometry(0.03, 0.02, 0.02);
    const portMaterial = new THREE.MeshBasicMaterial({
      color: i === 0 ? accentColor : 0x333333,
    });
    const port = new THREE.Mesh(portGeometry, portMaterial);
    port.position.set(0.26, 0.3 + i * 0.1, 0);
    group.add(port);
  }

  // Top vent
  const topVentGeometry = new THREE.PlaneGeometry(0.4, 0.3);
  const topVentMaterial = new THREE.MeshToonMaterial({
    color: 0x222222,
  });
  const topVent = new THREE.Mesh(topVentGeometry, topVentMaterial);
  topVent.rotation.x = -Math.PI / 2;
  topVent.position.y = 0.81;
  group.add(topVent);

  // Power button
  const powerGeometry = new THREE.CircleGeometry(0.025, 12);
  const powerMaterial = new THREE.MeshBasicMaterial({
    color: status === 'idle' ? 0x004400 : 0x00ff00,
  });
  const power = new THREE.Mesh(powerGeometry, powerMaterial);
  power.position.set(-0.18, 0.2, 0.2);
  group.add(power);

  // Data flow particles
  const particleCount = 40;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);
  const velocities = new Float32Array(particleCount * 3);

  const particleColor = new THREE.Color(accentColor);

  for (let i = 0; i < particleCount; i++) {
    // Start around the server
    positions[i * 3] = (Math.random() - 0.5) * 0.6;
    positions[i * 3 + 1] = Math.random() * 0.8;
    positions[i * 3 + 2] = 0.2 + Math.random() * 0.3;

    colors[i * 3] = particleColor.r;
    colors[i * 3 + 1] = particleColor.g;
    colors[i * 3 + 2] = particleColor.b;

    sizes[i] = 0.02 + Math.random() * 0.02;

    // Random velocity
    velocities[i * 3] = (Math.random() - 0.5) * 0.02;
    velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
    velocities[i * 3 + 2] = 0.01 + Math.random() * 0.02;
  }

  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const particleMaterial = new THREE.PointsMaterial({
    size: 0.04,
    vertexColors: true,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
  });

  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.visible = false;
  particles.userData = { velocities };
  group.add(particles);

  group.userData = { status, accentColor, isProcessing: status === 'processing' };

  return {
    group,
    leds,
    particles,
  };
}

/**
 * Animate the server rack
 */
export function animateServerRack3D(
  refs: ServerRack3DRefs,
  time: number,
  options?: {
    isProcessing?: boolean;
    status?: 'idle' | 'processing' | 'ready' | 'error';
    loadLevel?: number;  // 0-1
  }
): void {
  const isProcessing = options?.isProcessing ?? refs.group.userData.isProcessing ?? false;
  const status = options?.status ?? refs.group.userData.status ?? 'idle';
  const loadLevel = options?.loadLevel ?? (isProcessing ? 0.7 : 0.2);

  refs.group.userData.isProcessing = isProcessing;
  refs.group.userData.status = status;

  // Animate LEDs (blinking patterns based on load)
  refs.leds.forEach((led) => {
    const material = led.material as THREE.MeshBasicMaterial;
    const { serverIndex, ledIndex } = led.userData;

    if (isProcessing) {
      // Activity pattern
      const phase = (time * 5 + serverIndex * 2 + ledIndex) % 4;
      if (ledIndex === 0) {
        // Power LED - steady green
        material.opacity = 0.8;
      } else if (ledIndex === 1) {
        // Activity LED - blinking based on load
        material.opacity = phase < 2 ? 0.9 : 0.2;
      } else {
        // Network LED - fast blink
        material.opacity = Math.sin(time * 20 + serverIndex) > 0 ? 0.8 : 0.2;
      }
    } else {
      // Idle state
      material.opacity = ledIndex === 0 ? 0.5 : 0.15;
    }

    // Error state
    if (status === 'error' && ledIndex === 1) {
      material.color.setHex(0xff0000);
      material.opacity = Math.sin(time * 10) > 0 ? 0.9 : 0.3;
    }
  });

  // Animate particles
  if (refs.particles) {
    refs.particles.visible = isProcessing;

    if (isProcessing) {
      const positions = refs.particles.geometry.attributes.position.array as Float32Array;
      const velocities = refs.particles.userData.velocities as Float32Array;
      const particleCount = positions.length / 3;

      for (let i = 0; i < particleCount; i++) {
        // Move particles
        positions[i * 3] += velocities[i * 3] * loadLevel;
        positions[i * 3 + 1] += velocities[i * 3 + 1] * loadLevel;
        positions[i * 3 + 2] += velocities[i * 3 + 2] * loadLevel;

        // Reset if too far
        if (
          positions[i * 3 + 2] > 1.5 ||
          Math.abs(positions[i * 3]) > 1 ||
          Math.abs(positions[i * 3 + 1]) > 1.5
        ) {
          positions[i * 3] = (Math.random() - 0.5) * 0.3;
          positions[i * 3 + 1] = Math.random() * 0.6 + 0.1;
          positions[i * 3 + 2] = 0.2;
        }
      }

      refs.particles.geometry.attributes.position.needsUpdate = true;

      const particleMaterial = refs.particles.material as THREE.PointsMaterial;
      particleMaterial.opacity = 0.4 + loadLevel * 0.4;
    }
  }
}

/**
 * Dispose server rack resources
 */
export function disposeServerRack3D(refs: ServerRack3DRefs): void {
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

  if (refs.particles) {
    refs.particles.geometry.dispose();
    (refs.particles.material as THREE.Material).dispose();
  }
}
