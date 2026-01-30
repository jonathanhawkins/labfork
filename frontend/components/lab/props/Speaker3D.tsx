// Speaker3D - Cute cartoon speaker for generation/playback activities
// Features: Bouncing animation when playing, musical note particles

import * as THREE from 'three';

export interface Speaker3DRefs {
  group: THREE.Group;
  cone: THREE.Mesh;
  body: THREE.Mesh;
  particles?: THREE.Points;
}

export interface Speaker3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  isPlaying?: boolean;
}

/**
 * Create a cute cartoon speaker
 */
export function createSpeaker3D(options: Speaker3DOptions): Speaker3DRefs {
  const {
    position,
    scale = 1,
    accentColor = 0xffe66d,
    isPlaying = false,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  // Speaker body (rounded box shape)
  const bodyGeometry = new THREE.BoxGeometry(0.6, 0.8, 0.4);
  const bodyMaterial = new THREE.MeshToonMaterial({
    color: 0x2d2d2d,
  });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.position.y = 0.4;
  body.castShadow = true;
  group.add(body);

  // Speaker front panel (slightly lighter)
  const frontGeometry = new THREE.BoxGeometry(0.58, 0.78, 0.02);
  const frontMaterial = new THREE.MeshToonMaterial({
    color: 0x1a1a1a,
  });
  const front = new THREE.Mesh(frontGeometry, frontMaterial);
  front.position.set(0, 0.4, 0.2);
  group.add(front);

  // Main speaker cone (large)
  const coneGeometry = new THREE.CylinderGeometry(0.15, 0.2, 0.08, 24);
  const coneMaterial = new THREE.MeshToonMaterial({
    color: 0x555555,
  });
  const cone = new THREE.Mesh(coneGeometry, coneMaterial);
  cone.rotation.x = Math.PI / 2;
  cone.position.set(0, 0.5, 0.22);
  group.add(cone);

  // Cone center (dust cap)
  const dustCapGeometry = new THREE.SphereGeometry(0.06, 12, 12, 0, Math.PI * 2, 0, Math.PI / 2);
  const dustCapMaterial = new THREE.MeshToonMaterial({
    color: 0x333333,
  });
  const dustCap = new THREE.Mesh(dustCapGeometry, dustCapMaterial);
  dustCap.rotation.x = -Math.PI / 2;
  dustCap.position.set(0, 0.5, 0.25);
  group.add(dustCap);

  // Cone ring (accent color)
  const ringGeometry = new THREE.TorusGeometry(0.18, 0.015, 8, 24);
  const ringMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.8,
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.position.set(0, 0.5, 0.22);
  group.add(ring);

  // Tweeter (small speaker at top)
  const tweeterGeometry = new THREE.CylinderGeometry(0.05, 0.07, 0.04, 16);
  const tweeterMaterial = new THREE.MeshToonMaterial({
    color: 0x444444,
  });
  const tweeter = new THREE.Mesh(tweeterGeometry, tweeterMaterial);
  tweeter.rotation.x = Math.PI / 2;
  tweeter.position.set(0, 0.7, 0.22);
  group.add(tweeter);

  // Port (bass reflex hole)
  const portGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.05, 12);
  const portMaterial = new THREE.MeshBasicMaterial({
    color: 0x0a0a0a,
  });
  const port = new THREE.Mesh(portGeometry, portMaterial);
  port.rotation.x = Math.PI / 2;
  port.position.set(0, 0.2, 0.2);
  group.add(port);

  // Status LED
  const ledGeometry = new THREE.CircleGeometry(0.02, 12);
  const ledMaterial = new THREE.MeshBasicMaterial({
    color: isPlaying ? 0x00ff00 : 0x004400,
  });
  const led = new THREE.Mesh(ledGeometry, ledMaterial);
  led.position.set(0.2, 0.1, 0.21);
  group.add(led);

  // Musical note particles
  const particleCount = 20;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);
  const phases = new Float32Array(particleCount);

  const particleColor = new THREE.Color(accentColor);

  for (let i = 0; i < particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 0.3;
    positions[i * 3 + 1] = 0.5 + Math.random() * 0.5;
    positions[i * 3 + 2] = 0.3 + Math.random() * 0.3;

    colors[i * 3] = particleColor.r;
    colors[i * 3 + 1] = particleColor.g;
    colors[i * 3 + 2] = particleColor.b;

    sizes[i] = 0.04 + Math.random() * 0.03;
    phases[i] = Math.random() * Math.PI * 2;
  }

  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  particleGeometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));

  const particleMaterial = new THREE.PointsMaterial({
    size: 0.06,
    vertexColors: true,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
  });

  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.visible = false;
  group.add(particles);

  group.userData = { isPlaying, accentColor, bouncePhase: 0 };

  return {
    group,
    cone,
    body,
    particles,
  };
}

/**
 * Animate the speaker
 */
export function animateSpeaker3D(
  refs: Speaker3DRefs,
  time: number,
  options?: {
    isPlaying?: boolean;
    audioLevel?: number;
  }
): void {
  const isPlaying = options?.isPlaying ?? refs.group.userData.isPlaying ?? false;
  const audioLevel = options?.audioLevel ?? (isPlaying ? 0.5 + Math.sin(time * 8) * 0.3 : 0);

  refs.group.userData.isPlaying = isPlaying;

  // Bounce the whole speaker when playing
  if (isPlaying) {
    refs.group.userData.bouncePhase += 0.2;
    const bounce = Math.abs(Math.sin(refs.group.userData.bouncePhase)) * 0.05;
    refs.body.position.y = 0.4 + bounce * audioLevel;
    refs.body.scale.x = 1 + Math.sin(time * 10) * 0.02 * audioLevel;
    refs.body.scale.z = 1 - Math.sin(time * 10) * 0.02 * audioLevel;
  } else {
    refs.body.position.y = 0.4;
    refs.body.scale.set(1, 1, 1);
  }

  // Cone pumping animation
  if (isPlaying) {
    const pumpAmount = audioLevel * 0.03;
    refs.cone.position.z = 0.22 + Math.sin(time * 15) * pumpAmount;
  } else {
    refs.cone.position.z = 0.22;
  }

  // Animate particles (floating musical notes)
  if (refs.particles) {
    refs.particles.visible = isPlaying;

    if (isPlaying) {
      const positions = refs.particles.geometry.attributes.position.array as Float32Array;
      const phases = refs.particles.geometry.attributes.phase.array as Float32Array;
      const particleCount = positions.length / 3;

      for (let i = 0; i < particleCount; i++) {
        const phase = phases[i];

        // Float upward and drift
        positions[i * 3] += Math.sin(time * 2 + phase) * 0.005;
        positions[i * 3 + 1] += 0.01 + audioLevel * 0.01;
        positions[i * 3 + 2] += 0.005;

        // Reset when too high or far
        if (positions[i * 3 + 1] > 2 || positions[i * 3 + 2] > 1.5) {
          positions[i * 3] = (Math.random() - 0.5) * 0.3;
          positions[i * 3 + 1] = 0.5;
          positions[i * 3 + 2] = 0.3;
        }
      }

      refs.particles.geometry.attributes.position.needsUpdate = true;

      const particleMaterial = refs.particles.material as THREE.PointsMaterial;
      particleMaterial.opacity = 0.5 + audioLevel * 0.3;
    }
  }
}

/**
 * Dispose speaker resources
 */
export function disposeSpeaker3D(refs: Speaker3DRefs): void {
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
