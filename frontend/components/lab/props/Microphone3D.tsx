// Microphone3D - Cute retro microphone for recording activities
// Features: Pulsing ring when active, sound wave particles

import * as THREE from 'three';

export interface Microphone3DRefs {
  group: THREE.Group;
  micHead: THREE.Mesh;
  pulseRing: THREE.Mesh;
  particles?: THREE.Points;
  recordingLight: THREE.Mesh;
}

export interface Microphone3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  isRecording?: boolean;
}

/**
 * Create a cute retro-style microphone
 */
export function createMicrophone3D(options: Microphone3DOptions): Microphone3DRefs {
  const {
    position,
    scale = 1,
    accentColor = 0x4ecdc4,
    isRecording = false,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  // Stand base (circular platform)
  const baseGeometry = new THREE.CylinderGeometry(0.25, 0.3, 0.08, 24);
  const baseMaterial = new THREE.MeshToonMaterial({
    color: 0x2d2d2d,
  });
  const base = new THREE.Mesh(baseGeometry, baseMaterial);
  base.position.y = 0.04;
  base.castShadow = true;
  group.add(base);

  // Stand pole
  const poleGeometry = new THREE.CylinderGeometry(0.03, 0.04, 0.6, 12);
  const poleMaterial = new THREE.MeshToonMaterial({
    color: 0x4a4a4a,
  });
  const pole = new THREE.Mesh(poleGeometry, poleMaterial);
  pole.position.y = 0.38;
  group.add(pole);

  // Mic holder/mount
  const mountGeometry = new THREE.TorusGeometry(0.15, 0.025, 8, 24);
  const mountMaterial = new THREE.MeshToonMaterial({
    color: 0x333333,
  });
  const mount = new THREE.Mesh(mountGeometry, mountMaterial);
  mount.position.y = 0.68;
  mount.rotation.x = Math.PI / 2;
  group.add(mount);

  // Microphone head (pill shape)
  const micHeadGeometry = new THREE.CapsuleGeometry(0.12, 0.25, 12, 24);
  const micHeadMaterial = new THREE.MeshToonMaterial({
    color: 0x666666,
  });
  const micHead = new THREE.Mesh(micHeadGeometry, micHeadMaterial);
  micHead.position.y = 0.95;
  micHead.castShadow = true;
  group.add(micHead);

  // Mesh grille pattern (darker band)
  const grilleGeometry = new THREE.CylinderGeometry(0.121, 0.121, 0.18, 24);
  const grilleMaterial = new THREE.MeshToonMaterial({
    color: 0x444444,
  });
  const grille = new THREE.Mesh(grilleGeometry, grilleMaterial);
  grille.position.y = 0.95;
  group.add(grille);

  // Recording indicator light
  const lightGeometry = new THREE.SphereGeometry(0.03, 8, 8);
  const lightMaterial = new THREE.MeshBasicMaterial({
    color: isRecording ? 0xff0000 : 0x440000,
    transparent: true,
    opacity: isRecording ? 1 : 0.3,
  });
  const recordingLight = new THREE.Mesh(lightGeometry, lightMaterial);
  recordingLight.position.set(0, 0.7, 0.06);
  group.add(recordingLight);

  // Pulse ring (for recording animation)
  const pulseRingGeometry = new THREE.TorusGeometry(0.2, 0.02, 8, 32);
  const pulseRingMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0,
  });
  const pulseRing = new THREE.Mesh(pulseRingGeometry, pulseRingMaterial);
  pulseRing.position.y = 0.95;
  pulseRing.rotation.x = Math.PI / 2;
  group.add(pulseRing);

  // Accent ring on base
  const accentRingGeometry = new THREE.TorusGeometry(0.27, 0.015, 8, 32);
  const accentRingMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.7,
  });
  const accentRing = new THREE.Mesh(accentRingGeometry, accentRingMaterial);
  accentRing.position.y = 0.08;
  accentRing.rotation.x = Math.PI / 2;
  group.add(accentRing);

  // Sound wave particles
  const particleCount = 60;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);
  const phases = new Float32Array(particleCount);

  const particleColor = new THREE.Color(accentColor);

  for (let i = 0; i < particleCount; i++) {
    // Start at mic head
    const angle = (i / particleCount) * Math.PI * 2;
    positions[i * 3] = Math.cos(angle) * 0.15;
    positions[i * 3 + 1] = 0.95;
    positions[i * 3 + 2] = Math.sin(angle) * 0.15;

    colors[i * 3] = particleColor.r;
    colors[i * 3 + 1] = particleColor.g;
    colors[i * 3 + 2] = particleColor.b;

    sizes[i] = 0.03 + Math.random() * 0.02;
    phases[i] = Math.random() * Math.PI * 2;
  }

  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  particleGeometry.setAttribute('phase', new THREE.BufferAttribute(phases, 1));

  const particleMaterial = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
  });

  const particles = new THREE.Points(particleGeometry, particleMaterial);
  particles.visible = false;
  group.add(particles);

  // Store state
  group.userData = { isRecording, accentColor, pulsePhase: 0 };

  return {
    group,
    micHead,
    pulseRing,
    particles,
    recordingLight,
  };
}

/**
 * Animate the microphone
 */
export function animateMicrophone3D(
  refs: Microphone3DRefs,
  time: number,
  options?: {
    isRecording?: boolean;
    audioLevel?: number;  // 0-1, for visualizing audio input
  }
): void {
  const isRecording = options?.isRecording ?? refs.group.userData.isRecording ?? false;
  const audioLevel = options?.audioLevel ?? (isRecording ? 0.5 + Math.sin(time * 10) * 0.3 : 0);

  refs.group.userData.isRecording = isRecording;

  // Recording light pulse
  const lightMaterial = refs.recordingLight.material as THREE.MeshBasicMaterial;
  if (isRecording) {
    lightMaterial.color.setHex(0xff0000);
    lightMaterial.opacity = 0.7 + Math.sin(time * 6) * 0.3;
  } else {
    lightMaterial.color.setHex(0x440000);
    lightMaterial.opacity = 0.3;
  }

  // Pulse ring animation
  const ringMaterial = refs.pulseRing.material as THREE.MeshBasicMaterial;
  if (isRecording) {
    refs.group.userData.pulsePhase += 0.05;
    const pulsePhase = refs.group.userData.pulsePhase;

    // Expand and fade
    const scale = 1 + (pulsePhase % 2) * 0.5;
    const opacity = Math.max(0, 0.8 - (pulsePhase % 2) * 0.4);

    refs.pulseRing.scale.setScalar(scale);
    ringMaterial.opacity = opacity * (0.5 + audioLevel * 0.5);
  } else {
    refs.pulseRing.scale.setScalar(1);
    ringMaterial.opacity = 0;
  }

  // Mic head bob when recording
  if (isRecording) {
    refs.micHead.rotation.z = Math.sin(time * 2) * 0.05;
    refs.micHead.position.y = 0.95 + audioLevel * 0.02;
  } else {
    refs.micHead.rotation.z = 0;
    refs.micHead.position.y = 0.95;
  }

  // Animate particles (sound waves)
  if (refs.particles) {
    refs.particles.visible = isRecording;

    if (isRecording) {
      const positions = refs.particles.geometry.attributes.position.array as Float32Array;
      const phases = refs.particles.geometry.attributes.phase.array as Float32Array;
      const particleCount = positions.length / 3;

      for (let i = 0; i < particleCount; i++) {
        const phase = phases[i];
        const waveProgress = ((time * 2 + phase) % 2);

        // Expand outward in waves
        const angle = (i / particleCount) * Math.PI * 2;
        const radius = 0.15 + waveProgress * 0.8;
        const waveHeight = Math.sin(waveProgress * Math.PI) * 0.3 * audioLevel;

        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] = 0.95 + waveHeight;
        positions[i * 3 + 2] = Math.sin(angle) * radius;
      }

      refs.particles.geometry.attributes.position.needsUpdate = true;

      const particleMaterial = refs.particles.material as THREE.PointsMaterial;
      particleMaterial.opacity = 0.4 + audioLevel * 0.4;
    }
  }
}

/**
 * Set recording state
 */
export function setMicrophoneRecording(refs: Microphone3DRefs, isRecording: boolean): void {
  refs.group.userData.isRecording = isRecording;
  refs.group.userData.pulsePhase = 0;
}

/**
 * Update microphone accent color
 */
export function setMicrophone3DColor(refs: Microphone3DRefs, color: number): void {
  refs.group.userData.accentColor = color;

  // Update pulse ring
  const ringMaterial = refs.pulseRing.material as THREE.MeshBasicMaterial;
  ringMaterial.color.setHex(color);

  // Update particles
  if (refs.particles) {
    const colors = refs.particles.geometry.attributes.color.array as Float32Array;
    const particleColor = new THREE.Color(color);
    const particleCount = colors.length / 3;

    for (let i = 0; i < particleCount; i++) {
      colors[i * 3] = particleColor.r;
      colors[i * 3 + 1] = particleColor.g;
      colors[i * 3 + 2] = particleColor.b;
    }

    refs.particles.geometry.attributes.color.needsUpdate = true;
  }
}

/**
 * Dispose microphone resources
 */
export function disposeMicrophone3D(refs: Microphone3DRefs): void {
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
