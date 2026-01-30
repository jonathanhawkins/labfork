// WaveformDisplay3D - Floating 3D audio waveform visualization
// Features: Real-time audio visualization, glow effects

import * as THREE from 'three';

export interface WaveformDisplay3DRefs {
  group: THREE.Group;
  waveformLine: THREE.Line;
  glowMesh?: THREE.Mesh;
  particles?: THREE.Points;
}

export interface WaveformDisplay3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  isActive?: boolean;
  sampleCount?: number;
}

/**
 * Create a floating 3D waveform display
 */
export function createWaveformDisplay3D(options: WaveformDisplay3DOptions): WaveformDisplay3DRefs {
  const {
    position,
    scale = 1,
    accentColor = 0xa855f7,
    isActive = false,
    sampleCount = 64,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  // Frame/border for the display
  const frameGeometry = new THREE.TorusGeometry(0.8, 0.02, 8, 48);
  const frameMaterial = new THREE.MeshToonMaterial({
    color: 0x333333,
  });
  const frame = new THREE.Mesh(frameGeometry, frameMaterial);
  frame.rotation.x = Math.PI / 2;
  group.add(frame);

  // Glow behind waveform
  const glowGeometry = new THREE.CircleGeometry(0.75, 32);
  const glowMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.1,
    side: THREE.DoubleSide,
  });
  const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
  glowMesh.rotation.x = Math.PI / 2;
  glowMesh.position.y = -0.01;
  group.add(glowMesh);

  // Inner ring accent
  const innerRingGeometry = new THREE.TorusGeometry(0.6, 0.01, 8, 32);
  const innerRingMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.5,
  });
  const innerRing = new THREE.Mesh(innerRingGeometry, innerRingMaterial);
  innerRing.rotation.x = Math.PI / 2;
  innerRing.position.y = 0.01;
  group.add(innerRing);

  // Create waveform line
  const waveformPoints: THREE.Vector3[] = [];
  for (let i = 0; i < sampleCount; i++) {
    const angle = (i / sampleCount) * Math.PI * 2;
    const radius = 0.5;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    waveformPoints.push(new THREE.Vector3(x, 0, z));
  }
  // Close the loop
  waveformPoints.push(waveformPoints[0].clone());

  const waveformGeometry = new THREE.BufferGeometry().setFromPoints(waveformPoints);
  const waveformMaterial = new THREE.LineBasicMaterial({
    color: accentColor,
    linewidth: 2,
    transparent: true,
    opacity: 0.9,
  });
  const waveformLine = new THREE.Line(waveformGeometry, waveformMaterial);
  group.add(waveformLine);

  // Center orb
  const orbGeometry = new THREE.SphereGeometry(0.08, 16, 16);
  const orbMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.8,
  });
  const orb = new THREE.Mesh(orbGeometry, orbMaterial);
  group.add(orb);

  // Sparkle particles around the waveform
  const particleCount = 30;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);

  const particleColor = new THREE.Color(accentColor);

  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const radius = 0.4 + Math.random() * 0.3;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 0.2;
    positions[i * 3 + 2] = Math.sin(angle) * radius;

    colors[i * 3] = particleColor.r;
    colors[i * 3 + 1] = particleColor.g;
    colors[i * 3 + 2] = particleColor.b;

    sizes[i] = 0.02 + Math.random() * 0.02;
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
  group.add(particles);

  // Store state
  group.userData = {
    isActive,
    accentColor,
    sampleCount,
    audioData: new Float32Array(sampleCount).fill(0),
  };

  return {
    group,
    waveformLine,
    glowMesh,
    particles,
  };
}

/**
 * Update waveform with audio data
 */
export function updateWaveformData(
  refs: WaveformDisplay3DRefs,
  audioData: Float32Array | number[]
): void {
  const sampleCount = refs.group.userData.sampleCount;
  const data = refs.group.userData.audioData as Float32Array;

  // Normalize and store data
  const inputLength = audioData.length;
  for (let i = 0; i < sampleCount; i++) {
    const srcIndex = Math.floor((i / sampleCount) * inputLength);
    data[i] = Math.abs(audioData[srcIndex] || 0);
  }
}

/**
 * Animate the waveform display
 */
export function animateWaveformDisplay3D(
  refs: WaveformDisplay3DRefs,
  time: number,
  options?: {
    isActive?: boolean;
    audioData?: Float32Array | number[];
    intensity?: number;
  }
): void {
  const isActive = options?.isActive ?? refs.group.userData.isActive ?? false;
  const intensity = options?.intensity ?? (isActive ? 0.7 : 0);

  refs.group.userData.isActive = isActive;

  // Update audio data if provided
  if (options?.audioData) {
    updateWaveformData(refs, options.audioData);
  }

  const audioData = refs.group.userData.audioData as Float32Array;
  const sampleCount = refs.group.userData.sampleCount;

  // Slowly rotate the whole display
  refs.group.rotation.y += 0.005;

  // Float up and down
  refs.group.position.y += Math.sin(time * 2) * 0.001;

  // Update waveform geometry
  const positions = refs.waveformLine.geometry.attributes.position.array as Float32Array;
  const baseRadius = 0.5;

  for (let i = 0; i < sampleCount; i++) {
    const angle = (i / sampleCount) * Math.PI * 2;

    // Get audio amplitude (or generate fake waveform when active)
    let amplitude = audioData[i];
    if (isActive && amplitude === 0) {
      // Generate fake waveform when no real data
      amplitude = Math.sin(time * 10 + i * 0.5) * 0.3 + 0.5;
      amplitude *= Math.sin(time * 3 + i * 0.1) * 0.3 + 0.7;
    }

    const radius = baseRadius + amplitude * 0.2 * intensity;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = amplitude * 0.1 * intensity * Math.sin(time * 5 + i);
    positions[i * 3 + 2] = Math.sin(angle) * radius;
  }

  // Close the loop
  positions[sampleCount * 3] = positions[0];
  positions[sampleCount * 3 + 1] = positions[1];
  positions[sampleCount * 3 + 2] = positions[2];

  refs.waveformLine.geometry.attributes.position.needsUpdate = true;

  // Glow intensity
  if (refs.glowMesh) {
    const glowMaterial = refs.glowMesh.material as THREE.MeshBasicMaterial;
    glowMaterial.opacity = isActive ? 0.15 + intensity * 0.1 : 0.05;

    // Pulse glow
    refs.glowMesh.scale.setScalar(1 + Math.sin(time * 3) * 0.02 * intensity);
  }

  // Animate particles
  if (refs.particles) {
    refs.particles.visible = isActive;

    if (isActive) {
      const particlePositions = refs.particles.geometry.attributes.position.array as Float32Array;
      const particleCount = particlePositions.length / 3;

      for (let i = 0; i < particleCount; i++) {
        const angle = time * 0.5 + (i / particleCount) * Math.PI * 2;
        const pulseRadius = 0.5 + Math.sin(time * 4 + i) * 0.1;

        particlePositions[i * 3] = Math.cos(angle) * pulseRadius;
        particlePositions[i * 3 + 1] = Math.sin(time * 3 + i * 0.5) * 0.1;
        particlePositions[i * 3 + 2] = Math.sin(angle) * pulseRadius;
      }

      refs.particles.geometry.attributes.position.needsUpdate = true;

      const particleMaterial = refs.particles.material as THREE.PointsMaterial;
      particleMaterial.opacity = 0.4 + intensity * 0.4;
    }
  }
}

/**
 * Update waveform color
 */
export function setWaveformColor(refs: WaveformDisplay3DRefs, color: number): void {
  refs.group.userData.accentColor = color;

  const lineMaterial = refs.waveformLine.material as THREE.LineBasicMaterial;
  lineMaterial.color.setHex(color);

  if (refs.glowMesh) {
    const glowMaterial = refs.glowMesh.material as THREE.MeshBasicMaterial;
    glowMaterial.color.setHex(color);
  }

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
 * Dispose waveform display resources
 */
export function disposeWaveformDisplay3D(refs: WaveformDisplay3DRefs): void {
  refs.group.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
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
