// EmotionVerify3D - V7 verification visualization showing Happy vs Sad F0 comparison
// Features: Two emotion orbs with pitch bars, verification status indicator

import * as THREE from 'three';

export interface EmotionVerify3DRefs {
  group: THREE.Group;
  happyOrb: THREE.Mesh;
  sadOrb: THREE.Mesh;
  happyBar: THREE.Mesh;
  sadBar: THREE.Mesh;
  happyBarFill: THREE.Mesh;
  sadBarFill: THREE.Mesh;
  comparisonArrow?: THREE.Group;
  particles?: THREE.Points;
  statusIndicator?: THREE.Mesh;
}

export interface EmotionVerify3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  happyF0?: number;  // 0-1 normalized F0 value
  sadF0?: number;    // 0-1 normalized F0 value
  isVerifying?: boolean;
  verified?: boolean;
}

// Pastel colors for emotions
const HAPPY_COLOR = 0xfcd34d;  // Warm yellow
const SAD_COLOR = 0x93c5fd;   // Soft blue
const SUCCESS_COLOR = 0x4ade80;  // Green for verified
const PENDING_COLOR = 0xfbbf24;  // Amber for verifying

/**
 * Create cute emotion face on an orb
 */
function createFace(parent: THREE.Group, isHappy: boolean, color: number): void {
  const faceColor = 0x1f2937;  // Dark gray for features

  // Eyes
  const eyeGeometry = new THREE.SphereGeometry(0.06, 8, 8);
  const eyeMaterial = new THREE.MeshToonMaterial({ color: faceColor });

  const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
  leftEye.position.set(-0.12, 0.08, 0.22);
  parent.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
  rightEye.position.set(0.12, 0.08, 0.22);
  parent.add(rightEye);

  // Eye highlights
  const highlightGeometry = new THREE.SphereGeometry(0.025, 6, 6);
  const highlightMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });

  const leftHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
  leftHighlight.position.set(-0.10, 0.10, 0.26);
  parent.add(leftHighlight);

  const rightHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
  rightHighlight.position.set(0.14, 0.10, 0.26);
  parent.add(rightHighlight);

  // Mouth (curved line using torus segment)
  if (isHappy) {
    // Happy smile - same as frown rotation but flipped with rotation.z
    const smileGeometry = new THREE.TorusGeometry(0.08, 0.02, 8, 12, Math.PI);
    const smileMaterial = new THREE.MeshToonMaterial({ color: faceColor });
    const smile = new THREE.Mesh(smileGeometry, smileMaterial);
    smile.position.set(0, -0.04, 0.23);
    smile.rotation.x = Math.PI / 2;  // Same as frown
    smile.rotation.z = Math.PI;       // Flip to curve upward
    parent.add(smile);

    // Rosy cheeks
    const cheekGeometry = new THREE.CircleGeometry(0.05, 8);
    const cheekMaterial = new THREE.MeshBasicMaterial({
      color: 0xfca5a5,
      transparent: true,
      opacity: 0.5,
      side: THREE.DoubleSide,
    });

    const leftCheek = new THREE.Mesh(cheekGeometry, cheekMaterial);
    leftCheek.position.set(-0.18, -0.02, 0.21);
    parent.add(leftCheek);

    const rightCheek = new THREE.Mesh(cheekGeometry, cheekMaterial);
    rightCheek.position.set(0.18, -0.02, 0.21);
    parent.add(rightCheek);
  } else {
    // Sad frown - downward curve
    // Use the same arc but don't add rotation.z to keep curve opening downward
    const frownGeometry = new THREE.TorusGeometry(0.08, 0.02, 8, 12, Math.PI);
    const frownMaterial = new THREE.MeshToonMaterial({ color: faceColor });
    const frown = new THREE.Mesh(frownGeometry, frownMaterial);
    frown.position.set(0, -0.1, 0.23);
    // Only rotate on X to face camera, Z=0 keeps curve opening down (opposite of smile's Z=PI)
    frown.rotation.x = Math.PI / 2;
    parent.add(frown);

    // Sad eyebrows (tilted)
    const browGeometry = new THREE.BoxGeometry(0.08, 0.015, 0.02);
    const browMaterial = new THREE.MeshToonMaterial({ color: faceColor });

    const leftBrow = new THREE.Mesh(browGeometry, browMaterial);
    leftBrow.position.set(-0.12, 0.18, 0.22);
    leftBrow.rotation.z = 0.3;
    parent.add(leftBrow);

    const rightBrow = new THREE.Mesh(browGeometry, browMaterial);
    rightBrow.position.set(0.12, 0.18, 0.22);
    rightBrow.rotation.z = -0.3;
    parent.add(rightBrow);
  }
}

/**
 * Create a pitch bar with fill indicator
 */
function createPitchBar(
  height: number,
  color: number,
  fillColor: number,
): { bar: THREE.Mesh; fill: THREE.Mesh } {
  // Bar background
  const barGeometry = new THREE.BoxGeometry(0.15, height, 0.08);
  const barMaterial = new THREE.MeshToonMaterial({
    color: 0x374151,
    transparent: true,
    opacity: 0.3,
  });
  const bar = new THREE.Mesh(barGeometry, barMaterial);

  // Bar fill (starts at bottom)
  const fillGeometry = new THREE.BoxGeometry(0.13, 0.01, 0.06);
  const fillMaterial = new THREE.MeshToonMaterial({ color: fillColor });
  const fill = new THREE.Mesh(fillGeometry, fillMaterial);
  fill.position.y = -height / 2 + 0.02;

  return { bar, fill };
}

/**
 * Create an arrow showing comparison direction
 */
function createComparisonArrow(): THREE.Group {
  const arrowGroup = new THREE.Group();

  // Arrow shaft
  const shaftGeometry = new THREE.BoxGeometry(0.3, 0.04, 0.04);
  const shaftMaterial = new THREE.MeshToonMaterial({ color: SUCCESS_COLOR });
  const shaft = new THREE.Mesh(shaftGeometry, shaftMaterial);
  arrowGroup.add(shaft);

  // Arrow head (cone)
  const headGeometry = new THREE.ConeGeometry(0.06, 0.12, 8);
  const headMaterial = new THREE.MeshToonMaterial({ color: SUCCESS_COLOR });
  const head = new THREE.Mesh(headGeometry, headMaterial);
  head.position.x = 0.2;
  head.rotation.z = -Math.PI / 2;
  arrowGroup.add(head);

  // ">" symbol glow
  const glowGeometry = new THREE.SphereGeometry(0.08, 8, 8);
  const glowMaterial = new THREE.MeshBasicMaterial({
    color: SUCCESS_COLOR,
    transparent: true,
    opacity: 0.3,
  });
  const glow = new THREE.Mesh(glowGeometry, glowMaterial);
  glow.position.x = 0.2;
  arrowGroup.add(glow);

  return arrowGroup;
}

/**
 * Create the emotion verification display
 */
export function createEmotionVerify3D(options: EmotionVerify3DOptions): EmotionVerify3DRefs {
  const {
    position,
    scale = 1,
    happyF0 = 0.7,
    sadF0 = 0.4,
    isVerifying = false,
    verified = false,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  // Base platform
  const platformGeometry = new THREE.CylinderGeometry(1.2, 1.3, 0.1, 16);
  const platformMaterial = new THREE.MeshToonMaterial({
    color: 0x4b5563,
  });
  const platform = new THREE.Mesh(platformGeometry, platformMaterial);
  platform.position.y = -0.05;
  group.add(platform);

  // Platform ring accent
  const ringGeometry = new THREE.TorusGeometry(1.25, 0.03, 8, 32);
  const ringMaterial = new THREE.MeshBasicMaterial({
    color: verified ? SUCCESS_COLOR : PENDING_COLOR,
    transparent: true,
    opacity: 0.6,
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.01;
  group.add(ring);

  // === HAPPY EMOTION ORB (left side) ===
  const happyGroup = new THREE.Group();
  happyGroup.position.set(-0.5, 0.5, 0);

  const happyOrbGeometry = new THREE.SphereGeometry(0.3, 16, 16);
  const happyOrbMaterial = new THREE.MeshToonMaterial({ color: HAPPY_COLOR });
  const happyOrb = new THREE.Mesh(happyOrbGeometry, happyOrbMaterial);
  happyGroup.add(happyOrb);

  createFace(happyGroup, true, HAPPY_COLOR);
  group.add(happyGroup);

  // Happy pitch bar
  const happyBarHeight = 0.8;
  const { bar: happyBar, fill: happyBarFill } = createPitchBar(happyBarHeight, 0x374151, HAPPY_COLOR);
  happyBar.position.set(-0.5, 1.0, 0);
  happyBarFill.position.set(-0.5, 0.62, 0);
  group.add(happyBar);
  group.add(happyBarFill);

  // "F0" label for happy
  const happyLabelGeometry = new THREE.PlaneGeometry(0.2, 0.1);
  const happyLabelMaterial = new THREE.MeshBasicMaterial({
    color: HAPPY_COLOR,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
  });
  const happyLabel = new THREE.Mesh(happyLabelGeometry, happyLabelMaterial);
  happyLabel.position.set(-0.5, 1.5, 0);
  group.add(happyLabel);

  // === SAD EMOTION ORB (right side) ===
  const sadGroup = new THREE.Group();
  sadGroup.position.set(0.5, 0.5, 0);

  const sadOrbGeometry = new THREE.SphereGeometry(0.3, 16, 16);
  const sadOrbMaterial = new THREE.MeshToonMaterial({ color: SAD_COLOR });
  const sadOrb = new THREE.Mesh(sadOrbGeometry, sadOrbMaterial);
  sadGroup.add(sadOrb);

  createFace(sadGroup, false, SAD_COLOR);
  group.add(sadGroup);

  // Sad pitch bar
  const sadBarHeight = 0.8;
  const { bar: sadBar, fill: sadBarFill } = createPitchBar(sadBarHeight, 0x374151, SAD_COLOR);
  sadBar.position.set(0.5, 1.0, 0);
  sadBarFill.position.set(0.5, 0.62, 0);
  group.add(sadBar);
  group.add(sadBarFill);

  // "F0" label for sad
  const sadLabelGeometry = new THREE.PlaneGeometry(0.2, 0.1);
  const sadLabelMaterial = new THREE.MeshBasicMaterial({
    color: SAD_COLOR,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
  });
  const sadLabel = new THREE.Mesh(sadLabelGeometry, sadLabelMaterial);
  sadLabel.position.set(0.5, 1.5, 0);
  group.add(sadLabel);

  // === COMPARISON ARROW (center) ===
  const comparisonArrow = createComparisonArrow();
  comparisonArrow.position.set(0, 1.0, 0.3);
  comparisonArrow.rotation.y = Math.PI;  // Point from happy to sad
  comparisonArrow.visible = verified;
  group.add(comparisonArrow);

  // === STATUS INDICATOR (top center) ===
  const statusGeometry = new THREE.SphereGeometry(0.12, 12, 12);
  const statusMaterial = new THREE.MeshBasicMaterial({
    color: verified ? SUCCESS_COLOR : (isVerifying ? PENDING_COLOR : 0x6b7280),
    transparent: true,
    opacity: 0.9,
  });
  const statusIndicator = new THREE.Mesh(statusGeometry, statusMaterial);
  statusIndicator.position.set(0, 1.7, 0);
  group.add(statusIndicator);

  // Status glow ring
  const statusGlowGeometry = new THREE.TorusGeometry(0.18, 0.02, 8, 24);
  const statusGlowMaterial = new THREE.MeshBasicMaterial({
    color: verified ? SUCCESS_COLOR : PENDING_COLOR,
    transparent: true,
    opacity: 0.4,
  });
  const statusGlow = new THREE.Mesh(statusGlowGeometry, statusGlowMaterial);
  statusGlow.position.copy(statusIndicator.position);
  statusGlow.rotation.x = Math.PI / 2;
  group.add(statusGlow);

  // === CELEBRATION PARTICLES ===
  const particleCount = 50;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const radius = 0.8 + Math.random() * 0.4;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = 0.5 + Math.random() * 1.0;
    positions[i * 3 + 2] = Math.sin(angle) * radius;

    // Alternate between happy and success colors
    const particleColor = i % 2 === 0
      ? new THREE.Color(HAPPY_COLOR)
      : new THREE.Color(SUCCESS_COLOR);
    colors[i * 3] = particleColor.r;
    colors[i * 3 + 1] = particleColor.g;
    colors[i * 3 + 2] = particleColor.b;

    sizes[i] = 0.03 + Math.random() * 0.03;
  }

  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

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
  group.userData = {
    happyF0,
    sadF0,
    isVerifying,
    verified,
    celebrationPhase: 0,
  };

  return {
    group,
    happyOrb,
    sadOrb,
    happyBar,
    sadBar,
    happyBarFill,
    sadBarFill,
    comparisonArrow,
    particles,
    statusIndicator,
  };
}

/**
 * Animate the emotion verification display
 */
export function animateEmotionVerify3D(
  refs: EmotionVerify3DRefs,
  time: number,
  options?: {
    happyF0?: number;
    sadF0?: number;
    isVerifying?: boolean;
    verified?: boolean;
  }
): void {
  const happyF0 = options?.happyF0 ?? refs.group.userData.happyF0 ?? 0.7;
  const sadF0 = options?.sadF0 ?? refs.group.userData.sadF0 ?? 0.4;
  const isVerifying = options?.isVerifying ?? refs.group.userData.isVerifying ?? false;
  const verified = options?.verified ?? refs.group.userData.verified ?? false;

  // Update state
  refs.group.userData.happyF0 = happyF0;
  refs.group.userData.sadF0 = sadF0;
  refs.group.userData.isVerifying = isVerifying;
  refs.group.userData.verified = verified;

  // Animate emotion orbs - gentle bobbing
  const happyOrb = refs.happyOrb;
  const sadOrb = refs.sadOrb;

  // Happy orb bobs higher when verified
  const happyBob = verified ? 0.08 : 0.04;
  happyOrb.parent!.position.y = 0.5 + Math.sin(time * 3) * happyBob;
  happyOrb.parent!.rotation.y = Math.sin(time * 0.5) * 0.1;

  // Sad orb has slower, lower bob
  const sadBob = verified ? 0.02 : 0.04;
  sadOrb.parent!.position.y = 0.5 + Math.sin(time * 2 + 1) * sadBob;
  sadOrb.parent!.rotation.y = Math.sin(time * 0.4) * 0.1;

  // Animate pitch bar fills
  const barHeight = 0.8;
  const happyFillHeight = happyF0 * barHeight * 0.9;
  const sadFillHeight = sadF0 * barHeight * 0.9;

  // Smooth interpolation for bar fills
  const currentHappyHeight = refs.happyBarFill.scale.y;
  const targetHappyHeight = happyFillHeight / 0.01;  // Original height is 0.01
  refs.happyBarFill.scale.y = THREE.MathUtils.lerp(currentHappyHeight, targetHappyHeight, 0.1);
  refs.happyBarFill.position.y = 0.62 + (refs.happyBarFill.scale.y * 0.01) / 2;

  const currentSadHeight = refs.sadBarFill.scale.y;
  const targetSadHeight = sadFillHeight / 0.01;
  refs.sadBarFill.scale.y = THREE.MathUtils.lerp(currentSadHeight, targetSadHeight, 0.1);
  refs.sadBarFill.position.y = 0.62 + (refs.sadBarFill.scale.y * 0.01) / 2;

  // Status indicator animation
  if (refs.statusIndicator) {
    const statusMaterial = refs.statusIndicator.material as THREE.MeshBasicMaterial;

    if (verified) {
      statusMaterial.color.setHex(SUCCESS_COLOR);
      refs.statusIndicator.scale.setScalar(1 + Math.sin(time * 4) * 0.1);
    } else if (isVerifying) {
      statusMaterial.color.setHex(PENDING_COLOR);
      refs.statusIndicator.scale.setScalar(1 + Math.sin(time * 6) * 0.15);
      // Pulsing effect
      statusMaterial.opacity = 0.7 + Math.sin(time * 8) * 0.3;
    } else {
      statusMaterial.color.setHex(0x6b7280);
      refs.statusIndicator.scale.setScalar(1);
      statusMaterial.opacity = 0.6;
    }
  }

  // Comparison arrow animation
  if (refs.comparisonArrow) {
    refs.comparisonArrow.visible = verified;
    if (verified) {
      refs.comparisonArrow.rotation.z = Math.sin(time * 2) * 0.1;
      refs.comparisonArrow.position.y = 1.0 + Math.sin(time * 3) * 0.05;
    }
  }

  // Celebration particles
  if (refs.particles) {
    refs.particles.visible = verified;

    if (verified) {
      const particlePositions = refs.particles.geometry.attributes.position.array as Float32Array;
      const particleCount = particlePositions.length / 3;

      for (let i = 0; i < particleCount; i++) {
        const baseAngle = (i / particleCount) * Math.PI * 2;
        const angle = baseAngle + time * 0.5;
        const radius = 0.8 + Math.sin(time * 2 + i * 0.5) * 0.2;

        particlePositions[i * 3] = Math.cos(angle) * radius;
        particlePositions[i * 3 + 1] = 0.8 + Math.sin(time * 3 + i * 0.3) * 0.4 +
          (Math.sin(time + i) * 0.1);
        particlePositions[i * 3 + 2] = Math.sin(angle) * radius;
      }

      refs.particles.geometry.attributes.position.needsUpdate = true;

      const particleMaterial = refs.particles.material as THREE.PointsMaterial;
      particleMaterial.opacity = 0.6 + Math.sin(time * 4) * 0.2;
    }
  }

  // Slight rotation of whole display when verifying
  if (isVerifying) {
    refs.group.rotation.y += 0.002;
  }
}

/**
 * Update verification state
 */
export function setVerificationState(
  refs: EmotionVerify3DRefs,
  state: {
    happyF0?: number;
    sadF0?: number;
    isVerifying?: boolean;
    verified?: boolean;
  }
): void {
  if (state.happyF0 !== undefined) refs.group.userData.happyF0 = state.happyF0;
  if (state.sadF0 !== undefined) refs.group.userData.sadF0 = state.sadF0;
  if (state.isVerifying !== undefined) refs.group.userData.isVerifying = state.isVerifying;
  if (state.verified !== undefined) refs.group.userData.verified = state.verified;
}

/**
 * Dispose emotion verification display resources
 */
export function disposeEmotionVerify3D(refs: EmotionVerify3DRefs): void {
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
