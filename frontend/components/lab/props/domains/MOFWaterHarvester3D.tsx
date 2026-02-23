// MOFWaterHarvester3D - Physics-accurate MOF atmospheric water harvester
// Based on MIT/Berkeley MOF-801 design (Yaghi et al. 2017)
//
// Real physics cycle:
// NIGHT: MOF sorbent adsorbs water from humid air (even at 20% RH)
// DAY: Sunlight heats MOF → releases water vapor → condenses on cool dome → drips to collector
//
// Components:
// - Transparent dome enclosure (acrylic/glass)
// - MOF sorbent bed (dark porous material)
// - Condensation surface (inner dome walls, cooler than sorbent)
// - Collection trough at base

import * as THREE from 'three';

// ============================================================================
// Types
// ============================================================================

export interface MOFHarvesterProps {
  // Design parameters
  domeRadius: number;        // cm, dome size
  sorbentMass: number;       // kg of MOF material
  humidity: number;          // ambient humidity 0-100%

  // Time of day (0-1, 0=midnight, 0.5=noon)
  timeOfDay: number;

  // Simulation results
  dailyYield?: number;       // liters collected
  sorbentTemp?: number;      // celsius
  domeTemp?: number;         // celsius
  waterContent?: number;     // % of MOF capacity
}

export interface MOFHarvesterRefs {
  group: THREE.Group;

  // Components
  dome: DomeRefs;
  sorbent: SorbentRefs;
  collector: CollectorRefs;
  particles: ParticleRefs;
  environment: EnvironmentRefs;

  // State
  isNight: boolean;
  cyclePhase: 'adsorbing' | 'heating' | 'releasing' | 'condensing';
}

interface DomeRefs {
  mesh: THREE.Mesh;
  condensationDroplets: THREE.Points;
}

interface SorbentRefs {
  mesh: THREE.Mesh;
  glowMesh: THREE.Mesh;  // Heat glow during day
  temperature: number;
}

interface CollectorRefs {
  trough: THREE.Mesh;
  water: THREE.Mesh;
  fillLevel: number;
}

interface ParticleRefs {
  // Vapor particles (rising from sorbent)
  vapor: THREE.Points;
  vaporPositions: Float32Array;
  vaporVelocities: Float32Array;
  vaporStates: Float32Array;  // 0=adsorbing, 1=releasing, 2=condensing, 3=dripping

  // Water droplets (falling from dome)
  droplets: THREE.Points;
  dropletPositions: Float32Array;
  dropletVelocities: Float32Array;
}

interface EnvironmentRefs {
  sun: THREE.DirectionalLight;
  ambient: THREE.AmbientLight;
  sky: THREE.Mesh;
  ground: THREE.Mesh;
}

// ============================================================================
// Colors
// ============================================================================

const COLORS = {
  // Dome
  domeGlass: 0xb8d4e8,      // Light blue-tinted glass
  domeEdge: 0x88aabb,

  // MOF Sorbent
  mofCold: 0x2d3436,        // Dark gray when cold (night)
  mofWarm: 0x6d4c41,        // Brown when warming
  mofHot: 0xd84315,         // Orange-red when hot (releasing water)

  // Water
  vapor: 0xadd8e6,          // Light blue vapor
  droplet: 0x4fc3f7,        // Bright blue droplet
  poolWater: 0x29b6f6,      // Water in collector

  // Environment
  daySky: 0x87ceeb,
  nightSky: 0x0a1628,
  ground: 0xd4a574,         // Desert sand
  sunlight: 0xfffaf0,
};

// ============================================================================
// Layout Constants
// ============================================================================

const LAYOUT = {
  domeRadius: 0.6,
  domeHeight: 0.7,

  sorbentY: -0.15,          // Sorbent bed sits low in dome
  sorbentRadius: 0.35,
  sorbentHeight: 0.12,

  collectorY: -0.4,         // Trough below sorbent
  collectorRadius: 0.45,

  groundY: -0.5,
};

// ============================================================================
// Main Creation Function
// ============================================================================

export function createMOFHarvester(props: MOFHarvesterProps): MOFHarvesterRefs {
  const group = new THREE.Group();

  // Create components
  const dome = createDome();
  const sorbent = createSorbent();
  const collector = createCollector();
  const particles = createParticles();
  const environment = createEnvironment();

  // Add to group
  group.add(dome.mesh);
  group.add(dome.condensationDroplets);
  group.add(sorbent.mesh);
  group.add(sorbent.glowMesh);
  group.add(collector.trough);
  group.add(collector.water);
  group.add(particles.vapor);
  group.add(particles.droplets);
  group.add(environment.sky);
  group.add(environment.ground);

  // Lights added to scene separately

  const isNight = props.timeOfDay < 0.25 || props.timeOfDay > 0.75;

  return {
    group,
    dome,
    sorbent,
    collector,
    particles,
    environment,
    isNight,
    cyclePhase: isNight ? 'adsorbing' : 'releasing',
  };
}

// ============================================================================
// Component Creation
// ============================================================================

function createDome(): DomeRefs {
  // Transparent hemisphere dome
  const domeGeo = new THREE.SphereGeometry(
    LAYOUT.domeRadius,
    32, 24,
    0, Math.PI * 2,
    0, Math.PI / 2  // Only top hemisphere
  );

  const domeMat = new THREE.MeshPhysicalMaterial({
    color: COLORS.domeGlass,
    transparent: true,
    opacity: 0.3,
    roughness: 0.1,
    metalness: 0,
    transmission: 0.6,
    thickness: 0.02,
    side: THREE.DoubleSide,
  });

  const dome = new THREE.Mesh(domeGeo, domeMat);
  dome.position.y = LAYOUT.sorbentY + 0.1;

  // Condensation droplets on inner dome surface
  const dropletCount = 200;
  const dropletPositions = new Float32Array(dropletCount * 3);

  // Initially no condensation (populated during animation)
  for (let i = 0; i < dropletCount; i++) {
    dropletPositions[i * 3] = 0;
    dropletPositions[i * 3 + 1] = -10; // Hidden below
    dropletPositions[i * 3 + 2] = 0;
  }

  const dropletGeo = new THREE.BufferGeometry();
  dropletGeo.setAttribute('position', new THREE.BufferAttribute(dropletPositions, 3));

  const dropletMat = new THREE.PointsMaterial({
    color: COLORS.droplet,
    size: 0.02,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true,
  });

  const condensationDroplets = new THREE.Points(dropletGeo, dropletMat);

  return { mesh: dome, condensationDroplets };
}

function createSorbent(): SorbentRefs {
  // MOF sorbent bed - cylindrical porous material
  const sorbentGeo = new THREE.CylinderGeometry(
    LAYOUT.sorbentRadius,
    LAYOUT.sorbentRadius * 0.9,
    LAYOUT.sorbentHeight,
    32
  );

  const sorbentMat = new THREE.MeshStandardMaterial({
    color: COLORS.mofCold,
    roughness: 0.9,
    metalness: 0.1,
  });

  const sorbent = new THREE.Mesh(sorbentGeo, sorbentMat);
  sorbent.position.y = LAYOUT.sorbentY;

  // Heat glow mesh (slightly larger, emissive)
  const glowGeo = new THREE.CylinderGeometry(
    LAYOUT.sorbentRadius * 1.05,
    LAYOUT.sorbentRadius * 0.95,
    LAYOUT.sorbentHeight * 1.1,
    32
  );

  const glowMat = new THREE.MeshBasicMaterial({
    color: COLORS.mofHot,
    transparent: true,
    opacity: 0,
    side: THREE.BackSide,
  });

  const glowMesh = new THREE.Mesh(glowGeo, glowMat);
  glowMesh.position.y = LAYOUT.sorbentY;

  return { mesh: sorbent, glowMesh, temperature: 25 };
}

function createCollector(): CollectorRefs {
  // Collection trough - ring shape
  const troughGeo = new THREE.TorusGeometry(
    LAYOUT.collectorRadius,
    0.05,
    8, 32
  );

  const troughMat = new THREE.MeshStandardMaterial({
    color: 0x5d4037,
    roughness: 0.7,
    metalness: 0.3,
  });

  const trough = new THREE.Mesh(troughGeo, troughMat);
  trough.rotation.x = Math.PI / 2;
  trough.position.y = LAYOUT.collectorY;

  // Water in collector
  const waterGeo = new THREE.CylinderGeometry(
    LAYOUT.collectorRadius + 0.03,
    LAYOUT.collectorRadius + 0.03,
    0.01,
    32
  );

  const waterMat = new THREE.MeshPhysicalMaterial({
    color: COLORS.poolWater,
    transparent: true,
    opacity: 0.7,
    roughness: 0.1,
    metalness: 0,
  });

  const water = new THREE.Mesh(waterGeo, waterMat);
  water.position.y = LAYOUT.collectorY;
  water.scale.y = 0.1; // Start nearly empty

  return { trough, water, fillLevel: 0 };
}

function createParticles(): ParticleRefs {
  const VAPOR_COUNT = 1000;
  const DROPLET_COUNT = 100;

  // Vapor particles
  const vaporPositions = new Float32Array(VAPOR_COUNT * 3);
  const vaporVelocities = new Float32Array(VAPOR_COUNT * 3);
  const vaporStates = new Float32Array(VAPOR_COUNT);

  // Initialize vapor around sorbent
  for (let i = 0; i < VAPOR_COUNT; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * LAYOUT.sorbentRadius;

    vaporPositions[i * 3] = Math.cos(angle) * radius;
    vaporPositions[i * 3 + 1] = LAYOUT.sorbentY + Math.random() * 0.3;
    vaporPositions[i * 3 + 2] = Math.sin(angle) * radius;

    vaporVelocities[i * 3] = (Math.random() - 0.5) * 0.01;
    vaporVelocities[i * 3 + 1] = Math.random() * 0.02;
    vaporVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.01;

    vaporStates[i] = 0; // Start adsorbing
  }

  const vaporGeo = new THREE.BufferGeometry();
  vaporGeo.setAttribute('position', new THREE.BufferAttribute(vaporPositions, 3));

  const vaporMat = new THREE.PointsMaterial({
    color: COLORS.vapor,
    size: 0.015,
    transparent: true,
    opacity: 0.6,
    sizeAttenuation: true,
    blending: THREE.AdditiveBlending,
  });

  const vapor = new THREE.Points(vaporGeo, vaporMat);

  // Water droplets (falling)
  const dropletPositions = new Float32Array(DROPLET_COUNT * 3);
  const dropletVelocities = new Float32Array(DROPLET_COUNT * 3);

  for (let i = 0; i < DROPLET_COUNT; i++) {
    dropletPositions[i * 3] = 0;
    dropletPositions[i * 3 + 1] = -10; // Hidden initially
    dropletPositions[i * 3 + 2] = 0;

    dropletVelocities[i * 3] = 0;
    dropletVelocities[i * 3 + 1] = 0;
    dropletVelocities[i * 3 + 2] = 0;
  }

  const dropletGeo = new THREE.BufferGeometry();
  dropletGeo.setAttribute('position', new THREE.BufferAttribute(dropletPositions, 3));

  const dropletMat = new THREE.PointsMaterial({
    color: COLORS.droplet,
    size: 0.03,
    transparent: true,
    opacity: 0.9,
    sizeAttenuation: true,
  });

  const droplets = new THREE.Points(dropletGeo, dropletMat);

  return {
    vapor,
    vaporPositions,
    vaporVelocities,
    vaporStates,
    droplets,
    dropletPositions,
    dropletVelocities,
  };
}

function createEnvironment(): EnvironmentRefs {
  // Sun light
  const sun = new THREE.DirectionalLight(COLORS.sunlight, 1);
  sun.position.set(2, 3, 1);
  sun.castShadow = true;

  // Ambient
  const ambient = new THREE.AmbientLight(0x404040, 0.5);

  // Sky dome
  const skyGeo = new THREE.SphereGeometry(10, 32, 32);
  const skyMat = new THREE.MeshBasicMaterial({
    color: COLORS.daySky,
    side: THREE.BackSide,
  });
  const sky = new THREE.Mesh(skyGeo, skyMat);

  // Ground plane (desert)
  const groundGeo = new THREE.CircleGeometry(5, 32);
  const groundMat = new THREE.MeshStandardMaterial({
    color: COLORS.ground,
    roughness: 0.9,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = LAYOUT.groundY;

  return { sun, ambient, sky, ground };
}

// ============================================================================
// Animation
// ============================================================================

export function animateMOFHarvester(
  refs: MOFHarvesterRefs,
  time: number,
  props: MOFHarvesterProps
): void {
  // Determine cycle phase based on time of day
  const timeOfDay = props.timeOfDay;
  refs.isNight = timeOfDay < 0.25 || timeOfDay > 0.75;

  // Sun intensity (0 at night, 1 at noon)
  const sunIntensity = refs.isNight ? 0 : Math.sin((timeOfDay - 0.25) * 2 * Math.PI);

  // Update environment
  updateEnvironment(refs.environment, refs.isNight, sunIntensity);

  // Update sorbent temperature and appearance
  const targetTemp = refs.isNight ? 25 : 25 + sunIntensity * 75; // 25°C night, up to 100°C day
  refs.sorbent.temperature += (targetTemp - refs.sorbent.temperature) * 0.02;
  updateSorbent(refs.sorbent, refs.sorbent.temperature);

  // Determine cycle phase
  if (refs.isNight) {
    refs.cyclePhase = 'adsorbing';
  } else if (refs.sorbent.temperature < 50) {
    refs.cyclePhase = 'heating';
  } else if (refs.sorbent.temperature < 80) {
    refs.cyclePhase = 'releasing';
  } else {
    refs.cyclePhase = 'condensing';
  }

  // Animate particles based on phase
  animateParticles(refs.particles, refs.cyclePhase, refs.sorbent.temperature, time);

  // Update condensation on dome
  updateCondensation(refs.dome, refs.cyclePhase, time);

  // Animate droplets falling
  animateDroplets(refs.particles, refs.collector, time);

  // Update water level
  const targetFill = Math.min(1, (props.dailyYield || 0) / 2.8); // Max 2.8L per MIT paper
  refs.collector.fillLevel += (targetFill - refs.collector.fillLevel) * 0.01;
  refs.collector.water.scale.y = 0.1 + refs.collector.fillLevel * 2;
}

function updateEnvironment(env: EnvironmentRefs, isNight: boolean, sunIntensity: number): void {
  // Sky color
  const skyMat = env.sky.material as THREE.MeshBasicMaterial;
  if (isNight) {
    skyMat.color.setHex(COLORS.nightSky);
  } else {
    skyMat.color.lerpColors(
      new THREE.Color(COLORS.nightSky),
      new THREE.Color(COLORS.daySky),
      sunIntensity
    );
  }

  // Sun intensity
  env.sun.intensity = sunIntensity;
}

function updateSorbent(sorbent: SorbentRefs, temperature: number): void {
  const mat = sorbent.mesh.material as THREE.MeshStandardMaterial;
  const glowMat = sorbent.glowMesh.material as THREE.MeshBasicMaterial;

  // Color transitions with temperature
  const t = Math.max(0, Math.min(1, (temperature - 25) / 75));

  const coldColor = new THREE.Color(COLORS.mofCold);
  const warmColor = new THREE.Color(COLORS.mofWarm);
  const hotColor = new THREE.Color(COLORS.mofHot);

  if (t < 0.5) {
    mat.color.lerpColors(coldColor, warmColor, t * 2);
  } else {
    mat.color.lerpColors(warmColor, hotColor, (t - 0.5) * 2);
  }

  // Glow effect when hot
  glowMat.opacity = Math.max(0, t - 0.3) * 0.5;
}

function animateParticles(
  particles: ParticleRefs,
  phase: string,
  temperature: number,
  time: number
): void {
  const positions = particles.vaporPositions;
  const velocities = particles.vaporVelocities;
  const states = particles.vaporStates;
  const count = positions.length / 3;

  for (let i = 0; i < count; i++) {
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];

    // Distance from center
    const dist = Math.sqrt(px * px + pz * pz);

    switch (phase) {
      case 'adsorbing':
        // Particles drift toward sorbent (being absorbed)
        velocities[i * 3] = -px * 0.01;
        velocities[i * 3 + 1] = (LAYOUT.sorbentY - py) * 0.02;
        velocities[i * 3 + 2] = -pz * 0.01;
        states[i] = 0;
        break;

      case 'heating':
        // Particles stay near sorbent, slight vibration
        velocities[i * 3] = (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.002;
        states[i] = 0.5;
        break;

      case 'releasing':
        // Particles rise from sorbent as vapor
        velocities[i * 3] += (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 1] = 0.005 + Math.random() * 0.01; // Rise up
        velocities[i * 3 + 2] += (Math.random() - 0.5) * 0.002;
        states[i] = 1;
        break;

      case 'condensing':
        // Particles drift toward dome walls and slow down
        const toDomeX = px * (LAYOUT.domeRadius / (dist + 0.01) - 1) * 0.01;
        const toDomeZ = pz * (LAYOUT.domeRadius / (dist + 0.01) - 1) * 0.01;

        velocities[i * 3] = toDomeX;
        velocities[i * 3 + 1] *= 0.95; // Slow vertical movement
        velocities[i * 3 + 2] = toDomeZ;
        states[i] = 2;
        break;
    }

    // Apply velocities
    positions[i * 3] += velocities[i * 3];
    positions[i * 3 + 1] += velocities[i * 3 + 1];
    positions[i * 3 + 2] += velocities[i * 3 + 2];

    // Bounds checking
    const newDist = Math.sqrt(
      positions[i * 3] ** 2 + positions[i * 3 + 2] ** 2
    );

    // Keep within dome
    if (newDist > LAYOUT.domeRadius * 0.9) {
      const scale = (LAYOUT.domeRadius * 0.9) / newDist;
      positions[i * 3] *= scale;
      positions[i * 3 + 2] *= scale;
    }

    // Keep in vertical range
    if (positions[i * 3 + 1] > LAYOUT.domeHeight * 0.8) {
      positions[i * 3 + 1] = LAYOUT.domeHeight * 0.8;
      velocities[i * 3 + 1] *= -0.5;
    }
    if (positions[i * 3 + 1] < LAYOUT.sorbentY - 0.05) {
      positions[i * 3 + 1] = LAYOUT.sorbentY - 0.05;
      velocities[i * 3 + 1] *= -0.5;
    }
  }

  // Update geometry
  particles.vapor.geometry.attributes.position.needsUpdate = true;

  // Update color based on average state
  const avgState = Array.from(states).reduce((a, b) => a + b, 0) / count;
  const vaporMat = particles.vapor.material as THREE.PointsMaterial;

  if (avgState < 0.5) {
    vaporMat.color.setHex(0x90a4ae); // Gray when adsorbing
    vaporMat.opacity = 0.3;
  } else if (avgState < 1.5) {
    vaporMat.color.setHex(COLORS.vapor); // Blue when releasing
    vaporMat.opacity = 0.6;
  } else {
    vaporMat.color.setHex(COLORS.droplet); // Brighter when condensing
    vaporMat.opacity = 0.8;
  }
}

function updateCondensation(dome: DomeRefs, phase: string, time: number): void {
  const positions = dome.condensationDroplets.geometry.attributes.position.array as Float32Array;
  const count = positions.length / 3;

  if (phase === 'condensing') {
    // Show droplets on dome surface
    for (let i = 0; i < count; i++) {
      if (positions[i * 3 + 1] < -5) {
        // Spawn new droplet on dome surface
        const angle = Math.random() * Math.PI * 2;
        const height = 0.1 + Math.random() * 0.4;
        const radius = LAYOUT.domeRadius * 0.95 * Math.sin(Math.acos(height / LAYOUT.domeHeight));

        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] = height;
        positions[i * 3 + 2] = Math.sin(angle) * radius;
      } else {
        // Slowly slide down
        positions[i * 3 + 1] -= 0.001;
        if (positions[i * 3 + 1] < LAYOUT.collectorY + 0.1) {
          positions[i * 3 + 1] = -10; // Reset
        }
      }
    }
  } else {
    // Hide droplets
    for (let i = 0; i < count; i++) {
      if (positions[i * 3 + 1] > -5) {
        positions[i * 3 + 1] -= 0.01;
      }
    }
  }

  dome.condensationDroplets.geometry.attributes.position.needsUpdate = true;
}

function animateDroplets(particles: ParticleRefs, collector: CollectorRefs, time: number): void {
  const positions = particles.dropletPositions;
  const velocities = particles.dropletVelocities;
  const count = positions.length / 3;

  for (let i = 0; i < count; i++) {
    if (positions[i * 3 + 1] > -5) {
      // Active droplet - apply gravity
      velocities[i * 3 + 1] -= 0.001; // Gravity

      positions[i * 3] += velocities[i * 3];
      positions[i * 3 + 1] += velocities[i * 3 + 1];
      positions[i * 3 + 2] += velocities[i * 3 + 2];

      // Hit collector
      if (positions[i * 3 + 1] < LAYOUT.collectorY) {
        positions[i * 3 + 1] = -10; // Reset
        velocities[i * 3 + 1] = 0;
      }
    } else if (Math.random() < 0.02) {
      // Spawn new droplet from dome edge
      const angle = Math.random() * Math.PI * 2;
      const radius = LAYOUT.domeRadius * 0.3 + Math.random() * LAYOUT.domeRadius * 0.5;

      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = 0.1 + Math.random() * 0.2;
      positions[i * 3 + 2] = Math.sin(angle) * radius;

      velocities[i * 3] = 0;
      velocities[i * 3 + 1] = -0.01;
      velocities[i * 3 + 2] = 0;
    }
  }

  particles.droplets.geometry.attributes.position.needsUpdate = true;
}

// ============================================================================
// GPU Particle Update (from Warp server)
// ============================================================================

let gpuUpdateCount = 0;

export function updateParticlesFromGPU(
  refs: MOFHarvesterRefs,
  gpuPositions: number[][],
): void {
  if (!refs.particles || !gpuPositions || gpuPositions.length === 0) return;

  gpuUpdateCount++;

  const positions = refs.particles.vaporPositions;
  const maxParticles = Math.min(gpuPositions.length, positions.length / 3);

  // Map GPU positions to scene
  for (let i = 0; i < maxParticles; i++) {
    const gp = gpuPositions[i];

    // GPU uses 0-1 range, map to scene coordinates
    positions[i * 3] = (gp[0] - 0.5) * LAYOUT.domeRadius * 2;
    positions[i * 3 + 1] = LAYOUT.sorbentY + gp[1] * LAYOUT.domeHeight;
    positions[i * 3 + 2] = (gp[2] - 0.5) * LAYOUT.domeRadius * 2;
  }

  refs.particles.vapor.geometry.attributes.position.needsUpdate = true;

  // Visual indicator: cyan color when GPU active
  const mat = refs.particles.vapor.material as THREE.PointsMaterial;
  mat.color.setHex(0x00ffff);
  mat.opacity = 0.8;
  mat.size = 0.02;

  if (gpuUpdateCount % 100 === 0) {
    console.log(`[GPU→MOF] Update #${gpuUpdateCount}: ${maxParticles} particles`);
  }
}

// ============================================================================
// Cleanup
// ============================================================================

export function disposeMOFHarvester(refs: MOFHarvesterRefs): void {
  refs.group.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Points) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach(m => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}
