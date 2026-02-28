// MOFWaterHarvester3D - Parabolic solar concentrator MOF water harvester
//
// Design: Parabolic dish with mirror segments on the inner face,
// all focusing sunlight to a single focal point where the MOF sorbent sits.
//
// Cycle:
// NIGHT: MOF sorbent adsorbs water from humid air (even at 20% RH)
// DAY: Mirror segments concentrate sunlight → heats MOF → releases water vapor
//       → vapor rises, condenses on cooler surfaces → drips to collector below
//
// Components:
// - Parabolic dish with mirror facets on inner surface
// - MOF sorbent bed at focal point (center of dish)
// - Water collection basin below sorbent
// - Vapor/condensation particle system

import * as THREE from 'three';

// ============================================================================
// Types
// ============================================================================

export interface MOFHarvesterProps {
  domeRadius: number;
  sorbentMass: number;
  humidity: number;
  mirrorCount: number;       // number of mirror facets on the dish (2-8)
  timeOfDay: number;         // 0-1, 0=midnight, 0.5=noon
  dailyYield?: number;
  sorbentTemp?: number;
  domeTemp?: number;
  waterContent?: number;
}

export interface MOFHarvesterRefs {
  group: THREE.Group;
  dome: DomeRefs;
  sorbent: SorbentRefs;
  collector: CollectorRefs;
  particles: ParticleRefs;
  mirrors: MirrorRefs;
  environment: EnvironmentRefs;
  isNight: boolean;
  cyclePhase: 'adsorbing' | 'heating' | 'releasing' | 'condensing';
}

interface DomeRefs {
  mesh: THREE.Mesh;
  condensationDroplets: THREE.Points;
}

interface SorbentRefs {
  mesh: THREE.Mesh;
  glowMesh: THREE.Mesh;
  temperature: number;
}

interface CollectorRefs {
  trough: THREE.Mesh;
  water: THREE.Mesh;
  fillLevel: number;
}

interface ParticleRefs {
  vapor: THREE.Points;
  vaporPositions: Float32Array;
  vaporVelocities: Float32Array;
  vaporStates: Float32Array;
  droplets: THREE.Points;
  dropletPositions: Float32Array;
  dropletVelocities: Float32Array;
}

interface EnvironmentRefs {
  sun: THREE.DirectionalLight;
  ambient: THREE.AmbientLight;
  hemisphere: THREE.HemisphereLight;
  fill: THREE.DirectionalLight;
  sky: THREE.Mesh;
  ground: THREE.Mesh;
}

interface MirrorRefs {
  group: THREE.Group;
  panels: THREE.Mesh[];
  supports: THREE.Mesh[];
  glintMeshes: THREE.Mesh[];
  godRays: THREE.Mesh[];
  count: number;
}

// ============================================================================
// Colors
// ============================================================================

const COLORS = {
  domeGlass: 0xb8d4e8,
  domeEdge: 0x88aabb,
  mofCold: 0x2d3436,
  mofWarm: 0x6d4c41,
  mofHot: 0xd84315,
  vapor: 0xadd8e6,
  droplet: 0x4fc3f7,
  poolWater: 0x29b6f6,
  daySky: 0x87ceeb,
  nightSky: 0x0a1628,
  ground: 0xd4a574,
  sunlight: 0xfffaf0,
  mirrorSurface: 0xc0d0e0,
  mirrorFrame: 0x5d6d7e,
  mirrorGlint: 0xfffff0,
  dishOuter: 0x3a3f4a,       // Outer shell — dark so mirrors contrast
};

// ============================================================================
// Layout Constants
// ============================================================================

const LAYOUT = {
  // Parabolic dish
  dishRadius: 0.75,           // Radius of the dish opening
  dishDepth: 0.35,            // How deep the parabola is
  dishRimY: 0.05,             // Y position of dish rim
  dishThickness: 0.03,        // Shell thickness

  // Sorbent at focal point
  sorbentY: -0.12,            // Focal point inside the dish
  sorbentRadius: 0.15,        // Smaller — concentrated target
  sorbentHeight: 0.08,

  // Collector below dish
  collectorY: -0.5,
  collectorRadius: 0.3,

  groundY: -0.6,
};

// ============================================================================
// Main Creation Function
// ============================================================================

export function createMOFHarvester(props: MOFHarvesterProps): MOFHarvesterRefs {
  const group = new THREE.Group();

  const dome = createDish();
  const sorbent = createSorbent();
  const collector = createCollector();
  const particles = createParticles();
  const mirrors = createMirrorFacets(props.mirrorCount);
  const environment = createEnvironment();

  group.add(dome.mesh);
  group.add(dome.condensationDroplets);
  group.add(sorbent.mesh);
  group.add(sorbent.glowMesh);
  group.add(collector.trough);
  group.add(collector.water);
  group.add(particles.vapor);
  group.add(particles.droplets);
  group.add(mirrors.group);
  group.add(environment.sky);
  group.add(environment.ground);

  const isNight = props.timeOfDay < 0.25 || props.timeOfDay > 0.75;

  return {
    group,
    dome,
    sorbent,
    collector,
    particles,
    mirrors,
    environment,
    isNight,
    cyclePhase: isNight ? 'adsorbing' : 'releasing',
  };
}

// ============================================================================
// Component Creation
// ============================================================================

function createDish(): DomeRefs {
  // Parabolic dish — open at the top, reflective inside
  // Use a LatheGeometry with a parabolic profile
  const segments = 48;
  const profilePoints: THREE.Vector2[] = [];
  const steps = 20;

  for (let i = 0; i <= steps; i++) {
    const t = i / steps; // 0 = center, 1 = rim
    const r = t * LAYOUT.dishRadius;
    // Parabolic curve: y = depth * (r/radius)^2, inverted so center is lowest
    const y = LAYOUT.dishRimY - LAYOUT.dishDepth * (1 - t * t);
    profilePoints.push(new THREE.Vector2(r, y));
  }

  const dishGeo = new THREE.LatheGeometry(profilePoints, segments);

  // Outer shell — dark matte so mirror facets pop
  const dishMat = new THREE.MeshStandardMaterial({
    color: COLORS.dishOuter,
    roughness: 0.8,
    metalness: 0.2,
    side: THREE.DoubleSide,
  });

  const dish = new THREE.Mesh(dishGeo, dishMat);

  // Condensation droplets (vapor condensing and falling)
  const dropletCount = 200;
  const dropletPositions = new Float32Array(dropletCount * 3);
  for (let i = 0; i < dropletCount; i++) {
    dropletPositions[i * 3] = 0;
    dropletPositions[i * 3 + 1] = -10;
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

  return { mesh: dish, condensationDroplets };
}

function createSorbent(): SorbentRefs {
  // MOF sorbent at focal point — small concentrated target
  const sorbentGeo = new THREE.CylinderGeometry(
    LAYOUT.sorbentRadius,
    LAYOUT.sorbentRadius * 0.85,
    LAYOUT.sorbentHeight,
    24
  );

  const sorbentMat = new THREE.MeshStandardMaterial({
    color: COLORS.mofCold,
    roughness: 0.9,
    metalness: 0.1,
  });

  const sorbent = new THREE.Mesh(sorbentGeo, sorbentMat);
  sorbent.position.y = LAYOUT.sorbentY;

  // Support post holding sorbent at focal point
  const postGeo = new THREE.CylinderGeometry(0.02, 0.025, 0.25, 8);
  const postMat = new THREE.MeshStandardMaterial({ color: 0x444444, metalness: 0.5, roughness: 0.5 });
  const post = new THREE.Mesh(postGeo, postMat);
  post.position.y = LAYOUT.sorbentY - LAYOUT.sorbentHeight / 2 - 0.125;
  sorbent.add(post);

  // Heat glow
  const glowGeo = new THREE.CylinderGeometry(
    LAYOUT.sorbentRadius * 1.15,
    LAYOUT.sorbentRadius,
    LAYOUT.sorbentHeight * 1.3,
    24
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
  // Water collection basin below the dish
  const troughGeo = new THREE.CylinderGeometry(
    LAYOUT.collectorRadius,
    LAYOUT.collectorRadius * 0.8,
    0.08,
    24, 1, true // Open-ended cylinder for trough walls
  );

  const troughMat = new THREE.MeshStandardMaterial({
    color: 0x5d4037,
    roughness: 0.7,
    metalness: 0.3,
    side: THREE.DoubleSide,
  });

  const trough = new THREE.Mesh(troughGeo, troughMat);
  trough.position.y = LAYOUT.collectorY;

  // Trough bottom
  const bottomGeo = new THREE.CircleGeometry(LAYOUT.collectorRadius * 0.8, 24);
  const bottom = new THREE.Mesh(bottomGeo, troughMat);
  bottom.rotation.x = -Math.PI / 2;
  bottom.position.y = LAYOUT.collectorY - 0.04;
  trough.add(bottom);

  // Water surface
  const waterGeo = new THREE.CylinderGeometry(
    LAYOUT.collectorRadius * 0.75,
    LAYOUT.collectorRadius * 0.75,
    0.01, 24
  );

  const waterMat = new THREE.MeshPhysicalMaterial({
    color: COLORS.poolWater,
    transparent: true,
    opacity: 0.7,
    roughness: 0.1,
    metalness: 0,
  });

  const water = new THREE.Mesh(waterGeo, waterMat);
  water.position.y = LAYOUT.collectorY - 0.02;
  water.scale.y = 0.1;

  return { trough, water, fillLevel: 0 };
}

function createParticles(): ParticleRefs {
  const VAPOR_COUNT = 800;
  const DROPLET_COUNT = 80;

  const vaporPositions = new Float32Array(VAPOR_COUNT * 3);
  const vaporVelocities = new Float32Array(VAPOR_COUNT * 3);
  const vaporStates = new Float32Array(VAPOR_COUNT);

  for (let i = 0; i < VAPOR_COUNT; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * LAYOUT.sorbentRadius * 2;
    vaporPositions[i * 3] = Math.cos(angle) * radius;
    vaporPositions[i * 3 + 1] = LAYOUT.sorbentY + Math.random() * 0.4;
    vaporPositions[i * 3 + 2] = Math.sin(angle) * radius;
    vaporVelocities[i * 3] = (Math.random() - 0.5) * 0.01;
    vaporVelocities[i * 3 + 1] = Math.random() * 0.02;
    vaporVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.01;
    vaporStates[i] = 0;
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

  const dropletPositions = new Float32Array(DROPLET_COUNT * 3);
  const dropletVelocities = new Float32Array(DROPLET_COUNT * 3);

  for (let i = 0; i < DROPLET_COUNT; i++) {
    dropletPositions[i * 3 + 1] = -10;
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

  return { vapor, vaporPositions, vaporVelocities, vaporStates, droplets, dropletPositions, dropletVelocities };
}

/**
 * Create mirror facets on the inner surface of the parabolic dish.
 * Each facet is a curved segment of the dish's inner surface, highly reflective.
 * All facets angle toward the focal point (where the sorbent sits).
 */
function createMirrorFacets(count: number): MirrorRefs {
  const group = new THREE.Group();
  const panels: THREE.Mesh[] = [];
  const supports: THREE.Mesh[] = [];
  const glintMeshes: THREE.Mesh[] = [];
  const godRays: THREE.Mesh[] = [];

  const panelMat = new THREE.MeshStandardMaterial({
    color: 0xe8eef5,
    metalness: 0.3,
    roughness: 0.2,
    emissive: new THREE.Color(0x8899bb),
    emissiveIntensity: 0.5,
    side: THREE.DoubleSide,
  });

  const glintMat = new THREE.MeshBasicMaterial({
    color: COLORS.mirrorGlint,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  // Create mirror facets as curved segments on the dish inner surface
  const angularGap = 0.18; // Visible gap between facets so dark dish shows through

  for (let i = 0; i < count; i++) {
    const startAngle = (i / count) * Math.PI * 2 + angularGap / 2;
    const endAngle = ((i + 1) / count) * Math.PI * 2 - angularGap / 2;
    const midAngle = (startAngle + endAngle) / 2;

    // Create a curved panel using LatheGeometry for each facet
    const profilePoints: THREE.Vector2[] = [];
    const radialSteps = 10;
    const innerR = 0.08; // Leave hole at center for sorbent
    const outerR = LAYOUT.dishRadius * 0.92;

    for (let j = 0; j <= radialSteps; j++) {
      const t = j / radialSteps;
      const r = innerR + t * (outerR - innerR);
      const normalizedR = r / LAYOUT.dishRadius;
      // Same parabolic curve as dish, offset well above surface so facets are clearly visible
      const y = LAYOUT.dishRimY - LAYOUT.dishDepth * (1 - normalizedR * normalizedR) + 0.025;
      profilePoints.push(new THREE.Vector2(r, y));
    }

    const facetAngleSpan = endAngle - startAngle;
    const facetSegments = Math.max(6, Math.round(facetAngleSpan * 8));

    const facetGeo = new THREE.LatheGeometry(
      profilePoints,
      facetSegments,
      startAngle,
      facetAngleSpan
    );

    const panel = new THREE.Mesh(facetGeo, panelMat.clone());
    panels.push(panel);
    group.add(panel);

    // Glint spot at the center of each facet
    const glintR = (innerR + outerR) * 0.45;
    const normalizedGlintR = glintR / LAYOUT.dishRadius;
    const glintY = LAYOUT.dishRimY - LAYOUT.dishDepth * (1 - normalizedGlintR * normalizedGlintR) + 0.02;

    const glintGeo = new THREE.PlaneGeometry(0.06, 0.06);
    const glint = new THREE.Mesh(glintGeo, glintMat.clone());
    glint.position.set(
      Math.cos(midAngle) * glintR,
      glintY,
      Math.sin(midAngle) * glintR
    );
    // Face upward toward the sun
    glint.rotation.x = -Math.PI / 2;
    glintMeshes.push(glint);
    group.add(glint);

    // No external supports needed — facets are part of the dish
    supports.push(panel); // Reuse panel ref for compat

    // God ray — a tapered cone of light from this facet region to the sorbent
    // Origin: midpoint of the facet surface
    // Target: sorbent focal point at (0, sorbentY, 0)
    const rayOriginR = (innerR + outerR) * 0.5;
    const normalizedRayR = rayOriginR / LAYOUT.dishRadius;
    const rayOriginY = LAYOUT.dishRimY - LAYOUT.dishDepth * (1 - normalizedRayR * normalizedRayR) + 0.025;
    const rayOriginX = Math.cos(midAngle) * rayOriginR;
    const rayOriginZ = Math.sin(midAngle) * rayOriginR;

    const rayTarget = new THREE.Vector3(0, LAYOUT.sorbentY, 0);
    const rayOrigin = new THREE.Vector3(rayOriginX, rayOriginY, rayOriginZ);
    const rayDir = new THREE.Vector3().subVectors(rayTarget, rayOrigin);
    const rayLength = rayDir.length();

    // Cone: wide end at the mirror facet, tapers to a tiny point at the sorbent
    const rayTopRadius = 0.008; // Very narrow at focal point
    const rayBottomRadius = 0.05 + (outerR - innerR) * 0.2; // Wide at mirror
    const rayGeo = new THREE.CylinderGeometry(
      rayTopRadius,
      rayBottomRadius,
      rayLength,
      8,
      1,
      true // Open-ended for translucent light shaft look
    );

    const rayMat = new THREE.MeshBasicMaterial({
      color: 0xfffce0, // Warm sunlight yellow-white
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    const ray = new THREE.Mesh(rayGeo, rayMat);

    // Position at midpoint between origin and target
    const midPoint = new THREE.Vector3().addVectors(rayOrigin, rayTarget).multiplyScalar(0.5);
    ray.position.copy(midPoint);

    // Orient the cylinder to point from origin to target
    // CylinderGeometry is aligned along Y axis by default
    // We need to rotate it so Y axis aligns with rayDir
    rayDir.normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(up, rayDir);
    ray.quaternion.copy(quat);

    godRays.push(ray);
    group.add(ray);
  }

  return { group, panels, supports, glintMeshes, godRays, count };
}

function createEnvironment(): EnvironmentRefs {
  const sun = new THREE.DirectionalLight(COLORS.sunlight, 2);
  sun.position.set(2, 3, 1);
  sun.castShadow = true;

  const ambient = new THREE.AmbientLight(0x8899aa, 0.8);

  const hemisphere = new THREE.HemisphereLight(0x87ceeb, 0xd4a574, 0.6);

  const fill = new THREE.DirectionalLight(0x6688aa, 0.5);
  fill.position.set(-1.5, 1, -1);

  const skyGeo = new THREE.SphereGeometry(10, 32, 32);
  const skyMat = new THREE.MeshBasicMaterial({
    color: COLORS.daySky,
    side: THREE.BackSide,
  });
  const sky = new THREE.Mesh(skyGeo, skyMat);

  const groundGeo = new THREE.CircleGeometry(5, 32);
  const groundMat = new THREE.MeshStandardMaterial({
    color: COLORS.ground,
    roughness: 0.9,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = LAYOUT.groundY;

  return { sun, ambient, hemisphere, fill, sky, ground };
}

// ============================================================================
// Animation
// ============================================================================

export function animateMOFHarvester(
  refs: MOFHarvesterRefs,
  time: number,
  props: MOFHarvesterProps
): void {
  const timeOfDay = props.timeOfDay;
  refs.isNight = timeOfDay < 0.25 || timeOfDay > 0.75;

  const sunIntensity = refs.isNight ? 0 : Math.sin((timeOfDay - 0.25) * 2 * Math.PI);

  updateEnvironment(refs.environment, refs.isNight, sunIntensity);
  animateMirrors(refs.mirrors, sunIntensity, time);

  const targetTemp = refs.isNight ? 25 : 25 + sunIntensity * 75;
  refs.sorbent.temperature += (targetTemp - refs.sorbent.temperature) * 0.02;
  updateSorbent(refs.sorbent, refs.sorbent.temperature);

  if (refs.isNight) {
    refs.cyclePhase = 'adsorbing';
  } else if (refs.sorbent.temperature < 50) {
    refs.cyclePhase = 'heating';
  } else if (refs.sorbent.temperature < 80) {
    refs.cyclePhase = 'releasing';
  } else {
    refs.cyclePhase = 'condensing';
  }

  animateParticles(refs.particles, refs.cyclePhase, refs.sorbent.temperature, time);
  updateCondensation(refs.dome, refs.cyclePhase, time);
  animateDroplets(refs.particles, refs.collector, time);

  const targetFill = Math.min(1, (props.dailyYield || 0) / 2.8);
  refs.collector.fillLevel += (targetFill - refs.collector.fillLevel) * 0.01;
  refs.collector.water.scale.y = 0.1 + refs.collector.fillLevel * 2;
}

function updateEnvironment(env: EnvironmentRefs, isNight: boolean, sunIntensity: number): void {
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

  env.sun.intensity = 0.4 + sunIntensity * 1.6;
  env.hemisphere.intensity = isNight ? 0.3 : 0.4 + sunIntensity * 0.3;
  env.fill.intensity = isNight ? 0.3 : 0.5;
  env.ambient.intensity = isNight ? 1.0 : 0.8;
}

function animateMirrors(mirrors: MirrorRefs, sunIntensity: number, time: number): void {
  const { panels, glintMeshes, godRays } = mirrors;

  for (let i = 0; i < panels.length; i++) {
    const panel = panels[i];
    const glint = glintMeshes[i];
    const panelMat = panel.material as THREE.MeshStandardMaterial;
    const glintMat = glint.material as THREE.MeshBasicMaterial;

    if (sunIntensity > 0.05) {
      panelMat.color.setHex(0xeef3fa);
      panelMat.emissive.setHex(0x99aacc);
      panelMat.emissiveIntensity = 0.4 + sunIntensity * 0.6;

      const phase = (i / panels.length) * Math.PI * 2;
      const glintPulse = Math.max(0, Math.sin(time * 1.5 + phase));
      glintMat.opacity = sunIntensity * glintPulse * 0.9;
    } else {
      panelMat.color.setHex(0xb0bec5);
      panelMat.emissive.setHex(0x445566);
      panelMat.emissiveIntensity = 0.2;
      glintMat.opacity = 0;
    }

    // God rays — visible light beams from mirror facets to sorbent
    if (i < godRays.length) {
      const rayMat = godRays[i].material as THREE.MeshBasicMaterial;
      if (sunIntensity > 0.05) {
        // Subtle shimmer per ray — staggered phase so they pulse independently
        const rayPhase = (i / panels.length) * Math.PI * 2;
        const shimmer = 0.85 + 0.15 * Math.sin(time * 2.0 + rayPhase);
        // Base opacity scales with sun intensity — clearly visible beams
        rayMat.opacity = sunIntensity * 0.4 * shimmer;
        godRays[i].visible = true;
      } else {
        rayMat.opacity = 0;
        godRays[i].visible = false;
      }
    }
  }
}

function updateSorbent(sorbent: SorbentRefs, temperature: number): void {
  const mat = sorbent.mesh.material as THREE.MeshStandardMaterial;
  const glowMat = sorbent.glowMesh.material as THREE.MeshBasicMaterial;

  const t = Math.max(0, Math.min(1, (temperature - 25) / 75));

  const coldColor = new THREE.Color(COLORS.mofCold);
  const warmColor = new THREE.Color(COLORS.mofWarm);
  const hotColor = new THREE.Color(COLORS.mofHot);

  if (t < 0.5) {
    mat.color.lerpColors(coldColor, warmColor, t * 2);
  } else {
    mat.color.lerpColors(warmColor, hotColor, (t - 0.5) * 2);
  }

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

  // Containment radius — particles must stay centered over the dish
  const maxRadius = LAYOUT.dishRadius * 0.4;
  // Soft boundary where particles start getting pushed back inward
  const softRadius = maxRadius * 0.75;

  for (let i = 0; i < count; i++) {
    const px = positions[i * 3];
    const py = positions[i * 3 + 1];
    const pz = positions[i * 3 + 2];
    const dist = Math.sqrt(px * px + pz * pz);

    switch (phase) {
      case 'adsorbing':
        // Drift toward sorbent (being absorbed from air)
        velocities[i * 3] = -px * 0.008;
        velocities[i * 3 + 1] = (LAYOUT.sorbentY - py) * 0.015;
        velocities[i * 3 + 2] = -pz * 0.008;
        states[i] = 0;
        break;

      case 'heating':
        velocities[i * 3] = (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 1] = (Math.random() - 0.5) * 0.002;
        velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.002;
        states[i] = 0.5;
        break;

      case 'releasing': {
        // Rise upward from heated sorbent as vapor
        // Strong damping + centering force to keep cloud above dish center
        velocities[i * 3] *= 0.8;
        velocities[i * 3 + 2] *= 0.8;
        // Always pull back toward center (x=0, z=0)
        velocities[i * 3] -= px * 0.015;
        velocities[i * 3 + 2] -= pz * 0.015;
        velocities[i * 3] += (Math.random() - 0.5) * 0.001;
        velocities[i * 3 + 1] = 0.004 + Math.random() * 0.008;
        velocities[i * 3 + 2] += (Math.random() - 0.5) * 0.001;
        states[i] = 1;
        break;
      }

      case 'condensing': {
        // Drift slightly outward and downward, but stay centered
        const outward = dist < 0.01 ? 0.002 : 0.001;
        velocities[i * 3] = px * outward - px * 0.008 + (Math.random() - 0.5) * 0.0005;
        velocities[i * 3 + 1] = -0.003 - Math.random() * 0.004;
        velocities[i * 3 + 2] = pz * outward - pz * 0.008 + (Math.random() - 0.5) * 0.0005;
        states[i] = 2;
        break;
      }
    }

    // Clamp velocity magnitude to prevent runaway speeds
    const vx = velocities[i * 3];
    const vz = velocities[i * 3 + 2];
    const hSpeed = Math.sqrt(vx * vx + vz * vz);
    const maxHSpeed = 0.008;
    if (hSpeed > maxHSpeed) {
      const vScale = maxHSpeed / hSpeed;
      velocities[i * 3] *= vScale;
      velocities[i * 3 + 2] *= vScale;
    }

    // Soft radial containment: push particles back toward center when near boundary
    if (dist > softRadius) {
      const overshot = (dist - softRadius) / (maxRadius - softRadius);
      const pushBack = Math.min(overshot, 1) * 0.02;
      const invDist = dist > 0.001 ? 1 / dist : 0;
      velocities[i * 3] -= px * invDist * pushBack;
      velocities[i * 3 + 2] -= pz * invDist * pushBack;
    }

    positions[i * 3] += velocities[i * 3];
    positions[i * 3 + 1] += velocities[i * 3 + 1];
    positions[i * 3 + 2] += velocities[i * 3 + 2];

    const newDist = Math.sqrt(positions[i * 3] ** 2 + positions[i * 3 + 2] ** 2);

    // Hard containment: clamp position and kill outward velocity
    if (newDist > maxRadius) {
      const clampScale = maxRadius / newDist;
      positions[i * 3] *= clampScale;
      positions[i * 3 + 2] *= clampScale;
      // Reverse and heavily damp outward velocity
      velocities[i * 3] *= -0.3;
      velocities[i * 3 + 2] *= -0.3;
    }

    // Vertical bounds
    if (positions[i * 3 + 1] > LAYOUT.dishRimY + 0.35) {
      positions[i * 3 + 1] = LAYOUT.dishRimY + 0.35;
      velocities[i * 3 + 1] *= -0.3;
    }
    if (positions[i * 3 + 1] < LAYOUT.collectorY) {
      // Reset to sorbent area
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * LAYOUT.sorbentRadius;
      positions[i * 3] = Math.cos(angle) * r;
      positions[i * 3 + 1] = LAYOUT.sorbentY + 0.05;
      positions[i * 3 + 2] = Math.sin(angle) * r;
      velocities[i * 3] = 0;
      velocities[i * 3 + 1] = 0;
      velocities[i * 3 + 2] = 0;
    }
  }

  particles.vapor.geometry.attributes.position.needsUpdate = true;

  const avgState = Array.from(states).reduce((a, b) => a + b, 0) / count;
  const vaporMat = particles.vapor.material as THREE.PointsMaterial;

  if (avgState < 0.5) {
    vaporMat.color.setHex(0x90a4ae);
    vaporMat.opacity = 0.3;
  } else if (avgState < 1.5) {
    vaporMat.color.setHex(COLORS.vapor);
    vaporMat.opacity = 0.6;
  } else {
    vaporMat.color.setHex(COLORS.droplet);
    vaporMat.opacity = 0.8;
  }
}

function updateCondensation(dome: DomeRefs, phase: string, time: number): void {
  const positions = dome.condensationDroplets.geometry.attributes.position.array as Float32Array;
  const count = positions.length / 3;

  if (phase === 'condensing') {
    for (let i = 0; i < count; i++) {
      if (positions[i * 3 + 1] < -5) {
        // Spawn on dish inner wall
        const angle = Math.random() * Math.PI * 2;
        const r = LAYOUT.sorbentRadius + Math.random() * (LAYOUT.dishRadius * 0.7 - LAYOUT.sorbentRadius);
        const normalizedR = r / LAYOUT.dishRadius;
        const y = LAYOUT.dishRimY - LAYOUT.dishDepth * (1 - normalizedR * normalizedR);
        positions[i * 3] = Math.cos(angle) * r;
        positions[i * 3 + 1] = y + 0.01;
        positions[i * 3 + 2] = Math.sin(angle) * r;
      } else {
        // Slide down the dish toward collector
        positions[i * 3 + 1] -= 0.002;
        if (positions[i * 3 + 1] < LAYOUT.collectorY + 0.05) {
          positions[i * 3 + 1] = -10;
        }
      }
    }
  } else {
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
      velocities[i * 3 + 1] -= 0.001;
      positions[i * 3] += velocities[i * 3];
      positions[i * 3 + 1] += velocities[i * 3 + 1];
      positions[i * 3 + 2] += velocities[i * 3 + 2];

      if (positions[i * 3 + 1] < LAYOUT.collectorY) {
        positions[i * 3 + 1] = -10;
        velocities[i * 3 + 1] = 0;
      }
    } else if (Math.random() < 0.02) {
      const angle = Math.random() * Math.PI * 2;
      const radius = LAYOUT.sorbentRadius + Math.random() * 0.1;
      positions[i * 3] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = LAYOUT.sorbentY - 0.05;
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

  for (let i = 0; i < maxParticles; i++) {
    const gp = gpuPositions[i];
    positions[i * 3] = (gp[0] - 0.5) * LAYOUT.dishRadius * 2;
    positions[i * 3 + 1] = LAYOUT.sorbentY + gp[1] * 0.7;
    positions[i * 3 + 2] = (gp[2] - 0.5) * LAYOUT.dishRadius * 2;
  }

  refs.particles.vapor.geometry.attributes.position.needsUpdate = true;

  const mat = refs.particles.vapor.material as THREE.PointsMaterial;
  mat.color.setHex(0x00ffff);
  mat.opacity = 0.8;
  mat.size = 0.02;

  if (gpuUpdateCount % 100 === 0) {
    console.log(`[GPU→MOF] Update #${gpuUpdateCount}: ${maxParticles} particles`);
  }
}

// ============================================================================
// Mirror Count Update (hot-swap when slider changes)
// ============================================================================

export function updateMirrorCount(refs: MOFHarvesterRefs, newCount: number): void {
  if (refs.mirrors.count === newCount) return;

  refs.mirrors.group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach(m => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });

  refs.group.remove(refs.mirrors.group);

  const newMirrors = createMirrorFacets(newCount);
  refs.mirrors = newMirrors;
  refs.group.add(newMirrors.group);
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
