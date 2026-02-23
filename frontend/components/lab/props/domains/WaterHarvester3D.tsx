// WaterHarvester3D - "Parabolic Solar Dish Concentrator" 3D Visualization
// An atmospheric water harvesting system using parabolic dish mirror concentration
//
// Layout (top to bottom):
// - Sun above
// - Parabolic dish (smooth curved mirror, tracks sun position)
// - God rays (visible light beams converging to focal point)
// - Focal point with HOT sorbent (glowing orange/red)
// - Condenser (beetle surface, COOL, in shadow)
// - Collection bottle at bottom
//
// Components:
// - ParabolicDish: Smooth parabolic mirror dish that tracks sun position
// - GodRays: Volumetric light beams from dish to focal point
// - SorbentCore3D: Focal point receiving concentrated heat, glowing hot
// - CondenserSurface3D: Beetle-textured cool surface for droplet collection
// - VaporParticleSystem: Heat shimmer rising from sorbent, condensing below
// - DropletParticleSystem: Water forming on condenser and falling to bottle
// - HarvesterEnvironment3D: Day/night cycle, volumetric fog
// - HarvesterGauges3D: Floating holographic stats

import * as THREE from 'three';

// ============================================================================
// Types & Interfaces
// ============================================================================

export interface WaterHarvester3DProps {
  // Design parameters
  sorbentWidth: number;      // cm
  sorbentDepth: number;      // cm
  mirrorCount: number;       // 2-8
  mirrorAngle: number;       // degrees
  surfacePattern: 'beetle' | 'flat';
  humidity: number;          // 0-100%

  // Simulation results (optional - drives visualization)
  dailyYield?: number;       // liters
  efficiency?: number;       // 0-100%
  peakTemp?: number;         // celsius
  collectionRate?: number;   // ml/hr

  // Simulation state
  isSimulating?: boolean;
  simulationStatus?: 'pending' | 'running' | 'completed' | 'failed';
}

export interface WaterHarvester3DRefs {
  group: THREE.Group;

  // Sub-components
  sorbentCore: SorbentCoreRefs;
  condenserSurface: CondenserSurfaceRefs;
  parabolicDish: ParabolicDishRefs;
  vaporSystem: VaporSystemRefs;
  droplets: DropletSystemRefs;
  environment: EnvironmentRefs;
  gauges: GaugesRefs;
  collectionBottle: CollectionBottleRefs;

  // Animation state
  dayNightCycle: number;     // 0-1 (0=midnight, 0.5=noon)
  isNight: boolean;
}

interface SorbentCoreRefs {
  group: THREE.Group;
  mesh: THREE.Mesh;
  glowMesh: THREE.Mesh;
  heatShimmer: THREE.Points;
  temperature: number;
}

interface CondenserSurfaceRefs {
  group: THREE.Group;
  mesh: THREE.Mesh;
  bumpTexture: THREE.CanvasTexture;
  coolGlow: THREE.Mesh;
}

interface ParabolicDishRefs {
  group: THREE.Group;
  dish: THREE.Mesh;
  dishFrame: THREE.LineSegments;
  supportArm: THREE.Mesh;
  supportBase: THREE.Mesh;
  godRays: THREE.Mesh[];
  godRaysCone: THREE.Mesh;
  focalPoint: THREE.Vector3;
  sunPosition: THREE.Vector3;
  focalLength: number;
}

interface VaporSystemRefs {
  particles: THREE.Points;
  positions: Float32Array;
  velocities: Float32Array;
  lifetimes: Float32Array;
}

interface DropletSystemRefs {
  particles: THREE.Points;
  positions: Float32Array;
  velocities: Float32Array;
  sizes: Float32Array;
  coalescenceCount: Float32Array;
  splashParticles: THREE.Points;
}

interface EnvironmentRefs {
  skyMesh: THREE.Mesh;
  sunLight: THREE.DirectionalLight;
  ambientLight: THREE.AmbientLight;
  stars: THREE.Points;
  fog: THREE.FogExp2;
  groundPlane: THREE.Mesh;
}

interface GaugesRefs {
  group: THREE.Group;
  yieldGauge: GaugeRefs;
  tempGauge: GaugeRefs;
  humidityGauge: GaugeRefs;
  efficiencyGauge: GaugeRefs;
}

interface GaugeRefs {
  group: THREE.Group;
  needle: THREE.Group;
  arcFill: THREE.Mesh;
  valueSprite: THREE.Sprite;
  labelSprite: THREE.Sprite;
}

interface CollectionBottleRefs {
  group: THREE.Group;
  waterLevel: THREE.Mesh;
  glass: THREE.Mesh;
  fillPercent: number;
}

// ============================================================================
// Color Palette - Archimedes Solar Concentrator Theme
// ============================================================================

const COLORS = {
  // Sky colors
  daySky: 0x87ceeb,
  sunsetSky: 0xff7e5f,
  nightSky: 0x0a1628,

  // Water/droplets
  waterLight: 0x7dd3fc,
  waterMedium: 0x38bdf8,
  waterDark: 0x0284c7,
  waterDeep: 0x0369a1,

  // Temperature gradient
  tempCold: 0x3b82f6,     // Blue
  tempWarm: 0xf97316,     // Orange
  tempHot: 0xef4444,      // Red
  tempGlow: 0xff6b35,     // Orange glow

  // Materials
  sorbentBase: 0x8b7355,  // Sandy brown
  sorbentHot: 0xff4500,   // Red-orange when heated
  condenserCool: 0x4a6fa5, // Cool blue-gray
  condenserSurface: 0x5a7a9a, // Blue-gray beetle surface
  mirror: 0xe8e8e8,       // Bright silver
  mirrorReflect: 0xfffaf0, // Warm white reflection

  // Environment
  groundDay: 0xb8e6c1,    // Pastel green
  groundNight: 0x1e3a29,  // Dark green

  // UI/Gauges
  gaugeBackground: 0x1e293b,
  gaugeBorder: 0x475569,
  gaugeNeedle: 0xef4444,
  gaugeGreen: 0x22c55e,
  gaugeYellow: 0xeab308,
  gaugeRed: 0xef4444,

  // Accent
  accentTeal: 0x14b8a6,
  accentBlue: 0x3b82f6,
};

// ============================================================================
// Scene Layout Constants
// ============================================================================

const LAYOUT = {
  // Parabolic dish faces UP - catches sunlight and focuses it DOWN to focal point
  dishVertexY: 0.0,          // Vertex (bottom of bowl) at ground level
  dishRadius: 1.2,           // Radius of the dish opening
  focalLength: 0.35,         // SHORT focal = focal point INSIDE the bowl

  // With r=1.2, f=0.35: rim height = 1.44/(4*0.35) = 1.03
  // Focal point at 0.35 is WELL INSIDE the bowl (rim at 1.03)
  // This means light rays converge DOWNWARD into the bowl center!

  // Sorbent at focal point (INSIDE the bowl, below rim level)
  sorbentY: 0.35,            // = focalLength (at focal point inside dish)

  // NO separate condenser - the dish surface IS the condenser!
  // At night, the dish cools and water condenses on its inner surface
  // Water droplets slide DOWN the parabolic curve toward the vertex (center bottom)
  condenserY: 0.5,           // Droplets form on dish inner surface (mid-height)
  condenserX: 0,             // Centered

  // Drain at dish vertex (bottom center) leads to collection bottle BELOW
  drainY: -0.05,             // Just below dish vertex
  bottleY: -0.6,             // Collection bottle below the dish
  bottleX: 0,                // Centered directly below drain

  // Support structure (tripod legs around the dish)
  supportBaseY: -0.2,        // Ground level
};

// ============================================================================
// Parabolic Dish Concentrator
// ============================================================================

/**
 * Creates a smooth parabolic dish geometry using parametric equations.
 * Parabola formula: y = (x² + z²) / (4 * focalLength)
 * The dish is oriented facing UPWARD (concave side UP) to catch sunlight from above.
 * Vertex (deepest point) is at y=0, rim is at positive y.
 */
function createParabolicDishGeometry(
  radius: number,
  focalLength: number,
  segments: number = 48
): THREE.BufferGeometry {
  const vertices: number[] = [];
  const indices: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];

  // Create vertices in concentric rings from center outward
  for (let ring = 0; ring <= segments; ring++) {
    const ringRadius = (ring / segments) * radius;
    const ringSegments = Math.max(8, Math.floor(segments * (ring / segments) * 2));

    for (let seg = 0; seg <= ringSegments; seg++) {
      const angle = (seg / ringSegments) * Math.PI * 2;
      const x = ringRadius * Math.cos(angle);
      const z = ringRadius * Math.sin(angle);
      // Parabolic height: POSITIVE so dish opens upward like a bowl
      // y = (x² + z²) / (4f) - center is at y=0, rim rises up
      const y = (x * x + z * z) / (4 * focalLength);

      vertices.push(x, y, z);

      // Calculate inward-facing normal for concave surface
      // For y = (x² + z²)/(4f), gradient is (x/(2f), -1, z/(2f))
      // Normal points INTO the dish (upward and inward)
      const nx = -x / (2 * focalLength);
      const nz = -z / (2 * focalLength);
      const ny = 1;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      normals.push(nx / len, ny / len, nz / len);

      // UV coordinates
      uvs.push(
        0.5 + (x / radius) * 0.5,
        0.5 + (z / radius) * 0.5
      );
    }
  }

  // Create faces by connecting rings
  let vertexOffset = 0;
  for (let ring = 0; ring < segments; ring++) {
    const currentRingSegments = Math.max(8, Math.floor(segments * (ring / segments) * 2));
    const nextRingSegments = Math.max(8, Math.floor(segments * ((ring + 1) / segments) * 2));

    const currentRingStart = vertexOffset;
    const nextRingStart = vertexOffset + currentRingSegments + 1;

    // Connect current ring to next ring
    for (let seg = 0; seg < Math.max(currentRingSegments, nextRingSegments); seg++) {
      const currentSeg = Math.floor(seg * currentRingSegments / Math.max(currentRingSegments, nextRingSegments));
      const nextSeg = Math.floor(seg * nextRingSegments / Math.max(currentRingSegments, nextRingSegments));

      const a = currentRingStart + (currentSeg % (currentRingSegments + 1));
      const b = currentRingStart + ((currentSeg + 1) % (currentRingSegments + 1));
      const c = nextRingStart + (nextSeg % (nextRingSegments + 1));
      const d = nextRingStart + ((nextSeg + 1) % (nextRingSegments + 1));

      if (ring === 0) {
        // First ring - triangles from center
        indices.push(a, d, c);  // Reversed winding for upward-facing
      } else {
        // Other rings - quads
        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }

    vertexOffset = nextRingStart;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();

  return geometry;
}

function createParabolicDish(mirrorCount: number, angle: number): ParabolicDishRefs {
  const group = new THREE.Group();
  const focalLength = LAYOUT.focalLength;
  const sunPosition = new THREE.Vector3(0, 10, 0);

  // Position dish at ground level - vertex (bottom of bowl) at dishVertexY
  group.position.y = LAYOUT.dishVertexY;

  // Focal point is ABOVE the dish at focalLength distance
  const focalPointLocalY = focalLength;  // In local coords, focal point is above vertex
  const focalPoint = new THREE.Vector3(0, LAYOUT.sorbentY, 0);  // World coords

  // Create the parabolic dish mesh - now facing UP
  const dishGeometry = createParabolicDishGeometry(LAYOUT.dishRadius, focalLength, 32);

  // Create mirror material with high reflectivity
  const dishMaterial = new THREE.MeshPhongMaterial({
    color: COLORS.mirror,
    specular: 0xffffff,
    shininess: 200,
    emissive: 0x445566,
    emissiveIntensity: 0.2,
    side: THREE.DoubleSide,
    reflectivity: 1.0,
  });

  const dish = new THREE.Mesh(dishGeometry, dishMaterial);
  dish.castShadow = true;
  dish.receiveShadow = true;
  group.add(dish);

  // Add structural frame lines on the dish surface
  const frameGeometry = new THREE.EdgesGeometry(dishGeometry, 15);
  const frameMaterial = new THREE.LineBasicMaterial({
    color: 0x667788,
    transparent: true,
    opacity: 0.5,
  });
  const dishFrame = new THREE.LineSegments(frameGeometry, frameMaterial);
  group.add(dishFrame);

  // Rim height in local coords (at the top edge of the bowl)
  const rimHeight = (LAYOUT.dishRadius * LAYOUT.dishRadius) / (4 * focalLength);

  // Add rim ring around the dish edge (at top of the bowl)
  const rimGeometry = new THREE.TorusGeometry(LAYOUT.dishRadius, 0.04, 8, 48);
  const rimMaterial = new THREE.MeshToonMaterial({ color: 0x64748b });
  const rim = new THREE.Mesh(rimGeometry, rimMaterial);
  rim.rotation.x = Math.PI / 2;
  rim.position.y = rimHeight;
  group.add(rim);

  // Support struts from rim going INWARD and DOWN to hold sorbent at focal point
  // Focal point is INSIDE the bowl (below rim level)
  const strutCount = 4;
  const strutMaterial = new THREE.MeshToonMaterial({ color: 0x556677 });

  // Create a small support ring at the focal point to hold the sorbent
  const supportRingGeometry = new THREE.TorusGeometry(0.08, 0.015, 8, 16);
  const supportRing = new THREE.Mesh(supportRingGeometry, strutMaterial);
  supportRing.rotation.x = Math.PI / 2;
  supportRing.position.y = focalLength;
  group.add(supportRing);

  // Thin struts from rim down to the focal point support ring
  for (let i = 0; i < strutCount; i++) {
    const strutAngle = (i / strutCount) * Math.PI * 2;

    // Start at rim (outer edge, high up)
    const strutStartX = Math.cos(strutAngle) * LAYOUT.dishRadius * 0.95;
    const strutStartZ = Math.sin(strutAngle) * LAYOUT.dishRadius * 0.95;
    const strutStartY = rimHeight;  // At rim level

    // End at focal point (center, lower down inside bowl)
    const strutEndX = Math.cos(strutAngle) * 0.08;  // Small ring at center
    const strutEndZ = Math.sin(strutAngle) * 0.08;
    const strutEndY = focalLength;  // At focal point

    // Calculate strut vector
    const dx = strutEndX - strutStartX;
    const dy = strutEndY - strutStartY;  // Negative! Going DOWN from rim to focal
    const dz = strutEndZ - strutStartZ;
    const strutLength = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const strutGeometry = new THREE.CylinderGeometry(0.012, 0.015, strutLength, 6);
    const strut = new THREE.Mesh(strutGeometry, strutMaterial);

    // Position at midpoint of strut
    strut.position.set(
      (strutStartX + strutEndX) / 2,
      (strutStartY + strutEndY) / 2,
      (strutStartZ + strutEndZ) / 2
    );

    // Orient strut along direction
    const direction = new THREE.Vector3(dx, dy, dz).normalize();
    strut.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

    group.add(strut);
  }

  // Placeholder for supportArm (not used, but needed for refs)
  const supportArm = new THREE.Mesh(
    new THREE.CylinderGeometry(0.01, 0.01, 0.01, 4),
    new THREE.MeshBasicMaterial({ visible: false })
  );
  group.add(supportArm);

  // Support base (tripod legs holding dish above ground)
  const legCount = 3;
  const legMaterial = new THREE.MeshToonMaterial({ color: 0x374151 });
  for (let i = 0; i < legCount; i++) {
    const legAngle = (i / legCount) * Math.PI * 2 + Math.PI / 6;
    const legGeometry = new THREE.CylinderGeometry(0.04, 0.05, 0.8, 8);
    const leg = new THREE.Mesh(legGeometry, legMaterial);
    leg.position.set(
      Math.cos(legAngle) * LAYOUT.dishRadius * 0.7,
      -0.35,
      Math.sin(legAngle) * LAYOUT.dishRadius * 0.7
    );
    leg.rotation.z = Math.cos(legAngle) * 0.15;
    leg.rotation.x = Math.sin(legAngle) * 0.15;
    group.add(leg);
  }
  const supportBase = new THREE.Mesh(
    new THREE.CylinderGeometry(0.01, 0.01, 0.01, 4),
    new THREE.MeshBasicMaterial({ visible: false })
  );

  // DRAIN TUBE from dish center (vertex) down to collection bottle
  // Water condenses on dish, flows down to center, drains through this tube
  const drainLength = Math.abs(LAYOUT.bottleY) + 0.15;
  const drainGeometry = new THREE.CylinderGeometry(0.05, 0.04, drainLength, 12);
  const drainMaterial = new THREE.MeshStandardMaterial({
    color: 0x4a6a8a,
    transparent: true,
    opacity: 0.7,
    metalness: 0.3,
    roughness: 0.4,
  });
  const drainTube = new THREE.Mesh(drainGeometry, drainMaterial);
  drainTube.position.y = -drainLength / 2 - 0.02;  // From dish vertex downward
  group.add(drainTube);

  // Funnel at top of drain (catches water at dish vertex)
  const funnelGeometry = new THREE.ConeGeometry(0.12, 0.1, 16, 1, true);
  const funnelMaterial = new THREE.MeshStandardMaterial({
    color: 0x5a7a9a,
    transparent: true,
    opacity: 0.8,
    side: THREE.DoubleSide,
  });
  const funnel = new THREE.Mesh(funnelGeometry, funnelMaterial);
  funnel.rotation.x = Math.PI;  // Open side up
  funnel.position.y = 0.02;  // Just at dish vertex
  group.add(funnel);

  // Create GOD RAYS - light beams from dish rim converging DOWN to focal point
  // With short focal length, focal point is INSIDE the bowl, so rays go INWARD and DOWN
  const godRays: THREE.Mesh[] = [];
  const rayCount = Math.max(12, mirrorCount * 3);

  for (let i = 0; i < rayCount; i++) {
    const rayAngle = (i / rayCount) * Math.PI * 2;
    const rayRadius = LAYOUT.dishRadius * 0.8;  // Near rim

    // Start point on dish surface (near rim, high up)
    const startX = Math.cos(rayAngle) * rayRadius;
    const startZ = Math.sin(rayAngle) * rayRadius;
    // Height on dish at this radius - this is HIGH (near rim)
    const startY = (startX * startX + startZ * startZ) / (4 * focalLength);

    // End point is the focal point (center, LOWER than startY)
    const endX = 0;
    const endY = focalPointLocalY;  // Lower than startY - rays converge DOWN
    const endZ = 0;

    // Vector from start to end (dx toward center, dy NEGATIVE = downward, dz toward center)
    const dx = endX - startX;
    const dy = endY - startY;  // This is NEGATIVE - rays go DOWN!
    const dz = endZ - startZ;
    const rayLength = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Thin cone for each ray - tip points toward focal point
    const rayGeometry = new THREE.ConeGeometry(0.05, rayLength, 6, 1, true);
    const rayMaterial = new THREE.MeshBasicMaterial({
      color: 0xffdd44,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
    });

    const ray = new THREE.Mesh(rayGeometry, rayMaterial);

    // Position at midpoint between dish surface and focal point
    ray.position.set(
      (startX + endX) / 2,
      (startY + endY) / 2,
      (startZ + endZ) / 2
    );

    // Orient the cone so tip points toward focal point (DOWNWARD and INWARD)
    const direction = new THREE.Vector3(dx, dy, dz).normalize();
    ray.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

    group.add(ray);
    godRays.push(ray);
  }

  // Central convergence glow at focal point (intense light concentration)
  // This is a sphere glow showing where all rays meet
  const convergeSphereGeometry = new THREE.SphereGeometry(0.15, 16, 16);
  const convergeSphereMaterial = new THREE.MeshBasicMaterial({
    color: 0xffaa33,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
  });
  const godRaysCone = new THREE.Mesh(convergeSphereGeometry, convergeSphereMaterial);
  godRaysCone.position.y = focalPointLocalY;  // At focal point
  group.add(godRaysCone);

  return {
    group,
    dish,
    dishFrame,
    supportArm,
    supportBase,
    godRays,
    godRaysCone,
    focalPoint,
    sunPosition,
    focalLength,
  };
}

function animateParabolicDish(
  refs: ParabolicDishRefs,
  time: number,
  isNight: boolean,
  dayNightCycle: number
): void {
  // Update sun position based on day cycle
  const sunAngle = dayNightCycle * Math.PI * 2 - Math.PI / 2;
  refs.sunPosition.set(
    Math.cos(sunAngle) * 10,
    Math.sin(sunAngle) * 10 + 5,
    Math.sin(sunAngle * 0.3) * 3
  );

  // Calculate sun intensity (0 at night, 1 at noon)
  const sunIntensity = isNight ? 0 : Math.max(0, Math.sin(dayNightCycle * Math.PI));

  // Dish tracks the sun - tilt toward sun position
  if (!isNight && sunIntensity > 0.1) {
    // Calculate direction from dish to sun
    const dishWorldPos = new THREE.Vector3();
    refs.group.getWorldPosition(dishWorldPos);

    const toSun = refs.sunPosition.clone().sub(dishWorldPos).normalize();

    // Calculate tilt angles (limited range to keep dish roughly upward)
    const maxTilt = Math.PI / 6; // 30 degrees max tilt
    const targetTiltX = Math.max(-maxTilt, Math.min(maxTilt, -toSun.z * maxTilt));
    const targetTiltZ = Math.max(-maxTilt, Math.min(maxTilt, toSun.x * maxTilt));

    // Smoothly interpolate dish rotation
    refs.group.rotation.x = THREE.MathUtils.lerp(refs.group.rotation.x, targetTiltX, 0.02);
    refs.group.rotation.z = THREE.MathUtils.lerp(refs.group.rotation.z, targetTiltZ, 0.02);
  } else {
    // Return to neutral position at night
    refs.group.rotation.x = THREE.MathUtils.lerp(refs.group.rotation.x, 0, 0.01);
    refs.group.rotation.z = THREE.MathUtils.lerp(refs.group.rotation.z, 0, 0.01);
  }

  // Animate GOD RAYS
  const rayOpacity = isNight ? 0 : sunIntensity * 0.6;
  refs.godRays.forEach((ray, i) => {
    const material = ray.material as THREE.MeshBasicMaterial;
    // Pulsing shimmer effect
    const shimmer = 0.7 + Math.sin(time * 3 + i * 0.8) * 0.3;
    material.opacity = rayOpacity * shimmer;

    // Subtle color shift toward white at peak intensity
    if (sunIntensity > 0.8) {
      material.color.setHex(0xffffaa);
    } else if (sunIntensity > 0.5) {
      material.color.setHex(0xffee77);
    } else {
      material.color.setHex(0xffdd66);
    }
  });

  // Animate convergence cone
  const coneMaterial = refs.godRaysCone.material as THREE.MeshBasicMaterial;
  const coneShimmer = 0.6 + Math.sin(time * 5) * 0.4;
  coneMaterial.opacity = rayOpacity * coneShimmer * 0.8;

  // Subtle scale pulsing for the convergence cone
  const scalePulse = 1 + Math.sin(time * 4) * 0.05;
  refs.godRaysCone.scale.set(scalePulse, 1, scalePulse);

  // Mirror surface shimmer
  const dishMaterial = refs.dish.material as THREE.MeshPhongMaterial;
  if (!isNight) {
    dishMaterial.emissiveIntensity = 0.15 + Math.sin(time * 2) * 0.05;
    // Increase specular when sun is up
    dishMaterial.shininess = 180 + sunIntensity * 40;
  } else {
    dishMaterial.emissiveIntensity = 0.05;
    dishMaterial.shininess = 100;
  }
}

// ============================================================================
// Sorbent Core (at Focal Point - VERY HOT)
// ============================================================================

function createSorbentCore(width: number, depth: number): SorbentCoreRefs {
  const group = new THREE.Group();
  group.position.y = LAYOUT.sorbentY;

  // Scale dimensions
  const scaleW = Math.max(0.3, width * 0.015);
  const scaleD = Math.max(0.3, depth * 0.015);
  const height = 0.15;

  // Main sorbent mesh - cylindrical for focal point
  const geometry = new THREE.CylinderGeometry(scaleW, scaleW * 0.9, height, 16);
  const material = new THREE.MeshStandardMaterial({
    color: COLORS.sorbentBase,
    roughness: 0.8,
    metalness: 0.1,
    emissive: COLORS.tempHot,
    emissiveIntensity: 0,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);

  // Outer glow mesh (for intense heat visualization)
  const glowGeometry = new THREE.CylinderGeometry(scaleW * 1.3, scaleW * 1.2, height * 0.8, 16);
  const glowMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.tempGlow,
    transparent: true,
    opacity: 0,
    side: THREE.BackSide,
  });
  const glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
  group.add(glowMesh);

  // Heat shimmer particles (rise upward from hot surface)
  const shimmerCount = 300;
  const shimmerGeometry = new THREE.BufferGeometry();
  const shimmerPositions = new Float32Array(shimmerCount * 3);
  const shimmerColors = new Float32Array(shimmerCount * 3);

  for (let i = 0; i < shimmerCount; i++) {
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * scaleW;
    shimmerPositions[i * 3] = Math.cos(angle) * radius;
    shimmerPositions[i * 3 + 1] = Math.random() * 0.8;
    shimmerPositions[i * 3 + 2] = Math.sin(angle) * radius;

    // Orange-yellow heat color
    shimmerColors[i * 3] = 1.0;
    shimmerColors[i * 3 + 1] = 0.6 + Math.random() * 0.3;
    shimmerColors[i * 3 + 2] = 0.2 + Math.random() * 0.2;
  }

  shimmerGeometry.setAttribute('position', new THREE.BufferAttribute(shimmerPositions, 3));
  shimmerGeometry.setAttribute('color', new THREE.BufferAttribute(shimmerColors, 3));

  const shimmerMaterial = new THREE.PointsMaterial({
    size: 0.03,
    vertexColors: true,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
  });

  const heatShimmer = new THREE.Points(shimmerGeometry, shimmerMaterial);
  group.add(heatShimmer);

  // Container ring
  const ringGeometry = new THREE.TorusGeometry(scaleW * 1.1, 0.02, 8, 24);
  const ringMaterial = new THREE.MeshToonMaterial({ color: 0x4a4a4a });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = -height / 2;
  group.add(ring);

  return {
    group,
    mesh,
    glowMesh,
    heatShimmer,
    temperature: 25,
  };
}

function animateSorbentCore(
  refs: SorbentCoreRefs,
  time: number,
  isNight: boolean,
  targetTemp: number
): void {
  // Smooth temperature transition
  refs.temperature = THREE.MathUtils.lerp(refs.temperature, targetTemp, 0.03);

  const material = refs.mesh.material as THREE.MeshStandardMaterial;
  const glowMaterial = refs.glowMesh.material as THREE.MeshBasicMaterial;
  const shimmerMaterial = refs.heatShimmer.material as THREE.PointsMaterial;

  // Temperature normalization (20-150C range for focal point heating)
  const tempNorm = Math.min(1, Math.max(0, (refs.temperature - 20) / 130));

  // Color transition: brown -> orange -> bright red-orange
  const coldColor = new THREE.Color(COLORS.sorbentBase);
  const warmColor = new THREE.Color(COLORS.tempWarm);
  const hotColor = new THREE.Color(COLORS.sorbentHot);

  if (tempNorm < 0.5) {
    material.color.lerpColors(coldColor, warmColor, tempNorm * 2);
  } else {
    material.color.lerpColors(warmColor, hotColor, (tempNorm - 0.5) * 2);
  }

  // Emissive glow when hot
  material.emissiveIntensity = tempNorm * 0.8;

  // Outer glow pulse
  glowMaterial.opacity = tempNorm * 0.4 * (0.8 + Math.sin(time * 3) * 0.2);

  // Animate heat shimmer
  const positions = refs.heatShimmer.geometry.attributes.position.array as Float32Array;

  if (!isNight && refs.temperature > 60) {
    shimmerMaterial.opacity = Math.min(0.6, (refs.temperature - 60) / 100);

    for (let i = 0; i < positions.length / 3; i++) {
      // Rise upward with turbulence
      positions[i * 3 + 1] += 0.015 + Math.random() * 0.01;

      // Horizontal drift (heat shimmer effect)
      positions[i * 3] += Math.sin(time * 5 + i * 0.3) * 0.003;
      positions[i * 3 + 2] += Math.cos(time * 4 + i * 0.4) * 0.003;

      // Reset if too high (vapor rises toward condenser above, but most dissipates)
      if (positions[i * 3 + 1] > 1.2) {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 0.2;
        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = Math.sin(angle) * radius;
      }
    }

    refs.heatShimmer.geometry.attributes.position.needsUpdate = true;
  } else {
    shimmerMaterial.opacity = Math.max(0, shimmerMaterial.opacity - 0.02);
  }
}

// ============================================================================
// Condenser Surface (Dome above sorbent - catches rising vapor)
// ============================================================================

function createCondenserSurface(pattern: 'beetle' | 'flat'): CondenserSurfaceRefs {
  const group = new THREE.Group();
  // Position at rim level as a transparent cover over the bowl opening
  // Vapor rises from sorbent, hits this cool surface, condenses
  group.position.set(LAYOUT.condenserX, LAYOUT.condenserY, 0);

  // Create beetle bump pattern texture
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d')!;

  // Base color - cool blue-gray
  ctx.fillStyle = '#4a6a8a';
  ctx.fillRect(0, 0, 256, 256);

  if (pattern === 'beetle') {
    // Hydrophilic bumps (lighter spots - water-attracting)
    ctx.fillStyle = '#7899bb';
    for (let i = 0; i < 120; i++) {
      const x = (i % 12) * 21 + 10;
      const y = Math.floor(i / 12) * 21 + 10;
      const radius = 3 + Math.random() * 4;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }

    // Water channels (grooves leading to center/drip point)
    ctx.strokeStyle = '#2a4a6a';
    ctx.lineWidth = 2;
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 6) {
      ctx.beginPath();
      ctx.moveTo(128, 128);
      ctx.lineTo(128 + Math.cos(angle) * 120, 128 + Math.sin(angle) * 120);
      ctx.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 2);

  // Create bump map
  const bumpCanvas = document.createElement('canvas');
  bumpCanvas.width = 256;
  bumpCanvas.height = 256;
  const bumpCtx = bumpCanvas.getContext('2d')!;

  bumpCtx.fillStyle = '#808080';
  bumpCtx.fillRect(0, 0, 256, 256);

  if (pattern === 'beetle') {
    bumpCtx.fillStyle = '#ffffff';
    for (let i = 0; i < 120; i++) {
      const x = (i % 12) * 21 + 10;
      const y = Math.floor(i / 12) * 21 + 10;
      const radius = 3 + Math.random() * 4;

      bumpCtx.beginPath();
      bumpCtx.arc(x, y, radius, 0, Math.PI * 2);
      bumpCtx.fill();
    }

    bumpCtx.strokeStyle = '#404040';
    bumpCtx.lineWidth = 3;
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 6) {
      bumpCtx.beginPath();
      bumpCtx.moveTo(128, 128);
      bumpCtx.lineTo(128 + Math.cos(angle) * 120, 128 + Math.sin(angle) * 120);
      bumpCtx.stroke();
    }
  }

  const bumpTexture = new THREE.CanvasTexture(bumpCanvas);
  bumpTexture.wrapS = THREE.RepeatWrapping;
  bumpTexture.wrapT = THREE.RepeatWrapping;
  bumpTexture.repeat.set(2, 2);

  // The parabolic DISH itself is the condenser surface!
  // At night, the dish cools, vapor condenses on its inner surface,
  // and water droplets slide DOWN the parabolic curve to the drain at center.
  // This "condenser" component is now just a visual indicator of where condensation happens.

  // Create a subtle inner glow on the dish to show condensation zone
  // (This is a visual effect - the dish geometry is the actual condenser)
  const condenserRadius = LAYOUT.dishRadius * 0.6;  // Mid-section of dish
  const geometry = new THREE.RingGeometry(0.15, condenserRadius, 32);

  const material = new THREE.MeshStandardMaterial({
    color: 0x6a9acc,  // Cool blue tint showing condensation
    map: texture,
    bumpMap: bumpTexture,
    bumpScale: 0.005,
    roughness: 0.3,
    metalness: 0.1,
    transparent: true,
    opacity: 0.0,  // Invisible by default, shows at night
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;  // Flat, horizontal
  mesh.position.y = 0.3;  // Inside dish, above vertex
  group.add(mesh);

  // Cool glow effect (shows condensation happening on dish at night)
  const coolGlowGeometry = new THREE.RingGeometry(0.1, condenserRadius * 1.1, 32);
  const coolGlowMaterial = new THREE.MeshBasicMaterial({
    color: 0x4488cc,
    transparent: true,
    opacity: 0.0,  // Only visible at night
  });
  const coolGlow = new THREE.Mesh(coolGlowGeometry, coolGlowMaterial);
  coolGlow.rotation.x = -Math.PI / 2;
  coolGlow.position.y = 0.28;
  group.add(coolGlow);

  // No separate components needed - the dish itself is the condenser
  // Water drains through the center drain tube (created in parabolic dish)

  return {
    group,
    mesh,
    bumpTexture,
    coolGlow,
  };
}

function animateCondenser(refs: CondenserSurfaceRefs, time: number, isNight: boolean): void {
  // Condensation effect only shows at night when dish is cool
  const meshMaterial = refs.mesh.material as THREE.MeshStandardMaterial;
  const glowMaterial = refs.coolGlow.material as THREE.MeshBasicMaterial;

  if (isNight) {
    // At night: dish cools, vapor condenses - show blue condensation glow
    meshMaterial.opacity = 0.15 + Math.sin(time * 1.5) * 0.05;
    glowMaterial.opacity = 0.2 + Math.sin(time * 1.2) * 0.08;
  } else {
    // During day: no condensation visible (sorbent heating, vapor releasing)
    meshMaterial.opacity = 0;
    glowMaterial.opacity = 0;
  }
}

// ============================================================================
// Vapor Particle System (Rises from sorbent, drifts toward condenser)
// ============================================================================

function createVaporSystem(count: number = 400): VaporSystemRefs {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  const velocities = new Float32Array(count * 3);
  const lifetimes = new Float32Array(count);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    // Start at sorbent level (focal point)
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.random() * 0.2;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = LAYOUT.sorbentY + Math.random() * 0.1;
    positions[i * 3 + 2] = Math.sin(angle) * radius;

    // Vapor rises straight UP from hot sorbent to condenser above
    velocities[i * 3] = (Math.random() - 0.5) * 0.002;  // Slight horizontal drift
    velocities[i * 3 + 1] = 0.003 + Math.random() * 0.004; // Rise upward
    velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.002;

    lifetimes[i] = Math.random();

    // White/light blue vapor color
    colors[i * 3] = 0.9;
    colors[i * 3 + 1] = 0.95;
    colors[i * 3 + 2] = 1.0;
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.04,
    vertexColors: true,
    transparent: true,
    opacity: 0.3,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });

  const particles = new THREE.Points(geometry, material);

  return {
    particles,
    positions,
    velocities,
    lifetimes,
  };
}

function animateVaporSystem(
  refs: VaporSystemRefs,
  time: number,
  isNight: boolean,
  temperature: number
): void {
  const positions = refs.particles.geometry.attributes.position.array as Float32Array;
  const material = refs.particles.material as THREE.PointsMaterial;

  // Vapor only visible when sorbent is hot (releasing moisture)
  const vaporIntensity = Math.max(0, (temperature - 50) / 100);
  material.opacity = vaporIntensity * (isNight ? 0.2 : 0.4);

  if (vaporIntensity > 0.1) {
    for (let i = 0; i < positions.length / 3; i++) {
      // Apply velocity - vapor rises from sorbent toward condenser dome
      positions[i * 3] += refs.velocities[i * 3];
      positions[i * 3 + 1] += refs.velocities[i * 3 + 1];
      positions[i * 3 + 2] += refs.velocities[i * 3 + 2];

      // Turbulence inside the bowl
      positions[i * 3] += Math.sin(time * 3 + i * 0.2) * 0.002;
      positions[i * 3 + 2] += Math.cos(time * 2.5 + i * 0.3) * 0.002;

      // Get distance from center
      const px = positions[i * 3];
      const pz = positions[i * 3 + 2];
      const distFromCenter = Math.sqrt(px * px + pz * pz);

      // Reset when vapor reaches condenser dome or dish inner wall
      // Condenser dome is at rim level (~1.0), dish wall curves up to rim
      const maxHeight = LAYOUT.condenserY - 0.1;
      const maxRadius = LAYOUT.dishRadius * 0.8;  // Stay inside dish

      if (positions[i * 3 + 1] > maxHeight || distFromCenter > maxRadius) {
        // Respawn at sorbent (focal point inside bowl)
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 0.15;
        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] = LAYOUT.sorbentY + 0.05 + Math.random() * 0.15;
        positions[i * 3 + 2] = Math.sin(angle) * radius;

        // Rise upward toward dome
        refs.velocities[i * 3] = (Math.random() - 0.5) * 0.003;
        refs.velocities[i * 3 + 1] = 0.004 + Math.random() * 0.006;  // Faster rise
        refs.velocities[i * 3 + 2] = (Math.random() - 0.5) * 0.003;
      }
    }

    refs.particles.geometry.attributes.position.needsUpdate = true;
  }
}

// ============================================================================
// Droplet Particle System (Forms on condenser, falls to bottle)
// ============================================================================

function createDropletSystem(count: number = 3000): DropletSystemRefs {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  const velocities = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const coalescenceCount = new Float32Array(count);

  const colorLight = new THREE.Color(COLORS.waterLight);
  const colorMedium = new THREE.Color(COLORS.waterMedium);
  const colorDark = new THREE.Color(COLORS.waterDark);

  for (let i = 0; i < count; i++) {
    // Start on inner surface of parabolic dish (distributed across dish)
    const angle = Math.random() * Math.PI * 2;
    // Droplets form across dish surface, from near rim down toward center
    const radius = 0.15 + Math.random() * (LAYOUT.dishRadius * 0.8 - 0.15);
    positions[i * 3] = Math.cos(angle) * radius;
    // Height on dish surface: y = r²/(4f) - parabolic curve
    const dishHeight = (radius * radius) / (4 * LAYOUT.focalLength);
    positions[i * 3 + 1] = dishHeight + 0.02;  // Slightly above dish surface
    positions[i * 3 + 2] = Math.sin(angle) * radius;

    // Droplets slide INWARD and DOWN along dish surface toward center drain
    // Velocity points toward center (0,0) and down
    const inwardSpeed = 0.001 + Math.random() * 0.001;
    velocities[i * 3] = -Math.cos(angle) * inwardSpeed;  // Toward center
    velocities[i * 3 + 1] = -0.0003 - Math.random() * 0.0005;  // Gravity
    velocities[i * 3 + 2] = -Math.sin(angle) * inwardSpeed;  // Toward center

    const colorMix = Math.random();
    const color = colorMix < 0.5 ? colorLight : colorMix < 0.8 ? colorMedium : colorDark;
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;

    sizes[i] = 0.01 + Math.random() * 0.015;
    coalescenceCount[i] = 0;
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const material = new THREE.PointsMaterial({
    size: 0.025,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    blending: THREE.NormalBlending,
    sizeAttenuation: true,
  });

  const particles = new THREE.Points(geometry, material);

  // Splash particles
  const splashGeometry = new THREE.BufferGeometry();
  const splashPositions = new Float32Array(150 * 3);
  const splashColors = new Float32Array(150 * 3);

  for (let i = 0; i < 150; i++) {
    splashPositions[i * 3] = 0;
    splashPositions[i * 3 + 1] = -10;
    splashPositions[i * 3 + 2] = 0;

    splashColors[i * 3] = colorLight.r;
    splashColors[i * 3 + 1] = colorLight.g;
    splashColors[i * 3 + 2] = colorLight.b;
  }

  splashGeometry.setAttribute('position', new THREE.BufferAttribute(splashPositions, 3));
  splashGeometry.setAttribute('color', new THREE.BufferAttribute(splashColors, 3));

  const splashMaterial = new THREE.PointsMaterial({
    size: 0.015,
    vertexColors: true,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
  });

  const splashParticles = new THREE.Points(splashGeometry, splashMaterial);

  return {
    particles,
    positions,
    velocities,
    sizes,
    coalescenceCount,
    splashParticles,
  };
}

function animateDroplets(
  refs: DropletSystemRefs,
  time: number,
  isNight: boolean,
  humidity: number,
  temperature: number
): void {
  const positions = refs.particles.geometry.attributes.position.array as Float32Array;
  const sizes = refs.particles.geometry.attributes.size.array as Float32Array;
  const colors = refs.particles.geometry.attributes.color.array as Float32Array;

  const colorLight = new THREE.Color(COLORS.waterLight);
  const colorDark = new THREE.Color(COLORS.waterDeep);

  // Condensation happens when vapor hits cool condenser
  // More condensation when: high humidity, night time, or high temp differential
  const condensationRate = (humidity / 100) * (isNight ? 0.9 : 0.4) * Math.min(1, temperature / 80);

  const gravity = 0.00012;
  const terminalVelocity = 0.015;
  const collectionY = LAYOUT.bottleY + 0.3;

  for (let i = 0; i < positions.length / 3; i++) {
    const idx = i * 3;

    // Apply gravity
    refs.velocities[idx + 1] -= gravity;
    if (refs.velocities[idx + 1] < -terminalVelocity) {
      refs.velocities[idx + 1] = -terminalVelocity;
    }

    // Update position
    positions[idx] += refs.velocities[idx];
    positions[idx + 1] += refs.velocities[idx + 1];
    positions[idx + 2] += refs.velocities[idx + 2];

    // Slight drift
    positions[idx] += Math.sin(time * 1.5 + i * 0.1) * 0.0001;
    positions[idx + 2] += Math.cos(time * 1.5 + i * 0.1) * 0.0001;

    // Calculate distance from center (drain is at center)
    const px = positions[idx];
    const pz = positions[idx + 2];
    const distFromCenter = Math.sqrt(px * px + pz * pz);

    // Keep droplets on dish surface as they slide toward center
    if (distFromCenter > 0.12 && positions[idx + 1] > -0.1) {
      // Still on dish - follow the parabolic surface
      const surfaceY = (distFromCenter * distFromCenter) / (4 * LAYOUT.focalLength);
      positions[idx + 1] = Math.max(positions[idx + 1], surfaceY + 0.015);

      // Accelerate toward center (gravity pulls down the curve)
      const toCenter = 0.0004 / Math.max(0.3, distFromCenter);
      refs.velocities[idx] -= (px / distFromCenter) * toCenter;
      refs.velocities[idx + 2] -= (pz / distFromCenter) * toCenter;
    }

    // At center drain - droplet falls straight down through tube
    if (distFromCenter < 0.12) {
      refs.velocities[idx] *= 0.8;  // Stop horizontal movement
      refs.velocities[idx + 2] *= 0.8;
      refs.velocities[idx + 1] -= gravity * 2;  // Accelerate down through drain
    }

    // Droplet reaches collection bottle
    if (positions[idx + 1] < collectionY) {
      if (Math.random() < condensationRate) {
        // Reset to dish inner surface (condensation)
        const angle = Math.random() * Math.PI * 2;
        const radius = 0.2 + Math.random() * (LAYOUT.dishRadius * 0.75 - 0.2);
        positions[idx] = Math.cos(angle) * radius;
        const dishHeight = (radius * radius) / (4 * LAYOUT.focalLength);
        positions[idx + 1] = dishHeight + 0.02;
        positions[idx + 2] = Math.sin(angle) * radius;

        // Slide toward center drain
        const inwardVel = 0.0008 + Math.random() * 0.0006;
        refs.velocities[idx] = -Math.cos(angle) * inwardVel;
        refs.velocities[idx + 1] = -0.0002;
        refs.velocities[idx + 2] = -Math.sin(angle) * inwardVel;
        sizes[i] = 0.008 + Math.random() * 0.012;
        refs.coalescenceCount[i] = 0;

        colors[idx] = colorLight.r;
        colors[idx + 1] = colorLight.g;
        colors[idx + 2] = colorLight.b;
      } else {
        positions[idx + 1] = -10; // Hide
      }
    }

    // Coalescence check
    if (i < positions.length / 3 - 1 && sizes[i] < 0.04) {
      const nextIdx = (i + 1) * 3;
      const dx = positions[idx] - positions[nextIdx];
      const dy = positions[idx + 1] - positions[nextIdx + 1];
      const dz = positions[idx + 2] - positions[nextIdx + 2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (dist < 0.025) {
        sizes[i] += sizes[i + 1] * 0.4;
        sizes[i + 1] = 0.008;
        refs.coalescenceCount[i]++;

        const mixFactor = Math.min(1, refs.coalescenceCount[i] / 4);
        colors[idx] = THREE.MathUtils.lerp(colorLight.r, colorDark.r, mixFactor);
        colors[idx + 1] = THREE.MathUtils.lerp(colorLight.g, colorDark.g, mixFactor);
        colors[idx + 2] = THREE.MathUtils.lerp(colorLight.b, colorDark.b, mixFactor);

        refs.velocities[idx + 1] *= 1.15;
      }
    }
  }

  refs.particles.geometry.attributes.position.needsUpdate = true;
  refs.particles.geometry.attributes.size.needsUpdate = true;
  refs.particles.geometry.attributes.color.needsUpdate = true;
}

// ============================================================================
// Environment (Sky, Lighting, Ground)
// ============================================================================

function createEnvironment(): EnvironmentRefs {
  // Sky dome
  const skyGeometry = new THREE.SphereGeometry(50, 32, 32);
  const skyMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.daySky,
    side: THREE.BackSide,
  });
  const skyMesh = new THREE.Mesh(skyGeometry, skyMaterial);

  // Sun directional light
  const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
  sunLight.position.set(5, 10, 5);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 1024;
  sunLight.shadow.mapSize.height = 1024;
  sunLight.shadow.camera.near = 0.5;
  sunLight.shadow.camera.far = 50;
  sunLight.shadow.camera.left = -6;
  sunLight.shadow.camera.right = 6;
  sunLight.shadow.camera.top = 6;
  sunLight.shadow.camera.bottom = -6;

  // Ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);

  // Stars
  const starsCount = 600;
  const starsGeometry = new THREE.BufferGeometry();
  const starsPositions = new Float32Array(starsCount * 3);

  for (let i = 0; i < starsCount; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = 45;

    starsPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starsPositions[i * 3 + 1] = Math.abs(r * Math.cos(phi));
    starsPositions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }

  starsGeometry.setAttribute('position', new THREE.BufferAttribute(starsPositions, 3));

  const starsMaterial = new THREE.PointsMaterial({
    size: 0.2,
    color: 0xffffff,
    transparent: true,
    opacity: 0,
  });

  const stars = new THREE.Points(starsGeometry, starsMaterial);

  // Ground plane
  const groundGeometry = new THREE.PlaneGeometry(25, 25);
  const groundMaterial = new THREE.MeshToonMaterial({
    color: COLORS.groundDay,
  });
  const groundPlane = new THREE.Mesh(groundGeometry, groundMaterial);
  groundPlane.rotation.x = -Math.PI / 2;
  groundPlane.position.y = -0.5;
  groundPlane.receiveShadow = true;

  // Grid
  const gridHelper = new THREE.GridHelper(25, 25, 0x94a3b8, 0x94a3b8);
  gridHelper.position.y = -0.49;
  (gridHelper.material as THREE.Material).transparent = true;
  (gridHelper.material as THREE.Material).opacity = 0.08;
  groundPlane.add(gridHelper);

  // Fog
  const fog = new THREE.FogExp2(COLORS.daySky, 0.015);

  return {
    skyMesh,
    sunLight,
    ambientLight,
    stars,
    fog,
    groundPlane,
  };
}

function animateEnvironment(refs: EnvironmentRefs, dayNightCycle: number): void {
  const isNight = dayNightCycle < 0.25 || dayNightCycle > 0.75;

  const skyMaterial = refs.skyMesh.material as THREE.MeshBasicMaterial;
  const daySkyColor = new THREE.Color(COLORS.daySky);
  const sunsetColor = new THREE.Color(COLORS.sunsetSky);
  const nightSkyColor = new THREE.Color(COLORS.nightSky);

  if (dayNightCycle < 0.2) {
    skyMaterial.color.lerpColors(nightSkyColor, sunsetColor, dayNightCycle / 0.2);
  } else if (dayNightCycle < 0.3) {
    skyMaterial.color.lerpColors(sunsetColor, daySkyColor, (dayNightCycle - 0.2) / 0.1);
  } else if (dayNightCycle < 0.7) {
    skyMaterial.color.copy(daySkyColor);
  } else if (dayNightCycle < 0.8) {
    skyMaterial.color.lerpColors(daySkyColor, sunsetColor, (dayNightCycle - 0.7) / 0.1);
  } else {
    skyMaterial.color.lerpColors(sunsetColor, nightSkyColor, (dayNightCycle - 0.8) / 0.2);
  }

  // Sun position
  const sunAngle = dayNightCycle * Math.PI * 2 - Math.PI / 2;
  refs.sunLight.position.set(
    Math.cos(sunAngle) * 10,
    Math.sin(sunAngle) * 10 + 3,
    4
  );

  const sunHeight = Math.sin(sunAngle);
  refs.sunLight.intensity = Math.max(0, sunHeight * 1.4);

  refs.ambientLight.intensity = isNight ? 0.12 : 0.35;
  refs.ambientLight.color.copy(isNight ? new THREE.Color(0x3355aa) : new THREE.Color(0xffffff));

  const starsMaterial = refs.stars.material as THREE.PointsMaterial;
  starsMaterial.opacity = isNight ? 0.85 : 0;
  if (isNight) {
    starsMaterial.size = 0.15 + Math.sin(Date.now() * 0.002) * 0.05;
  }

  const groundMaterial = refs.groundPlane.material as THREE.MeshToonMaterial;
  groundMaterial.color.lerpColors(
    new THREE.Color(COLORS.groundNight),
    new THREE.Color(COLORS.groundDay),
    isNight ? 0.25 : 1
  );

  refs.fog.density = isNight ? 0.025 : 0.01;
  refs.fog.color.copy(skyMaterial.color);
}

// ============================================================================
// Gauges (HUD)
// ============================================================================

function createGauge(label: string, color: number, position: THREE.Vector3): GaugeRefs {
  const group = new THREE.Group();
  group.position.copy(position);

  const radius = 0.22;
  const arcWidth = 0.04;

  const bgGeometry = new THREE.RingGeometry(radius - arcWidth, radius, 32, 1, 0, Math.PI);
  const bgMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.gaugeBackground,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.8,
  });
  const bgArc = new THREE.Mesh(bgGeometry, bgMaterial);
  bgArc.position.z = 0.01;
  group.add(bgArc);

  const fillGeometry = new THREE.RingGeometry(radius - arcWidth, radius, 32, 1, Math.PI, 0);
  const fillMaterial = new THREE.MeshBasicMaterial({
    color: color,
    side: THREE.DoubleSide,
  });
  const arcFill = new THREE.Mesh(fillGeometry, fillMaterial);
  arcFill.position.z = 0.015;
  group.add(arcFill);

  const borderGeometry = new THREE.RingGeometry(radius + 0.01, radius + 0.025, 32, 1, 0, Math.PI);
  const borderMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.gaugeBorder,
    side: THREE.DoubleSide,
  });
  const border = new THREE.Mesh(borderGeometry, borderMaterial);
  border.position.z = 0.02;
  group.add(border);

  const needleGeometry = new THREE.ConeGeometry(0.015, radius - 0.04, 4);
  const needleMaterial = new THREE.MeshBasicMaterial({ color: COLORS.gaugeNeedle });
  const needleMesh = new THREE.Mesh(needleGeometry, needleMaterial);
  needleMesh.position.y = (radius - 0.02) / 2;

  const needle = new THREE.Group();
  needle.add(needleMesh);
  needle.rotation.z = Math.PI / 2;
  needle.position.z = 0.025;
  group.add(needle);

  const centerGeometry = new THREE.CircleGeometry(0.015, 16);
  const centerMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const centerDot = new THREE.Mesh(centerGeometry, centerMaterial);
  centerDot.position.z = 0.03;
  group.add(centerDot);

  const valueCanvas = document.createElement('canvas');
  valueCanvas.width = 128;
  valueCanvas.height = 64;
  const valueTexture = new THREE.CanvasTexture(valueCanvas);
  const valueMaterial = new THREE.SpriteMaterial({ map: valueTexture, transparent: true });
  const valueSprite = new THREE.Sprite(valueMaterial);
  valueSprite.position.set(0, -0.05, 0.04);
  valueSprite.scale.set(0.1, 0.05, 1);
  group.add(valueSprite);

  const labelCanvas = document.createElement('canvas');
  labelCanvas.width = 128;
  labelCanvas.height = 64;
  const labelCtx = labelCanvas.getContext('2d')!;
  labelCtx.fillStyle = '#94a3b8';
  labelCtx.font = 'bold 24px Arial';
  labelCtx.textAlign = 'center';
  labelCtx.textBaseline = 'middle';
  labelCtx.fillText(label, 64, 32);

  const labelTexture = new THREE.CanvasTexture(labelCanvas);
  const labelMaterial = new THREE.SpriteMaterial({ map: labelTexture, transparent: true });
  const labelSprite = new THREE.Sprite(labelMaterial);
  labelSprite.position.set(0, -0.1, 0.04);
  labelSprite.scale.set(0.12, 0.04, 1);
  group.add(labelSprite);

  return {
    group,
    needle,
    arcFill,
    valueSprite,
    labelSprite,
  };
}

function updateGauge(gauge: GaugeRefs, value: number, displayText: string, maxValue: number = 100): void {
  const normalizedValue = Math.min(1, Math.max(0, value / maxValue));

  const needleAngle = Math.PI / 2 - normalizedValue * Math.PI;
  gauge.needle.rotation.z = needleAngle;

  const radius = 0.12;
  const arcWidth = 0.02;
  gauge.arcFill.geometry.dispose();
  gauge.arcFill.geometry = new THREE.RingGeometry(
    radius - arcWidth,
    radius,
    32,
    1,
    Math.PI - normalizedValue * Math.PI,
    normalizedValue * Math.PI
  );

  const fillMaterial = gauge.arcFill.material as THREE.MeshBasicMaterial;
  if (normalizedValue < 0.5) {
    fillMaterial.color.setHex(COLORS.gaugeGreen);
  } else if (normalizedValue < 0.8) {
    fillMaterial.color.setHex(COLORS.gaugeYellow);
  } else {
    fillMaterial.color.setHex(COLORS.gaugeRed);
  }

  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, 128, 64);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 28px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(displayText, 64, 32);

  const texture = new THREE.CanvasTexture(canvas);
  (gauge.valueSprite.material as THREE.SpriteMaterial).map?.dispose();
  (gauge.valueSprite.material as THREE.SpriteMaterial).map = texture;
  (gauge.valueSprite.material as THREE.SpriteMaterial).needsUpdate = true;
}

function createGauges(): GaugesRefs {
  const group = new THREE.Group();
  // Position gauges to the right side of the scene
  group.position.set(2.2, 1.5, 0);

  const yieldGauge = createGauge('YIELD', COLORS.accentBlue, new THREE.Vector3(0, 0.55, 0));
  const tempGauge = createGauge('TEMP', COLORS.tempWarm, new THREE.Vector3(0.6, 0.55, 0));
  const humidityGauge = createGauge('HUMIDITY', COLORS.accentTeal, new THREE.Vector3(0, 0, 0));
  const efficiencyGauge = createGauge('EFFICIENCY', COLORS.gaugeGreen, new THREE.Vector3(0.6, 0, 0));

  group.add(yieldGauge.group);
  group.add(tempGauge.group);
  group.add(humidityGauge.group);
  group.add(efficiencyGauge.group);

  return {
    group,
    yieldGauge,
    tempGauge,
    humidityGauge,
    efficiencyGauge,
  };
}

function animateGauges(refs: GaugesRefs, time: number): void {
  refs.group.rotation.y = Math.sin(time * 0.5) * 0.05;
  refs.group.position.y = 1.5 + Math.sin(time) * 0.03;
}

// ============================================================================
// Collection Bottle
// ============================================================================

function createCollectionBottle(): CollectionBottleRefs {
  const group = new THREE.Group();
  // Position below the condenser to catch drips
  group.position.set(LAYOUT.bottleX, LAYOUT.bottleY, 0);

  // Glass bottle
  const glassGeometry = new THREE.CylinderGeometry(0.18, 0.14, 0.6, 16, 1, true);
  const glassMaterial = new THREE.MeshStandardMaterial({
    color: 0xaaddff,
    transparent: true,
    opacity: 0.25,
    metalness: 0.1,
    roughness: 0.1,
    side: THREE.DoubleSide,
  });
  const glass = new THREE.Mesh(glassGeometry, glassMaterial);
  glass.position.y = 0.3;
  group.add(glass);

  // Bottle bottom
  const bottomGeometry = new THREE.CircleGeometry(0.14, 16);
  const bottomMaterial = new THREE.MeshStandardMaterial({
    color: 0xaaddff,
    transparent: true,
    opacity: 0.35,
  });
  const bottom = new THREE.Mesh(bottomGeometry, bottomMaterial);
  bottom.rotation.x = -Math.PI / 2;
  group.add(bottom);

  // Water inside
  const waterGeometry = new THREE.CylinderGeometry(0.16, 0.13, 0.01, 16);
  const waterMaterial = new THREE.MeshStandardMaterial({
    color: COLORS.waterMedium,
    transparent: true,
    opacity: 0.85,
    metalness: 0.1,
    roughness: 0.2,
  });
  const waterLevel = new THREE.Mesh(waterGeometry, waterMaterial);
  waterLevel.position.y = 0.01;
  waterLevel.scale.y = 0.01;
  group.add(waterLevel);

  // Measurement lines
  for (let i = 1; i <= 5; i++) {
    const lineGeometry = new THREE.PlaneGeometry(0.06, 0.002);
    const lineMaterial = new THREE.MeshBasicMaterial({ color: 0x555555 });
    const line = new THREE.Mesh(lineGeometry, lineMaterial);
    line.position.set(0.18, i * 0.1, 0);
    line.rotation.z = Math.PI / 2;
    group.add(line);
  }

  // Bottle neck/opening to catch drips
  const neckGeometry = new THREE.CylinderGeometry(0.12, 0.18, 0.15, 16, 1, true);
  const neckMaterial = new THREE.MeshStandardMaterial({
    color: 0xaaddff,
    transparent: true,
    opacity: 0.2,
    side: THREE.DoubleSide,
  });
  const neck = new THREE.Mesh(neckGeometry, neckMaterial);
  neck.position.y = 0.65;
  group.add(neck);

  return {
    group,
    waterLevel,
    glass,
    fillPercent: 0,
  };
}

function animateBottle(refs: CollectionBottleRefs, targetFill: number): void {
  refs.fillPercent = THREE.MathUtils.lerp(refs.fillPercent, targetFill, 0.02);

  const fillHeight = 0.5 * refs.fillPercent;
  refs.waterLevel.scale.y = Math.max(0.01, fillHeight);
  refs.waterLevel.position.y = fillHeight / 2;

  const topRadius = THREE.MathUtils.lerp(0.13, 0.16, refs.fillPercent);
  refs.waterLevel.geometry.dispose();
  refs.waterLevel.geometry = new THREE.CylinderGeometry(topRadius, 0.13, Math.max(0.01, fillHeight), 16);
}

// ============================================================================
// Main Export: Create Water Harvester 3D
// ============================================================================

export function createWaterHarvester3D(props: WaterHarvester3DProps): WaterHarvester3DRefs {
  const group = new THREE.Group();

  // Create all sub-components in correct vertical order
  const parabolicDish = createParabolicDish(props.mirrorCount, props.mirrorAngle);
  const sorbentCore = createSorbentCore(props.sorbentWidth, props.sorbentDepth);
  const condenserSurface = createCondenserSurface(props.surfacePattern);
  const vaporSystem = createVaporSystem(400);
  const droplets = createDropletSystem(3000);
  const environment = createEnvironment();
  const gauges = createGauges();
  const collectionBottle = createCollectionBottle();

  // Add to main group (order matters for rendering)
  group.add(environment.skyMesh);
  group.add(environment.sunLight);
  group.add(environment.ambientLight);
  group.add(environment.stars);
  group.add(environment.groundPlane);
  group.add(parabolicDish.group);
  group.add(sorbentCore.group);
  group.add(condenserSurface.group);
  group.add(vaporSystem.particles);
  group.add(droplets.particles);
  group.add(droplets.splashParticles);
  group.add(collectionBottle.group);
  group.add(gauges.group);

  // Initial gauge values
  updateGauge(gauges.yieldGauge, props.dailyYield || 0, `${(props.dailyYield || 0).toFixed(1)}L`, 2);
  updateGauge(gauges.tempGauge, props.peakTemp || 25, `${props.peakTemp || 25}C`, 150);
  updateGauge(gauges.humidityGauge, props.humidity, `${props.humidity}%`, 100);
  updateGauge(gauges.efficiencyGauge, props.efficiency || 0, `${props.efficiency || 0}%`, 100);

  return {
    group,
    sorbentCore,
    condenserSurface,
    parabolicDish,
    vaporSystem,
    droplets,
    environment,
    gauges,
    collectionBottle,
    dayNightCycle: 0.5,
    isNight: false,
  };
}

// ============================================================================
// Animation Loop
// ============================================================================

export function animateWaterHarvester3D(
  refs: WaterHarvester3DRefs,
  time: number,
  props: WaterHarvester3DProps,
  cycleSpeed: number = 0.0003
): void {
  // Update day/night cycle
  refs.dayNightCycle = (refs.dayNightCycle + cycleSpeed) % 1;
  refs.isNight = refs.dayNightCycle < 0.25 || refs.dayNightCycle > 0.75;

  // Temperature at focal point - much higher due to parabolic concentration
  const baseTemp = 25;
  const maxTemp = props.peakTemp || 120;
  const tempMultiplier = refs.isNight ? 0.2 : 1;
  const sunIntensity = Math.max(0, Math.sin(refs.dayNightCycle * Math.PI));
  // Parabolic dish concentration creates very high temperatures at focal point
  const concentrationFactor = 1.8;
  const currentTemp = baseTemp + (maxTemp - baseTemp) * sunIntensity * tempMultiplier * concentrationFactor;

  // Animate all components
  animateEnvironment(refs.environment, refs.dayNightCycle);
  animateParabolicDish(refs.parabolicDish, time, refs.isNight, refs.dayNightCycle);
  animateSorbentCore(refs.sorbentCore, time, refs.isNight, currentTemp);
  animateCondenser(refs.condenserSurface, time, refs.isNight);
  animateVaporSystem(refs.vaporSystem, time, refs.isNight, currentTemp);
  animateDroplets(refs.droplets, time, refs.isNight, props.humidity, currentTemp);
  animateGauges(refs.gauges, time);

  // Bottle fills based on daily yield
  const fillTarget = Math.min(1, (props.dailyYield || 0) / 2);
  animateBottle(refs.collectionBottle, fillTarget);

  // Update gauges
  updateGauge(refs.gauges.yieldGauge, props.dailyYield || 0, `${(props.dailyYield || 0).toFixed(1)}L`, 2);
  updateGauge(refs.gauges.tempGauge, currentTemp, `${Math.round(currentTemp)}C`, 150);
  updateGauge(refs.gauges.humidityGauge, props.humidity, `${props.humidity}%`, 100);
  updateGauge(refs.gauges.efficiencyGauge, props.efficiency || 0, `${props.efficiency || 0}%`, 100);
}

// ============================================================================
// GPU Particle Update (from Warp server)
// ============================================================================

// Track GPU update count for debugging
let gpuUpdateCount = 0;

export function updateParticlesFromGPU(
  refs: WaterHarvester3DRefs,
  gpuPositions: number[][],
  scale: number = 0.8
): void {
  if (!refs.vaporSystem || !gpuPositions || gpuPositions.length === 0) return;

  gpuUpdateCount++;

  const positions = refs.vaporSystem.particles.geometry.attributes.position.array as Float32Array;
  const maxParticles = Math.min(gpuPositions.length, positions.length / 3);

  // Map GPU positions (0-1 range) to scene coordinates
  // GPU sim uses: x,z = horizontal plane, y = vertical (0=bottom, 1=top)
  // Scene uses: similar but with specific offsets for sorbent/condenser positions
  const LAYOUT = {
    sorbentY: 0.3,     // Focal point height
    condenserY: -0.3,  // Condenser below
    scaleXZ: 0.8,      // Horizontal spread
  };

  for (let i = 0; i < maxParticles; i++) {
    const gp = gpuPositions[i];
    // Center around origin and scale
    positions[i * 3] = (gp[0] - 0.5) * scale * 2;
    // Map y: GPU 0-1 to scene range (condenser to above sorbent)
    positions[i * 3 + 1] = LAYOUT.condenserY + gp[1] * (LAYOUT.sorbentY - LAYOUT.condenserY + 0.8);
    positions[i * 3 + 2] = (gp[2] - 0.5) * scale * 2;
  }

  refs.vaporSystem.particles.geometry.attributes.position.needsUpdate = true;

  // VISUAL PROOF: Change particle color to BRIGHT CYAN when GPU data is active
  // This makes it obvious that real GPU simulation data is being rendered
  const material = refs.vaporSystem.particles.material as THREE.PointsMaterial;
  material.color.setHex(0x00ffff); // Bright cyan = GPU active
  material.opacity = 0.8;
  material.size = 0.03;

  // Log every 100th update to prove data flow
  if (gpuUpdateCount % 100 === 0) {
    console.log(`[GPU→3D] Update #${gpuUpdateCount}: ${maxParticles} particles, Y range: ${Math.min(...gpuPositions.map(p => p[1])).toFixed(2)} - ${Math.max(...gpuPositions.map(p => p[1])).toFixed(2)}`);
  }
}

// ============================================================================
// Cleanup
// ============================================================================

export function disposeWaterHarvester3D(refs: WaterHarvester3DRefs): void {
  refs.group.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
    if (child instanceof THREE.Points) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
    if (child instanceof THREE.Line) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
  });

  refs.condenserSurface.bumpTexture.dispose();
}
