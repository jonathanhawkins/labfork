// Supercomputer3D - Towering server rack / supercomputer for the lab
// Features: Blinking status lights, cooling fans, data flow particles, imposing presence

import * as THREE from 'three';

export interface GpuGaugeRefs {
  group: THREE.Group;
  needle: THREE.Group;
  arcFill: THREE.Mesh;
  valueText?: THREE.Sprite;
  labelText?: THREE.Sprite;
}

export interface Supercomputer3DRefs {
  group: THREE.Group;
  fans: THREE.Group[];
  statusLights: THREE.Mesh[];
  dataParticles?: THREE.Points;
  screenMesh?: THREE.Mesh;
  // Diegetic GPU stats gauges
  gauges?: {
    utilization: GpuGaugeRefs;
    temperature: GpuGaugeRefs;
    vram: GpuGaugeRefs;
    power: GpuGaugeRefs;
  };
  statsDisplay?: THREE.Group;
}

export interface Supercomputer3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
  isProcessing?: boolean;
}

// Colors
const COLORS = {
  chassis: 0x1a1a2e,
  chassisAccent: 0x16213e,
  ventGrill: 0x0f0f1a,
  ledOff: 0x333344,
  ledGreen: 0x00ff88,
  ledBlue: 0x00aaff,
  ledRed: 0xff4444,
  ledYellow: 0xffaa00,
  screen: 0x001122,
  screenGlow: 0x00ffaa,
  fan: 0x2a2a3a,
  fanBlade: 0x444455,
  // Gauge colors - bright and visible
  gaugeBackground: 0x2a3a4a,
  gaugeBorder: 0x5588aa,
  gaugeNeedle: 0xff6666,
  gaugeArcLow: 0x00ff88,
  gaugeArcMid: 0xffcc00,
  gaugeArcHigh: 0xff4444,
};

/**
 * Create a cartoon-style semicircular gauge (like a speedometer)
 * Arc is at TOP, 0% on left, 100% on right
 */
function createGauge(
  label: string,
  color: number,
  position: [number, number, number],
  scale: number = 1
): GpuGaugeRefs {
  const gaugeGroup = new THREE.Group();
  gaugeGroup.position.set(...position);
  gaugeGroup.scale.setScalar(scale);

  const gaugeRadius = 0.12;
  const arcWidth = 0.025;

  // Background arc (full semicircle at TOP)
  // RingGeometry: thetaStart=0 is +X (right), angles go counterclockwise
  // Start at 0 (right), sweep PI (180°) counterclockwise = right -> top -> left
  const bgArcGeometry = new THREE.RingGeometry(
    gaugeRadius - arcWidth,
    gaugeRadius,
    32,
    1,
    0,            // Start at right (+X)
    Math.PI       // Sweep 180° counterclockwise to left (through top)
  );
  const bgArcMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.gaugeBackground,
    side: THREE.DoubleSide,
  });
  const bgArc = new THREE.Mesh(bgArcGeometry, bgArcMaterial);
  bgArc.position.z = 0.005;
  gaugeGroup.add(bgArc);

  // Fill arc (animated based on value) - fills from left toward right
  // Start position will be adjusted in updateGauge
  const fillArcGeometry = new THREE.RingGeometry(
    gaugeRadius - arcWidth,
    gaugeRadius,
    32,
    1,
    Math.PI,      // Start at left (will be updated)
    0             // No fill initially
  );
  const fillArcMaterial = new THREE.MeshBasicMaterial({
    color: color,
    side: THREE.DoubleSide,
  });
  const fillArc = new THREE.Mesh(fillArcGeometry, fillArcMaterial);
  fillArc.position.z = 0.01;
  gaugeGroup.add(fillArc);

  // Gauge border ring (at top)
  const borderGeometry = new THREE.RingGeometry(
    gaugeRadius + 0.005,
    gaugeRadius + 0.015,
    32,
    1,
    0,            // Start at right
    Math.PI       // Sweep to left through top
  );
  const borderMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.gaugeBorder,
    side: THREE.DoubleSide,
  });
  const border = new THREE.Mesh(borderGeometry, borderMaterial);
  border.position.z = 0.015;
  gaugeGroup.add(border);

  // Needle (triangular pointer) - points upward in local space
  const needleLength = gaugeRadius - 0.02;
  const needleGeometry = new THREE.ConeGeometry(0.012, needleLength, 4);
  const needleMaterial = new THREE.MeshBasicMaterial({ color: COLORS.gaugeNeedle });
  const needle = new THREE.Mesh(needleGeometry, needleMaterial);
  needle.position.y = needleLength / 2;  // Offset so base is at origin

  // Needle pivot group (rotates around Z axis)
  // rotation.z = PI/2 points left (0%), rotation.z = -PI/2 points right (100%)
  const needlePivot = new THREE.Group();
  needlePivot.add(needle);
  needlePivot.rotation.z = Math.PI / 2;  // 0% = pointing left
  needlePivot.position.z = 0.025;
  gaugeGroup.add(needlePivot);

  // Center dot
  const centerGeometry = new THREE.CircleGeometry(0.02, 16);
  const centerMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const centerDot = new THREE.Mesh(centerGeometry, centerMaterial);
  centerDot.position.z = 0.03;  // Pushed out more
  gaugeGroup.add(centerDot);

  // Create text label using canvas texture
  const createTextSprite = (text: string, fontSize: number, color: string): THREE.Sprite => {
    const canvas = document.createElement('canvas');
    const size = 128;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, size, size);

    ctx.font = `bold ${fontSize}px Arial`;
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, size / 2, size / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(0.1, 0.1, 1);
    return sprite;
  };

  // Label text (below gauge) - pushed out
  const labelSprite = createTextSprite(label, 28, '#aaaaaa');
  labelSprite.position.set(0, -0.08, 0.04);
  labelSprite.scale.set(0.12, 0.06, 1);
  gaugeGroup.add(labelSprite);

  // Value text (in center) - pushed out
  const valueSprite = createTextSprite('0%', 36, '#ffffff');
  valueSprite.position.set(0, -0.03, 0.04);
  valueSprite.scale.set(0.08, 0.04, 1);
  valueSprite.name = 'valueText';
  gaugeGroup.add(valueSprite);

  return {
    group: gaugeGroup,
    needle: needlePivot,
    arcFill: fillArc,
    valueText: valueSprite,
    labelText: labelSprite,
  };
}

/**
 * Update a gauge to show a value (0-100)
 * Needle sweeps from left (0%) to right (100%) along top arc
 */
function updateGauge(
  gauge: GpuGaugeRefs,
  value: number,
  displayValue: string,
  colorLow: number = COLORS.gaugeArcLow,
  colorMid: number = COLORS.gaugeArcMid,
  colorHigh: number = COLORS.gaugeArcHigh
): void {
  const clampedValue = Math.max(0, Math.min(100, value));
  const normalizedValue = clampedValue / 100;

  // Update needle rotation
  // At 0%: rotation.z = PI/2 (pointing left, 9 o'clock)
  // At 100%: rotation.z = -PI/2 (pointing right, 3 o'clock)
  // Linear interpolation: PI/2 - PI * normalizedValue
  const needleAngle = Math.PI / 2 - Math.PI * normalizedValue;
  gauge.needle.rotation.z = needleAngle;

  // Update arc fill by recreating geometry
  // Arc fills from left (PI) toward right (0) through top
  // At 0%: no arc (thetaLength = 0)
  // At 100%: full arc from left to right (thetaLength = PI)
  const gaugeRadius = 0.12;
  const arcWidth = 0.025;
  const arcAngle = Math.PI * normalizedValue;  // How much to fill

  gauge.arcFill.geometry.dispose();
  gauge.arcFill.geometry = new THREE.RingGeometry(
    gaugeRadius - arcWidth,
    gaugeRadius,
    32,
    1,
    Math.PI - arcAngle,  // Start closer to right as value increases
    arcAngle             // Fill amount
  );
  // Maintain Z offset
  gauge.arcFill.position.z = 0.01;

  // Update color based on value
  const arcMaterial = gauge.arcFill.material as THREE.MeshBasicMaterial;
  if (normalizedValue < 0.5) {
    arcMaterial.color.setHex(colorLow);
  } else if (normalizedValue < 0.8) {
    arcMaterial.color.setHex(colorMid);
  } else {
    arcMaterial.color.setHex(colorHigh);
  }

  // Update value text
  if (gauge.valueText) {
    const canvas = document.createElement('canvas');
    const size = 128;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, size, size);

    ctx.font = 'bold 36px Arial';
    ctx.fillStyle = normalizedValue > 0.8 ? '#ff6666' : normalizedValue > 0.5 ? '#ffaa00' : '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(displayValue, size / 2, size / 2);

    const texture = new THREE.CanvasTexture(canvas);
    (gauge.valueText.material as THREE.SpriteMaterial).map?.dispose();
    (gauge.valueText.material as THREE.SpriteMaterial).map = texture;
    (gauge.valueText.material as THREE.SpriteMaterial).needsUpdate = true;
  }
}

/**
 * Create a towering supercomputer that dominates the scene
 */
export function createSupercomputer3D(options: Supercomputer3DOptions): Supercomputer3DRefs {
  const {
    position,
    scale = 1,
    accentColor = 0x00ffaa,
    isProcessing = true,
  } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const fans: THREE.Group[] = [];
  const statusLights: THREE.Mesh[] = [];

  // === MAIN CHASSIS (tall server tower) ===
  const chassisWidth = 1.2;
  const chassisHeight = 3.0;
  const chassisDepth = 0.8;

  // Main body
  const chassisGeometry = new THREE.BoxGeometry(chassisWidth, chassisHeight, chassisDepth);
  const chassisMaterial = new THREE.MeshToonMaterial({ color: COLORS.chassis });
  const chassis = new THREE.Mesh(chassisGeometry, chassisMaterial);
  chassis.position.y = chassisHeight / 2;
  chassis.castShadow = true;
  chassis.receiveShadow = true;
  group.add(chassis);

  // Front panel (slightly recessed)
  const frontPanelGeometry = new THREE.BoxGeometry(chassisWidth - 0.1, chassisHeight - 0.1, 0.05);
  const frontPanelMaterial = new THREE.MeshToonMaterial({ color: COLORS.chassisAccent });
  const frontPanel = new THREE.Mesh(frontPanelGeometry, frontPanelMaterial);
  frontPanel.position.set(0, chassisHeight / 2, chassisDepth / 2 - 0.02);
  group.add(frontPanel);

  // === TOP SECTION - Cooling fans ===
  const topFanY = chassisHeight - 0.4;

  // Fan housings
  for (let i = 0; i < 2; i++) {
    const fanX = (i - 0.5) * 0.45;

    // Fan housing circle
    const housingGeometry = new THREE.CylinderGeometry(0.18, 0.18, 0.08, 24);
    const housingMaterial = new THREE.MeshToonMaterial({ color: COLORS.ventGrill });
    const housing = new THREE.Mesh(housingGeometry, housingMaterial);
    housing.rotation.x = Math.PI / 2;
    housing.position.set(fanX, topFanY, chassisDepth / 2 + 0.01);
    group.add(housing);

    // Fan blades
    const fanGroup = new THREE.Group();
    const bladeGeometry = new THREE.BoxGeometry(0.28, 0.02, 0.06);
    const bladeMaterial = new THREE.MeshToonMaterial({ color: COLORS.fanBlade });

    for (let b = 0; b < 5; b++) {
      const blade = new THREE.Mesh(bladeGeometry, bladeMaterial);
      blade.rotation.y = (b / 5) * Math.PI * 2;
      fanGroup.add(blade);
    }

    // Fan center
    const hubGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.03, 12);
    const hubMaterial = new THREE.MeshToonMaterial({ color: accentColor });
    const hub = new THREE.Mesh(hubGeometry, hubMaterial);
    hub.rotation.x = Math.PI / 2;
    fanGroup.add(hub);

    fanGroup.position.set(fanX, topFanY, chassisDepth / 2 + 0.02);
    fanGroup.rotation.x = Math.PI / 2;
    fanGroup.userData = { speed: 3 + i * 0.5 };
    group.add(fanGroup);
    fans.push(fanGroup);
  }

  // === MIDDLE SECTION - Status display and lights ===
  const middleY = chassisHeight / 2 + 0.3;

  // Screen border (behind screen)
  const screenBorderGeometry = new THREE.BoxGeometry(0.65, 0.4, 0.01);
  const screenBorderMaterial = new THREE.MeshToonMaterial({ color: 0x333344 });
  const screenBorder = new THREE.Mesh(screenBorderGeometry, screenBorderMaterial);
  screenBorder.position.set(0, middleY, chassisDepth / 2 + 0.01);
  screenBorder.renderOrder = 1;
  group.add(screenBorder);

  // Screen glow layer (soft glow behind the screen)
  const screenGlowGeometry = new THREE.PlaneGeometry(0.7, 0.45);
  const screenGlowMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.screenGlow,
    transparent: true,
    opacity: 0.3,
    side: THREE.FrontSide,
  });
  const screenGlow = new THREE.Mesh(screenGlowGeometry, screenGlowMaterial);
  screenGlow.position.set(0, middleY, chassisDepth / 2 + 0.02);
  screenGlow.renderOrder = 1;
  group.add(screenGlow);

  // Status screen (in front of border, glowing teal)
  // Using PlaneGeometry to ensure front face is always visible
  const screenGeometry = new THREE.PlaneGeometry(0.6, 0.35);
  const screenMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.screenGlow,
    transparent: true,
    opacity: 0.9,
    side: THREE.FrontSide,
  });
  const screen = new THREE.Mesh(screenGeometry, screenMaterial);
  screen.position.set(0, middleY, chassisDepth / 2 + 0.03);  // Positioned in front of glow layer
  screen.renderOrder = 3;  // Render after glow
  group.add(screen);

  // Status light strips (vertical columns)
  const lightColumns = 3;
  const lightsPerColumn = 8;
  const lightStartY = 0.4;
  const lightEndY = middleY - 0.35;
  const lightSpacing = (lightEndY - lightStartY) / (lightsPerColumn - 1);

  for (let col = 0; col < lightColumns; col++) {
    const colX = (col - 1) * 0.35;

    for (let row = 0; row < lightsPerColumn; row++) {
      const ledGeometry = new THREE.BoxGeometry(0.04, 0.025, 0.015);
      const ledMaterial = new THREE.MeshBasicMaterial({
        color: COLORS.ledGreen,
        transparent: true,
        opacity: 0.3,
      });
      const led = new THREE.Mesh(ledGeometry, ledMaterial);
      led.position.set(colX, lightStartY + row * lightSpacing, chassisDepth / 2 + 0.03);
      led.userData = { column: col, row: row, type: 'status' };
      group.add(led);
      statusLights.push(led);
    }
  }

  // === BOTTOM SECTION - Power indicators and vents ===
  const bottomY = 0.25;

  // Large power indicator
  const powerGeometry = new THREE.CylinderGeometry(0.06, 0.06, 0.02, 16);
  const powerMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.ledGreen,
    transparent: true,
    opacity: 0.8,
  });
  const powerLed = new THREE.Mesh(powerGeometry, powerMaterial);
  powerLed.rotation.x = Math.PI / 2;
  powerLed.position.set(-0.4, bottomY, chassisDepth / 2 + 0.02);
  powerLed.userData = { type: 'power' };
  group.add(powerLed);
  statusLights.push(powerLed);

  // Activity indicator
  const activityGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.02, 16);
  const activityMaterial = new THREE.MeshBasicMaterial({
    color: COLORS.ledBlue,
    transparent: true,
    opacity: 0.5,
  });
  const activityLed = new THREE.Mesh(activityGeometry, activityMaterial);
  activityLed.rotation.x = Math.PI / 2;
  activityLed.position.set(-0.25, bottomY, chassisDepth / 2 + 0.02);
  activityLed.userData = { type: 'activity' };
  group.add(activityLed);
  statusLights.push(activityLed);

  // Vent grills on sides
  for (let side = -1; side <= 1; side += 2) {
    for (let v = 0; v < 6; v++) {
      const ventGeometry = new THREE.BoxGeometry(0.03, 0.15, chassisDepth - 0.1);
      const ventMaterial = new THREE.MeshToonMaterial({ color: COLORS.ventGrill });
      const vent = new THREE.Mesh(ventGeometry, ventMaterial);
      vent.position.set(
        side * (chassisWidth / 2 + 0.01),
        0.5 + v * 0.4,
        0
      );
      group.add(vent);
    }
  }

  // === ACCENT DETAILS ===

  // Top accent strip
  const topStripGeometry = new THREE.BoxGeometry(chassisWidth + 0.05, 0.05, chassisDepth + 0.05);
  const topStripMaterial = new THREE.MeshToonMaterial({ color: accentColor });
  const topStrip = new THREE.Mesh(topStripGeometry, topStripMaterial);
  topStrip.position.y = chassisHeight + 0.025;
  group.add(topStrip);

  // Bottom accent strip
  const bottomStripGeometry = new THREE.BoxGeometry(chassisWidth + 0.05, 0.03, chassisDepth + 0.05);
  const bottomStrip = new THREE.Mesh(bottomStripGeometry, topStripMaterial);
  bottomStrip.position.y = 0.015;
  group.add(bottomStrip);

  // Side accent lines
  for (let side = -1; side <= 1; side += 2) {
    const sideLineGeometry = new THREE.BoxGeometry(0.02, chassisHeight, 0.02);
    const sideLine = new THREE.Mesh(sideLineGeometry, topStripMaterial);
    sideLine.position.set(side * chassisWidth / 2, chassisHeight / 2, chassisDepth / 2);
    group.add(sideLine);
  }

  // === DATA FLOW PARTICLES ===
  const particleCount = 80;
  const particleGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  const colors = new Float32Array(particleCount * 3);
  const sizes = new Float32Array(particleCount);

  const particleColor1 = new THREE.Color(accentColor);
  const particleColor2 = new THREE.Color(COLORS.ledBlue);

  for (let i = 0; i < particleCount; i++) {
    // Particles flow upward around the supercomputer
    const angle = Math.random() * Math.PI * 2;
    const radius = 0.8 + Math.random() * 0.4;
    positions[i * 3] = Math.cos(angle) * radius;
    positions[i * 3 + 1] = Math.random() * chassisHeight;
    positions[i * 3 + 2] = Math.sin(angle) * radius;

    const mixFactor = Math.random();
    const color = particleColor1.clone().lerp(particleColor2, mixFactor);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;

    sizes[i] = 0.02 + Math.random() * 0.03;
  }

  particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  particleGeometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const particleMaterial = new THREE.PointsMaterial({
    size: 0.04,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    blending: THREE.AdditiveBlending,
  });

  const dataParticles = new THREE.Points(particleGeometry, particleMaterial);
  dataParticles.visible = isProcessing;
  group.add(dataParticles);

  // === DIEGETIC GPU STATS PANEL (floating beside the supercomputer, billboard-style) ===
  const statsPanel = new THREE.Group();
  // Position floating to the right of the supercomputer
  statsPanel.position.set(chassisWidth / 2 + 1.0, chassisHeight / 2 + 0.5, 0);
  // Mark as billboard so it always faces camera
  statsPanel.userData.isBillboard = true;

  // Panel backing - larger and brighter
  const panelGeometry = new THREE.BoxGeometry(0.9, 1.0, 0.05);
  const panelMaterial = new THREE.MeshBasicMaterial({
    color: 0x1a2a3a,
    transparent: true,
    opacity: 0.95,
  });
  const panel = new THREE.Mesh(panelGeometry, panelMaterial);
  statsPanel.add(panel);

  // Panel border glow - brighter
  const panelBorderGeometry = new THREE.BoxGeometry(0.95, 1.05, 0.04);
  const panelBorderMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.6,
  });
  const panelBorder = new THREE.Mesh(panelBorderGeometry, panelBorderMaterial);
  panelBorder.position.z = -0.01;
  statsPanel.add(panelBorder);

  // Title text - pushed out from panel
  const titleCanvas = document.createElement('canvas');
  titleCanvas.width = 256;
  titleCanvas.height = 64;
  const titleCtx = titleCanvas.getContext('2d')!;
  titleCtx.fillStyle = '#00ffaa';
  titleCtx.font = 'bold 32px Arial';
  titleCtx.textAlign = 'center';
  titleCtx.fillText('RTX 4090', 128, 40);
  const titleTexture = new THREE.CanvasTexture(titleCanvas);
  const titleSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: titleTexture, transparent: true }));
  titleSprite.position.set(0, 0.4, 0.08);  // Pushed out more
  titleSprite.scale.set(0.5, 0.12, 1);
  statsPanel.add(titleSprite);

  // Create gauges in a 2x2 grid - larger and pushed out
  const gaugeScale = 1.2;
  const gaugeSpacingX = 0.35;
  const gaugeSpacingY = 0.35;
  const gaugeZOffset = 0.08;  // Push gauges out from panel

  // GPU Utilization gauge (top-left)
  const utilizationGauge = createGauge('GPU', COLORS.ledGreen, [-gaugeSpacingX / 2, gaugeSpacingY / 2 - 0.05, gaugeZOffset], gaugeScale);
  statsPanel.add(utilizationGauge.group);

  // Temperature gauge (top-right)
  const tempGauge = createGauge('TEMP', COLORS.ledYellow, [gaugeSpacingX / 2, gaugeSpacingY / 2 - 0.05, gaugeZOffset], gaugeScale);
  statsPanel.add(tempGauge.group);

  // VRAM gauge (bottom-left)
  const vramGauge = createGauge('VRAM', COLORS.ledBlue, [-gaugeSpacingX / 2, -gaugeSpacingY / 2 - 0.1, gaugeZOffset], gaugeScale);
  statsPanel.add(vramGauge.group);

  // Power gauge (bottom-right)
  const powerGauge = createGauge('PWR', 0xff6600, [gaugeSpacingX / 2, -gaugeSpacingY / 2 - 0.1, gaugeZOffset], gaugeScale);
  statsPanel.add(powerGauge.group);

  group.add(statsPanel);

  // Store state
  group.userData = { isProcessing, accentColor, chassisHeight };

  return {
    group,
    fans,
    statusLights,
    dataParticles,
    screenMesh: screen,
    gauges: {
      utilization: utilizationGauge,
      temperature: tempGauge,
      vram: vramGauge,
      power: powerGauge,
    },
    statsDisplay: statsPanel,
  };
}

export interface TrainingData {
  isTraining: boolean;
  epoch?: number;
  loss?: number;
  script?: string;
}

export interface GpuStatsData {
  connected: boolean;
  utilization: number;  // 0-100
  temperature: number;  // degrees C
  memoryUsed: number;   // MiB
  memoryTotal: number;  // MiB
  powerDraw: number;    // Watts
  powerLimit: number;   // Watts
  training?: TrainingData;
}

// Floating text particle for loss values
interface FloatingText {
  sprite: THREE.Sprite;
  velocity: THREE.Vector3;
  life: number;
  maxLife: number;
}

// Smoke puff particle for epoch changes
interface SmokePuff {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  life: number;
  maxLife: number;
  scale: number;
}

// Store for active particles (managed outside refs for simplicity)
const floatingTexts: FloatingText[] = [];
const smokePuffs: SmokePuff[] = [];
let lastEpoch: number | null = null;
let lastLossSpawnTime = 0;

/**
 * Create a floating text sprite for loss values
 */
function createFloatingLossText(
  parent: THREE.Group,
  loss: number,
  chassisHeight: number
): FloatingText {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext('2d')!;

  // Clear
  ctx.fillStyle = 'transparent';
  ctx.fillRect(0, 0, 128, 64);

  // Draw loss value with glow effect
  const lossText = loss.toFixed(4);
  ctx.font = 'bold 24px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  // Glow
  ctx.shadowColor = '#00ffaa';
  ctx.shadowBlur = 10;
  ctx.fillStyle = '#00ffaa';
  ctx.fillText(lossText, 64, 32);

  const texture = new THREE.CanvasTexture(canvas);
  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 1,
  });
  const sprite = new THREE.Sprite(spriteMaterial);

  // Start position at top of supercomputer with random X offset
  const offsetX = (Math.random() - 0.5) * 0.4;
  sprite.position.set(offsetX, chassisHeight + 0.3, 0.5);
  sprite.scale.set(0.4, 0.2, 1);

  parent.add(sprite);

  return {
    sprite,
    velocity: new THREE.Vector3(
      (Math.random() - 0.5) * 0.01,
      0.015 + Math.random() * 0.01,
      (Math.random() - 0.5) * 0.005
    ),
    life: 1,
    maxLife: 1,
  };
}

/**
 * Create smoke puff particles for epoch completion
 */
function createEpochSmokePuffs(
  parent: THREE.Group,
  chassisHeight: number,
  count: number = 8
): void {
  for (let i = 0; i < count; i++) {
    // Create a simple sphere for smoke puff
    const geometry = new THREE.SphereGeometry(0.08, 8, 8);
    const material = new THREE.MeshBasicMaterial({
      color: 0xaaffcc,
      transparent: true,
      opacity: 0.7,
    });
    const mesh = new THREE.Mesh(geometry, material);

    // Position at top with spread
    const angle = (i / count) * Math.PI * 2;
    const radius = 0.3 + Math.random() * 0.2;
    mesh.position.set(
      Math.cos(angle) * radius,
      chassisHeight + 0.2,
      Math.sin(angle) * radius
    );

    parent.add(mesh);

    smokePuffs.push({
      mesh,
      velocity: new THREE.Vector3(
        Math.cos(angle) * 0.02,
        0.03 + Math.random() * 0.02,
        Math.sin(angle) * 0.02
      ),
      life: 1,
      maxLife: 1,
      scale: 0.08 + Math.random() * 0.05,
    });
  }
}

/**
 * Update floating text particles
 */
function updateFloatingTexts(parent: THREE.Group, deltaTime: number): void {
  for (let i = floatingTexts.length - 1; i >= 0; i--) {
    const ft = floatingTexts[i];

    // Update position
    ft.sprite.position.add(ft.velocity);

    // Update life and opacity
    ft.life -= deltaTime * 0.5;
    const opacity = Math.max(0, ft.life / ft.maxLife);
    (ft.sprite.material as THREE.SpriteMaterial).opacity = opacity;

    // Scale up slightly as it rises
    const scale = 0.4 + (1 - ft.life) * 0.2;
    ft.sprite.scale.set(scale, scale * 0.5, 1);

    // Remove if dead
    if (ft.life <= 0) {
      parent.remove(ft.sprite);
      ft.sprite.material.dispose();
      (ft.sprite.material as THREE.SpriteMaterial).map?.dispose();
      floatingTexts.splice(i, 1);
    }
  }
}

/**
 * Update smoke puff particles
 */
function updateSmokePuffs(parent: THREE.Group, deltaTime: number): void {
  for (let i = smokePuffs.length - 1; i >= 0; i--) {
    const puff = smokePuffs[i];

    // Update position
    puff.mesh.position.add(puff.velocity);

    // Slow down horizontal velocity
    puff.velocity.x *= 0.98;
    puff.velocity.z *= 0.98;

    // Update life and opacity
    puff.life -= deltaTime * 0.8;
    const opacity = Math.max(0, puff.life / puff.maxLife) * 0.7;
    (puff.mesh.material as THREE.MeshBasicMaterial).opacity = opacity;

    // Expand as it rises
    const expandScale = puff.scale * (1 + (1 - puff.life) * 2);
    puff.mesh.scale.setScalar(expandScale);

    // Remove if dead
    if (puff.life <= 0) {
      parent.remove(puff.mesh);
      puff.mesh.geometry.dispose();
      (puff.mesh.material as THREE.Material).dispose();
      smokePuffs.splice(i, 1);
    }
  }
}

/**
 * Animate the supercomputer (fans, lights, particles, GPU stats, training effects)
 */
export function animateSupercomputer3D(
  refs: Supercomputer3DRefs,
  time: number,
  options?: {
    isProcessing?: boolean;
    loadLevel?: number; // 0-1
    progress?: number;  // 0-100
    gpuStats?: GpuStatsData;
    camera?: THREE.Camera;  // Pass camera for billboard effect
  }
): void {
  const gpuStats = options?.gpuStats;
  const training = gpuStats?.training;
  // Explicit isProcessing option takes precedence, then gpuStats.connected, then defaults
  const isProcessing = options?.isProcessing !== undefined
    ? options.isProcessing
    : (gpuStats?.connected ?? refs.group.userData.isProcessing ?? true);
  const loadLevel = gpuStats ? gpuStats.utilization / 100 : (options?.loadLevel ?? 0.5);
  const progress = options?.progress ?? 50;
  const chassisHeight = refs.group.userData.chassisHeight ?? 3.0;
  const deltaTime = 0.016; // Approximate 60fps

  // === TRAINING VISUAL EFFECTS ===
  if (training?.isTraining && training.loss !== undefined) {
    // Spawn floating loss numbers periodically (every ~2 seconds)
    if (time - lastLossSpawnTime > 2) {
      createFloatingLossText(refs.group, training.loss, chassisHeight);
      lastLossSpawnTime = time;
    }

    // Check for epoch change - spawn smoke puffs
    if (training.epoch !== undefined && training.epoch !== lastEpoch) {
      if (lastEpoch !== null) {
        // Epoch changed! Spawn celebration smoke puffs
        createEpochSmokePuffs(refs.group, chassisHeight, 12);
      }
      lastEpoch = training.epoch;
    }
  }

  // Update floating text particles
  updateFloatingTexts(refs.group, deltaTime);

  // Update smoke puff particles
  updateSmokePuffs(refs.group, deltaTime);

  // Billboard effect for stats panel - always face camera
  if (refs.statsDisplay && options?.camera) {
    // Get world position of the stats panel
    const panelWorldPos = new THREE.Vector3();
    refs.statsDisplay.getWorldPosition(panelWorldPos);

    // Get camera world position
    const cameraWorldPos = new THREE.Vector3();
    options.camera.getWorldPosition(cameraWorldPos);

    // Calculate direction to camera (only Y rotation for billboard)
    const direction = new THREE.Vector3();
    direction.subVectors(cameraWorldPos, panelWorldPos);
    direction.y = 0; // Keep panel upright
    direction.normalize();

    // Calculate the angle to face camera
    const angle = Math.atan2(direction.x, direction.z);
    refs.statsDisplay.rotation.y = angle;

    // Add gentle float animation
    const baseY = refs.group.userData.chassisHeight / 2 + 0.5;
    refs.statsDisplay.position.y = baseY + Math.sin(time * 1.5) * 0.05;
  }

  // Update GPU stats gauges if available
  if (refs.gauges && gpuStats?.connected) {
    // GPU Utilization gauge
    updateGauge(
      refs.gauges.utilization,
      gpuStats.utilization,
      `${gpuStats.utilization}%`,
      COLORS.ledGreen,
      COLORS.ledYellow,
      COLORS.ledRed
    );

    // Temperature gauge (0-90°C range)
    const tempPercent = Math.min(100, (gpuStats.temperature / 90) * 100);
    updateGauge(
      refs.gauges.temperature,
      tempPercent,
      `${gpuStats.temperature}°C`,
      COLORS.ledGreen,
      COLORS.ledYellow,
      COLORS.ledRed
    );

    // VRAM gauge
    const vramPercent = gpuStats.memoryTotal > 0
      ? (gpuStats.memoryUsed / gpuStats.memoryTotal) * 100
      : 0;
    const vramGB = (gpuStats.memoryUsed / 1024).toFixed(1);
    updateGauge(
      refs.gauges.vram,
      vramPercent,
      `${vramGB}G`,
      COLORS.ledBlue,
      COLORS.ledYellow,
      COLORS.ledRed
    );

    // Power gauge
    const powerPercent = gpuStats.powerLimit > 0
      ? (gpuStats.powerDraw / gpuStats.powerLimit) * 100
      : 0;
    updateGauge(
      refs.gauges.power,
      powerPercent,
      `${gpuStats.powerDraw}W`,
      COLORS.ledGreen,
      COLORS.ledYellow,
      COLORS.ledRed
    );
  } else if (refs.gauges && !gpuStats?.connected) {
    // Show "offline" state - all gauges at 0 with dimmed appearance
    updateGauge(refs.gauges.utilization, 0, '--', 0x444444, 0x444444, 0x444444);
    updateGauge(refs.gauges.temperature, 0, '--', 0x444444, 0x444444, 0x444444);
    updateGauge(refs.gauges.vram, 0, '--', 0x444444, 0x444444, 0x444444);
    updateGauge(refs.gauges.power, 0, '--', 0x444444, 0x444444, 0x444444);
  }

  // Animate fans - speed based on load
  const fanSpeed = 0.5 + loadLevel * 3;
  refs.fans.forEach((fan, idx) => {
    const baseSpeed = fan.userData?.speed || 3;
    fan.rotation.y += baseSpeed * fanSpeed * 0.05;
  });

  // Animate status lights
  refs.statusLights.forEach((led, idx) => {
    const material = led.material as THREE.MeshBasicMaterial;
    const userData = led.userData;

    if (userData.type === 'power') {
      // Power LED - steady glow with slight pulse
      material.opacity = 0.7 + Math.sin(time * 2) * 0.2;
      material.color.setHex(isProcessing ? COLORS.ledGreen : COLORS.ledYellow);
    } else if (userData.type === 'activity') {
      // Activity LED - rapid blink when processing
      if (isProcessing) {
        material.opacity = Math.sin(time * 15) > 0 ? 0.9 : 0.2;
        material.color.setHex(COLORS.ledBlue);
      } else {
        material.opacity = 0.2;
      }
    } else if (userData.type === 'status') {
      // Status LEDs - show progress/activity
      const col = userData.column;
      const row = userData.row;
      const ledIndex = col * 8 + row;
      const totalLeds = 24;
      const litThreshold = (progress / 100) * totalLeds;

      if (ledIndex < litThreshold) {
        // Lit LED with wave effect
        const wave = Math.sin(time * 4 + ledIndex * 0.3) * 0.3 + 0.7;
        material.opacity = wave;

        // Color varies by column
        if (col === 0) material.color.setHex(COLORS.ledGreen);
        else if (col === 1) material.color.setHex(COLORS.ledBlue);
        else material.color.setHex(refs.group.userData.accentColor || COLORS.ledGreen);
      } else {
        material.opacity = 0.1;
        material.color.setHex(COLORS.ledOff);
      }
    }
  });

  // Animate screen - ensure it always glows teal/cyan
  if (refs.screenMesh) {
    const screenMat = refs.screenMesh.material as THREE.MeshBasicMaterial;
    if (isProcessing) {
      // Pulsing glow with high visibility
      const pulse = 0.8 + Math.sin(time * 3) * 0.15;
      screenMat.opacity = pulse;
      // Slight color shift (cyan/teal range) with higher lightness for visibility
      const hue = 0.45 + Math.sin(time * 0.5) * 0.05;
      screenMat.color.setHSL(hue, 1.0, 0.55);  // Full saturation, higher lightness
    } else {
      // Idle state - bright teal glow (visible even when not processing)
      screenMat.opacity = 0.7;
      screenMat.color.setHex(COLORS.screenGlow);  // Use the defined teal color
    }
  }

  // Animate data particles
  if (refs.dataParticles) {
    refs.dataParticles.visible = isProcessing;

    if (isProcessing) {
      const positions = refs.dataParticles.geometry.attributes.position.array as Float32Array;

      for (let i = 0; i < positions.length / 3; i++) {
        // Spiral upward
        const x = positions[i * 3];
        const z = positions[i * 3 + 2];
        const angle = Math.atan2(z, x) + 0.02 * loadLevel;
        const radius = Math.sqrt(x * x + z * z);

        positions[i * 3] = Math.cos(angle) * radius;
        positions[i * 3 + 1] += 0.02 + loadLevel * 0.03;
        positions[i * 3 + 2] = Math.sin(angle) * radius;

        // Reset when too high
        if (positions[i * 3 + 1] > chassisHeight + 0.5) {
          const newAngle = Math.random() * Math.PI * 2;
          const newRadius = 0.8 + Math.random() * 0.4;
          positions[i * 3] = Math.cos(newAngle) * newRadius;
          positions[i * 3 + 1] = 0;
          positions[i * 3 + 2] = Math.sin(newAngle) * newRadius;
        }
      }

      refs.dataParticles.geometry.attributes.position.needsUpdate = true;

      // Opacity based on load
      const material = refs.dataParticles.material as THREE.PointsMaterial;
      material.opacity = 0.4 + loadLevel * 0.4;
    }
  }
}

/**
 * Dispose supercomputer resources
 */
export function disposeSupercomputer3D(refs: Supercomputer3DRefs): void {
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

  if (refs.dataParticles) {
    refs.dataParticles.geometry.dispose();
    (refs.dataParticles.material as THREE.Material).dispose();
  }
}
