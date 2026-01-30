/**
 * TextCorpus3D - Text display with embeddings for NLP domain
 * Features: Floating text panels, embedding space visualization
 */

import * as THREE from "three";

export interface TextCorpus3DRefs {
  group: THREE.Group;
  panels: THREE.Mesh[];
  embeddings: THREE.Points;
}

export interface TextCorpus3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

export function createTextCorpus3D(options: TextCorpus3DOptions): TextCorpus3DRefs {
  const { position, scale = 1, accentColor = 0x6366f1 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const panels: THREE.Mesh[] = [];

  // Base platform
  const platformGeometry = new THREE.BoxGeometry(1.5, 0.05, 1);
  const platformMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const platform = new THREE.Mesh(platformGeometry, platformMaterial);
  platform.position.y = 0.025;
  platform.castShadow = true;
  group.add(platform);

  // Text panels (floating documents)
  const panelPositions = [
    { x: -0.5, y: 0.6, z: 0.2, rotY: 0.2 },
    { x: 0, y: 0.8, z: 0, rotY: 0 },
    { x: 0.5, y: 0.5, z: 0.3, rotY: -0.2 },
    { x: -0.3, y: 1.1, z: -0.2, rotY: 0.1 },
    { x: 0.4, y: 1.0, z: -0.1, rotY: -0.15 },
  ];

  panelPositions.forEach((pos, idx) => {
    // Panel (document)
    const panelGeometry = new THREE.BoxGeometry(0.4, 0.5, 0.02);
    const panelMaterial = new THREE.MeshToonMaterial({
      color: 0x2d2d2d,
    });
    const panel = new THREE.Mesh(panelGeometry, panelMaterial);
    panel.position.set(pos.x, pos.y, pos.z);
    panel.rotation.y = pos.rotY;
    panel.userData = { panelIndex: idx, baseY: pos.y };
    group.add(panel);
    panels.push(panel);

    // Text lines on panel
    const lineCount = 6;
    for (let i = 0; i < lineCount; i++) {
      const lineWidth = 0.25 + Math.random() * 0.1;
      const lineGeometry = new THREE.PlaneGeometry(lineWidth, 0.02);
      const lineMaterial = new THREE.MeshBasicMaterial({
        color: accentColor,
        transparent: true,
        opacity: 0.4 + Math.random() * 0.3,
      });
      const line = new THREE.Mesh(lineGeometry, lineMaterial);
      line.position.set(
        pos.x + (lineWidth - 0.35) / 2,
        pos.y + 0.18 - i * 0.06,
        pos.z + 0.015
      );
      line.rotation.y = pos.rotY;
      group.add(line);
    }

    // Highlight bracket
    const bracketGeometry = new THREE.PlaneGeometry(0.02, 0.12);
    const bracketMaterial = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.6,
    });
    const bracket = new THREE.Mesh(bracketGeometry, bracketMaterial);
    bracket.position.set(pos.x - 0.18, pos.y - 0.05, pos.z + 0.015);
    bracket.rotation.y = pos.rotY;
    group.add(bracket);
  });

  // Embedding space visualization (point cloud)
  const pointCount = 100;
  const embeddingGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointCount * 3);
  const colors = new Float32Array(pointCount * 3);

  const color1 = new THREE.Color(accentColor);
  const color2 = new THREE.Color(0xec4899);
  const color3 = new THREE.Color(0x22c55e);
  const clusterColors = [color1, color2, color3];

  for (let i = 0; i < pointCount; i++) {
    // Create 3 clusters
    const cluster = Math.floor(i / (pointCount / 3));
    const clusterCenter = [
      new THREE.Vector3(-0.3, 0.4, -0.4),
      new THREE.Vector3(0.3, 0.6, -0.5),
      new THREE.Vector3(0, 0.3, -0.3),
    ][cluster] || new THREE.Vector3(0, 0.5, -0.4);

    positions[i * 3] = clusterCenter.x + (Math.random() - 0.5) * 0.3;
    positions[i * 3 + 1] = clusterCenter.y + (Math.random() - 0.5) * 0.3;
    positions[i * 3 + 2] = clusterCenter.z + (Math.random() - 0.5) * 0.2;

    const color = clusterColors[cluster] || color1;
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  embeddingGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  embeddingGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const embeddingMaterial = new THREE.PointsMaterial({
    size: 0.03,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
  });
  const embeddings = new THREE.Points(embeddingGeometry, embeddingMaterial);
  group.add(embeddings);

  // Connection lines between clusters
  const lineMaterial = new THREE.LineBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.2,
  });

  const connections = [
    [new THREE.Vector3(-0.3, 0.4, -0.4), new THREE.Vector3(0.3, 0.6, -0.5)],
    [new THREE.Vector3(0.3, 0.6, -0.5), new THREE.Vector3(0, 0.3, -0.3)],
    [new THREE.Vector3(0, 0.3, -0.3), new THREE.Vector3(-0.3, 0.4, -0.4)],
  ];

  connections.forEach((conn) => {
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(conn);
    const line = new THREE.Line(lineGeometry, lineMaterial);
    group.add(line);
  });

  return { group, panels, embeddings };
}

export function animateTextCorpus3D(
  refs: TextCorpus3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Float panels gently
  refs.panels.forEach((panel, idx) => {
    const baseY = panel.userData.baseY || 0.7;
    panel.position.y = baseY + Math.sin(time * 0.8 + idx * 0.5) * 0.05 * activity;
    panel.rotation.z = Math.sin(time * 0.5 + idx) * 0.02 * activity;
  });

  // Rotate embedding space
  refs.embeddings.rotation.y = time * 0.2 * activity;

  // Pulse embedding opacity
  const material = refs.embeddings.material as THREE.PointsMaterial;
  material.opacity = 0.5 + Math.sin(time * 1.5) * 0.2;
}

export function disposeTextCorpus3D(refs: TextCorpus3DRefs): void {
  refs.group.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Line || child instanceof THREE.Points) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}
