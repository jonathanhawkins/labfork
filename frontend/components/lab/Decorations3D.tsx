"use client";

import * as THREE from "three";
import { LAB_COLORS, getRainbowColor } from "./colors";

/**
 * Create a cute potted plant decoration
 */
export function createPlant3D(position: THREE.Vector3, scale: number = 1): THREE.Group {
  const group = new THREE.Group();

  // Pot
  const potGeometry = new THREE.CylinderGeometry(0.15 * scale, 0.12 * scale, 0.2 * scale, 12);
  const potMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.pot });
  const pot = new THREE.Mesh(potGeometry, potMaterial);
  pot.position.y = 0.1 * scale;
  group.add(pot);

  // Plant (stacked spheres for a cute bush look)
  const plantMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.plant });
  for (let i = 0; i < 3; i++) {
    const leafGeometry = new THREE.SphereGeometry((0.12 - i * 0.02) * scale, 8, 8);
    const leaf = new THREE.Mesh(leafGeometry, plantMaterial);
    leaf.position.y = (0.25 + i * 0.12) * scale;
    leaf.position.x = (Math.random() - 0.5) * 0.1 * scale;
    leaf.position.z = (Math.random() - 0.5) * 0.1 * scale;
    group.add(leaf);
  }

  group.position.copy(position);
  return group;
}

/**
 * Create a floating decorative cube (rainbow colored)
 */
export function createFloatingCube3D(
  index: number,
  total: number,
  orbitRadius: number = 6,
  heightVariation: number = 0.5
): THREE.Mesh {
  const cubeGeometry = new THREE.BoxGeometry(0.2, 0.2, 0.2);
  const color = getRainbowColor(index, total);
  const cubeMaterial = new THREE.MeshToonMaterial({
    color,
    transparent: true,
    opacity: 0.7,
  });

  const cube = new THREE.Mesh(cubeGeometry, cubeMaterial);
  const angle = (index / total) * Math.PI * 2;
  cube.position.set(
    Math.cos(angle) * orbitRadius,
    2 + Math.sin(index * 1.5) * heightVariation,
    Math.sin(angle) * orbitRadius
  );
  cube.userData = { floatPhase: index * 0.5, orbitAngle: angle };
  cube.name = `floatingCube${index}`;

  return cube;
}

/**
 * Animate floating cubes with rotation and bobbing
 */
export function animateFloatingCube(cube: THREE.Mesh, time: number) {
  const phase = cube.userData.floatPhase || 0;
  cube.rotation.x = time * 0.5 + phase;
  cube.rotation.y = time * 0.7 + phase;
  cube.position.y = 2 + Math.sin(time + phase) * 0.3;
}

/**
 * Create ground pattern (circles on the grass)
 */
export function createGroundPattern3D(gridSize: number = 8, spacing: number = 4): THREE.Group {
  const group = new THREE.Group();

  for (let x = -gridSize; x <= gridSize; x += spacing) {
    for (let z = -gridSize; z <= gridSize; z += spacing) {
      const circleGeometry = new THREE.CircleGeometry(0.5, 16);
      const circleMaterial = new THREE.MeshToonMaterial({
        color: LAB_COLORS.groundAccent,
      });
      const circle = new THREE.Mesh(circleGeometry, circleMaterial);
      circle.rotation.x = -Math.PI / 2;
      circle.position.set(x, 0.01, z);
      group.add(circle);
    }
  }

  return group;
}

/**
 * Create all decorations for the lab scene
 */
export function createAllDecorations(
  plantPositions: THREE.Vector3[] = [
    new THREE.Vector3(-5, 0, 0),
    new THREE.Vector3(5, 0, 0),
    new THREE.Vector3(0, 0, -5),
    new THREE.Vector3(0, 0, 5),
  ],
  cubeCount: number = 8
): THREE.Group {
  const group = new THREE.Group();

  // Add plants
  plantPositions.forEach((pos) => {
    group.add(createPlant3D(pos));
  });

  // Add floating cubes
  for (let i = 0; i < cubeCount; i++) {
    group.add(createFloatingCube3D(i, cubeCount));
  }

  // Add ground pattern
  group.add(createGroundPattern3D());

  return group;
}
