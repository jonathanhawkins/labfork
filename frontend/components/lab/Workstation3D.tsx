"use client";

import * as THREE from "three";
import { Workstation } from "./types";
import { LAB_COLORS } from "./colors";

export interface WorkstationRefs {
  group: THREE.Group;
  screen: THREE.Mesh;
}

/**
 * Create a cute desk workstation with computer
 */
export function createWorkstation3D(config: Workstation): WorkstationRefs {
  const group = new THREE.Group();

  // Desk top
  const deskGeometry = new THREE.BoxGeometry(1.5, 0.1, 0.8);
  const deskMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.desk });
  const desk = new THREE.Mesh(deskGeometry, deskMaterial);
  desk.position.y = 0.7;
  desk.castShadow = true;
  desk.receiveShadow = true;
  group.add(desk);

  // Desk legs
  const legGeometry = new THREE.BoxGeometry(0.1, 0.7, 0.1);
  const legMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.deskLeg });

  const legPositions: [number, number, number][] = [
    [-0.6, 0.35, 0.3],
    [0.6, 0.35, 0.3],
    [-0.6, 0.35, -0.3],
    [0.6, 0.35, -0.3],
  ];

  legPositions.forEach(([x, y, z]) => {
    const leg = new THREE.Mesh(legGeometry, legMaterial);
    leg.position.set(x, y, z);
    leg.castShadow = true;
    group.add(leg);
  });

  // Monitor frame
  const monitorGeometry = new THREE.BoxGeometry(0.8, 0.5, 0.05);
  const monitorMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.monitor });
  const monitor = new THREE.Mesh(monitorGeometry, monitorMaterial);
  monitor.position.set(0, 1.1, -0.2);
  monitor.castShadow = true;
  group.add(monitor);

  // Monitor screen (glowing)
  const screenGeometry = new THREE.BoxGeometry(0.7, 0.4, 0.01);
  const screenMaterial = new THREE.MeshBasicMaterial({
    color: config.screenColor || LAB_COLORS.screenGlow,
    transparent: true,
    opacity: 0.9,
  });
  const screen = new THREE.Mesh(screenGeometry, screenMaterial);
  screen.position.set(0, 1.1, -0.17);
  screen.name = "screen";
  group.add(screen);

  // Monitor stand
  const standGeometry = new THREE.BoxGeometry(0.1, 0.25, 0.1);
  const stand = new THREE.Mesh(standGeometry, monitorMaterial);
  stand.position.set(0, 0.87, -0.2);
  group.add(stand);

  // Keyboard
  const keyboardGeometry = new THREE.BoxGeometry(0.5, 0.03, 0.2);
  const keyboardMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.keyboard });
  const keyboard = new THREE.Mesh(keyboardGeometry, keyboardMaterial);
  keyboard.position.set(0, 0.77, 0.15);
  group.add(keyboard);

  // Position and rotate
  group.position.set(...config.position);
  group.rotation.y = config.rotation;

  return { group, screen };
}

/**
 * Animate workstation screen with flickering glow effect
 */
export function animateWorkstationScreen(screen: THREE.Mesh, time: number, index: number = 0) {
  const material = screen.material as THREE.MeshBasicMaterial;
  const brightness = 0.7 + Math.sin(time * 3 + index) * 0.3;
  material.color.setHSL(0.4 + Math.sin(time * 0.5 + index) * 0.1, 0.8, brightness * 0.5);
}
