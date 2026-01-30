/**
 * GameController3D - Game controller and screen for Game AI domain
 * Features: Glowing buttons, analog sticks, game screen
 */

import * as THREE from "three";

export interface GameController3DRefs {
  group: THREE.Group;
  buttons: THREE.Mesh[];
  sticks: THREE.Mesh[];
  screen: THREE.Mesh;
}

export interface GameController3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

export function createGameController3D(options: GameController3DOptions): GameController3DRefs {
  const { position, scale = 1, accentColor = 0xec4899 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const buttons: THREE.Mesh[] = [];
  const sticks: THREE.Mesh[] = [];

  // Game screen/TV
  const tvFrameGeometry = new THREE.BoxGeometry(2, 1.2, 0.1);
  const tvFrameMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const tvFrame = new THREE.Mesh(tvFrameGeometry, tvFrameMaterial);
  tvFrame.position.set(0, 1.5, -1);
  tvFrame.castShadow = true;
  group.add(tvFrame);

  // Screen
  const screenGeometry = new THREE.PlaneGeometry(1.8, 1.0);
  const screenMaterial = new THREE.MeshBasicMaterial({
    color: 0x1a0a20,
    transparent: true,
    opacity: 0.95,
  });
  const screen = new THREE.Mesh(screenGeometry, screenMaterial);
  screen.position.set(0, 1.5, -0.94);
  group.add(screen);

  // Game character on screen (simple block)
  const characterGeometry = new THREE.BoxGeometry(0.15, 0.2, 0.02);
  const characterMaterial = new THREE.MeshBasicMaterial({ color: accentColor });
  const character = new THREE.Mesh(characterGeometry, characterMaterial);
  character.position.set(-0.3, 1.4, -0.93);
  group.add(character);

  // Controller body
  const controllerGeometry = new THREE.BoxGeometry(0.8, 0.12, 0.4);
  const controllerMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const controller = new THREE.Mesh(controllerGeometry, controllerMaterial);
  controller.position.set(0, 0.5, 0.5);
  controller.rotation.x = -0.3;
  group.add(controller);

  // Controller grips
  const gripGeometry = new THREE.CylinderGeometry(0.08, 0.1, 0.25, 8);
  const gripMaterial = new THREE.MeshToonMaterial({ color: 0x222222 });

  const leftGrip = new THREE.Mesh(gripGeometry, gripMaterial);
  leftGrip.position.set(-0.35, 0.38, 0.6);
  leftGrip.rotation.x = 0.5;
  group.add(leftGrip);

  const rightGrip = new THREE.Mesh(gripGeometry, gripMaterial);
  rightGrip.position.set(0.35, 0.38, 0.6);
  rightGrip.rotation.x = 0.5;
  group.add(rightGrip);

  // Analog sticks
  const stickGeometry = new THREE.CylinderGeometry(0.04, 0.04, 0.05, 12);
  const stickMaterial = new THREE.MeshToonMaterial({ color: 0x444444 });

  const leftStick = new THREE.Mesh(stickGeometry, stickMaterial);
  leftStick.position.set(-0.2, 0.57, 0.4);
  leftStick.userData = { stickIndex: 0 };
  group.add(leftStick);
  sticks.push(leftStick);

  const rightStick = new THREE.Mesh(stickGeometry, stickMaterial);
  rightStick.position.set(0.15, 0.57, 0.55);
  rightStick.userData = { stickIndex: 1 };
  group.add(rightStick);
  sticks.push(rightStick);

  // Face buttons (ABXY)
  const buttonColors = [0x22c55e, 0xef4444, 0x3b82f6, 0xf59e0b]; // Green, Red, Blue, Yellow
  const buttonOffsets = [
    { x: 0.05, y: 0 },
    { x: 0, y: -0.05 },
    { x: -0.05, y: 0 },
    { x: 0, y: 0.05 },
  ];

  buttonOffsets.forEach((offset, idx) => {
    const buttonGeometry = new THREE.CylinderGeometry(0.025, 0.025, 0.015, 12);
    const buttonMaterial = new THREE.MeshBasicMaterial({
      color: buttonColors[idx],
      transparent: true,
      opacity: 0.8,
    });
    const button = new THREE.Mesh(buttonGeometry, buttonMaterial);
    button.position.set(0.25 + offset.x, 0.58, 0.42 + offset.y);
    button.userData = { buttonIndex: idx };
    group.add(button);
    buttons.push(button);
  });

  // D-pad
  const dpadGeometry = new THREE.BoxGeometry(0.08, 0.015, 0.025);
  const dpadMaterial = new THREE.MeshToonMaterial({ color: 0x444444 });

  const dpadH = new THREE.Mesh(dpadGeometry, dpadMaterial);
  dpadH.position.set(-0.25, 0.57, 0.55);
  group.add(dpadH);

  const dpadV = new THREE.Mesh(dpadGeometry, dpadMaterial);
  dpadV.position.set(-0.25, 0.57, 0.55);
  dpadV.rotation.y = Math.PI / 2;
  group.add(dpadV);

  return { group, buttons, sticks, screen };
}

export function animateGameController3D(
  refs: GameController3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Animate buttons (random presses)
  refs.buttons.forEach((button, idx) => {
    const material = button.material as THREE.MeshBasicMaterial;
    const press = Math.sin(time * 4 + idx * 1.5) > 0.7 ? 1 : 0.5;
    material.opacity = 0.5 + press * 0.5 * activity;
    button.position.y = 0.58 - press * 0.01;
  });

  // Animate sticks (circular motion simulating AI input)
  refs.sticks.forEach((stick, idx) => {
    const phase = time * 2 + idx * Math.PI;
    stick.rotation.x = Math.sin(phase) * 0.2 * activity;
    stick.rotation.z = Math.cos(phase) * 0.2 * activity;
  });
}

export function disposeGameController3D(refs: GameController3DRefs): void {
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
}
