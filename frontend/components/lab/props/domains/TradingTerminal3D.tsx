/**
 * TradingTerminal3D - Multi-monitor trading setup for Quant Trading domain
 * Features: Multiple screens with chart animations, ticker tape, LED accents
 */

import * as THREE from "three";

export interface TradingTerminal3DRefs {
  group: THREE.Group;
  screens: THREE.Mesh[];
  ticker: THREE.Mesh;
  leds: THREE.Mesh[];
}

export interface TradingTerminal3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

export function createTradingTerminal3D(options: TradingTerminal3DOptions): TradingTerminal3DRefs {
  const { position, scale = 1, accentColor = 0x10b981 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const screens: THREE.Mesh[] = [];
  const leds: THREE.Mesh[] = [];

  // Desk
  const deskGeometry = new THREE.BoxGeometry(3, 0.1, 1.2);
  const deskMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const desk = new THREE.Mesh(deskGeometry, deskMaterial);
  desk.position.y = 0.8;
  desk.castShadow = true;
  group.add(desk);

  // Monitor stand
  const standGeometry = new THREE.BoxGeometry(0.8, 0.05, 0.3);
  const standMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const stand = new THREE.Mesh(standGeometry, standMaterial);
  stand.position.set(0, 0.88, -0.3);
  group.add(stand);

  // Create 3 monitors
  const monitorPositions = [
    { x: -0.9, rot: 0.15 },
    { x: 0, rot: 0 },
    { x: 0.9, rot: -0.15 },
  ];

  monitorPositions.forEach((pos, idx) => {
    // Monitor frame
    const frameGeometry = new THREE.BoxGeometry(0.8, 0.5, 0.05);
    const frameMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
    const frame = new THREE.Mesh(frameGeometry, frameMaterial);
    frame.position.set(pos.x, 1.2, -0.4);
    frame.rotation.y = pos.rot;
    group.add(frame);

    // Screen (emissive)
    const screenGeometry = new THREE.PlaneGeometry(0.72, 0.42);
    const screenMaterial = new THREE.MeshBasicMaterial({
      color: 0x0a1a0f,
      transparent: true,
      opacity: 0.9,
    });
    const screen = new THREE.Mesh(screenGeometry, screenMaterial);
    screen.position.set(pos.x, 1.2, -0.37);
    screen.rotation.y = pos.rot;
    screen.userData = { screenIndex: idx };
    group.add(screen);
    screens.push(screen);

    // Chart line on screen
    const chartPoints = [];
    for (let i = 0; i < 20; i++) {
      chartPoints.push(
        new THREE.Vector3(
          pos.x - 0.3 + i * 0.03,
          1.1 + Math.sin(i * 0.5 + idx) * 0.1,
          -0.36
        )
      );
    }
    const chartGeometry = new THREE.BufferGeometry().setFromPoints(chartPoints);
    const chartMaterial = new THREE.LineBasicMaterial({ color: accentColor });
    const chart = new THREE.Line(chartGeometry, chartMaterial);
    chart.rotation.y = pos.rot;
    group.add(chart);
  });

  // Ticker tape at bottom
  const tickerGeometry = new THREE.BoxGeometry(2.8, 0.08, 0.02);
  const tickerMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.8,
  });
  const ticker = new THREE.Mesh(tickerGeometry, tickerMaterial);
  ticker.position.set(0, 0.92, -0.35);
  group.add(ticker);

  // LED strip accents
  for (let i = 0; i < 12; i++) {
    const ledGeometry = new THREE.BoxGeometry(0.02, 0.02, 0.02);
    const ledMaterial = new THREE.MeshBasicMaterial({
      color: accentColor,
      transparent: true,
      opacity: 0.5,
    });
    const led = new THREE.Mesh(ledGeometry, ledMaterial);
    led.position.set(-1.2 + i * 0.22, 0.86, -0.55);
    led.userData = { ledIndex: i };
    group.add(led);
    leds.push(led);
  }

  // Keyboard
  const keyboardGeometry = new THREE.BoxGeometry(0.5, 0.02, 0.2);
  const keyboardMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const keyboard = new THREE.Mesh(keyboardGeometry, keyboardMaterial);
  keyboard.position.set(0, 0.87, 0.2);
  group.add(keyboard);

  return { group, screens, ticker, leds };
}

export function animateTradingTerminal3D(
  refs: TradingTerminal3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Animate LEDs in wave pattern
  refs.leds.forEach((led, idx) => {
    const material = led.material as THREE.MeshBasicMaterial;
    const wave = Math.sin(time * 3 + idx * 0.5) * 0.3 + 0.5;
    material.opacity = wave * activity;
  });

  // Pulse ticker
  const tickerMat = refs.ticker.material as THREE.MeshBasicMaterial;
  tickerMat.opacity = 0.6 + Math.sin(time * 2) * 0.2;
}

export function disposeTradingTerminal3D(refs: TradingTerminal3DRefs): void {
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
}
