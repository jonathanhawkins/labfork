/**
 * EarthGlobe3D - Rotating Earth globe for Climate Modeling domain
 * Features: Rotating sphere, cloud layer, data visualization overlay
 */

import * as THREE from "three";

export interface EarthGlobe3DRefs {
  group: THREE.Group;
  globe: THREE.Mesh;
  clouds: THREE.Mesh;
  dataPoints: THREE.Points;
}

export interface EarthGlobe3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

export function createEarthGlobe3D(options: EarthGlobe3DOptions): EarthGlobe3DRefs {
  const { position, scale = 1, accentColor = 0x0ea5e9 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  // Stand
  const standGeometry = new THREE.CylinderGeometry(0.3, 0.4, 0.1, 24);
  const standMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const stand = new THREE.Mesh(standGeometry, standMaterial);
  stand.position.y = 0.05;
  stand.castShadow = true;
  group.add(stand);

  // Support ring
  const ringGeometry = new THREE.TorusGeometry(0.65, 0.02, 8, 48);
  const ringMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.position.y = 0.8;
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  // Vertical support
  const supportGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.7, 8);
  const supportMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const support = new THREE.Mesh(supportGeometry, supportMaterial);
  support.position.set(0, 0.45, 0.65);
  group.add(support);

  // Globe (Earth)
  const globeGeometry = new THREE.SphereGeometry(0.6, 32, 32);
  const globeMaterial = new THREE.MeshToonMaterial({
    color: 0x1e3a5f, // Deep ocean blue
  });
  const globe = new THREE.Mesh(globeGeometry, globeMaterial);
  globe.position.y = 0.8;
  group.add(globe);

  // Continents (simplified shapes using additional meshes)
  const continentMaterial = new THREE.MeshToonMaterial({ color: 0x22c55e });

  // North America
  const na = createContinent(0.15, 0.12, { lat: 45, lon: -100 }, 0.61);
  na.material = continentMaterial;
  globe.add(na);

  // South America
  const sa = createContinent(0.08, 0.15, { lat: -15, lon: -60 }, 0.61);
  sa.material = continentMaterial;
  globe.add(sa);

  // Europe
  const eu = createContinent(0.1, 0.06, { lat: 50, lon: 10 }, 0.61);
  eu.material = continentMaterial;
  globe.add(eu);

  // Africa
  const af = createContinent(0.12, 0.15, { lat: 5, lon: 20 }, 0.61);
  af.material = continentMaterial;
  globe.add(af);

  // Asia
  const asia = createContinent(0.2, 0.12, { lat: 40, lon: 90 }, 0.61);
  asia.material = continentMaterial;
  globe.add(asia);

  // Australia
  const aus = createContinent(0.08, 0.06, { lat: -25, lon: 135 }, 0.61);
  aus.material = continentMaterial;
  globe.add(aus);

  // Cloud layer
  const cloudGeometry = new THREE.SphereGeometry(0.65, 24, 24);
  const cloudMaterial = new THREE.MeshToonMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.3,
  });
  const clouds = new THREE.Mesh(cloudGeometry, cloudMaterial);
  clouds.position.y = 0.8;
  group.add(clouds);

  // Data points (weather stations / measurement points)
  const pointCount = 50;
  const pointsGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointCount * 3);
  const colors = new Float32Array(pointCount * 3);

  const warmColor = new THREE.Color(0xef4444);
  const coolColor = new THREE.Color(accentColor);

  for (let i = 0; i < pointCount; i++) {
    const lat = (Math.random() - 0.5) * 180;
    const lon = Math.random() * 360;
    const radius = 0.68;

    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);

    positions[i * 3] = -radius * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = radius * Math.cos(phi) + 0.8;
    positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);

    // Color based on latitude (warm near equator)
    const temp = 1 - Math.abs(lat) / 90;
    const color = coolColor.clone().lerp(warmColor, temp);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }

  pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  pointsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const pointsMaterial = new THREE.PointsMaterial({
    size: 0.03,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
  });
  const dataPoints = new THREE.Points(pointsGeometry, pointsMaterial);
  group.add(dataPoints);

  // Latitude/longitude lines
  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0x3b82f6,
    transparent: true,
    opacity: 0.3,
  });

  // Equator
  const equatorPoints = [];
  for (let i = 0; i <= 64; i++) {
    const angle = (i / 64) * Math.PI * 2;
    equatorPoints.push(new THREE.Vector3(
      Math.cos(angle) * 0.62,
      0.8,
      Math.sin(angle) * 0.62
    ));
  }
  const equatorGeometry = new THREE.BufferGeometry().setFromPoints(equatorPoints);
  const equator = new THREE.Line(equatorGeometry, lineMaterial);
  group.add(equator);

  return { group, globe, clouds, dataPoints };
}

function createContinent(
  width: number,
  height: number,
  coords: { lat: number; lon: number },
  radius: number
): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshToonMaterial({ color: 0x22c55e });
  const mesh = new THREE.Mesh(geometry, material);

  const phi = (90 - coords.lat) * (Math.PI / 180);
  const theta = (coords.lon + 180) * (Math.PI / 180);

  mesh.position.set(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );

  mesh.lookAt(0, 0, 0);

  return mesh;
}

export function animateEarthGlobe3D(
  refs: EarthGlobe3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Rotate globe
  refs.globe.rotation.y = time * 0.1 * activity;

  // Rotate clouds slightly faster
  refs.clouds.rotation.y = time * 0.12 * activity;
  refs.clouds.rotation.x = Math.sin(time * 0.1) * 0.05;

  // Pulse data points
  const material = refs.dataPoints.material as THREE.PointsMaterial;
  material.opacity = 0.6 + Math.sin(time * 2) * 0.2;
}

export function disposeEarthGlobe3D(refs: EarthGlobe3DRefs): void {
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
