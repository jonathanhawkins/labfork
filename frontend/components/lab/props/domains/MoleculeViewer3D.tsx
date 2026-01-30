/**
 * MoleculeViewer3D - Rotating molecule display for Drug Discovery domain
 * Features: Atoms, bonds, rotation, highlighting
 */

import * as THREE from "three";

export interface MoleculeViewer3DRefs {
  group: THREE.Group;
  atoms: THREE.Mesh[];
  bonds: THREE.Mesh[];
  display: THREE.Mesh;
}

export interface MoleculeViewer3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

// Simple caffeine-like molecule structure
const moleculeData = {
  atoms: [
    { pos: [0, 0, 0], color: 0x3b82f6, size: 0.12 }, // N - blue
    { pos: [0.4, 0.2, 0], color: 0x444444, size: 0.1 }, // C - gray
    { pos: [0.8, 0, 0.1], color: 0x444444, size: 0.1 }, // C
    { pos: [0.6, -0.4, 0], color: 0xef4444, size: 0.11 }, // O - red
    { pos: [-0.3, 0.3, 0.2], color: 0x444444, size: 0.1 }, // C
    { pos: [-0.5, -0.2, 0], color: 0x3b82f6, size: 0.12 }, // N
    { pos: [-0.2, -0.5, 0.1], color: 0x444444, size: 0.1 }, // C
    { pos: [0.2, -0.3, -0.1], color: 0x444444, size: 0.1 }, // C
  ],
  bonds: [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [6, 7], [7, 0],
  ],
};

export function createMoleculeViewer3D(options: MoleculeViewer3DOptions): MoleculeViewer3DRefs {
  const { position, scale = 1, accentColor = 0x14b8a6 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const atoms: THREE.Mesh[] = [];
  const bonds: THREE.Mesh[] = [];

  // Display stand
  const standGeometry = new THREE.CylinderGeometry(0.4, 0.5, 0.1, 24);
  const standMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const stand = new THREE.Mesh(standGeometry, standMaterial);
  stand.position.y = 0.05;
  stand.castShadow = true;
  group.add(stand);

  // Display post
  const postGeometry = new THREE.CylinderGeometry(0.03, 0.03, 0.8, 8);
  const postMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const post = new THREE.Mesh(postGeometry, postMaterial);
  post.position.y = 0.5;
  group.add(post);

  // Display sphere (container)
  const displayGeometry = new THREE.SphereGeometry(0.6, 24, 24);
  const displayMaterial = new THREE.MeshToonMaterial({
    color: 0x1a2a3a,
    transparent: true,
    opacity: 0.2,
  });
  const display = new THREE.Mesh(displayGeometry, displayMaterial);
  display.position.y = 1.2;
  group.add(display);

  // Molecule container (will rotate)
  const moleculeGroup = new THREE.Group();
  moleculeGroup.position.y = 1.2;
  moleculeGroup.userData = { isMolecule: true };
  group.add(moleculeGroup);

  // Create atoms
  moleculeData.atoms.forEach((atomData, idx) => {
    const atomGeometry = new THREE.SphereGeometry(atomData.size, 16, 16);
    const atomMaterial = new THREE.MeshToonMaterial({
      color: atomData.color,
      emissive: atomData.color,
      emissiveIntensity: 0.2,
    });
    const atom = new THREE.Mesh(atomGeometry, atomMaterial);
    atom.position.set(atomData.pos[0], atomData.pos[1], atomData.pos[2]);
    atom.userData = { atomIndex: idx };
    moleculeGroup.add(atom);
    atoms.push(atom);
  });

  // Create bonds
  moleculeData.bonds.forEach(([a1, a2], idx) => {
    const start = new THREE.Vector3(...moleculeData.atoms[a1].pos);
    const end = new THREE.Vector3(...moleculeData.atoms[a2].pos);
    const mid = start.clone().add(end).multiplyScalar(0.5);
    const length = start.distanceTo(end);

    const bondGeometry = new THREE.CylinderGeometry(0.02, 0.02, length, 8);
    const bondMaterial = new THREE.MeshToonMaterial({ color: 0x888888 });
    const bond = new THREE.Mesh(bondGeometry, bondMaterial);

    bond.position.copy(mid);
    bond.lookAt(end);
    bond.rotateX(Math.PI / 2);
    bond.userData = { bondIndex: idx };
    moleculeGroup.add(bond);
    bonds.push(bond);
  });

  // Highlight ring around display
  const ringGeometry = new THREE.TorusGeometry(0.62, 0.02, 8, 32);
  const ringMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.6,
  });
  const ring = new THREE.Mesh(ringGeometry, ringMaterial);
  ring.position.y = 1.2;
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  // Label
  const labelGeometry = new THREE.PlaneGeometry(0.6, 0.15);
  const labelMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.8,
  });
  const label = new THREE.Mesh(labelGeometry, labelMaterial);
  label.position.set(0, 0.2, 0.45);
  group.add(label);

  return { group, atoms, bonds, display };
}

export function animateMoleculeViewer3D(
  refs: MoleculeViewer3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Find and rotate molecule group
  refs.group.children.forEach((child) => {
    if (child instanceof THREE.Group && child.userData.isMolecule) {
      child.rotation.y = time * 0.5 * activity;
      child.rotation.x = Math.sin(time * 0.3) * 0.2;
    }
  });

  // Pulse atoms
  refs.atoms.forEach((atom, idx) => {
    const material = atom.material as THREE.MeshToonMaterial;
    const pulse = Math.sin(time * 2 + idx * 0.5) * 0.1 + 0.2;
    material.emissiveIntensity = pulse * activity;
  });
}

export function disposeMoleculeViewer3D(refs: MoleculeViewer3DRefs): void {
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
