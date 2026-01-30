/**
 * ImageGrid3D - Image display wall for Computer Vision domain
 * Features: Grid of images, bounding boxes, feature highlighting
 */

import * as THREE from "three";

export interface ImageGrid3DRefs {
  group: THREE.Group;
  images: THREE.Mesh[];
  boundingBoxes: THREE.LineSegments[];
  featurePoints: THREE.Points;
}

export interface ImageGrid3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

export function createImageGrid3D(options: ImageGrid3DOptions): ImageGrid3DRefs {
  const { position, scale = 1, accentColor = 0xef4444 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const images: THREE.Mesh[] = [];
  const boundingBoxes: THREE.LineSegments[] = [];

  // Display frame
  const frameGeometry = new THREE.BoxGeometry(2.2, 1.6, 0.1);
  const frameMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const frame = new THREE.Mesh(frameGeometry, frameMaterial);
  frame.position.set(0, 1, -0.5);
  frame.castShadow = true;
  group.add(frame);

  // Create 3x2 image grid
  const gridCols = 3;
  const gridRows = 2;
  const imageWidth = 0.6;
  const imageHeight = 0.6;
  const gap = 0.08;

  for (let row = 0; row < gridRows; row++) {
    for (let col = 0; col < gridCols; col++) {
      const x = (col - (gridCols - 1) / 2) * (imageWidth + gap);
      const y = 1 + ((gridRows - 1) / 2 - row) * (imageHeight + gap);

      // Image placeholder
      const imageGeometry = new THREE.PlaneGeometry(imageWidth, imageHeight);
      const hue = (row * gridCols + col) * 0.1;
      const imageColor = new THREE.Color().setHSL(hue, 0.3, 0.2);
      const imageMaterial = new THREE.MeshBasicMaterial({
        color: imageColor,
      });
      const image = new THREE.Mesh(imageGeometry, imageMaterial);
      image.position.set(x, y, -0.44);
      image.userData = { imageIndex: row * gridCols + col, col, row };
      group.add(image);
      images.push(image);

      // Simulate image content with random shapes
      const shapeCount = 2 + Math.floor(Math.random() * 3);
      for (let s = 0; s < shapeCount; s++) {
        const shapeGeometry = new THREE.PlaneGeometry(
          0.1 + Math.random() * 0.15,
          0.1 + Math.random() * 0.15
        );
        const shapeMaterial = new THREE.MeshBasicMaterial({
          color: new THREE.Color().setHSL(Math.random(), 0.5, 0.4),
        });
        const shape = new THREE.Mesh(shapeGeometry, shapeMaterial);
        shape.position.set(
          x + (Math.random() - 0.5) * 0.4,
          y + (Math.random() - 0.5) * 0.4,
          -0.43
        );
        group.add(shape);
      }

      // Bounding box (detection result)
      if (Math.random() > 0.3) {
        const boxWidth = 0.15 + Math.random() * 0.2;
        const boxHeight = 0.15 + Math.random() * 0.2;
        const boxX = x + (Math.random() - 0.5) * 0.3;
        const boxY = y + (Math.random() - 0.5) * 0.3;

        const boxPoints = [
          // Top
          new THREE.Vector3(boxX - boxWidth / 2, boxY + boxHeight / 2, -0.42),
          new THREE.Vector3(boxX + boxWidth / 2, boxY + boxHeight / 2, -0.42),
          // Right
          new THREE.Vector3(boxX + boxWidth / 2, boxY + boxHeight / 2, -0.42),
          new THREE.Vector3(boxX + boxWidth / 2, boxY - boxHeight / 2, -0.42),
          // Bottom
          new THREE.Vector3(boxX + boxWidth / 2, boxY - boxHeight / 2, -0.42),
          new THREE.Vector3(boxX - boxWidth / 2, boxY - boxHeight / 2, -0.42),
          // Left
          new THREE.Vector3(boxX - boxWidth / 2, boxY - boxHeight / 2, -0.42),
          new THREE.Vector3(boxX - boxWidth / 2, boxY + boxHeight / 2, -0.42),
        ];

        const boxGeometry = new THREE.BufferGeometry().setFromPoints(boxPoints);
        const boxMaterial = new THREE.LineBasicMaterial({
          color: accentColor,
          linewidth: 2,
        });
        const box = new THREE.LineSegments(boxGeometry, boxMaterial);
        box.userData = { boxIndex: images.length - 1 };
        group.add(box);
        boundingBoxes.push(box);

        // Confidence label
        const labelGeometry = new THREE.PlaneGeometry(0.12, 0.04);
        const labelMaterial = new THREE.MeshBasicMaterial({
          color: accentColor,
          transparent: true,
          opacity: 0.8,
        });
        const label = new THREE.Mesh(labelGeometry, labelMaterial);
        label.position.set(boxX - boxWidth / 2 + 0.06, boxY + boxHeight / 2 + 0.03, -0.41);
        group.add(label);
      }
    }
  }

  // Feature points (keypoint detection)
  const pointCount = 80;
  const featureGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(pointCount * 3);
  const colors = new Float32Array(pointCount * 3);

  const featureColor = new THREE.Color(0x22c55e);

  for (let i = 0; i < pointCount; i++) {
    const imgIdx = Math.floor(Math.random() * images.length);
    const img = images[imgIdx];

    positions[i * 3] = img.position.x + (Math.random() - 0.5) * 0.5;
    positions[i * 3 + 1] = img.position.y + (Math.random() - 0.5) * 0.5;
    positions[i * 3 + 2] = -0.41;

    colors[i * 3] = featureColor.r;
    colors[i * 3 + 1] = featureColor.g;
    colors[i * 3 + 2] = featureColor.b;
  }

  featureGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  featureGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const featureMaterial = new THREE.PointsMaterial({
    size: 0.02,
    vertexColors: true,
    transparent: true,
    opacity: 0.6,
  });
  const featurePoints = new THREE.Points(featureGeometry, featureMaterial);
  group.add(featurePoints);

  // Processing indicator
  const processingGeometry = new THREE.RingGeometry(0.08, 0.1, 24);
  const processingMaterial = new THREE.MeshBasicMaterial({
    color: accentColor,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  });
  const processing = new THREE.Mesh(processingGeometry, processingMaterial);
  processing.position.set(0.9, 0.3, -0.4);
  processing.userData = { isProcessing: true };
  group.add(processing);

  return { group, images, boundingBoxes, featurePoints };
}

export function animateImageGrid3D(
  refs: ImageGrid3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Animate bounding boxes (pulse)
  refs.boundingBoxes.forEach((box, idx) => {
    const material = box.material as THREE.LineBasicMaterial;
    const pulse = Math.sin(time * 3 + idx * 0.5) * 0.3 + 0.7;
    material.opacity = pulse * activity;
  });

  // Animate feature points
  const featureMat = refs.featurePoints.material as THREE.PointsMaterial;
  featureMat.opacity = 0.4 + Math.sin(time * 2) * 0.2;

  // Rotate processing indicator
  refs.group.children.forEach((child) => {
    if (child instanceof THREE.Mesh && child.userData.isProcessing) {
      child.rotation.z = time * 2;
    }
  });
}

export function disposeImageGrid3D(refs: ImageGrid3DRefs): void {
  refs.group.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments || child instanceof THREE.Points) {
      child.geometry.dispose();
      if (Array.isArray(child.material)) {
        child.material.forEach((m) => m.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}
