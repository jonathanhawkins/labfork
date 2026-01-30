/**
 * RobotArm3D - Articulated robot arm for Robotics domain
 * Features: Multiple joints, gripper, workspace platform
 */

import * as THREE from "three";

export interface RobotArm3DRefs {
  group: THREE.Group;
  joints: THREE.Group[];
  gripper: THREE.Group;
  platform: THREE.Mesh;
}

export interface RobotArm3DOptions {
  position: [number, number, number];
  scale?: number;
  accentColor?: number;
}

export function createRobotArm3D(options: RobotArm3DOptions): RobotArm3DRefs {
  const { position, scale = 1, accentColor = 0xf97316 } = options;

  const group = new THREE.Group();
  group.position.set(...position);
  group.scale.setScalar(scale);

  const joints: THREE.Group[] = [];

  // Base platform
  const platformGeometry = new THREE.CylinderGeometry(0.6, 0.7, 0.15, 24);
  const platformMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const platform = new THREE.Mesh(platformGeometry, platformMaterial);
  platform.position.y = 0.075;
  platform.castShadow = true;
  group.add(platform);

  // Base rotation joint
  const baseJoint = new THREE.Group();
  baseJoint.position.y = 0.15;

  const baseGeometry = new THREE.CylinderGeometry(0.2, 0.25, 0.2, 16);
  const baseMaterial = new THREE.MeshToonMaterial({ color: 0x1a1a1a });
  const base = new THREE.Mesh(baseGeometry, baseMaterial);
  base.position.y = 0.1;
  baseJoint.add(base);

  // First arm segment
  const arm1Geometry = new THREE.BoxGeometry(0.12, 0.6, 0.12);
  const arm1Material = new THREE.MeshToonMaterial({ color: accentColor });
  const arm1 = new THREE.Mesh(arm1Geometry, arm1Material);
  arm1.position.y = 0.5;
  baseJoint.add(arm1);

  group.add(baseJoint);
  joints.push(baseJoint);

  // Elbow joint
  const elbowJoint = new THREE.Group();
  elbowJoint.position.y = 0.8;

  const elbowGeometry = new THREE.SphereGeometry(0.1, 12, 12);
  const elbowMaterial = new THREE.MeshToonMaterial({ color: 0x444444 });
  const elbow = new THREE.Mesh(elbowGeometry, elbowMaterial);
  elbowJoint.add(elbow);

  // Second arm segment
  const arm2Geometry = new THREE.BoxGeometry(0.1, 0.5, 0.1);
  const arm2Material = new THREE.MeshToonMaterial({ color: accentColor });
  const arm2 = new THREE.Mesh(arm2Geometry, arm2Material);
  arm2.position.y = 0.25;
  elbowJoint.add(arm2);

  baseJoint.add(elbowJoint);
  joints.push(elbowJoint);

  // Wrist joint
  const wristJoint = new THREE.Group();
  wristJoint.position.y = 0.5;

  const wristGeometry = new THREE.SphereGeometry(0.06, 12, 12);
  const wristMaterial = new THREE.MeshToonMaterial({ color: 0x444444 });
  const wrist = new THREE.Mesh(wristGeometry, wristMaterial);
  wristJoint.add(wrist);

  elbowJoint.add(wristJoint);
  joints.push(wristJoint);

  // Gripper
  const gripper = new THREE.Group();
  gripper.position.y = 0.05;

  // Gripper base
  const gripperBaseGeometry = new THREE.CylinderGeometry(0.04, 0.05, 0.08, 8);
  const gripperBaseMaterial = new THREE.MeshToonMaterial({ color: 0x2d2d2d });
  const gripperBase = new THREE.Mesh(gripperBaseGeometry, gripperBaseMaterial);
  gripper.add(gripperBase);

  // Gripper fingers
  const fingerGeometry = new THREE.BoxGeometry(0.02, 0.1, 0.03);
  const fingerMaterial = new THREE.MeshToonMaterial({ color: 0x666666 });

  const leftFinger = new THREE.Mesh(fingerGeometry, fingerMaterial);
  leftFinger.position.set(-0.03, -0.08, 0);
  leftFinger.userData = { finger: "left" };
  gripper.add(leftFinger);

  const rightFinger = new THREE.Mesh(fingerGeometry, fingerMaterial);
  rightFinger.position.set(0.03, -0.08, 0);
  rightFinger.userData = { finger: "right" };
  gripper.add(rightFinger);

  wristJoint.add(gripper);

  // Work object (cube to pick up)
  const objectGeometry = new THREE.BoxGeometry(0.12, 0.12, 0.12);
  const objectMaterial = new THREE.MeshToonMaterial({ color: 0x3b82f6 });
  const workObject = new THREE.Mesh(objectGeometry, objectMaterial);
  workObject.position.set(0.4, 0.21, 0.3);
  workObject.castShadow = true;
  group.add(workObject);

  // Grid lines on platform
  const gridMaterial = new THREE.LineBasicMaterial({ color: 0x444444 });
  for (let i = -5; i <= 5; i++) {
    const lineGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(i * 0.1, 0.16, -0.5),
      new THREE.Vector3(i * 0.1, 0.16, 0.5),
    ]);
    const line = new THREE.Line(lineGeometry, gridMaterial);
    group.add(line);

    const lineGeometry2 = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, 0.16, i * 0.1),
      new THREE.Vector3(0.5, 0.16, i * 0.1),
    ]);
    const line2 = new THREE.Line(lineGeometry2, gridMaterial);
    group.add(line2);
  }

  return { group, joints, gripper, platform };
}

export function animateRobotArm3D(
  refs: RobotArm3DRefs,
  time: number,
  options?: { activity?: number }
): void {
  const activity = options?.activity ?? 0.5;

  // Animate joints
  if (refs.joints[0]) {
    refs.joints[0].rotation.y = Math.sin(time * 0.5) * 0.8 * activity;
  }
  if (refs.joints[1]) {
    refs.joints[1].rotation.x = Math.sin(time * 0.7 + 0.5) * 0.4 * activity - 0.3;
  }
  if (refs.joints[2]) {
    refs.joints[2].rotation.x = Math.sin(time * 0.9 + 1) * 0.3 * activity;
    refs.joints[2].rotation.z = Math.sin(time * 0.6) * 0.2 * activity;
  }

  // Animate gripper (open/close)
  const gripperOpen = Math.sin(time * 2) > 0 ? 0.02 : -0.01;
  refs.gripper.children.forEach((child) => {
    if (child instanceof THREE.Mesh && child.userData.finger) {
      const offset = child.userData.finger === "left" ? -1 : 1;
      child.position.x = offset * (0.03 + gripperOpen);
    }
  });
}

export function disposeRobotArm3D(refs: RobotArm3DRefs): void {
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
