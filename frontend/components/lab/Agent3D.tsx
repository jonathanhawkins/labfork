"use client";

import * as THREE from "three";
import { Agent, AgentMeshRefs } from "./types";
import { LAB_COLORS } from "./colors";

/**
 * Create a cute robot agent from primitive shapes
 * Katamari Damacy style - simple, colorful, adorable
 */
export function createAgent3D(agent: Agent): AgentMeshRefs {
  const group = new THREE.Group();
  group.userData = { agent };

  // Body (rounded cylinder/capsule-like)
  const bodyGeometry = new THREE.CapsuleGeometry(0.35, 0.5, 8, 16);
  const bodyMaterial = new THREE.MeshToonMaterial({
    color: agent.color,
  });
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.position.y = 0.6;
  body.castShadow = true;
  group.add(body);

  // Head (sphere)
  const headGeometry = new THREE.SphereGeometry(0.3, 16, 16);
  const headMaterial = new THREE.MeshToonMaterial({
    color: agent.color,
  });
  const head = new THREE.Mesh(headGeometry, headMaterial);
  head.position.y = 1.3;
  head.castShadow = true;
  group.add(head);

  // Eyes (small dark spheres)
  const eyeGeometry = new THREE.SphereGeometry(0.06, 8, 8);
  const eyeMaterial = new THREE.MeshBasicMaterial({ color: LAB_COLORS.eyes });

  const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
  leftEye.position.set(-0.1, 1.35, 0.25);
  group.add(leftEye);

  const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
  rightEye.position.set(0.1, 1.35, 0.25);
  group.add(rightEye);

  // Eye highlights (tiny white spheres for that cute look)
  const highlightGeometry = new THREE.SphereGeometry(0.02, 4, 4);
  const highlightMaterial = new THREE.MeshBasicMaterial({ color: LAB_COLORS.eyeHighlight });

  const leftHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
  leftHighlight.position.set(-0.08, 1.37, 0.29);
  group.add(leftHighlight);

  const rightHighlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
  rightHighlight.position.set(0.12, 1.37, 0.29);
  group.add(rightHighlight);

  // Antenna (for working/thinking agents)
  let antennaBall: THREE.Mesh | undefined;
  if (agent.status === "working" || agent.status === "thinking") {
    const antennaGeometry = new THREE.CylinderGeometry(0.02, 0.02, 0.2, 8);
    const antennaMaterial = new THREE.MeshToonMaterial({ color: LAB_COLORS.antenna });
    const antenna = new THREE.Mesh(antennaGeometry, antennaMaterial);
    antenna.position.y = 1.7;
    group.add(antenna);

    const antennaBallGeometry = new THREE.SphereGeometry(0.05, 8, 8);
    const antennaBallMaterial = new THREE.MeshBasicMaterial({
      color: agent.status === "working" ? LAB_COLORS.antennaWorking : LAB_COLORS.antennaThinking,
      transparent: true,
      opacity: 1,
    });
    antennaBall = new THREE.Mesh(antennaBallGeometry, antennaBallMaterial);
    antennaBall.position.y = 1.85;
    antennaBall.name = "antennaBall";
    group.add(antennaBall);
  }

  // Arms (small capsules)
  const armGeometry = new THREE.CapsuleGeometry(0.08, 0.25, 4, 8);
  const armMaterial = new THREE.MeshToonMaterial({ color: agent.color });

  const leftArm = new THREE.Mesh(armGeometry, armMaterial);
  leftArm.position.set(-0.45, 0.6, 0);
  leftArm.rotation.z = Math.PI / 6;
  leftArm.name = "leftArm";
  group.add(leftArm);

  const rightArm = new THREE.Mesh(armGeometry, armMaterial);
  rightArm.position.set(0.45, 0.6, 0);
  rightArm.rotation.z = -Math.PI / 6;
  rightArm.name = "rightArm";
  group.add(rightArm);

  // Position the agent
  group.position.set(...agent.position);
  group.position.y = 0;

  return {
    group,
    leftArm,
    rightArm,
    antennaBall,
  };
}

/**
 * Animate an agent with bobbing, arm movement, and antenna glow
 */
export function animateAgent3D(refs: AgentMeshRefs, time: number) {
  const agent = refs.group.userData.agent as Agent;

  // Gentle bobbing motion
  refs.group.position.y = Math.sin(time * 2 + refs.group.position.x) * 0.05;

  // Arm animation for working agents (typing motion)
  if (agent.status === "working" && refs.leftArm && refs.rightArm) {
    refs.leftArm.rotation.x = Math.sin(time * 8) * 0.3;
    refs.rightArm.rotation.x = Math.sin(time * 8 + Math.PI) * 0.3;
  }

  // Antenna glow pulse
  if (refs.antennaBall) {
    const material = refs.antennaBall.material as THREE.MeshBasicMaterial;
    material.opacity = 0.5 + Math.sin(time * 4) * 0.5;
  }
}

/**
 * Update agent status (changes antenna color)
 */
export function updateAgentStatus(refs: AgentMeshRefs, status: "idle" | "working" | "thinking") {
  refs.group.userData.agent.status = status;

  if (refs.antennaBall) {
    const material = refs.antennaBall.material as THREE.MeshBasicMaterial;
    material.color.setHex(
      status === "working" ? LAB_COLORS.antennaWorking : LAB_COLORS.antennaThinking
    );
  }
}
