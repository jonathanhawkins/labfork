"use client";

import React, { useRef, useEffect, useMemo } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

interface ProsodyVisualizerProps {
  analyserNode?: AnalyserNode | null;
  isRecording?: boolean;
  isProcessing?: boolean;
  prosodyData?: {
    semantic?: {
      emotion?: string;
    };
    contour?: {
      smoothed?: number[];
    };
  } | null;
}

const EMOTION_COLORS: Record<string, number> = {
  neutral: 0x808080,
  happy: 0xffdd00,
  sad: 0x4488ff,
  angry: 0xff4444,
  fearful: 0x8844ff,
  surprised: 0xff8800,
  friendly: 0x44ff88,
  excited: 0xff6600,
  thoughtful: 0x6688ff,
  concerned: 0xffaa44,
  confident: 0x44ddff,
};

export default function ProsodyMatrixVisualizer({
  analyserNode,
  isRecording = false,
  isProcessing = false,
  prosodyData,
}: ProsodyVisualizerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animationRef = useRef<number>(0);
  
  // Core objects
  const coreRef = useRef<THREE.Mesh | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const waveformRef = useRef<THREE.Line | null>(null);
  const contourRef = useRef<THREE.Line | null>(null);
  
  // Animation state
  const timeRef = useRef(0);
  const particlePositionsRef = useRef<Float32Array | null>(null);

  useEffect(() => {
    const currentContainer = containerRef.current;
    if (!currentContainer) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(
      60,
      currentContainer.clientWidth / currentContainer.clientHeight,
      0.1,
      100
    );
    camera.position.set(3, 2, 4);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(currentContainer.clientWidth, currentContainer.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentContainer.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(5, 5, 5);
    scene.add(pointLight);

    // === Create Cube Structure ===
    
    // Wireframe cube
    const cubeGeometry = new THREE.BoxGeometry(2, 2, 2);
    const cubeEdges = new THREE.EdgesGeometry(cubeGeometry);
    const cubeMaterial = new THREE.LineBasicMaterial({ color: 0x44aaff, transparent: true, opacity: 0.6 });
    const cubeWireframe = new THREE.LineSegments(cubeEdges, cubeMaterial);
    scene.add(cubeWireframe);

    // Face labels (planes with slight transparency)
    const faceColors = [
      { position: [0, 0, 1.01], color: 0x00ffff, name: "INPUT" },      // Front
      { position: [0, 0, -1.01], color: 0xff00ff, name: "OUTPUT" },    // Back
      { position: [0, 1.01, 0], color: 0xffff00, name: "SEMANTIC" },   // Top
      { position: [0, -1.01, 0], color: 0x00ff00, name: "ACOUSTIC" },  // Bottom
      { position: [1.01, 0, 0], color: 0xff8800, name: "RHYTHM" },     // Right
      { position: [-1.01, 0, 0], color: 0x8800ff, name: "CONTOUR" },   // Left
    ];

    faceColors.forEach(({ position, color }) => {
      const faceGeometry = new THREE.PlaneGeometry(1.8, 1.8);
      const faceMaterial = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
      });
      const face = new THREE.Mesh(faceGeometry, faceMaterial);
      face.position.set(position[0], position[1], position[2]);
      
      // Rotate faces to align with cube
      if (position[1] !== 0) face.rotation.x = Math.PI / 2;
      if (position[0] !== 0) face.rotation.y = Math.PI / 2;
      
      scene.add(face);
    });

    // === Core (icosahedron that pulses) ===
    const coreGeometry = new THREE.IcosahedronGeometry(0.3, 1);
    const coreMaterial = new THREE.MeshPhongMaterial({
      color: 0x44aaff,
      emissive: 0x2266aa,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0.9,
    });
    const core = new THREE.Mesh(coreGeometry, coreMaterial);
    scene.add(core);
    coreRef.current = core;

    // === Particles ===
    const particleCount = 300;
    const particleGeometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount; i++) {
      // Random positions within extended cube bounds
      positions[i * 3] = (Math.random() - 0.5) * 6;     // X: spread for flow
      positions[i * 3 + 1] = (Math.random() - 0.5) * 2; // Y
      positions[i * 3 + 2] = (Math.random() - 0.5) * 2; // Z
      
      // Gradient color based on X position
      const t = (positions[i * 3] + 3) / 6;
      colors[i * 3] = 0 + t * 1;      // R: cyan to magenta
      colors[i * 3 + 1] = 1 - t * 0.5; // G
      colors[i * 3 + 2] = 1;           // B
    }
    
    particleGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    particleGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    particlePositionsRef.current = positions;
    
    const particleMaterial = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);
    particlesRef.current = particles;

    // === Waveform Line (left side input) ===
    const waveformPoints = [];
    for (let i = 0; i < 64; i++) {
      waveformPoints.push(new THREE.Vector3(-2.5, (i / 64 - 0.5) * 2, 0));
    }
    const waveformGeometry = new THREE.BufferGeometry().setFromPoints(waveformPoints);
    const waveformMaterial = new THREE.LineBasicMaterial({ color: 0x00ffff });
    const waveform = new THREE.Line(waveformGeometry, waveformMaterial);
    scene.add(waveform);
    waveformRef.current = waveform;

    // === Pitch Contour Line (inside cube) ===
    const contourPoints = [];
    for (let i = 0; i < 50; i++) {
      const x = (i / 50 - 0.5) * 2;
      contourPoints.push(new THREE.Vector3(x, 0, 0));
    }
    const contourGeometry = new THREE.BufferGeometry().setFromPoints(contourPoints);
    const contourMaterial = new THREE.LineBasicMaterial({ color: 0xffff00, linewidth: 2 });
    const contourLine = new THREE.Line(contourGeometry, contourMaterial);
    scene.add(contourLine);
    contourRef.current = contourLine;

    // === Animation Loop ===
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      timeRef.current += 0.016;
      
      controls.update();
      
      // Rotate core
      if (coreRef.current) {
        coreRef.current.rotation.x += 0.01;
        coreRef.current.rotation.y += 0.015;
      }
      
      // Move particles (left to right flow)
      if (particlesRef.current && particlePositionsRef.current) {
        const positions = particlePositionsRef.current;
        for (let i = 0; i < positions.length / 3; i++) {
          // Move right
          positions[i * 3] += 0.02;
          
          // Reset when past right boundary
          if (positions[i * 3] > 3) {
            positions[i * 3] = -3;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 2;
          }
          
          // Swirl toward center when inside cube
          if (Math.abs(positions[i * 3]) < 1) {
            const distToCenter = Math.sqrt(
              positions[i * 3 + 1] ** 2 + positions[i * 3 + 2] ** 2
            );
            if (distToCenter > 0.1) {
              positions[i * 3 + 1] *= 0.99;
              positions[i * 3 + 2] *= 0.99;
            }
          }
        }
        particlesRef.current.geometry.attributes.position.needsUpdate = true;
      }
      
      renderer.render(scene, camera);
    };
    
    animate();

    // Handle resize
    const handleResize = () => {
      if (!currentContainer) return;
      camera.aspect = currentContainer.clientWidth / currentContainer.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(currentContainer.clientWidth, currentContainer.clientHeight);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationRef.current);
      renderer.dispose();
      if (currentContainer && renderer.domElement) {
        currentContainer.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Update core color based on emotion
  useEffect(() => {
    if (coreRef.current && prosodyData?.semantic?.emotion) {
      const emotion = prosodyData.semantic.emotion;
      const color = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;
      (coreRef.current.material as THREE.MeshPhongMaterial).color.setHex(color);
      (coreRef.current.material as THREE.MeshPhongMaterial).emissive.setHex(color);
    }
  }, [prosodyData?.semantic?.emotion]);

  // Update core pulse based on recording/processing state
  useEffect(() => {
    if (coreRef.current) {
      const material = coreRef.current.material as THREE.MeshPhongMaterial;
      if (isRecording) {
        material.emissive.setHex(0xff0000);
        material.emissiveIntensity = 1.0;
      } else if (isProcessing) {
        material.emissive.setHex(0x00ff00);
        material.emissiveIntensity = 0.8;
      } else {
        material.emissiveIntensity = 0.5;
      }
    }
  }, [isRecording, isProcessing]);

  // Update waveform from analyser
  useEffect(() => {
    if (!analyserNode || !waveformRef.current) return;
    
    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    
    const updateWaveform = () => {
      analyserNode.getByteTimeDomainData(dataArray);
      
      const positions = waveformRef.current!.geometry.attributes.position.array as Float32Array;
      const step = Math.floor(dataArray.length / 64);
      
      for (let i = 0; i < 64; i++) {
        const value = (dataArray[i * step] / 128 - 1) * 0.5;
        positions[i * 3 + 2] = value; // Z displacement
      }
      
      waveformRef.current!.geometry.attributes.position.needsUpdate = true;
      
      if (isRecording) {
        requestAnimationFrame(updateWaveform);
      }
    };
    
    if (isRecording) {
      updateWaveform();
    }
  }, [analyserNode, isRecording]);

  // Update pitch contour
  useEffect(() => {
    if (!contourRef.current || !prosodyData?.contour?.smoothed) return;
    
    const contour = prosodyData.contour.smoothed;
    const positions = contourRef.current.geometry.attributes.position.array as Float32Array;
    
    const step = Math.max(1, Math.floor(contour.length / 50));
    const minPitch = Math.min(...contour.filter(v => v > 0));
    const maxPitch = Math.max(...contour);
    const range = maxPitch - minPitch || 1;
    
    for (let i = 0; i < 50; i++) {
      const idx = Math.min(i * step, contour.length - 1);
      const value = contour[idx];
      const normalized = value > 0 ? (value - minPitch) / range : 0;
      positions[i * 3 + 1] = (normalized - 0.5) * 1.5; // Y: pitch height
    }
    
    contourRef.current.geometry.attributes.position.needsUpdate = true;
  }, [prosodyData?.contour?.smoothed]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ minHeight: "300px" }}
    />
  );
}
