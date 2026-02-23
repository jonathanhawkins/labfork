"use client";

// WaterHarvesterScene - React wrapper for the 3D Water Harvester visualization
// Handles Three.js scene setup, OrbitControls, and mobile touch support

import React, { useRef, useEffect, useCallback, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import {
  createWaterHarvester3D,
  animateWaterHarvester3D,
  disposeWaterHarvester3D,
  updateParticlesFromGPU,
  WaterHarvester3DProps,
  WaterHarvester3DRefs,
} from "./lab/props/domains/WaterHarvester3D";
import { useWarpParticles } from "@/lib/simulations/use-warp-particles";

interface WaterHarvesterSceneProps {
  // Design parameters
  sorbentWidth: number;
  sorbentDepth: number;
  mirrorCount: number;
  mirrorAngle: number;
  surfacePattern: "beetle" | "flat";
  humidity: number;

  // Simulation results
  dailyYield?: number;
  efficiency?: number;
  peakTemp?: number;
  collectionRate?: number;

  // State
  isSimulating?: boolean;

  // Optional class for sizing
  className?: string;
}

export default function WaterHarvesterScene({
  sorbentWidth,
  sorbentDepth,
  mirrorCount,
  mirrorAngle,
  surfacePattern,
  humidity,
  dailyYield,
  efficiency,
  peakTemp,
  collectionRate,
  isSimulating,
  className = "",
}: WaterHarvesterSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationRef = useRef<number>(0);
  const harvesterRef = useRef<WaterHarvester3DRefs | null>(null);
  const timeRef = useRef(0);
  const propsRef = useRef<WaterHarvester3DProps>({
    sorbentWidth,
    sorbentDepth,
    mirrorCount,
    mirrorAngle,
    surfacePattern,
    humidity,
    dailyYield,
    efficiency,
    peakTemp,
    collectionRate,
    isSimulating,
  });

  // Track if we need to rebuild (structural changes)
  const [needsRebuild, setNeedsRebuild] = useState(false);

  // Stream GPU particle data from Warp server (always enabled for real-time visualization)
  const { particles: gpuParticles, isConnected: gpuConnected } = useWarpParticles({
    enabled: true, // Always stream for live visualization
    pollInterval: 50, // ~20 FPS for smooth animation without overloading
  });

  // Store GPU particles in ref for animation loop access
  const gpuParticlesRef = useRef<number[][] | null>(null);
  const [particleCount, setParticleCount] = useState(0);
  const [updateCount, setUpdateCount] = useState(0);

  useEffect(() => {
    if (gpuParticles?.positions) {
      gpuParticlesRef.current = gpuParticles.positions;
      setParticleCount(gpuParticles.positions.length);
      setUpdateCount(prev => prev + 1);
    }
  }, [gpuParticles]);

  // Update props ref without triggering rebuild
  useEffect(() => {
    propsRef.current = {
      sorbentWidth,
      sorbentDepth,
      mirrorCount,
      mirrorAngle,
      surfacePattern,
      humidity,
      dailyYield,
      efficiency,
      peakTemp,
      collectionRate,
      isSimulating,
    };
  }, [
    sorbentWidth,
    sorbentDepth,
    mirrorCount,
    mirrorAngle,
    surfacePattern,
    humidity,
    dailyYield,
    efficiency,
    peakTemp,
    collectionRate,
    isSimulating,
  ]);

  // Check for structural changes that require rebuild
  useEffect(() => {
    if (harvesterRef.current) {
      // These changes require a full rebuild
      setNeedsRebuild(true);
    }
  }, [sorbentWidth, sorbentDepth, mirrorCount, mirrorAngle, surfacePattern]);

  // Handle resize
  const handleResize = useCallback(() => {
    if (!containerRef.current || !rendererRef.current || !cameraRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(width, height);
  }, []);

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100);
    camera.position.set(3, 2.5, 3);
    camera.lookAt(0, 0.5, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls - mobile friendly
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 2;
    controls.maxDistance = 10;
    controls.maxPolarAngle = Math.PI / 2 + 0.2; // Slight ground view allowed
    controls.minPolarAngle = 0.2;
    controls.target.set(0, 0.5, 0);
    controls.enablePan = true;
    controls.panSpeed = 0.5;

    // Touch settings for mobile
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };

    controlsRef.current = controls;

    // Create harvester
    const harvester = createWaterHarvester3D(propsRef.current);
    scene.add(harvester.group);
    scene.fog = harvester.environment.fog;
    harvesterRef.current = harvester;

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      timeRef.current += 0.016; // ~60fps

      // Animate harvester
      if (harvesterRef.current) {
        animateWaterHarvester3D(harvesterRef.current, timeRef.current, propsRef.current);

        // Update with real GPU particle positions when available
        if (gpuParticlesRef.current && gpuParticlesRef.current.length > 0) {
          updateParticlesFromGPU(harvesterRef.current, gpuParticlesRef.current);
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // Resize handler
    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationRef.current);

      if (harvesterRef.current) {
        disposeWaterHarvester3D(harvesterRef.current);
        scene.remove(harvesterRef.current.group);
      }

      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle rebuild when structure changes
  useEffect(() => {
    if (!needsRebuild || !sceneRef.current || !harvesterRef.current) return;

    // Remove old harvester
    disposeWaterHarvester3D(harvesterRef.current);
    sceneRef.current.remove(harvesterRef.current.group);

    // Create new harvester with updated props
    const newHarvester = createWaterHarvester3D(propsRef.current);
    sceneRef.current.add(newHarvester.group);
    sceneRef.current.fog = newHarvester.environment.fog;
    harvesterRef.current = newHarvester;

    setNeedsRebuild(false);
  }, [needsRebuild]);

  return (
    <div className="relative w-full h-full">
      {/* GPU Connection Status Badge - Shows live particle count as proof */}
      {gpuConnected && (
        <div className="absolute top-2 left-2 z-10 flex flex-col gap-1 px-2 py-1.5 bg-green-500/20 border border-green-500/50 rounded text-xs text-green-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="font-medium">RTX 4090 GPU</span>
          </div>
          <div className="text-[10px] opacity-80">
            {particleCount.toLocaleString()} particles • {updateCount} updates
          </div>
        </div>
      )}
      <div
        ref={containerRef}
        className={`w-full h-full ${className}`}
        style={{
          touchAction: "none", // Prevent browser handling of touch
          userSelect: "none",
        }}
        aria-label="3D Water Harvester Simulation - Interactive visualization showing day/night water collection cycle"
        role="img"
      />
    </div>
  );
}
