"use client";

// MOFHarvesterScene - React wrapper for physics-accurate MOF water harvester
// Based on MIT/Berkeley MOF-801 design

import React, { useRef, useEffect, useCallback, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import {
  createMOFHarvester,
  animateMOFHarvester,
  disposeMOFHarvester,
  updateParticlesFromGPU,
  updateMirrorCount,
  MOFHarvesterProps,
  MOFHarvesterRefs,
} from "./lab/props/domains/MOFWaterHarvester3D";
import { useWarpParticles } from "@/lib/simulations/use-warp-particles";

interface MOFHarvesterSceneProps {
  // Design parameters
  humidity: number;
  mirrorCount?: number;       // Number of solar concentrator mirrors (2-8), default 4

  // Simulation results
  dailyYield?: number;
  sorbentTemp?: number;

  // Optional class
  className?: string;
}

export default function MOFHarvesterScene({
  humidity,
  mirrorCount = 4,
  dailyYield,
  sorbentTemp,
  className = "",
}: MOFHarvesterSceneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationRef = useRef<number>(0);
  const harvesterRef = useRef<MOFHarvesterRefs | null>(null);
  const timeRef = useRef(30); // Start at ~noon (0.5 * 60s cycle)

  // Time of day cycles automatically (0-1)
  const [timeOfDay, setTimeOfDay] = useState(0.5); // Start at noon

  // Props ref for animation loop
  const propsRef = useRef<MOFHarvesterProps>({
    domeRadius: 60,
    sorbentMass: 1,
    humidity,
    mirrorCount,
    timeOfDay,
    dailyYield,
    sorbentTemp,
  });

  // GPU particle streaming
  const { particles: gpuParticles, isConnected: gpuConnected } = useWarpParticles({
    enabled: true,
    pollInterval: 50,
  });

  const gpuParticlesRef = useRef<number[][] | null>(null);
  const [particleCount, setParticleCount] = useState(0);
  const [updateCount, setUpdateCount] = useState(0);

  useEffect(() => {
    if (gpuParticles?.positions) {
      gpuParticlesRef.current = gpuParticles.positions;
      setParticleCount(gpuParticles.positions.length);
      setUpdateCount((prev) => prev + 1);
    }
  }, [gpuParticles]);

  // Update props
  useEffect(() => {
    propsRef.current = {
      domeRadius: 60,
      sorbentMass: 1,
      humidity,
      mirrorCount,
      timeOfDay,
      dailyYield,
      sorbentTemp,
    };
  }, [humidity, mirrorCount, timeOfDay, dailyYield, sorbentTemp]);

  // Hot-swap mirrors when mirror count changes
  useEffect(() => {
    if (harvesterRef.current) {
      updateMirrorCount(harvesterRef.current, mirrorCount);
    }
  }, [mirrorCount]);

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
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Camera - pulled back to show mirrors around dome
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(2.0, 1.0, 2.0);
    camera.lookAt(0, -0.1, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.5;
    controls.maxDistance = 6;
    controls.target.set(0, -0.1, 0);
    controlsRef.current = controls;

    // Create harvester
    const harvester = createMOFHarvester(propsRef.current);
    scene.add(harvester.group);

    // Add lights to scene
    scene.add(harvester.environment.sun);
    scene.add(harvester.environment.ambient);
    scene.add(harvester.environment.hemisphere);
    scene.add(harvester.environment.fill);

    harvesterRef.current = harvester;

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      timeRef.current += 0.016;

      // Cycle time of day (complete cycle every 60 seconds)
      const newTimeOfDay = (timeRef.current * 0.0167) % 1;
      propsRef.current.timeOfDay = newTimeOfDay;

      // Update time state for UI (throttled)
      if (Math.floor(timeRef.current * 10) % 10 === 0) {
        setTimeOfDay(newTimeOfDay);
      }

      // Animate
      if (harvesterRef.current) {
        animateMOFHarvester(harvesterRef.current, timeRef.current, propsRef.current);

        // Apply GPU particles if available
        if (gpuParticlesRef.current && gpuParticlesRef.current.length > 0) {
          updateParticlesFromGPU(harvesterRef.current, gpuParticlesRef.current);
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationRef.current);

      if (harvesterRef.current) {
        disposeMOFHarvester(harvesterRef.current);
        scene.remove(harvesterRef.current.group);
      }

      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Get phase label - prefer GPU simulation data, fallback to local calculation
  const getPhaseLabel = () => {
    // Use GPU simulation phase if available
    if (gpuParticles?.phase) {
      const phaseLabels: Record<string, string> = {
        adsorbing: "NIGHT: Adsorbing moisture from air",
        heating: "DAWN: Heating sorbent with sunlight",
        releasing: "DAY: Releasing water vapor",
        condensing: "DAY: Condensing in dish",
        dripping: "DAY: Water dripping to collector",
      };
      return phaseLabels[gpuParticles.phase] || `Phase: ${gpuParticles.phase}`;
    }

    // Fallback to local calculation
    const isNight = timeOfDay < 0.25 || timeOfDay > 0.75;
    if (isNight) return "NIGHT: Adsorbing moisture";
    if (timeOfDay < 0.35) return "DAWN: Heating sorbent";
    if (timeOfDay < 0.5) return "MORNING: Releasing vapor";
    if (timeOfDay < 0.65) return "NOON: Condensing in dish";
    return "AFTERNOON: Collecting water";
  };

  const getTimeLabel = () => {
    // Use GPU simulation time if available
    if (gpuParticles?.timeLabel) {
      return gpuParticles.timeLabel;
    }

    // Fallback to local calculation
    const hours = Math.floor(timeOfDay * 24);
    const period = hours < 12 ? "AM" : "PM";
    const displayHour = hours % 12 || 12;
    return `${displayHour}:00 ${period}`;
  };

  const getSorbentTemp = () => {
    if (gpuParticles?.sorbentTemp) {
      return `${Math.round(gpuParticles.sorbentTemp)}°C`;
    }
    return sorbentTemp ? `${Math.round(sorbentTemp)}°C` : "25°C";
  };

  return (
    <div className="relative w-full h-full">
      {/* Info Panel */}
      <div className="absolute top-2 left-2 z-10 flex flex-col gap-2">
        {/* GPU Status */}
        {gpuConnected && (
          <div className="flex flex-col gap-1 px-2 py-1.5 bg-green-500/20 border border-green-500/50 rounded text-xs text-green-400">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              <span className="font-medium">RTX 4090 GPU</span>
            </div>
            <div className="text-[10px] opacity-80">
              {particleCount.toLocaleString()} particles • {updateCount} updates
            </div>
          </div>
        )}

        {/* Cycle Phase */}
        <div className="px-2 py-1.5 bg-slate-800/80 border border-slate-600/50 rounded text-xs">
          <div className="text-slate-400 text-[10px] uppercase tracking-wide">Cycle Phase</div>
          <div className="text-white font-medium">{getPhaseLabel()}</div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-slate-400 text-[10px]">{getTimeLabel()}</span>
            <span className="text-orange-400 text-[10px]">Sorbent: {getSorbentTemp()}</span>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="absolute bottom-2 left-2 z-10 px-2 py-1.5 bg-slate-800/80 border border-slate-600/50 rounded text-[10px] text-slate-300">
        <div className="font-medium text-white mb-1">MOF Water Harvester</div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-gray-600" />
          <span>MOF Sorbent (adsorbs H₂O)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-blue-400/60" />
          <span>Water Vapor (rising)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-cyan-400" />
          <span>Condensation (on dish)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-blue-500" />
          <span>Water Droplets (falling)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-slate-300" />
          <span>Solar Mirrors (concentrating)</span>
        </div>
      </div>

      {/* Stats */}
      {dailyYield !== undefined && (
        <div className="absolute top-2 right-2 z-10 px-2 py-1.5 bg-slate-800/80 border border-slate-600/50 rounded text-xs">
          <div className="text-slate-400 text-[10px] uppercase">Daily Yield</div>
          <div className="text-2xl font-bold text-blue-400">{dailyYield.toFixed(1)}L</div>
          <div className="text-slate-400 text-[10px]">of 2.8L max (MIT spec)</div>
        </div>
      )}

      <div
        ref={containerRef}
        className={`w-full h-full ${className}`}
        style={{ touchAction: "none", userSelect: "none" }}
        aria-label="3D MOF Water Harvester - Day/night water collection cycle"
        role="img"
      />
    </div>
  );
}
