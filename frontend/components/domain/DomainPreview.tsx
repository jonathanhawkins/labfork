"use client";

import { useRef, useEffect, useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import * as THREE from "three";

/**
 * Props for DomainPreview component
 */
export interface DomainPreviewProps {
  /** Primary color (hex string) */
  primaryColor: string;
  /** Accent color (hex string) */
  accentColor: string;
  /** Background style */
  backgroundStyle?: "sky" | "grid" | "gradient" | "particles" | "minimal";
  /** Props to show in the preview */
  props?: string[];
  /** Whether to animate the scene */
  animated?: boolean;
  /** Width of the preview */
  width?: number | string;
  /** Height of the preview */
  height?: number | string;
  /** Custom class name */
  className?: string;
}

/**
 * Create a simple cube prop
 */
function createCubeProp(
  position: [number, number, number],
  color: number,
  scale: number = 1
): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(0.3 * scale, 0.3 * scale, 0.3 * scale);
  const material = new THREE.MeshToonMaterial({
    color,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.castShadow = true;
  return mesh;
}

/**
 * Create a cylinder prop
 */
function createCylinderProp(
  position: [number, number, number],
  color: number,
  scale: number = 1
): THREE.Mesh {
  const geometry = new THREE.CylinderGeometry(
    0.15 * scale,
    0.15 * scale,
    0.4 * scale,
    12
  );
  const material = new THREE.MeshToonMaterial({
    color,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.castShadow = true;
  return mesh;
}

/**
 * Create a sphere prop
 */
function createSphereProp(
  position: [number, number, number],
  color: number,
  scale: number = 1
): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(0.2 * scale, 16, 16);
  const material = new THREE.MeshToonMaterial({
    color,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.castShadow = true;
  return mesh;
}

/**
 * Create floating particles
 */
function createParticles(count: number, color: number): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 3;
    positions[i * 3 + 1] = Math.random() * 2;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 3;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color,
    size: 0.03,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending,
  });

  return new THREE.Points(geometry, material);
}

/**
 * DomainPreview - Mini 3D scene preview for domain cards
 *
 * A lightweight Three.js preview showing domain colors and simplified props.
 */
export function DomainPreview({
  primaryColor,
  accentColor,
  backgroundStyle = "sky",
  props = [],
  animated = true,
  width = "100%",
  height = 120,
  className,
}: DomainPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const animationRef = useRef<number>(0);
  const propsGroupRef = useRef<THREE.Group | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Convert hex colors to numbers
  const primaryNum = useMemo(
    () => parseInt(primaryColor.replace("#", ""), 16),
    [primaryColor]
  );
  const accentNum = useMemo(
    () => parseInt(accentColor.replace("#", ""), 16),
    [accentColor]
  );

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    const containerWidth = rect.width || 200;
    const containerHeight =
      typeof height === "number" ? height : rect.height || 120;

    // Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Background based on style
    switch (backgroundStyle) {
      case "sky":
        scene.background = new THREE.Color(0x0a0a1a);
        scene.fog = new THREE.Fog(0x0a0a1a, 3, 8);
        break;
      case "grid":
        scene.background = new THREE.Color(0x0d0d0d);
        break;
      case "gradient":
        scene.background = new THREE.Color(0x0f0f1a);
        break;
      case "particles":
        scene.background = new THREE.Color(0x050510);
        break;
      case "minimal":
      default:
        scene.background = new THREE.Color(0x0a0a0a);
    }

    // Camera
    const camera = new THREE.PerspectiveCamera(
      50,
      containerWidth / containerHeight,
      0.1,
      100
    );
    camera.position.set(2, 1.5, 2);
    camera.lookAt(0, 0.3, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    renderer.setSize(containerWidth, containerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(3, 4, 2);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 256;
    mainLight.shadow.mapSize.height = 256;
    scene.add(mainLight);

    // Accent light using domain color
    const accentLight = new THREE.PointLight(primaryNum, 0.5, 5);
    accentLight.position.set(-1, 1, 1);
    scene.add(accentLight);

    // Ground plane
    const groundGeometry = new THREE.PlaneGeometry(10, 10);
    const groundMaterial = new THREE.MeshToonMaterial({
      color: 0x1a1a1a,
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0;
    ground.receiveShadow = true;
    scene.add(ground);

    // Grid helper for grid style
    if (backgroundStyle === "grid") {
      const gridHelper = new THREE.GridHelper(10, 20, primaryNum, 0x222222);
      gridHelper.position.y = 0.01;
      scene.add(gridHelper);
    }

    // Props group
    const propsGroup = new THREE.Group();
    propsGroupRef.current = propsGroup;

    // Add simplified props
    const propPositions: [number, number, number][] = [
      [-0.5, 0.2, 0],
      [0, 0.2, -0.3],
      [0.5, 0.3, 0.2],
      [0.2, 0.15, 0.5],
      [-0.3, 0.25, 0.4],
    ];

    const propsToAdd = Math.min(props.length || 3, 5);
    for (let i = 0; i < propsToAdd; i++) {
      const pos = propPositions[i];
      const isAccent = i % 2 === 1;

      // Alternate between different shapes
      let mesh: THREE.Mesh;
      switch (i % 3) {
        case 0:
          mesh = createCubeProp(pos, isAccent ? accentNum : primaryNum, 0.8);
          break;
        case 1:
          mesh = createCylinderProp(pos, isAccent ? primaryNum : accentNum, 0.8);
          break;
        default:
          mesh = createSphereProp(pos, isAccent ? accentNum : primaryNum, 0.7);
      }

      propsGroup.add(mesh);
    }

    scene.add(propsGroup);

    // Add particles for particles style
    if (backgroundStyle === "particles" || backgroundStyle === "sky") {
      const particles = createParticles(50, accentNum);
      particlesRef.current = particles;
      scene.add(particles);
    }

    setIsReady(true);

    // Cleanup
    return () => {
      cancelAnimationFrame(animationRef.current);

      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
        rendererRef.current.dispose();
      }

      // Dispose geometries and materials
      scene.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
    };
  }, [backgroundStyle, height]); // Only reinit on style change

  // Update colors when they change
  useEffect(() => {
    if (!isReady || !propsGroupRef.current) return;

    propsGroupRef.current.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshToonMaterial) {
        // Keep alternating pattern
        const index = propsGroupRef.current!.children.indexOf(child);
        const isAccent = index % 2 === 1;
        child.material.color.setHex(isAccent ? accentNum : primaryNum);
      }
    });

    if (particlesRef.current) {
      const material = particlesRef.current.material as THREE.PointsMaterial;
      material.color.setHex(accentNum);
    }
  }, [primaryNum, accentNum, isReady]);

  // Animation loop
  useEffect(() => {
    if (!isReady || !animated) return;

    let time = 0;

    const animate = () => {
      time += 0.01;

      // Rotate props group
      if (propsGroupRef.current) {
        propsGroupRef.current.rotation.y = time * 0.2;

        // Bob props up and down
        propsGroupRef.current.children.forEach((child, i) => {
          if (child instanceof THREE.Mesh) {
            child.position.y =
              child.userData.baseY || 0.2 + Math.sin(time * 2 + i) * 0.05;
          }
        });
      }

      // Animate particles
      if (particlesRef.current) {
        const positions = particlesRef.current.geometry.attributes.position
          .array as Float32Array;
        for (let i = 0; i < positions.length; i += 3) {
          positions[i + 1] += 0.003;
          if (positions[i + 1] > 2) {
            positions[i + 1] = 0;
          }
        }
        particlesRef.current.geometry.attributes.position.needsUpdate = true;
      }

      // Render
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [isReady, animated]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current)
        return;

      const rect = containerRef.current.getBoundingClientRect();
      const containerWidth = rect.width || 200;
      const containerHeight =
        typeof height === "number" ? height : rect.height || 120;

      cameraRef.current.aspect = containerWidth / containerHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(containerWidth, containerHeight);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [height]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-hidden rounded-lg",
        !isReady && "bg-background-card",
        className
      )}
      style={{
        width,
        height,
      }}
    >
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin"
            style={{ borderColor: `${primaryColor}40`, borderTopColor: "transparent" }}
          />
        </div>
      )}
    </div>
  );
}

export default DomainPreview;
