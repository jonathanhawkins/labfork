/**
 * FloatingFireflies
 *
 * Ambient canvas animation of floating firefly particles.
 * Creates a magical atmosphere for the Firefly Network hero section.
 */

"use client";

import { useEffect, useRef } from "react";

interface Firefly {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  brightness: number;
  phase: number;
  speed: number;
}

interface FloatingFirefliesProps {
  count?: number;
  className?: string;
}

export function FloatingFireflies({
  count = 40,
  className = "",
}: FloatingFirefliesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const firefliesRef = useRef<Firefly[]>([]);
  const animationRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Initialize fireflies
    const rect = canvas.getBoundingClientRect();
    firefliesRef.current = Array.from({ length: count }, () => ({
      x: Math.random() * rect.width,
      y: Math.random() * rect.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: 1.5 + Math.random() * 2.5,
      brightness: Math.random(),
      phase: Math.random() * Math.PI * 2,
      speed: 0.5 + Math.random() * 1.5,
    }));

    let time = 0;

    const animate = () => {
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      time += 0.016;

      firefliesRef.current.forEach((firefly) => {
        // Update position with gentle drift
        firefly.x += firefly.vx;
        firefly.y += firefly.vy;

        // Add slight wandering
        firefly.vx += (Math.random() - 0.5) * 0.02;
        firefly.vy += (Math.random() - 0.5) * 0.02;

        // Limit velocity
        const maxSpeed = 0.8;
        firefly.vx = Math.max(-maxSpeed, Math.min(maxSpeed, firefly.vx));
        firefly.vy = Math.max(-maxSpeed, Math.min(maxSpeed, firefly.vy));

        // Wrap around edges
        if (firefly.x < -20) firefly.x = rect.width + 20;
        if (firefly.x > rect.width + 20) firefly.x = -20;
        if (firefly.y < -20) firefly.y = rect.height + 20;
        if (firefly.y > rect.height + 20) firefly.y = -20;

        // Pulsing brightness (like real fireflies)
        const pulse = Math.sin(time * firefly.speed + firefly.phase);
        const brightness = 0.3 + 0.7 * Math.max(0, pulse);
        firefly.brightness = brightness;

        // Draw glow
        const glowRadius = firefly.size * 8 * brightness;
        const gradient = ctx.createRadialGradient(
          firefly.x,
          firefly.y,
          0,
          firefly.x,
          firefly.y,
          glowRadius
        );
        gradient.addColorStop(0, `rgba(255, 200, 80, ${0.6 * brightness})`);
        gradient.addColorStop(0.3, `rgba(255, 180, 60, ${0.3 * brightness})`);
        gradient.addColorStop(1, "rgba(255, 150, 50, 0)");

        ctx.beginPath();
        ctx.arc(firefly.x, firefly.y, glowRadius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw core
        ctx.beginPath();
        ctx.arc(firefly.x, firefly.y, firefly.size * brightness, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 240, 180, ${brightness})`;
        ctx.fill();
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationRef.current);
    };
  }, [count]);

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{ width: "100%", height: "100%" }}
    />
  );
}

export default FloatingFireflies;
