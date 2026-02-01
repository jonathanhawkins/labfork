/**
 * HeroSection
 *
 * Landing page hero with animated background, headline, and CTAs.
 */

"use client";

import React, { useEffect, useRef } from "react";
import Link from "next/link";

interface HeroSectionProps {
  labCount?: number;
  discoveryCount?: number;
}

export function HeroSection({
  labCount = 1247,
  discoveryCount = 89,
}: HeroSectionProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animated particle background
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      color: string;
    }> = [];

    const colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"];

    const resize = () => {
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    const initParticles = () => {
      particles.length = 0;
      const count = Math.floor((canvas.offsetWidth * canvas.offsetHeight) / 15000);
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.offsetWidth,
          y: Math.random() * canvas.offsetHeight,
          vx: (Math.random() - 0.5) * 0.5,
          vy: (Math.random() - 0.5) * 0.5,
          size: Math.random() * 2 + 1,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);

      // Update and draw particles
      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around
        if (p.x < 0) p.x = canvas.offsetWidth;
        if (p.x > canvas.offsetWidth) p.x = 0;
        if (p.y < 0) p.y = canvas.offsetHeight;
        if (p.y > canvas.offsetHeight) p.y = 0;

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color + "40";
        ctx.fill();

        // Draw connections
        particles.slice(i + 1).forEach((p2) => {
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = p.color + Math.floor((1 - dist / 100) * 30).toString(16).padStart(2, "0");
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      animationId = requestAnimationFrame(animate);
    };

    resize();
    initParticles();
    animate();

    window.addEventListener("resize", () => {
      resize();
      initParticles();
    });

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      {/* Animated background */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full opacity-60"
      />

      {/* Gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-transparent to-purple-600/10" />
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-slate-950 to-transparent" />

      {/* Content */}
      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
        {/* Featured Project Badge */}
        <Link
          href="/projects/firefly-network"
          className="inline-flex items-center gap-3 px-5 py-3 rounded-full bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 mb-8 hover:from-amber-500/30 hover:to-orange-500/30 transition-all group"
        >
          <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
          <span className="text-sm text-amber-300">
            Featured Project: The Firefly Network
          </span>
          <span className="text-amber-400/60 group-hover:text-amber-400 transition-colors">
            - Bringing light to 1B people
          </span>
          <svg
            className="w-4 h-4 text-amber-400 group-hover:translate-x-1 transition-transform"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </Link>

        {/* Active Labs Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8 ml-3">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-sm text-gray-300">
            {labCount.toLocaleString()} labs actively researching
          </span>
        </div>

        {/* Headline */}
        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
          <span className="bg-gradient-to-r from-blue-400 to-purple-400 text-transparent bg-clip-text">
            LabFork
          </span>
          <br />
          <span className="text-3xl md:text-5xl text-gray-300">
            Fork. Watch. Discover.
          </span>
        </h1>

        {/* Subheadline */}
        <p className="text-xl md:text-2xl text-gray-400 mb-10 max-w-3xl mx-auto leading-relaxed">
          Fork research labs. Watch AI agents implement papers in real-time.
          Discover synergies across domains. Collaborate globally.
        </p>

        {/* Stats */}
        <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-8 mb-10 px-4">
          <div className="text-center min-w-[80px]">
            <div className="text-2xl sm:text-3xl font-bold text-white">{labCount.toLocaleString()}</div>
            <div className="text-xs sm:text-sm text-gray-500">Active Labs</div>
          </div>
          <div className="hidden sm:block w-px h-12 bg-gray-700" />
          <div className="text-center min-w-[80px]">
            <div className="text-2xl sm:text-3xl font-bold text-white">{discoveryCount}</div>
            <div className="text-xs sm:text-sm text-gray-500">Synergies Found</div>
          </div>
          <div className="hidden sm:block w-px h-12 bg-gray-700" />
          <div className="text-center min-w-[80px]">
            <div className="text-2xl sm:text-3xl font-bold text-white">9</div>
            <div className="text-xs sm:text-sm text-gray-500">Domains</div>
          </div>
        </div>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 px-4">
          <Link
            href="/lab/new"
            className="w-full sm:w-auto px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all shadow-lg shadow-blue-600/25 hover:shadow-blue-600/40 text-center min-h-[56px] flex items-center justify-center gap-2"
          >
            <span>Fork a Lab</span>
            <span className="text-blue-200 text-sm font-normal">in 60 seconds</span>
          </Link>
          <Link
            href="/explore"
            className="w-full sm:w-auto px-8 py-4 bg-white/5 text-white font-semibold rounded-xl border border-white/10 hover:bg-white/10 transition-all text-center min-h-[56px] flex items-center justify-center"
          >
            Explore Public Labs
          </Link>
        </div>

        {/* Mobile-friendly quick action */}
        <div className="mt-6 sm:hidden">
          <Link
            href="/contribute"
            className="inline-flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
          >
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Or contribute your device's GPU power
          </Link>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <svg
            className="w-6 h-6 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
        </div>
      </div>
    </section>
  );
}

export default HeroSection;
