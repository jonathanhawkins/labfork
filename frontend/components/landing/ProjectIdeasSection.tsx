/**
 * ProjectIdeasSection
 *
 * Showcases inspiring real-world projects that visitors
 * can build using the LabFork platform.
 */

"use client";

import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";

export interface ProjectIdea {
  slug: string;
  emoji: string;
  name: string;
  description: string;
  impact: string;
  cost: string;
  domain: string;
  domainSlug: string;
  color: string;
  accentColor: string;
  featured?: boolean;
}

export const projectIdeas: ProjectIdea[] = [
  {
    slug: "firefly-network",
    emoji: "💡",
    name: "The Firefly Network",
    description: "Solar-powered mesh lights bringing illumination to communities worldwide",
    impact: "1B people",
    cost: "<$25",
    domain: "Firefly Network",
    domainSlug: "firefly-network",
    color: "#FFB84D",
    accentColor: "#FF6B35",
    featured: true,
  },
  {
    slug: "atmospheric-water-harvester",
    emoji: "🌊",
    name: "Atmospheric Water Harvester",
    description: "Extract clean water from air in any climate",
    impact: "2B people",
    cost: "<$100",
    domain: "Climate Modeling",
    domainSlug: "climate-modeling",
    color: "#0ea5e9",
    accentColor: "#22c55e",
  },
  {
    slug: "ai-micro-housing-factory",
    emoji: "🏠",
    name: "AI Micro-Housing Factory",
    description: "Automated robots build $5K homes in 48 hours",
    impact: "Billions",
    cost: "$5K/home",
    domain: "Robotics ML",
    domainSlug: "robotics-ml",
    color: "#f97316",
    accentColor: "#06b6d4",
  },
  {
    slug: "precision-micro-farming",
    emoji: "🌱",
    name: "Precision Micro-Farming",
    description: "Closet-sized farm feeds 4 people year-round",
    impact: "Food security",
    cost: "<$500",
    domain: "Computer Vision",
    domainSlug: "computer-vision",
    color: "#22c55e",
    accentColor: "#84cc16",
  },
  {
    slug: "community-mesh-power-grid",
    emoji: "⚡",
    name: "Community Mesh Power Grid",
    description: "Decentralized solar micro-grids with AI balancing",
    impact: "Energy independence",
    cost: "Free to join",
    domain: "Optimization",
    domainSlug: "quant-trading",
    color: "#eab308",
    accentColor: "#f59e0b",
  },
  {
    slug: "open-medical-diagnostic-ai",
    emoji: "🧠",
    name: "Open Medical Diagnostic AI",
    description: "Smartphone diagnoses 1000+ conditions offline",
    impact: "Healthcare for billions",
    cost: "Free",
    domain: "Computer Vision",
    domainSlug: "computer-vision",
    color: "#ec4899",
    accentColor: "#f43f5e",
  },
  {
    slug: "plastic-to-fuel-converter",
    emoji: "♻️",
    name: "Plastic-to-Fuel Converter",
    description: "Turn waste plastic into diesel at home",
    impact: "Clean oceans",
    cost: "<$2K",
    domain: "Chemistry ML",
    domainSlug: "drug-discovery",
    color: "#14b8a6",
    accentColor: "#10b981",
  },
  {
    slug: "disaster-response-drones",
    emoji: "🚁",
    name: "Disaster Response Drones",
    description: "Autonomous swarms find survivors in minutes",
    impact: "Save lives",
    cost: "<$500/drone",
    domain: "Robotics ML",
    domainSlug: "robotics-ml",
    color: "#ef4444",
    accentColor: "#f97316",
  },
  {
    slug: "regenerative-agriculture-ai",
    emoji: "🌾",
    name: "Regenerative Agriculture AI",
    description: "Smartphone app guides soil regeneration",
    impact: "Reverse climate change",
    cost: "Free",
    domain: "Agriculture ML",
    domainSlug: "climate-modeling",
    color: "#84cc16",
    accentColor: "#22c55e",
  },
  {
    slug: "local-manufacturing-platform",
    emoji: "🏭",
    name: "Local Manufacturing Platform",
    description: "Make anything locally with CNC + AI optimization",
    impact: "Local production",
    cost: "Design sharing",
    domain: "Generative Design",
    domainSlug: "robotics-ml",
    color: "#6366f1",
    accentColor: "#8b5cf6",
  },
  {
    slug: "ai-tutor-for-every-child",
    emoji: "📚",
    name: "AI Tutor for Every Child",
    description: "Personalized education that adapts to each kid",
    impact: "Education for all",
    cost: "Free",
    domain: "NLP Research",
    domainSlug: "nlp-research",
    color: "#8b5cf6",
    accentColor: "#a855f7",
  },
];

interface ProjectCardProps {
  project: ProjectIdea;
  index: number;
}

function ProjectCard({ project, index }: ProjectCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const isFeatured = project.featured;

  return (
    <div
      className="relative group"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Featured glow effect */}
      {isFeatured && (
        <div
          className="absolute -inset-1 rounded-2xl opacity-50 blur-xl animate-pulse"
          style={{
            background: `linear-gradient(135deg, ${project.color}40, ${project.accentColor}40)`,
          }}
        />
      )}

      <div
        className={`relative h-full rounded-2xl border overflow-hidden transition-all duration-300 ${
          isFeatured
            ? "border-amber-500/40"
            : isHovered
            ? "border-white/20 -translate-y-2 shadow-2xl"
            : "border-white/10"
        }`}
        style={{
          background: isFeatured
            ? `linear-gradient(135deg, ${project.color}15, ${project.accentColor}10)`
            : isHovered
            ? `linear-gradient(135deg, ${project.color}20, ${project.accentColor}10)`
            : "rgba(255, 255, 255, 0.03)",
          boxShadow: isFeatured
            ? `0 25px 50px -12px ${project.color}40`
            : isHovered
            ? `0 25px 50px -12px ${project.color}30`
            : undefined,
        }}
      >
        {/* Featured badge */}
        {isFeatured && (
          <div className="absolute top-3 right-3 z-10 px-2 py-1 rounded-full bg-amber-500/20 border border-amber-500/30">
            <span className="text-xs font-semibold text-amber-400">Featured</span>
          </div>
        )}

        {/* Gradient top accent */}
        <div
          className="absolute top-0 left-0 right-0 h-1"
          style={{
            background: `linear-gradient(90deg, ${project.color}, ${project.accentColor})`,
            opacity: isFeatured ? 1 : isHovered ? 1 : 0.6,
          }}
        />

        <div className="p-5">
          {/* Emoji & Name */}
          <div className="flex items-start gap-3 mb-3">
            <span className="text-3xl" role="img" aria-label={project.name}>
              {project.emoji}
            </span>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-white text-sm leading-tight mb-1">
                {project.name}
              </h3>
              <p className="text-xs text-gray-400 line-clamp-2">
                {project.description}
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap gap-2 mb-4">
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-white/5 text-xs">
              <svg
                className="w-3 h-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                style={{ color: project.color }}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                />
              </svg>
              <span className="text-gray-300">{project.impact}</span>
            </div>
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-white/5 text-xs">
              <svg
                className="w-3 h-3"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                style={{ color: project.accentColor }}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="text-gray-300">{project.cost}</span>
            </div>
          </div>

          {/* Domain Badge */}
          <div className="flex items-center justify-between">
            <span
              className="text-xs px-2 py-0.5 rounded-full"
              style={{
                backgroundColor: `${project.color}20`,
                color: project.color,
              }}
            >
              {project.domain}
            </span>
          </div>

          {/* Hover CTA */}
          <div
            className={`absolute inset-0 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm transition-opacity duration-200 ${
              isHovered ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
          >
            <Link
              href={`/lab/new?project=${project.slug}&domain=${project.domainSlug}`}
              className="flex items-center gap-2 px-4 py-2 rounded-full font-medium text-white text-sm transition-all"
              style={{
                background: `linear-gradient(135deg, ${project.color}, ${project.accentColor})`,
              }}
            >
              Start Building
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 8l4 4m0 0l-4 4m4-4H3"
                />
              </svg>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ProjectIdeasSectionProps {
  projects?: ProjectIdea[];
}

export function ProjectIdeasSection({
  projects = projectIdeas,
}: ProjectIdeasSectionProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const updateScrollButtons = () => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
      setCanScrollLeft(scrollLeft > 0);
      setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 10);
    }
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.addEventListener("scroll", updateScrollButtons);
      updateScrollButtons();
      return () => el.removeEventListener("scroll", updateScrollButtons);
    }
  }, []);

  const scroll = (direction: "left" | "right") => {
    if (scrollRef.current) {
      const scrollAmount = 320;
      scrollRef.current.scrollBy({
        left: direction === "left" ? -scrollAmount : scrollAmount,
        behavior: "smooth",
      });
    }
  };

  return (
    <section className="py-24 bg-slate-950 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 text-emerald-400 text-sm mb-4">
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
            Inspire Your Next Breakthrough
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Build Projects That Change The World
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Use LabFork to accelerate breakthrough projects in any field.
            Fork a lab and get started immediately.
          </p>
        </div>

        {/* Desktop Grid (hidden on mobile) - First row with featured project larger */}
        <div className="hidden lg:grid lg:grid-cols-4 gap-4 mb-4">
          {projects.slice(0, 4).map((project, index) => (
            <ProjectCard key={project.slug} project={project} index={index} />
          ))}
        </div>
        <div className="hidden lg:grid lg:grid-cols-4 gap-4 mb-4">
          {projects.slice(4, 8).map((project, index) => (
            <ProjectCard key={project.slug} project={project} index={index + 4} />
          ))}
        </div>
        <div className="hidden lg:flex lg:justify-center lg:gap-4">
          {projects.slice(8, 11).map((project, index) => (
            <div key={project.slug} className="w-1/4 max-w-[280px]">
              <ProjectCard project={project} index={index + 8} />
            </div>
          ))}
        </div>

        {/* Mobile Carousel (hidden on desktop) */}
        <div className="lg:hidden relative">
          {/* Scroll Buttons */}
          <button
            onClick={() => scroll("left")}
            className={`absolute left-0 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-slate-800/80 backdrop-blur border border-white/10 transition-opacity ${
              canScrollLeft ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
            aria-label="Scroll left"
          >
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15 19l-7-7 7-7"
              />
            </svg>
          </button>
          <button
            onClick={() => scroll("right")}
            className={`absolute right-0 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-slate-800/80 backdrop-blur border border-white/10 transition-opacity ${
              canScrollRight ? "opacity-100" : "opacity-0 pointer-events-none"
            }`}
            aria-label="Scroll right"
          >
            <svg
              className="w-5 h-5 text-white"
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
          </button>

          {/* Scrollable Container */}
          <div
            ref={scrollRef}
            className="flex gap-4 overflow-x-auto scrollbar-hide snap-x snap-mandatory pb-4"
            style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
          >
            {projects.map((project, index) => (
              <div
                key={project.slug}
                className="flex-shrink-0 w-72 snap-center"
              >
                <ProjectCard project={project} index={index} />
              </div>
            ))}
          </div>

          {/* Scroll Indicators */}
          <div className="flex justify-center gap-1.5 mt-4">
            {projects.map((_, index) => (
              <div
                key={index}
                className="w-1.5 h-1.5 rounded-full bg-white/20"
              />
            ))}
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-12">
          <p className="text-gray-500 mb-4">
            Have your own world-changing idea?
          </p>
          <Link
            href="/lab/new"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl font-medium text-white bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 transition-all hover:shadow-lg hover:shadow-emerald-500/25"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
              />
            </svg>
            Start Your Own Project
          </Link>
        </div>
      </div>
    </section>
  );
}

export default ProjectIdeasSection;
