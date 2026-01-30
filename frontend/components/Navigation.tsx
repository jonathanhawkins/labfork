"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import {
  Compass,
  Eye,
  Lightbulb,
  FlaskConical,
  ChevronDown,
  Plus,
  Sliders,
  Rss,
  Target,
  Layers,
  Bot,
  Cpu,
  BarChart3,
} from "lucide-react";

// Primary navigation items (always visible)
const primaryNavItems = [
  { href: "/explore", label: "Explore", icon: Compass, description: "Browse public labs" },
  { href: "/watch", label: "Watch", icon: Eye, description: "Live agent view" },
  { href: "/contribute", label: "Contribute", icon: Cpu, description: "Donate compute power" },
  { href: "/projects/firefly-network", label: "Firefly", icon: Lightbulb, featured: true, description: "Featured project" },
];

// Labs dropdown items
const labsDropdownItems = [
  { href: "/lab/new", label: "Create Lab", icon: Plus, description: "Start a new research lab" },
  { href: "/lab", label: "My Lab", icon: Bot, description: "Admin dashboard" },
  { href: "/labs", label: "Browse Labs", icon: FlaskConical, description: "Explore all labs" },
];

// More dropdown items
const moreDropdownItems = [
  { href: "/analytics", label: "Analytics", icon: BarChart3, description: "Network health & stats" },
  { href: "/demos", label: "Demos", icon: Sliders, description: "Research technique demos" },
  { href: "/feed", label: "Feed", icon: Rss, description: "Activity feed" },
  { href: "/research", label: "Research", icon: Target, description: "Research dashboard" },
  { href: "/domains", label: "Domains", icon: Layers, description: "Browse domains" },
];

interface DropdownProps {
  label: string;
  items: Array<{
    href: string;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    description?: string;
  }>;
  isActive: boolean;
}

function Dropdown({ label, items, isActive }: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={dropdownRef} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-1.5 px-3 py-1.5 text-sm transition-colors rounded-md",
          isActive
            ? "text-foreground-bright"
            : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
        )}
      >
        {isActive && <span className="w-1.5 h-1.5 rounded-full bg-foreground-bright" />}
        <span>{label}</span>
        <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-56 py-2 bg-background-elevated border border-border rounded-lg shadow-lg z-50 animate-in fade-in-0 zoom-in-95 duration-100">
          {items.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsOpen(false)}
                className="flex items-start gap-3 px-3 py-2 hover:bg-accent/50 transition-colors"
              >
                <Icon className="w-4 h-4 mt-0.5 text-muted-foreground" />
                <div>
                  <div className="text-sm text-foreground">{item.label}</div>
                  {item.description && (
                    <div className="text-xs text-muted-foreground">{item.description}</div>
                  )}
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function Navigation() {
  const pathname = usePathname();

  // Check if any labs dropdown item is active
  const isLabsActive = labsDropdownItems.some(item => pathname.startsWith(item.href));
  // Check if any more dropdown item is active
  const isMoreActive = moreDropdownItems.some(item => pathname.startsWith(item.href));

  return (
    <nav className="border-b border-border bg-background">
      <div className="max-w-[1400px] mx-auto px-4">
        <div className="flex items-center justify-between h-12">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 3v10M7 16l5-5 5 5M7 16v4M17 16v4" />
              </svg>
            </div>
            <span className="text-sm font-semibold bg-gradient-to-r from-blue-400 to-purple-400 text-transparent bg-clip-text">
              LabFork
            </span>
          </Link>

          {/* Navigation Items */}
          <div className="flex items-center gap-1">
            {/* Primary nav items */}
            {primaryNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
              const isFeatured = "featured" in item && item.featured;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 text-sm transition-colors rounded-md",
                    isFeatured
                      ? "text-amber-400 hover:text-amber-300 hover:bg-amber-400/10"
                      : isActive
                      ? "text-foreground-bright"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                  )}
                >
                  {isActive && !isFeatured && (
                    <span className="w-1.5 h-1.5 rounded-full bg-foreground-bright" />
                  )}
                  {isFeatured && (
                    <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                  )}
                  <Icon className={cn("w-3.5 h-3.5", isFeatured && "text-amber-400")} />
                  <span className="hidden md:inline">{item.label}</span>
                </Link>
              );
            })}

            {/* Separator */}
            <div className="w-px h-4 bg-border mx-1" />

            {/* Labs dropdown */}
            <Dropdown label="Labs" items={labsDropdownItems} isActive={isLabsActive} />

            {/* More dropdown */}
            <Dropdown label="More" items={moreDropdownItems} isActive={isMoreActive} />
          </div>

          {/* Right side: CTA + Status */}
          <div className="flex items-center gap-3">
            {/* Create Lab CTA */}
            <Link
              href="/lab/new"
              className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-md hover:from-blue-500 hover:to-purple-500 transition-all"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Create Lab</span>
            </Link>

            {/* Status indicator */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="hidden lg:inline">Ready</span>
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}
