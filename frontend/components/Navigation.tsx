"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Mic, GitCompare, GraduationCap, AudioWaveform, Sparkles, Film, Scroll } from "lucide-react";

const navItems = [
  { href: "/", label: "Studio", icon: Mic },
  { href: "/perform", label: "Perform", icon: Scroll },
  { href: "/generate", label: "Generate", icon: Sparkles },
  { href: "/author", label: "Author", icon: Film },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/training", label: "Training", icon: GraduationCap },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-600">
              <AudioWaveform className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">
              Voice Clone Pipeline
            </span>
          </div>

          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                    isActive
                      ? "bg-orange-500/20 text-orange-400"
                      : "text-zinc-400 hover:text-white hover:bg-zinc-800"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
