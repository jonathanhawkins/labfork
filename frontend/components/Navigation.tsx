"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Mic,
  GitCompare,
  GraduationCap,
  Sparkles,
  Film,
  Scroll,
  Radio,
  Home,
  FlaskConical,
  Bot,
  Target,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/studio", label: "Studio", icon: Mic },
  { href: "/perform", label: "Perform", icon: Scroll },
  { href: "/generate", label: "Generate", icon: Sparkles },
  { href: "/live", label: "Live", icon: Radio },
  { href: "/author", label: "Author", icon: Film },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/training", label: "Training", icon: GraduationCap },
  { href: "/evaluate", label: "A/B Test", icon: FlaskConical },
  { href: "/lab", label: "Lab", icon: Bot },
  { href: "/research", label: "Research", icon: Target },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-border bg-background">
      <div className="max-w-[1400px] mx-auto px-4">
        <div className="flex items-center justify-between h-12">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-6 h-6 rounded border border-border flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-foreground group-hover:bg-foreground-bright transition-colors" />
            </div>
            <span className="text-sm text-foreground-bright">
              Voice Clone Pipeline
            </span>
          </Link>

          {/* Navigation Items */}
          <div className="flex items-center">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 text-sm transition-colors",
                    isActive
                      ? "text-foreground-bright"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-foreground-bright" />
                  )}
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden lg:inline">{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="hidden sm:inline">Ready</span>
            <span className="w-1.5 h-1.5 rounded-full bg-foreground animate-pulse" />
          </div>
        </div>
      </div>
    </nav>
  );
}
