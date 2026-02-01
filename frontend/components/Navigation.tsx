"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useRef, useEffect } from "react";
import { useTranslations } from "next-intl";
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
  Menu,
  X,
} from "lucide-react";
import { LanguageSwitcher } from "./LanguageSwitcher";

interface NavItem {
  href: string;
  labelKey: string;
  descKey: string;
  icon: React.ComponentType<{ className?: string }>;
  featured?: boolean;
}

// Primary navigation items (always visible)
const primaryNavItems: NavItem[] = [
  { href: "/explore", labelKey: "explore", descKey: "exploreDesc", icon: Compass },
  { href: "/watch", labelKey: "watch", descKey: "watchDesc", icon: Eye },
  { href: "/contribute", labelKey: "contribute", descKey: "contributeDesc", icon: Cpu },
  { href: "/projects/firefly-network", labelKey: "firefly", descKey: "fireflyDesc", icon: Lightbulb, featured: true },
];

// Labs dropdown items
const labsDropdownItems: NavItem[] = [
  { href: "/lab/new", labelKey: "createLab", descKey: "createLabDesc", icon: Plus },
  { href: "/lab", labelKey: "myLab", descKey: "myLabDesc", icon: Bot },
  { href: "/labs", labelKey: "browseLabs", descKey: "browseLabsDesc", icon: FlaskConical },
];

// More dropdown items
const moreDropdownItems: NavItem[] = [
  { href: "/analytics", labelKey: "analytics", descKey: "analyticsDesc", icon: BarChart3 },
  { href: "/demos", labelKey: "demos", descKey: "demosDesc", icon: Sliders },
  { href: "/feed", labelKey: "feed", descKey: "feedDesc", icon: Rss },
  { href: "/research", labelKey: "research", descKey: "researchDesc", icon: Target },
  { href: "/domains", labelKey: "domains", descKey: "domainsDesc", icon: Layers },
];

interface DropdownProps {
  label: string;
  items: NavItem[];
  isActive: boolean;
  t: (key: string) => string;
}

function Dropdown({ label, items, isActive, t }: DropdownProps) {
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
          "flex items-center gap-1.5 px-3 py-2 text-sm transition-colors rounded-md min-h-[44px]",
          isActive
            ? "text-foreground-bright"
            : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
        )}
      >
        {isActive && <span className="w-1.5 h-1.5 rounded-full bg-foreground-bright" />}
        <span>{label}</span>
        <ChevronDown className={cn("w-4 h-4 transition-transform", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-64 py-2 bg-background-elevated border border-border rounded-lg shadow-lg z-50 animate-in fade-in-0 zoom-in-95 duration-100">
          {items.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsOpen(false)}
                className="flex items-start gap-3 px-4 py-3 hover:bg-accent/50 transition-colors min-h-[48px]"
              >
                <Icon className="w-5 h-5 mt-0.5 text-muted-foreground" />
                <div>
                  <div className="text-sm text-foreground font-medium">{t(item.labelKey)}</div>
                  <div className="text-xs text-muted-foreground">{t(item.descKey)}</div>
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
  const t = useTranslations("nav");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Check if any labs dropdown item is active
  const isLabsActive = labsDropdownItems.some(item => pathname.startsWith(item.href));
  // Check if any more dropdown item is active
  const isMoreActive = moreDropdownItems.some(item => pathname.startsWith(item.href));

  // Close mobile menu when route changes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [mobileMenuOpen]);

  return (
    <nav className="border-b border-border bg-background">
      <div className="max-w-[1400px] mx-auto px-4">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <svg className="w-4 h-4 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                <path d="M12 3v10M7 16l5-5 5 5M7 16v4M17 16v4" />
              </svg>
            </div>
            <span className="text-sm font-semibold bg-gradient-to-r from-blue-400 to-purple-400 text-transparent bg-clip-text">
              LabFork
            </span>
          </Link>

          {/* Desktop Navigation Items */}
          <div className="hidden md:flex items-center gap-1">
            {/* Primary nav items */}
            {primaryNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
              const isFeatured = item.featured;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2 px-3 py-2 text-sm transition-colors rounded-md min-h-[44px]",
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
                  <Icon className={cn("w-4 h-4", isFeatured && "text-amber-400")} />
                  <span>{t(item.labelKey)}</span>
                </Link>
              );
            })}

            {/* Separator */}
            <div className="w-px h-5 bg-border mx-1" />

            {/* Labs dropdown */}
            <Dropdown label={t("labs")} items={labsDropdownItems} isActive={isLabsActive} t={t} />

            {/* More dropdown */}
            <Dropdown label={t("more")} items={moreDropdownItems} isActive={isMoreActive} t={t} />

            {/* Language Switcher - Desktop */}
            <div className="w-px h-5 bg-border mx-1" />
            <LanguageSwitcher variant="dropdown" />
          </div>

          {/* Right side: CTA + Status + Mobile Menu */}
          <div className="flex items-center gap-2">
            {/* Create Lab CTA - always visible but responsive */}
            <Link
              href="/lab/new"
              className="flex items-center gap-1.5 px-3 py-2 text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-500 hover:to-purple-500 transition-all min-h-[44px]"
            >
              <Plus className="w-4 h-4" />
              <span className="hidden sm:inline">{t("createLab")}</span>
            </Link>

            {/* Status indicator - desktop only */}
            <div className="hidden lg:flex items-center gap-2 text-xs text-muted-foreground px-2">
              <span>{t("ready")}</span>
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            </div>

            {/* Language Switcher - Mobile (compact) */}
            <div className="md:hidden">
              <LanguageSwitcher variant="compact" />
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden flex items-center justify-center w-11 h-11 rounded-lg hover:bg-accent/50 transition-colors"
              aria-label={mobileMenuOpen ? t("closeMenu") : t("openMenu")}
            >
              {mobileMenuOpen ? (
                <X className="w-5 h-5 text-foreground" />
              ) : (
                <Menu className="w-5 h-5 text-foreground" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 top-14 z-50 bg-background/95 backdrop-blur-sm overflow-y-auto">
          <div className="px-4 py-6 space-y-6">
            {/* Primary Navigation */}
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-3 mb-2">
                {t("navigate")}
              </p>
              {primaryNavItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
                const isFeatured = item.featured;

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[48px]",
                      isFeatured
                        ? "text-amber-400 hover:bg-amber-400/10"
                        : isActive
                        ? "text-foreground-bright bg-accent/50"
                        : "text-foreground hover:bg-accent/50"
                    )}
                  >
                    <Icon className={cn("w-5 h-5", isFeatured && "text-amber-400")} />
                    <div className="flex-1">
                      <span className="font-medium">{t(item.labelKey)}</span>
                      <p className="text-xs text-muted-foreground mt-0.5">{t(item.descKey)}</p>
                    </div>
                    {isFeatured && (
                      <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                    )}
                  </Link>
                );
              })}
            </div>

            {/* Labs Section */}
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-3 mb-2">
                {t("labs")}
              </p>
              {labsDropdownItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href || pathname.startsWith(item.href + "/");

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[48px]",
                      isActive
                        ? "text-foreground-bright bg-accent/50"
                        : "text-foreground hover:bg-accent/50"
                    )}
                  >
                    <Icon className="w-5 h-5 text-muted-foreground" />
                    <div className="flex-1">
                      <span className="font-medium">{t(item.labelKey)}</span>
                      <p className="text-xs text-muted-foreground mt-0.5">{t(item.descKey)}</p>
                    </div>
                  </Link>
                );
              })}
            </div>

            {/* More Section */}
            <div className="space-y-1">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-3 mb-2">
                {t("more")}
              </p>
              {moreDropdownItems.map((item) => {
                const Icon = item.icon;
                const isActive = pathname === item.href || pathname.startsWith(item.href + "/");

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 px-3 py-3 rounded-lg transition-colors min-h-[48px]",
                      isActive
                        ? "text-foreground-bright bg-accent/50"
                        : "text-foreground hover:bg-accent/50"
                    )}
                  >
                    <Icon className="w-5 h-5 text-muted-foreground" />
                    <div className="flex-1">
                      <span className="font-medium">{t(item.labelKey)}</span>
                      <p className="text-xs text-muted-foreground mt-0.5">{t(item.descKey)}</p>
                    </div>
                  </Link>
                );
              })}
            </div>

            {/* Status */}
            <div className="px-3 pt-4 border-t border-border">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span>{t("networkReady")}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}
