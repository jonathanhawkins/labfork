"use client";

import { useState, useRef, useEffect } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Globe, ChevronDown, Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { locales, localeNames, type Locale } from "@/i18n/config";

interface LanguageSwitcherProps {
  variant?: "dropdown" | "compact";
  className?: string;
}

export function LanguageSwitcher({
  variant = "dropdown",
  className,
}: LanguageSwitcherProps) {
  const locale = useLocale() as Locale;
  const t = useTranslations("languageSwitcher");
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLocaleChange = (newLocale: Locale) => {
    // Set cookie for locale preference
    document.cookie = `NEXT_LOCALE=${newLocale};path=/;max-age=31536000`;
    // Reload to apply new locale
    window.location.reload();
  };

  if (variant === "compact") {
    return (
      <div ref={dropdownRef} className={cn("relative", className)}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center justify-center w-11 h-11 rounded-lg hover:bg-accent/50 transition-colors"
          aria-label={t("label")}
          aria-expanded={isOpen}
        >
          <Globe className="w-5 h-5 text-muted-foreground" />
        </button>

        {isOpen && (
          <div className="absolute top-full right-0 mt-1 w-40 py-1 bg-background-elevated border border-border rounded-lg shadow-lg z-50 animate-in fade-in-0 zoom-in-95 duration-100">
            {locales.map((loc) => (
              <button
                key={loc}
                onClick={() => {
                  handleLocaleChange(loc);
                  setIsOpen(false);
                }}
                className={cn(
                  "flex items-center justify-between w-full px-3 py-2 text-sm transition-colors min-h-[44px]",
                  loc === locale
                    ? "text-foreground-bright bg-accent/50"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent/30"
                )}
              >
                <span>{localeNames[loc]}</span>
                {loc === locale && <Check className="w-4 h-4" />}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div ref={dropdownRef} className={cn("relative", className)}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-2 px-3 py-2 text-sm transition-colors rounded-md min-h-[44px]",
          "text-muted-foreground hover:text-foreground hover:bg-accent/50"
        )}
        aria-label={t("label")}
        aria-expanded={isOpen}
      >
        <Globe className="w-4 h-4" />
        <span>{localeNames[locale]}</span>
        <ChevronDown
          className={cn("w-4 h-4 transition-transform", isOpen && "rotate-180")}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-48 py-2 bg-background-elevated border border-border rounded-lg shadow-lg z-50 animate-in fade-in-0 zoom-in-95 duration-100">
          <div className="px-3 py-2 text-xs text-muted-foreground uppercase tracking-wider">
            {t("label")}
          </div>
          {locales.map((loc) => (
            <button
              key={loc}
              onClick={() => {
                handleLocaleChange(loc);
                setIsOpen(false);
              }}
              className={cn(
                "flex items-center justify-between w-full px-3 py-3 text-sm transition-colors min-h-[48px]",
                loc === locale
                  ? "text-foreground-bright bg-accent/50"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent/30"
              )}
            >
              <div className="flex items-center gap-3">
                <span className="font-medium">{localeNames[loc]}</span>
              </div>
              {loc === locale && <Check className="w-4 h-4 text-green-500" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default LanguageSwitcher;
