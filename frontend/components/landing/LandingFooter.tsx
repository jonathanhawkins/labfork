/**
 * LandingFooter
 *
 * Footer with links, domain directory, and final CTA.
 * Fully internationalized for global accessibility.
 */

"use client";

import React from "react";
import Link from "next/link";
import { useTranslations } from "next-intl";

interface FooterLink {
  labelKey: string;
  href: string;
}

interface FooterSection {
  titleKey: string;
  links: FooterLink[];
}

const footerSections: FooterSection[] = [
  {
    titleKey: "product",
    links: [
      { labelKey: "features", href: "/features" },
      { labelKey: "domains", href: "/domains" },
      { labelKey: "pricing", href: "/pricing" },
      { labelKey: "changelog", href: "/changelog" },
    ],
  },
  {
    titleKey: "resources",
    links: [
      { labelKey: "documentation", href: "/docs" },
      { labelKey: "apiReference", href: "/docs/api" },
      { labelKey: "tutorials", href: "/tutorials" },
      { labelKey: "blog", href: "/blog" },
    ],
  },
  {
    titleKey: "community",
    links: [
      { labelKey: "discord", href: "https://discord.gg/labfork" },
      { labelKey: "github", href: "https://github.com/jonathanhawkins/labfork" },
      { labelKey: "twitter", href: "https://twitter.com/labfork" },
      { labelKey: "newsletter", href: "/newsletter" },
    ],
  },
  {
    titleKey: "company",
    links: [
      { labelKey: "about", href: "/about" },
      { labelKey: "contact", href: "mailto:hello@labfork.com" },
      { labelKey: "privacy", href: "/privacy" },
      { labelKey: "terms", href: "/terms" },
    ],
  },
];

const domainLinks = [
  { label: "Voice Clone", href: "/lab?domain=voice-clone", color: "#4ecdc4" },
  { label: "Quant Trading", href: "/lab?domain=quant-trading", color: "#10b981" },
  { label: "Game AI", href: "/lab?domain=game-ai", color: "#ec4899" },
  { label: "Robotics", href: "/lab?domain=robotics-ml", color: "#f97316" },
  { label: "Drug Discovery", href: "/lab?domain=drug-discovery", color: "#14b8a6" },
  { label: "Climate", href: "/lab?domain=climate-modeling", color: "#0ea5e9" },
  { label: "NLP", href: "/lab?domain=nlp-research", color: "#6366f1" },
  { label: "Computer Vision", href: "/lab?domain=computer-vision", color: "#ef4444" },
  { label: "Biotech NLP", href: "/lab?domain=biotech-nlp", color: "#8b5cf6" },
];

export function LandingFooter() {
  const t = useTranslations("footer");

  return (
    <footer className="bg-slate-950 border-t border-white/10">
      {/* Final CTA */}
      <div className="max-w-6xl mx-auto px-6 py-16 text-center">
        <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
          {t("finalCTA")}
        </h2>
        <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
          {t("finalCTADesc")}
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/lab/new"
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all shadow-lg shadow-blue-600/25"
          >
            {t("createLabFree")}
          </Link>
          <Link
            href="/explore"
            className="px-8 py-4 text-gray-300 font-medium hover:text-white transition-colors"
          >
            {t("viewExampleLabs")} &rarr;
          </Link>
        </div>
      </div>

      {/* Domain Quick Links */}
      <div className="border-t border-white/10 py-8">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex flex-wrap items-center justify-center gap-4">
            {domainLinks.map((domain) => (
              <Link
                key={domain.href}
                href={domain.href}
                className="px-4 py-2 rounded-full text-sm transition-colors"
                style={{
                  backgroundColor: `${domain.color}10`,
                  color: domain.color,
                }}
              >
                {domain.label}
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Main Footer */}
      <div className="border-t border-white/10 py-12">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-8">
            {/* Brand */}
            <div className="col-span-2 md:col-span-1">
              <Link href="/" className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M12 3v10M7 16l5-5 5 5M7 16v4M17 16v4" />
                  </svg>
                </div>
                <span className="font-bold text-white">LabFork</span>
              </Link>
              <p className="text-sm text-gray-500 mb-4">
                {t("tagline")}
              </p>
              <div className="flex items-center gap-3">
                <a
                  href="https://github.com/jonathanhawkins/labfork"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                  aria-label="GitHub"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                  </svg>
                </a>
                <a
                  href="https://discord.gg/labfork"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                  aria-label="Discord"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.317 4.37a19.791 19.791 0 00-4.885-1.515.074.074 0 00-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 00-5.487 0 12.64 12.64 0 00-.617-1.25.077.077 0 00-.079-.037A19.736 19.736 0 003.677 4.37a.07.07 0 00-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 00.031.057 19.9 19.9 0 005.993 3.03.078.078 0 00.084-.028 14.09 14.09 0 001.226-1.994.076.076 0 00-.041-.106 13.107 13.107 0 01-1.872-.892.077.077 0 01-.008-.128 10.2 10.2 0 00.372-.292.074.074 0 01.077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 01.078.01c.12.098.246.198.373.292a.077.077 0 01-.006.127 12.299 12.299 0 01-1.873.892.077.077 0 00-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 00.084.028 19.839 19.839 0 006.002-3.03.077.077 0 00.032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 00-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.956-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419 0-1.333.955-2.419 2.157-2.419 1.21 0 2.176 1.096 2.157 2.42 0 1.333-.946 2.418-2.157 2.418z" />
                  </svg>
                </a>
                <a
                  href="https://twitter.com/labfork"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg bg-white/5 text-gray-400 hover:text-white hover:bg-white/10 transition-colors"
                  aria-label="Twitter"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
                  </svg>
                </a>
              </div>
            </div>

            {/* Link Sections */}
            {footerSections.map((section) => (
              <div key={section.titleKey}>
                <h4 className="font-semibold text-white mb-4">{t(section.titleKey)}</h4>
                <ul className="space-y-2">
                  {section.links.map((link) => (
                    <li key={link.href}>
                      <Link
                        href={link.href}
                        className="text-sm text-gray-400 hover:text-white transition-colors"
                      >
                        {t(link.labelKey)}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-white/10 py-6">
        <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-gray-500">
            &copy; 2026 {t("copyright")}
          </p>
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span>{t("builtWith")}</span>
            <span className="hidden md:inline">|</span>
            <a
              href="https://github.com/jonathanhawkins/labfork"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              {t("starOnGitHub")}
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default LandingFooter;
