/**
 * DeploymentOptions
 *
 * Showcase of deployment options including cloud, docker, and local.
 */

"use client";

import React, { useState } from "react";
import Link from "next/link";

interface DeployOption {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  features: string[];
  cta: string;
  ctaLink: string;
  badge?: string;
  color: string;
}

const deployOptions: DeployOption[] = [
  {
    id: "cloud",
    name: "One-Click Cloud",
    description: "Deploy to AWS, GCP, or Azure in minutes",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
      </svg>
    ),
    features: [
      "Auto-scaling GPU instances",
      "Managed databases included",
      "SSL certificates automated",
      "Cost monitoring dashboard",
    ],
    cta: "Deploy Now",
    ctaLink: "/deploy/cloud",
    badge: "Recommended",
    color: "#3b82f6",
  },
  {
    id: "docker",
    name: "Docker Compose",
    description: "Run anywhere with containers",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
      </svg>
    ),
    features: [
      "Single docker-compose.yml",
      "NVIDIA GPU support",
      "Persistent volumes",
      "Easy environment config",
    ],
    cta: "Get Config",
    ctaLink: "/deploy/docker",
    color: "#8b5cf6",
  },
  {
    id: "local",
    name: "Local Development",
    description: "Run on your own hardware",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    features: [
      "Mac M-series optimized",
      "CUDA support for NVIDIA",
      "Minimal dependencies",
      "Full source access",
    ],
    cta: "Setup Guide",
    ctaLink: "/docs/local-setup",
    badge: "Free",
    color: "#10b981",
  },
];

const costEstimates = [
  { tier: "Hobby", cost: "$0", specs: "Local Mac/PC", features: ["Unlimited research", "5 papers/day", "CPU inference"] },
  { tier: "Pro", cost: "$49/mo", specs: "Cloud GPU (T4)", features: ["Unlimited papers", "GPU training", "Team of 3"] },
  { tier: "Team", cost: "$199/mo", specs: "Cloud GPU (A100)", features: ["Priority support", "Private labs", "Team of 10"] },
];

export function DeploymentOptions() {
  const [selectedOption, setSelectedOption] = useState<string>("cloud");

  return (
    <section className="py-24 bg-slate-950">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 text-green-400 text-sm mb-4">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2" />
            </svg>
            Easy Deployment
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Deploy Your Way
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Cloud, container, or local. Get your lab running in minutes with your preferred setup.
          </p>
        </div>

        {/* Deployment Options */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          {deployOptions.map((option) => (
            <button
              key={option.id}
              onClick={() => setSelectedOption(option.id)}
              className={`relative text-left p-6 rounded-2xl border transition-all ${
                selectedOption === option.id
                  ? "border-white/20 bg-white/10"
                  : "border-white/10 bg-white/5 hover:bg-white/10"
              }`}
            >
              {/* Badge */}
              {option.badge && (
                <div
                  className="absolute -top-3 right-4 px-3 py-1 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: `${option.color}20`,
                    color: option.color,
                  }}
                >
                  {option.badge}
                </div>
              )}

              {/* Icon */}
              <div
                className="p-3 rounded-xl inline-block mb-4"
                style={{ backgroundColor: `${option.color}20`, color: option.color }}
              >
                {option.icon}
              </div>

              {/* Content */}
              <h3 className="text-lg font-semibold text-white mb-2">{option.name}</h3>
              <p className="text-sm text-gray-400 mb-4">{option.description}</p>

              {/* Features */}
              <ul className="space-y-2 mb-6">
                {option.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center gap-2 text-sm text-gray-300">
                    <svg
                      className="w-4 h-4 flex-shrink-0"
                      style={{ color: option.color }}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>

              {/* CTA */}
              <Link
                href={option.ctaLink}
                className="inline-flex items-center gap-2 font-medium text-sm"
                style={{ color: option.color }}
              >
                {option.cta}
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </Link>
            </button>
          ))}
        </div>

        {/* Pricing Preview */}
        <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-white/10">
            <h3 className="font-semibold text-white">Simple Pricing</h3>
            <p className="text-sm text-gray-400">Start free, scale when ready</p>
          </div>
          <div className="grid md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-white/10">
            {costEstimates.map((tier) => (
              <div key={tier.tier} className="p-6">
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="text-2xl font-bold text-white">{tier.cost}</span>
                  {tier.cost !== "$0" && <span className="text-sm text-gray-500">/month</span>}
                </div>
                <div className="text-sm font-medium text-gray-300 mb-1">{tier.tier}</div>
                <div className="text-xs text-gray-500 mb-4">{tier.specs}</div>
                <ul className="space-y-2">
                  {tier.features.map((feature, idx) => (
                    <li key={idx} className="flex items-center gap-2 text-sm text-gray-400">
                      <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export default DeploymentOptions;
