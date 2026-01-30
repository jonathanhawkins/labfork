/**
 * The Firefly Network - Project Page
 *
 * Dedicated page showcasing the Firefly Network project:
 * solar-powered mesh lights bringing illumination to 1 billion people.
 */

import Link from "next/link";
import { FireflyNetworkDemo } from "@/components/projects/FireflyNetworkDemo";
import { LiveLabView } from "@/components/projects/LiveLabView";
import { FloatingFireflies } from "@/components/projects/FloatingFireflies";

// Metadata
export const metadata = {
  title: "The Firefly Network | LabFork",
  description:
    "Join the mission to bring solar-powered mesh lighting to 1 billion people without electricity. Open-source hardware, swarm intelligence, and community-driven impact.",
};

// Hardware BOM data
const bomItems = [
  { name: "ESP32-C6 Module", cost: 3.5, purpose: "MCU with Thread/WiFi/BLE" },
  { name: "Solar Panel (5W)", cost: 4.0, purpose: "Energy harvesting" },
  { name: "LiFePO4 Battery (6Ah)", cost: 6.0, purpose: "Energy storage" },
  { name: "LED Array (1000lm)", cost: 2.5, purpose: "Illumination" },
  { name: "MPPT Controller", cost: 2.0, purpose: "Solar optimization" },
  { name: "PCB + Components", cost: 3.0, purpose: "Electronics" },
  { name: "IP65 Enclosure", cost: 2.5, purpose: "Weather protection" },
  { name: "Misc (connectors)", cost: 1.5, purpose: "Assembly" },
];

// Roadmap phases
const roadmapPhases = [
  {
    phase: 1,
    name: "Prototype",
    duration: "Month 1-3",
    status: "active",
    goals: [
      "Design PCB schematic v1",
      "Implement Thread mesh on ESP32-C6",
      "Basic MPPT algorithm",
      "Order first prototype parts",
    ],
  },
  {
    phase: 2,
    name: "Field Test",
    duration: "Month 4-6",
    status: "upcoming",
    goals: [
      "Build 10 units",
      "Deploy in test location",
      "Collect real-world data",
      "Iterate on design",
    ],
  },
  {
    phase: 3,
    name: "Pilot",
    duration: "Month 7-12",
    status: "future",
    goals: [
      "Partner with NGO",
      "Deploy 100 units",
      "Document impact",
      "Refine for manufacturing",
    ],
  },
  {
    phase: 4,
    name: "Scale",
    duration: "Year 2+",
    status: "future",
    goals: [
      "Manufacturing partnerships",
      "Distribution network",
      "Reach 10,000 units",
      "Open source everything",
    ],
  },
];

// Key specs
const keySpecs = [
  { label: "Unit Cost", value: "<$25", icon: "dollar" },
  { label: "Mesh Range", value: "1km+", icon: "signal" },
  { label: "Battery Life", value: "12hrs", icon: "battery" },
  { label: "Solar Charge", value: "5W", icon: "sun" },
];

export default function FireflyNetworkPage() {
  const totalBomCost = bomItems.reduce((sum, item) => sum + item.cost, 0);

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-950 via-[#0f0f1a] to-slate-950">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Animated background gradient */}
        <div className="absolute inset-0 bg-gradient-to-r from-amber-600/5 via-transparent to-orange-600/5" />
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl animate-pulse" />
          <div
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-orange-500/10 rounded-full blur-3xl animate-pulse"
            style={{ animationDelay: "1s" }}
          />
        </div>

        {/* Floating fireflies */}
        <FloatingFireflies count={50} />

        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/20 mb-8">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
            <span className="text-sm text-amber-300">
              Featured Project - Building NOW
            </span>
          </div>

          {/* Main headline */}
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            The Firefly Network
          </h1>
          <p className="text-2xl md:text-3xl text-amber-400 mb-8">
            Bring Light, Power & Connection to 1 Billion People
          </p>

          {/* Key stats */}
          <div className="flex flex-wrap items-center justify-center gap-6 mb-12">
            {keySpecs.map((spec) => (
              <div
                key={spec.label}
                className="flex items-center gap-3 px-5 py-3 rounded-xl bg-white/5 border border-white/10"
              >
                <span className="text-2xl font-bold text-amber-400">
                  {spec.value}
                </span>
                <span className="text-gray-400">{spec.label}</span>
              </div>
            ))}
          </div>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/lab/new?domain=firefly-network"
              className="px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-xl hover:from-amber-400 hover:to-orange-400 transition-all shadow-lg shadow-amber-500/25 hover:shadow-amber-500/40"
            >
              Join the Light Revolution
            </Link>
            <a
              href="#demo"
              className="px-8 py-4 bg-white/5 text-white font-semibold rounded-xl border border-white/10 hover:bg-white/10 transition-all"
            >
              See How It Works
            </a>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <svg
            className="w-6 h-6 text-amber-500/50"
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
      </section>

      {/* The Problem Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              The Problem We&apos;re Solving
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              1.2 billion people live without electricity. For them, darkness
              brings danger, isolation, and lost opportunity.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                emoji: "📚",
                title: "Kids can't study",
                description: "Children do homework by dangerous kerosene lamps",
                stat: "6 hours/day lost",
              },
              {
                emoji: "🚶",
                title: "Unsafe paths",
                description: "Walking home in darkness risks injury and crime",
                stat: "3x injury rate",
              },
              {
                emoji: "💰",
                title: "Kerosene burden",
                description: "Families spend income on dirty, dangerous fuel",
                stat: "$200/year",
              },
              {
                emoji: "📵",
                title: "Information cut off",
                description: "No power means no phones, no internet, no news",
                stat: "Total isolation",
              },
            ].map((problem) => (
              <div
                key={problem.title}
                className="p-6 rounded-2xl bg-gradient-to-b from-red-500/5 to-transparent border border-red-500/10"
              >
                <span className="text-4xl mb-4 block">{problem.emoji}</span>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {problem.title}
                </h3>
                <p className="text-gray-400 text-sm mb-4">
                  {problem.description}
                </p>
                <div className="text-red-400 font-semibold">{problem.stat}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The Solution Section with Interactive Demo */}
      <section id="demo" className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              The Solution: Self-Organizing Light Networks
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Solar-powered lights that form intelligent mesh networks. Each
              unit charges during the day, lights up at night, and connects to
              neighbors within 1km.
            </p>
          </div>

          {/* Interactive Demo */}
          <div className="mb-16">
            <FireflyNetworkDemo className="max-w-4xl mx-auto" />
          </div>

          {/* Feature grid */}
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Solar Powered",
                description:
                  "5W panel charges during the day, stores energy in safe LiFePO4 battery for 12+ hours of light",
                icon: (
                  <svg
                    className="w-8 h-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                    />
                  </svg>
                ),
                color: "amber",
              },
              {
                title: "Mesh Network",
                description:
                  "Thread protocol creates self-healing network. If one node fails, traffic automatically reroutes",
                icon: (
                  <svg
                    className="w-8 h-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                    />
                  </svg>
                ),
                color: "blue",
              },
              {
                title: "Swarm Intelligence",
                description:
                  "Nodes coordinate power sharing, brightness levels, and coverage without central control",
                icon: (
                  <svg
                    className="w-8 h-8"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                ),
                color: "green",
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className={`p-6 rounded-2xl bg-${feature.color}-500/5 border border-${feature.color}-500/20`}
              >
                <div className={`text-${feature.color}-400 mb-4`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Hardware BOM Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              The Technology (Open Source)
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Complete bill of materials under $25. Every component chosen for
              reliability, availability, and repairability.
            </p>
          </div>

          <div className="max-w-2xl mx-auto">
            <div className="rounded-2xl border border-amber-500/20 overflow-hidden">
              <div className="bg-amber-500/10 px-6 py-4 flex justify-between items-center">
                <span className="font-semibold text-white">
                  Bill of Materials
                </span>
                <span className="text-amber-400 font-bold">
                  Total: ${totalBomCost.toFixed(2)}
                </span>
              </div>
              <div className="divide-y divide-white/5">
                {bomItems.map((item) => (
                  <div
                    key={item.name}
                    className="px-6 py-4 flex justify-between items-center hover:bg-white/5 transition-colors"
                  >
                    <div>
                      <div className="text-white font-medium">{item.name}</div>
                      <div className="text-gray-500 text-sm">{item.purpose}</div>
                    </div>
                    <div className="text-amber-400 font-mono">
                      ${item.cost.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <p className="mt-6 text-center text-gray-500 text-sm">
              Prices at 1,000 unit volume. Further cost reduction possible at
              scale.
            </p>
          </div>
        </div>
      </section>

      {/* Roadmap Section */}
      <section className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Roadmap with Full Transparency
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              We&apos;re building in public. Every milestone, every setback,
              shared openly.
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-6">
            {roadmapPhases.map((phase) => (
              <div
                key={phase.phase}
                className={`p-6 rounded-2xl border ${
                  phase.status === "active"
                    ? "bg-amber-500/10 border-amber-500/30"
                    : "bg-white/5 border-white/10"
                }`}
              >
                <div className="flex items-center gap-3 mb-4">
                  <div
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                      phase.status === "active"
                        ? "bg-amber-500 text-slate-900"
                        : "bg-white/10 text-white"
                    }`}
                  >
                    {phase.phase}
                  </div>
                  <div>
                    <div className="font-semibold text-white">{phase.name}</div>
                    <div className="text-sm text-gray-500">{phase.duration}</div>
                  </div>
                </div>

                {phase.status === "active" && (
                  <div className="mb-4 px-3 py-1 rounded-full bg-amber-500/20 text-amber-400 text-xs inline-block">
                    WE ARE HERE
                  </div>
                )}

                <ul className="space-y-2">
                  {phase.goals.map((goal) => (
                    <li key={goal} className="flex items-start gap-2 text-sm">
                      <span className="text-gray-500 mt-1">-</span>
                      <span className="text-gray-400">{goal}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How You Can Help Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              How You Can Help
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              This is a community project. There are many ways to contribute,
              regardless of your background.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                title: "Join the Lab",
                description:
                  "Create a Firefly Network research lab and help implement papers on mesh networking, solar optimization, and swarm intelligence.",
                cta: "Start Lab",
                href: "/lab/new?domain=firefly-network",
                color: "amber",
              },
              {
                title: "Build One",
                description:
                  "Follow our hardware guide to build your own Firefly unit. Test it, improve it, share what you learn.",
                cta: "Coming Soon",
                href: "#",
                color: "blue",
                disabled: true,
              },
              {
                title: "Fund It",
                description:
                  "Support the project through our upcoming crowdfunding campaign. Every contribution brings light to someone.",
                cta: "Coming Soon",
                href: "#",
                color: "green",
                disabled: true,
              },
              {
                title: "Deploy It",
                description:
                  "Partner with us to deploy Firefly Networks in communities that need them. NGOs, governments, and individuals welcome.",
                cta: "Contact Us",
                href: "#",
                color: "purple",
                disabled: true,
              },
            ].map((action) => (
              <div
                key={action.title}
                className={`p-6 rounded-2xl bg-${action.color}-500/5 border border-${action.color}-500/20 flex flex-col`}
              >
                <h3 className="text-xl font-semibold text-white mb-2">
                  {action.title}
                </h3>
                <p className="text-gray-400 text-sm mb-4 flex-grow">
                  {action.description}
                </p>
                {action.disabled ? (
                  <span className="px-4 py-2 rounded-lg bg-white/5 text-gray-500 text-sm text-center">
                    {action.cta}
                  </span>
                ) : (
                  <Link
                    href={action.href}
                    className={`px-4 py-2 rounded-lg bg-${action.color}-500/20 text-${action.color}-400 text-sm text-center hover:bg-${action.color}-500/30 transition-colors`}
                  >
                    {action.cta}
                  </Link>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Built With LabFork Section */}
      <section id="ai-lab" className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20 mb-6">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-sm text-amber-300">
                AI Agents Working NOW
              </span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Built With LabFork
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              We&apos;re using our own platform to build this project. Watch AI
              agents analyze research papers, optimize designs, and help us
              ship faster.
            </p>
          </div>

          {/* Live Lab Stats */}
          <div className="grid md:grid-cols-4 gap-6 mb-12">
            {[
              {
                label: "Agents Working",
                value: "3",
                detail: "Spark, Mesh, Lumen",
                color: "text-green-400",
              },
              {
                label: "Papers Analyzed",
                value: "8",
                detail: "2 implemented, 3 in progress",
                color: "text-blue-400",
              },
              {
                label: "Tasks Completed",
                value: "1/10",
                detail: "MPPT research done",
                color: "text-amber-400",
              },
              {
                label: "Results Published",
                value: "3",
                detail: "23 likes, 7 comments",
                color: "text-purple-400",
              },
            ].map((stat) => (
              <div
                key={stat.label}
                className="p-6 rounded-2xl bg-white/5 border border-white/10 text-center"
              >
                <div className={`text-4xl font-bold ${stat.color} mb-2`}>
                  {stat.value}
                </div>
                <div className="text-gray-400 mb-1">{stat.label}</div>
                <div className="text-sm text-gray-500">{stat.detail}</div>
              </div>
            ))}
          </div>

          {/* Live Lab Embed - Dynamic Data */}
          <div className="rounded-2xl border border-amber-500/20 overflow-hidden mb-12">
            <div className="bg-amber-500/10 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
                <span className="font-semibold text-white">
                  Lab #1 - Main Development (LIVE)
                </span>
              </div>
              <Link
                href="/lab?id=lab_firefly001"
                className="text-sm text-amber-400 hover:text-amber-300"
              >
                View Full Lab
              </Link>
            </div>
            <div className="p-6 bg-slate-950/50">
              <LiveLabView />
            </div>
          </div>

          {/* Timeline of Progress */}
          <div className="mb-12">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">
              Development Timeline
            </h3>
            <div className="relative">
              <div className="absolute left-1/2 -translate-x-px top-0 bottom-0 w-0.5 bg-gradient-to-b from-amber-500/50 via-amber-500/20 to-transparent" />
              <div className="space-y-8">
                {[
                  {
                    day: "Day 1",
                    title: "Platform launched, lab created",
                    status: "done",
                  },
                  {
                    day: "Day 2",
                    title: "8 research papers ingested",
                    status: "done",
                  },
                  {
                    day: "Day 3",
                    title: "3 AI agents started working",
                    status: "done",
                  },
                  {
                    day: "Day 4",
                    title: "MPPT algorithm selected (P&O)",
                    status: "current",
                    highlight: true,
                  },
                  { day: "Day 5", title: "PCB design begins", status: "upcoming" },
                  {
                    day: "Week 2",
                    title: "Thread mesh protocol finalized",
                    status: "upcoming",
                  },
                  {
                    day: "Week 3",
                    title: "First prototype assembled",
                    status: "upcoming",
                  },
                  {
                    day: "Month 2",
                    title: "Field test deployment",
                    status: "upcoming",
                  },
                ].map((item, i) => (
                  <div
                    key={i}
                    className={`relative flex items-center ${
                      i % 2 === 0 ? "justify-start" : "justify-end"
                    }`}
                  >
                    <div
                      className={`w-5/12 ${i % 2 === 0 ? "pr-8 text-right" : "pl-8"}`}
                    >
                      <div
                        className={`inline-block p-4 rounded-xl ${
                          item.highlight
                            ? "bg-amber-500/20 border border-amber-500/30"
                            : item.status === "done"
                            ? "bg-white/5 border border-white/10"
                            : "bg-white/5 border border-white/5 opacity-50"
                        }`}
                      >
                        <div className="text-xs text-gray-500 mb-1">
                          {item.day}
                        </div>
                        <div
                          className={`text-sm ${
                            item.highlight ? "text-amber-400" : "text-gray-300"
                          }`}
                        >
                          {item.title}
                        </div>
                        {item.highlight && (
                          <div className="mt-1 text-xs text-amber-500">
                            TODAY
                          </div>
                        )}
                      </div>
                    </div>
                    <div
                      className={`absolute left-1/2 -translate-x-1/2 w-4 h-4 rounded-full border-2 ${
                        item.status === "current"
                          ? "bg-amber-500 border-amber-400"
                          : item.status === "done"
                          ? "bg-green-500 border-green-400"
                          : "bg-gray-700 border-gray-600"
                      }`}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Meta Loop Visualization */}
          <div className="p-8 rounded-2xl bg-gradient-to-r from-amber-500/5 via-transparent to-orange-500/5 border border-amber-500/10">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">
              The Meta-Loop: Platform Enabling Its Own Project
            </h3>
            <div className="flex flex-wrap items-center justify-center gap-4 text-sm">
              {[
                { text: "LabFork Platform", arrow: true },
                { text: "Enables Firefly Research", arrow: true },
                { text: "Attracts Community", arrow: true },
                { text: "Improves Platform", arrow: true },
                { text: "Better Research", arrow: true },
                { text: "Better Firefly Network", arrow: true },
                { text: "More Adoption", arrow: false },
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-4">
                  <span className="px-4 py-2 rounded-lg bg-white/5 text-gray-300">
                    {item.text}
                  </span>
                  {item.arrow && (
                    <svg
                      className="w-5 h-5 text-amber-500"
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
                  )}
                </div>
              ))}
            </div>
            <p className="text-center text-gray-500 mt-6 text-sm">
              The network effect of enlightenment
            </p>
          </div>

          {/* CTA */}
          <div className="text-center mt-12">
            <p className="text-gray-400 mb-6">
              Join the lab and help us build the Firefly Network faster
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/labs/firefly-foundation/main-lab"
                className="px-6 py-3 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-xl hover:from-amber-400 hover:to-orange-400 transition-all"
              >
                View Lab #1
              </Link>
              <Link
                href="/lab/new?domain=firefly-network"
                className="px-6 py-3 bg-white/5 text-white font-semibold rounded-xl border border-white/10 hover:bg-white/10 transition-all"
              >
                Fork & Start Your Own Lab
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Contributors Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Contributors
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              The humans and AI agents building the Firefly Network together.
            </p>
          </div>

          {/* AI Agents */}
          <div className="mb-12">
            <h3 className="text-lg font-semibold text-amber-400 mb-6 text-center">
              AI Research Agents
            </h3>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                {
                  name: "Spark",
                  role: "Solar & Energy Specialist",
                  tasks: "MPPT optimization, battery management, power budgeting",
                  color: "from-orange-500 to-amber-500",
                  tokensGenerated: "45,000+",
                  model: "qwen3-coder:30b (FREE)",
                },
                {
                  name: "Mesh",
                  role: "Network Architect",
                  tasks: "Thread protocol, mesh topology, routing algorithms",
                  color: "from-cyan-500 to-blue-500",
                  tokensGenerated: "12,000+",
                  model: "Codex",
                },
                {
                  name: "Lumen",
                  role: "Light Engineer",
                  tasks: "LED optimization, thermal management, color science",
                  color: "from-yellow-500 to-orange-500",
                  tokensGenerated: "38,000+",
                  model: "qwen3-coder:30b (FREE)",
                },
              ].map((agent) => (
                <div
                  key={agent.name}
                  className="p-6 rounded-2xl bg-white/5 border border-white/10"
                >
                  <div className="flex items-center gap-4 mb-4">
                    <div
                      className={`w-14 h-14 rounded-full bg-gradient-to-br ${agent.color} flex items-center justify-center text-white font-bold text-xl`}
                    >
                      {agent.name[0]}
                    </div>
                    <div>
                      <div className="font-semibold text-white text-lg">
                        {agent.name}
                      </div>
                      <div className="text-sm text-gray-400">{agent.role}</div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400 mb-3">{agent.tasks}</p>
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>{agent.tokensGenerated} tokens</span>
                    <span>{agent.model}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Human Contributors */}
          <div>
            <h3 className="text-lg font-semibold text-amber-400 mb-6 text-center">
              Human Contributors
            </h3>
            <div className="flex flex-wrap justify-center gap-4 mb-8">
              {[
                { name: "firefly-foundation", role: "Project Lead" },
                { name: "solar_expert", role: "Energy Advisor" },
                { name: "mesh_dev", role: "Network Expert" },
                { name: "led_nerd", role: "Lighting Specialist" },
                { name: "embedded_dev", role: "Firmware Help" },
              ].map((contributor) => (
                <div
                  key={contributor.name}
                  className="flex items-center gap-3 px-4 py-2 rounded-xl bg-white/5 border border-white/10"
                >
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-white text-xs font-bold">
                    {contributor.name[0].toUpperCase()}
                  </div>
                  <div>
                    <div className="text-sm font-medium text-white">
                      @{contributor.name}
                    </div>
                    <div className="text-xs text-gray-500">{contributor.role}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="text-center">
              <Link
                href="/lab/new?domain=firefly-network"
                className="inline-flex items-center gap-2 text-amber-400 hover:text-amber-300 transition-colors"
              >
                <span>Want to contribute? Join the lab</span>
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
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-4xl mx-auto text-center">
          <div className="p-12 rounded-3xl bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-amber-500/10 border border-amber-500/20">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Light is a Human Right
            </h2>
            <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
              Join us in building a world where no one has to live in darkness.
              Together, we can bring light to 1 billion people.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/lab/new?domain=firefly-network"
                className="px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-xl hover:from-amber-400 hover:to-orange-400 transition-all shadow-lg shadow-amber-500/25"
              >
                Join the Firefly Network
              </Link>
              <Link
                href="/explore?domain=firefly-network"
                className="px-8 py-4 bg-white/5 text-white font-semibold rounded-xl border border-white/10 hover:bg-white/10 transition-all"
              >
                Explore Public Labs
              </Link>
            </div>

            <p className="mt-8 text-gray-500 text-sm">
              Open source. Community driven. Building in public.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5">
        <div className="max-w-6xl mx-auto text-center text-gray-500 text-sm">
          <p className="mb-4">
            The Firefly Network is an open-source project by LabFork
          </p>
          <div className="flex items-center justify-center gap-6">
            <Link href="/" className="hover:text-white transition-colors">
              Home
            </Link>
            <Link href="/domains" className="hover:text-white transition-colors">
              Domains
            </Link>
            <Link href="/explore" className="hover:text-white transition-colors">
              Explore
            </Link>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-white transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </footer>
    </main>
  );
}
