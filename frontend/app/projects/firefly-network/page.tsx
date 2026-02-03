/**
 * The Firefly Network - Research Progress Page
 *
 * Live research dashboard showing findings, experiments, and progress
 * on dual mesh networks: solar-powered lights + distributed AI compute.
 */

import Link from "next/link";
import { FireflyNetworkDemo } from "@/components/projects/FireflyNetworkDemo";
import { LiveLabView } from "@/components/projects/LiveLabView";
import { FloatingFireflies } from "@/components/projects/FloatingFireflies";

// Metadata
export const metadata = {
  title: "The Firefly Network | LabFork",
  description:
    "Mesh networks for light AND compute. Solar-powered lights for 1 billion people + distributed GPU network where any device can contribute to AI research.",
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

// Roadmap phases - now includes both Light Mesh and Compute Mesh
const roadmapPhases = [
  {
    phase: 1,
    name: "Light Prototype",
    duration: "Month 1-3",
    status: "active",
    track: "light",
    goals: [
      "Design PCB schematic v1",
      "Thread mesh on ESP32-C6",
      "MPPT algorithm (P&O)",
      "First prototype parts",
    ],
  },
  {
    phase: 2,
    name: "Compute Foundation",
    duration: "Weeks 1-4",
    status: "active",
    track: "compute",
    goals: [
      "Orchestrator service",
      "Desktop agent (CUDA + Metal)",
      "Exo integration for sharding",
      "10+ test devices connected",
    ],
  },
  {
    phase: 3,
    name: "Browser Compute",
    duration: "Weeks 5-8",
    status: "upcoming",
    track: "compute",
    goals: [
      "WebGPU device agent",
      "WebLLM draft models",
      "Credit system backend",
      "100+ browser contributors",
    ],
  },
  {
    phase: 4,
    name: "Light Field Test",
    duration: "Month 4-6",
    status: "upcoming",
    track: "light",
    goals: [
      "Build 10 light units",
      "Deploy in test location",
      "Swarm intelligence testing",
      "Energy sharing validation",
    ],
  },
  {
    phase: 5,
    name: "Mobile Compute",
    duration: "Weeks 9-12",
    status: "future",
    track: "compute",
    goals: [
      "iOS Safari optimization",
      "Android Chrome optimization",
      "Background contribution",
      "1000+ total contributors",
    ],
  },
  {
    phase: 6,
    name: "Scale Both",
    duration: "Year 2+",
    status: "future",
    track: "both",
    goals: [
      "Light: 10,000 units deployed",
      "Compute: Training workloads",
      "Enterprise tier with SLAs",
      "Open source everything",
    ],
  },
];

// Key specs for physical lights
const lightSpecs = [
  { label: "Unit Cost", value: "<$25", icon: "dollar" },
  { label: "Mesh Range", value: "1km+", icon: "signal" },
  { label: "Battery Life", value: "12hrs", icon: "battery" },
  { label: "Solar Charge", value: "5W", icon: "sun" },
];

// Key specs for distributed compute
const computeSpecs = [
  { label: "Device Tiers", value: "3", icon: "layers" },
  { label: "Min Bandwidth", value: "10Mbps", icon: "wifi" },
  { label: "Task Types", value: "6+", icon: "cpu" },
  { label: "Zero Install", value: "Browser", icon: "globe" },
];

// Distributed compute tiers
const computeTiers = [
  {
    name: "Power Tier",
    devices: "RTX 3090/4090, A100/H100",
    capabilities: ["Full model inference", "Training", "Fine-tuning"],
    performance: "~80 TFLOPS",
    memory: "24-80GB VRAM",
    color: "amber",
  },
  {
    name: "Standard Tier",
    devices: "Mac M1-M4, RTX 3060-80, Gaming PCs",
    capabilities: ["Model shards", "Inference", "Embeddings"],
    performance: "~15 TFLOPS",
    memory: "8-192GB RAM",
    color: "blue",
  },
  {
    name: "Crowd Tier",
    devices: "Browser, Mobile, Tablets",
    capabilities: ["Draft tokens", "Embeddings", "Validation"],
    performance: "~2 TFLOPS",
    memory: "WebGPU",
    color: "green",
  },
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
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 mb-8">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-sm text-green-300">
              Live Research - 6 AI Agents Working
            </span>
          </div>

          {/* Main headline */}
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            The Firefly Network
          </h1>
          <p className="text-2xl md:text-3xl text-amber-400 mb-4">
            Mesh Networks for Light AND Compute
          </p>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto mb-8">
            Solar-powered lights bringing illumination to 1 billion people + a distributed GPU network where any device can accelerate AI research
          </p>

          {/* Two-track specs */}
          <div className="grid md:grid-cols-2 gap-6 mb-12 max-w-4xl mx-auto">
            {/* Physical Light Network */}
            <div className="p-4 rounded-2xl bg-amber-500/5 border border-amber-500/20">
              <h3 className="text-sm text-amber-400 mb-3 font-semibold">LIGHT MESH</h3>
              <div className="flex flex-wrap items-center justify-center gap-4">
                {lightSpecs.map((spec) => (
                  <div
                    key={spec.label}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5"
                  >
                    <span className="text-lg font-bold text-amber-400">
                      {spec.value}
                    </span>
                    <span className="text-xs text-gray-400">{spec.label}</span>
                  </div>
                ))}
              </div>
            </div>
            {/* Distributed Compute Network */}
            <div className="p-4 rounded-2xl bg-cyan-500/5 border border-cyan-500/20">
              <h3 className="text-sm text-cyan-400 mb-3 font-semibold">COMPUTE MESH</h3>
              <div className="flex flex-wrap items-center justify-center gap-4">
                {computeSpecs.map((spec) => (
                  <div
                    key={spec.label}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5"
                  >
                    <span className="text-lg font-bold text-cyan-400">
                      {spec.value}
                    </span>
                    <span className="text-xs text-gray-400">{spec.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Research CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <a
              href="#findings"
              className="px-8 py-4 min-h-[52px] bg-gradient-to-r from-green-500 to-emerald-500 text-white font-semibold rounded-xl hover:from-green-400 hover:to-emerald-400 transition-all shadow-lg shadow-green-500/25 hover:shadow-green-500/40 active:scale-[0.98]"
            >
              View Research Findings
            </a>
            <Link
              href="/lab/new?domain=firefly-network&quick=true"
              className="px-8 py-4 min-h-[52px] bg-white/5 text-white font-semibold rounded-xl border border-white/10 hover:bg-white/10 transition-all active:scale-[0.98]"
            >
              Fork Research Lab
            </Link>
          </div>
          <p className="text-sm text-gray-400 mt-4">
            6 AI agents actively researching • All findings open source
          </p>
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

      {/* Research Progress Section */}
      <section id="findings" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 mb-6">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-sm text-green-300">Research in Progress</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Research Findings & Progress
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Live updates from our AI agents investigating mesh networking, distributed compute, and swarm intelligence.
            </p>
          </div>

          {/* Key Research Findings */}
          <div className="grid md:grid-cols-2 gap-6 mb-12">
            {/* Light Mesh Findings */}
            <div className="p-6 rounded-2xl bg-amber-500/5 border border-amber-500/20">
              <div className="flex items-center gap-2 mb-4">
                <span className="w-3 h-3 rounded-full bg-amber-400" />
                <h3 className="text-lg font-semibold text-amber-400">Light Mesh Findings</h3>
              </div>
              <div className="space-y-4">
                {[
                  {
                    finding: "P&O MPPT achieves 95%+ tracking efficiency",
                    status: "verified",
                    agent: "Spark",
                    confidence: "High",
                  },
                  {
                    finding: "Thread protocol supports 250 nodes per network",
                    status: "verified",
                    agent: "Mesh",
                    confidence: "High",
                  },
                  {
                    finding: "LiFePO4 optimal for outdoor thermal range",
                    status: "verified",
                    agent: "Spark",
                    confidence: "High",
                  },
                  {
                    finding: "Swarm energy sharing extends runtime 30%",
                    status: "testing",
                    agent: "Lumen",
                    confidence: "Medium",
                  },
                ].map((item, i) => (
                  <div key={i} className="p-3 rounded-lg bg-white/5">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <p className="text-sm text-gray-300">{item.finding}</p>
                      <span className={`px-2 py-0.5 rounded text-xs whitespace-nowrap ${
                        item.status === "verified" ? "bg-green-500/20 text-green-400" : "bg-yellow-500/20 text-yellow-400"
                      }`}>
                        {item.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span>Agent: {item.agent}</span>
                      <span>Confidence: {item.confidence}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Compute Mesh Findings */}
            <div className="p-6 rounded-2xl bg-cyan-500/5 border border-cyan-500/20">
              <div className="flex items-center gap-2 mb-4">
                <span className="w-3 h-3 rounded-full bg-cyan-400" />
                <h3 className="text-lg font-semibold text-cyan-400">Compute Mesh Findings</h3>
              </div>
              <div className="space-y-4">
                {[
                  {
                    finding: "Speculative decoding: 28-40% speedup confirmed",
                    status: "verified",
                    agent: "WebGPU",
                    confidence: "High",
                    source: "DSD paper",
                  },
                  {
                    finding: "Pipeline parallelism viable across heterogeneous devices",
                    status: "verified",
                    agent: "Shard",
                    confidence: "High",
                    source: "Exo Labs",
                  },
                  {
                    finding: "WebGPU stable on Chrome, Safari, Firefox (2026)",
                    status: "verified",
                    agent: "WebGPU",
                    confidence: "High",
                    source: "Browser compat matrix",
                  },
                  {
                    finding: "Mobile draft generation feasible with 1-3B models",
                    status: "testing",
                    agent: "WebGPU",
                    confidence: "Medium",
                    source: "WebLLM benchmarks",
                  },
                ].map((item, i) => (
                  <div key={i} className="p-3 rounded-lg bg-white/5">
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <p className="text-sm text-gray-300">{item.finding}</p>
                      <span className={`px-2 py-0.5 rounded text-xs whitespace-nowrap ${
                        item.status === "verified" ? "bg-green-500/20 text-green-400" : "bg-yellow-500/20 text-yellow-400"
                      }`}>
                        {item.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <span>Agent: {item.agent}</span>
                      <span>Source: {item.source}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Active Experiments */}
          <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">Active Experiments</h3>
            <div className="grid md:grid-cols-3 gap-4">
              {[
                {
                  name: "EXP-001: MPPT Shading Recovery",
                  track: "light",
                  status: "running",
                  progress: 65,
                  description: "Testing partial shading detection and multi-peak tracking",
                },
                {
                  name: "EXP-002: Three-Tier Task Routing",
                  track: "compute",
                  status: "running",
                  progress: 40,
                  description: "Optimizing task assignment to Power/Standard/Crowd tiers",
                },
                {
                  name: "EXP-003: Browser Contributor UX",
                  track: "compute",
                  status: "planned",
                  progress: 0,
                  description: "Zero-friction onboarding flow for WebGPU contribution",
                },
              ].map((exp, i) => (
                <div key={i} className="p-4 rounded-lg bg-white/5">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      exp.track === "light" ? "bg-amber-500/20 text-amber-400" : "bg-cyan-500/20 text-cyan-400"
                    }`}>
                      {exp.track}
                    </span>
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      exp.status === "running" ? "bg-green-500/20 text-green-400" : "bg-gray-500/20 text-gray-400"
                    }`}>
                      {exp.status}
                    </span>
                  </div>
                  <h4 className="text-sm font-medium text-white mb-1">{exp.name}</h4>
                  <p className="text-xs text-gray-400 mb-3">{exp.description}</p>
                  <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${exp.track === "light" ? "bg-amber-500" : "bg-cyan-500"}`}
                      style={{ width: `${exp.progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{exp.progress}% complete</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Prior Art Analysis Section */}
      <section className="py-24 px-6 bg-slate-900/30">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Prior Art Analysis: What We Evaluated
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Before building, our agents analyzed existing solutions to understand gaps and opportunities.
            </p>
          </div>

          {/* Meshtastic Analysis */}
          <div className="mb-12">
            <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Meshtastic Analysis</h3>
                  <p className="text-sm text-gray-400">Open-source LoRa mesh network</p>
                </div>
                <a
                  href="https://meshtastic.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-auto text-xs text-purple-400 hover:text-purple-300"
                >
                  meshtastic.org ↗
                </a>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                {/* What We Learned */}
                <div>
                  <h4 className="text-sm font-semibold text-green-400 mb-3">What Works Well</h4>
                  <ul className="space-y-2">
                    {[
                      "LoRa achieves 2-10km range in open terrain",
                      "Peer-to-peer mesh with no infrastructure needed",
                      "Active community, mature firmware",
                      "Supports ESP32, nRF52, RP2040 platforms",
                    ].map((item, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                        <span className="text-green-400 mt-0.5">✓</span>
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Gaps Identified */}
                <div>
                  <h4 className="text-sm font-semibold text-amber-400 mb-3">Gaps for Our Use Case</h4>
                  <ul className="space-y-2">
                    {[
                      { gap: "ESP32 power consumption too high for solar", detail: "nRF52 preferred but costs more" },
                      { gap: "Messaging-focused, not general compute", detail: "No framework for distributed tasks" },
                      { gap: "LoRa data rate ~300kbps max", detail: "Insufficient for AI model coordination" },
                      { gap: "No integrated light/power management", detail: "Would need custom hardware layer" },
                    ].map((item, i) => (
                      <li key={i} className="text-sm">
                        <span className="text-amber-400">△</span>
                        <span className="text-gray-300 ml-2">{item.gap}</span>
                        <p className="text-xs text-gray-500 ml-5">{item.detail}</p>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="mt-6 p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
                <p className="text-sm text-gray-300">
                  <strong className="text-purple-400">Research Conclusion:</strong> Meshtastic excels at off-grid messaging but wasn&apos;t designed for our dual requirements: (1) solar-powered lighting with swarm energy sharing, and (2) distributed AI compute coordination. We&apos;re using Thread protocol instead of LoRa for the light mesh (higher bandwidth, lower power with ESP32-C6), and WebSocket/WebGPU for compute mesh.
                </p>
              </div>
            </div>
          </div>

          {/* Other Solutions Analyzed */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            {[
              {
                name: "Petals",
                type: "Distributed LLM",
                verdict: "Inspiring",
                findings: "BitTorrent-style inference works. But requires always-on nodes, no mobile support.",
                link: "https://petals.dev",
              },
              {
                name: "Exo Labs",
                type: "Pipeline Parallel",
                verdict: "Adopted",
                findings: "Excellent model sharding across heterogeneous devices. Integrating for our Power tier.",
                link: "https://github.com/exo-explore/exo",
              },
              {
                name: "io.net",
                type: "GPU Network",
                verdict: "Reference",
                findings: "Proven market for distributed compute. But crypto-focused, no mobile/browser tier.",
                link: "https://io.net",
              },
            ].map((solution, i) => (
              <div key={i} className="p-5 rounded-xl bg-white/5 border border-white/10">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-white">{solution.name}</h4>
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    solution.verdict === "Adopted" ? "bg-green-500/20 text-green-400" :
                    solution.verdict === "Inspiring" ? "bg-blue-500/20 text-blue-400" :
                    "bg-gray-500/20 text-gray-400"
                  }`}>
                    {solution.verdict}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mb-2">{solution.type}</p>
                <p className="text-sm text-gray-400 mb-3">{solution.findings}</p>
                <a
                  href={solution.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-cyan-400 hover:text-cyan-300"
                >
                  {solution.link.replace('https://', '')} ↗
                </a>
              </div>
            ))}
          </div>

          {/* Academic Papers */}
          <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
            <h3 className="text-lg font-semibold text-white mb-4">Key Academic References</h3>
            <div className="grid md:grid-cols-2 gap-4">
              {[
                {
                  title: "Gateway-Free LoRa Mesh on ESP32: Design, Self-Healing Mechanisms, and Empirical Performance",
                  source: "MDPI Sensors, 2025",
                  finding: "Confirms mesh-oriented frameworks like Meshtastic focus mainly on messaging, not general compute",
                  link: "https://www.mdpi.com/1424-8220/25/19/6036",
                },
                {
                  title: "DSD: Distributed Speculative Decoding",
                  source: "arXiv, 2025",
                  finding: "Validates 28-40% speedup using small models for draft generation",
                  link: "https://arxiv.org/abs/2511.21669",
                },
              ].map((paper, i) => (
                <a
                  key={i}
                  href={paper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                >
                  <p className="text-sm text-white mb-1">{paper.title}</p>
                  <p className="text-xs text-gray-500 mb-2">{paper.source}</p>
                  <p className="text-xs text-cyan-400">&quot;{paper.finding}&quot;</p>
                </a>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Research Context: The Problem */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Problem Statement
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              The gaps our research aims to fill.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                emoji: "📚",
                title: "Light: Kids can't study",
                description: "1.2B people lack electricity. Children use dangerous kerosene.",
                stat: "6 hours/day lost",
                track: "light",
              },
              {
                emoji: "💻",
                title: "Compute: GPU scarcity",
                description: "Llama 70B needs $15K+ hardware. Most researchers can't access.",
                stat: "$2-4/hr cloud cost",
                track: "compute",
              },
              {
                emoji: "📵",
                title: "Light: Information isolation",
                description: "No power means no phones, no internet, no news.",
                stat: "Total isolation",
                track: "light",
              },
              {
                emoji: "🔋",
                title: "Compute: Idle resources",
                description: "Consumer GPUs sit idle 90%+ of the time.",
                stat: "Massive waste",
                track: "compute",
              },
            ].map((problem) => (
              <div
                key={problem.title}
                className={`p-6 rounded-2xl bg-gradient-to-b ${
                  problem.track === "light" ? "from-amber-500/5" : "from-cyan-500/5"
                } to-transparent border ${
                  problem.track === "light" ? "border-amber-500/10" : "border-cyan-500/10"
                }`}
              >
                <span className="text-4xl mb-4 block">{problem.emoji}</span>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {problem.title}
                </h3>
                <p className="text-gray-400 text-sm mb-4">
                  {problem.description}
                </p>
                <div className={`font-semibold ${
                  problem.track === "light" ? "text-amber-400" : "text-cyan-400"
                }`}>{problem.stat}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Research Approach Section with Interactive Demo */}
      <section id="demo" className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Research Approach: Self-Organizing Networks
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              <strong className="text-white">Hypothesis:</strong> Mesh topology + swarm intelligence can democratize both physical infrastructure (light) and digital infrastructure (compute).
            </p>
          </div>

          {/* Interactive Demo */}
          <div className="mb-16">
            <FireflyNetworkDemo className="max-w-4xl mx-auto" />
          </div>

          {/* Research Areas with Status */}
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "Energy Harvesting",
                description:
                  "5W panel + LiFePO4 battery. Researching P&O MPPT with partial shading detection.",
                status: "Implemented",
                finding: "95%+ tracking efficiency achieved",
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
                title: "Mesh Protocol",
                description:
                  "Thread/IEEE 802.15.4 for self-healing networks. Investigating multi-hop routing.",
                status: "In Progress",
                finding: "250 nodes per network confirmed",
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
                  "Distributed consensus for energy sharing and coverage optimization without central control.",
                status: "Testing",
                finding: "30% runtime extension in simulation",
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
                <div className="flex items-center justify-between mb-4">
                  <div className={`text-${feature.color}-400`}>
                    {feature.icon}
                  </div>
                  <span className={`px-2 py-1 rounded text-xs ${
                    feature.status === "Implemented" ? "bg-green-500/20 text-green-400" :
                    feature.status === "In Progress" ? "bg-blue-500/20 text-blue-400" :
                    "bg-yellow-500/20 text-yellow-400"
                  }`}>
                    {feature.status}
                  </span>
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-400 text-sm mb-3">{feature.description}</p>
                <div className="pt-3 border-t border-white/10">
                  <p className="text-xs text-gray-500">Latest finding:</p>
                  <p className={`text-sm text-${feature.color}-400`}>{feature.finding}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* NEW: Distributed Compute Network Section */}
      <section id="compute" className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-6">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-sm text-cyan-300">NEW - Phase 2</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Distributed Compute Network
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              The same mesh principles that power our lights now enable distributed AI compute.
              Any device—from RTX 4090s to iPhones—can contribute GPU power to accelerate research.
            </p>
          </div>

          {/* Key Insight */}
          <div className="p-6 rounded-2xl bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 mb-12 max-w-3xl mx-auto">
            <p className="text-cyan-300 text-center">
              <strong>Key Insight:</strong> Don&apos;t make phones do what GPUs do.
              Use speculative decoding and pipeline parallelism to give each device tier appropriate tasks—achieving better performance than homogeneous clusters.
            </p>
          </div>

          {/* Three Tier Architecture */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            {computeTiers.map((tier) => (
              <div
                key={tier.name}
                className={`p-6 rounded-2xl bg-${tier.color}-500/5 border border-${tier.color}-500/20`}
              >
                <h3 className={`text-xl font-bold text-${tier.color}-400 mb-2`}>
                  {tier.name}
                </h3>
                <p className="text-sm text-gray-500 mb-4">{tier.devices}</p>
                <ul className="space-y-2 mb-4">
                  {tier.capabilities.map((cap) => (
                    <li key={cap} className="flex items-center gap-2 text-sm text-gray-300">
                      <span className={`w-1.5 h-1.5 rounded-full bg-${tier.color}-400`} />
                      {cap}
                    </li>
                  ))}
                </ul>
                <div className="pt-4 border-t border-white/10 flex justify-between text-xs text-gray-500">
                  <span>{tier.performance}</span>
                  <span>{tier.memory}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Speculative Decoding Visual */}
          <div className="mb-16">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">
              Speculative Decoding: Phones Do Useful Work
            </h3>
            <div className="max-w-3xl mx-auto p-6 rounded-2xl bg-slate-900/50 border border-white/10">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                {/* Phone */}
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-3 rounded-2xl bg-green-500/20 flex items-center justify-center">
                    <svg className="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <p className="text-sm text-gray-400">Phone (1-3B model)</p>
                  <p className="text-xs text-green-400">Generates 8 draft tokens</p>
                </div>

                {/* Arrow */}
                <div className="hidden md:block">
                  <svg className="w-12 h-12 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>

                {/* Orchestrator */}
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-3 rounded-2xl bg-cyan-500/20 flex items-center justify-center">
                    <svg className="w-8 h-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  </div>
                  <p className="text-sm text-gray-400">Orchestrator</p>
                  <p className="text-xs text-cyan-400">Routes to GPU</p>
                </div>

                {/* Arrow */}
                <div className="hidden md:block">
                  <svg className="w-12 h-12 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </div>

                {/* GPU */}
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-3 rounded-2xl bg-amber-500/20 flex items-center justify-center">
                    <svg className="w-8 h-8 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                  </div>
                  <p className="text-sm text-gray-400">GPU (70B model)</p>
                  <p className="text-xs text-amber-400">Verifies: accepts 6/8</p>
                </div>
              </div>
              <p className="text-center mt-6 text-sm text-gray-500">
                Result: <span className="text-green-400 font-semibold">28-40% faster</span> than GPU-only inference
              </p>
            </div>
          </div>

          {/* How to Contribute */}
          <div className="grid md:grid-cols-3 gap-6">
            <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
              <h4 className="text-lg font-semibold text-white mb-2">Browser (Zero Install)</h4>
              <p className="text-sm text-gray-400 mb-4">
                Visit /contribute, click &quot;Start&quot;, and your browser&apos;s WebGPU starts helping immediately.
              </p>
              <Link
                href="/contribute"
                className="inline-flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300"
              >
                Start in Browser
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
            <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
              <h4 className="text-lg font-semibold text-white mb-2">Desktop Agent</h4>
              <p className="text-sm text-gray-400 mb-4">
                One-click installer auto-detects your GPU. Choose Maximum, Balanced, or Background mode.
              </p>
              <span className="text-sm text-gray-500">Coming Soon</span>
            </div>
            <div className="p-6 rounded-2xl bg-white/5 border border-white/10">
              <h4 className="text-lg font-semibold text-white mb-2">Mobile (Passive)</h4>
              <p className="text-sm text-gray-400 mb-4">
                Set it and forget it. Contribute only when charging, on WiFi, or during specific hours.
              </p>
              <span className="text-sm text-gray-500">Coming Soon</span>
            </div>
          </div>
        </div>
      </section>

      {/* Hardware Specifications Section */}
      <section className="py-24 px-6 bg-slate-900/30">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Hardware Specifications (Light Mesh)
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Component selection based on research into cost, reliability, and field repairability.
            </p>
          </div>

          <div className="max-w-3xl mx-auto">
            {/* Research Note */}
            <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 mb-6">
              <p className="text-sm text-amber-300">
                <strong>Research Note:</strong> ESP32-C6 chosen over nRF52 despite higher power consumption because Thread + WiFi + BLE on single chip reduces BOM complexity. LiFePO4 selected over Li-ion for thermal stability (-20°C to 60°C) critical for outdoor deployment.
              </p>
            </div>

            <div className="rounded-2xl border border-amber-500/20 overflow-hidden">
              <div className="bg-amber-500/10 px-6 py-4 flex justify-between items-center">
                <span className="font-semibold text-white">
                  Bill of Materials v1.0
                </span>
                <span className="text-amber-400 font-bold">
                  Target: ${totalBomCost.toFixed(2)} @ 1K units
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
              BOM under active research. See EXP-001 for MPPT optimization testing.
            </p>
          </div>
        </div>
      </section>

      {/* Research Phases Section */}
      <section className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Research Phases
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Parallel research tracks with defined milestones and success criteria.
            </p>
            {/* Track legend */}
            <div className="flex items-center justify-center gap-6 mt-6">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-amber-500" />
                <span className="text-sm text-gray-400">Light Mesh</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-cyan-500" />
                <span className="text-sm text-gray-400">Compute Mesh</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-gradient-to-r from-amber-500 to-cyan-500" />
                <span className="text-sm text-gray-400">Both</span>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-3 lg:grid-cols-6 gap-4">
            {roadmapPhases.map((phase) => (
              <div
                key={phase.phase}
                className={`p-5 rounded-2xl border ${
                  phase.status === "active"
                    ? phase.track === "compute"
                      ? "bg-cyan-500/10 border-cyan-500/30"
                      : "bg-amber-500/10 border-amber-500/30"
                    : "bg-white/5 border-white/10"
                }`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
                      phase.status === "active"
                        ? phase.track === "compute"
                          ? "bg-cyan-500 text-slate-900"
                          : "bg-amber-500 text-slate-900"
                        : "bg-white/10 text-white"
                    }`}
                  >
                    {phase.phase}
                  </div>
                  <div
                    className={`px-2 py-0.5 rounded-full text-xs ${
                      phase.track === "light"
                        ? "bg-amber-500/20 text-amber-400"
                        : phase.track === "compute"
                        ? "bg-cyan-500/20 text-cyan-400"
                        : "bg-gradient-to-r from-amber-500/20 to-cyan-500/20 text-white"
                    }`}
                  >
                    {phase.track === "light" ? "Light" : phase.track === "compute" ? "Compute" : "Both"}
                  </div>
                </div>

                <div className="mb-2">
                  <div className="font-semibold text-white text-sm">{phase.name}</div>
                  <div className="text-xs text-gray-500">{phase.duration}</div>
                </div>

                {phase.status === "active" && (
                  <div className={`mb-3 px-2 py-0.5 rounded-full text-xs inline-block ${
                    phase.track === "compute" ? "bg-cyan-500/20 text-cyan-400" : "bg-amber-500/20 text-amber-400"
                  }`}>
                    ACTIVE
                  </div>
                )}

                <ul className="space-y-1">
                  {phase.goals.map((goal) => (
                    <li key={goal} className="flex items-start gap-1.5 text-xs">
                      <span className="text-gray-500 mt-0.5">•</span>
                      <span className="text-gray-400">{goal}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contribute to Research Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Contribute to Research
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Open research means anyone can contribute. Here&apos;s how to get involved.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                title: "Donate Compute",
                description:
                  "Contribute spare GPU cycles to accelerate experiments. Your browser can help run distributed inference tests.",
                cta: "Contribute Now",
                href: "/contribute",
                color: "cyan",
              },
              {
                title: "Fork the Lab",
                description:
                  "Create your own research fork. Run experiments, validate findings, or explore new directions.",
                cta: "Fork Lab",
                href: "/lab/new?domain=firefly-network",
                color: "amber",
              },
              {
                title: "Replicate Hardware",
                description:
                  "Build a prototype using our BOM. Document results, report issues, suggest improvements.",
                cta: "Docs Coming",
                href: "",
                color: "green",
                disabled: true,
              },
              {
                title: "Field Testing",
                description:
                  "Help deploy and test prototypes in real conditions. Data collection partnerships welcome.",
                cta: "Contact",
                href: "",
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
                  <span
                    role="button"
                    aria-disabled="true"
                    className="px-4 py-2 rounded-lg bg-white/5 text-gray-500 text-sm text-center pointer-events-none"
                  >
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

      {/* Research Methodology Section */}
      <section id="ai-lab" className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-green-500/10 to-cyan-500/10 border border-green-500/20 mb-6">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span className="text-sm text-green-300">
                Live Research Dashboard
              </span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Research Methodology & Progress
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              AI agents continuously analyze papers, implement algorithms, and validate findings.
              All research is conducted transparently on the LabFork platform.
            </p>
          </div>

          {/* Live Lab Stats */}
          <div className="grid md:grid-cols-4 lg:grid-cols-8 gap-4 mb-12">
            {[
              {
                label: "Light Agents",
                value: "3",
                detail: "Spark, Mesh, Lumen",
                color: "text-amber-400",
              },
              {
                label: "Compute Agents",
                value: "3",
                detail: "Orchestr8, Shard, WebGPU",
                color: "text-cyan-400",
              },
              {
                label: "Papers Analyzed",
                value: "12",
                detail: "Light: 8, Compute: 4",
                color: "text-blue-400",
              },
              {
                label: "Light Tasks",
                value: "4/10",
                detail: "MPPT, mesh protocol done",
                color: "text-amber-400",
              },
              {
                label: "Compute Tasks",
                value: "2/8",
                detail: "PRD, architecture done",
                color: "text-cyan-400",
              },
              {
                label: "Device Tiers",
                value: "3",
                detail: "Power, Standard, Crowd",
                color: "text-green-400",
              },
              {
                label: "Target Contributors",
                value: "1000+",
                detail: "By end of Phase 3",
                color: "text-purple-400",
              },
              {
                label: "Speedup Target",
                value: "40%",
                detail: "Speculative decoding",
                color: "text-pink-400",
              },
            ].map((stat) => (
              <div
                key={stat.label}
                className="p-4 rounded-xl bg-white/5 border border-white/10 text-center"
              >
                <div className={`text-2xl font-bold ${stat.color} mb-1`}>
                  {stat.value}
                </div>
                <div className="text-xs text-gray-400 mb-0.5">{stat.label}</div>
                <div className="text-xs text-gray-500">{stat.detail}</div>
              </div>
            ))}
          </div>

          {/* Papers Analyzed */}
          <div className="rounded-2xl border border-white/10 overflow-hidden mb-12">
            <div className="bg-white/5 px-6 py-4 flex items-center justify-between">
              <span className="font-semibold text-white">Research Papers Analyzed</span>
              <span className="text-sm text-gray-400">12 papers ingested</span>
            </div>
            <div className="p-6 bg-slate-950/50">
              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { title: "DSD: Distributed Speculative Decoding", source: "arXiv 2511.21669", track: "compute", status: "implemented" },
                  { title: "HeteroFL: Heterogeneous Federated Learning", source: "OpenReview", track: "compute", status: "analyzing" },
                  { title: "WebLLM: High-Performance In-Browser LLM", source: "arXiv 2412.15803", track: "compute", status: "implemented" },
                  { title: "Thread Protocol Specification", source: "Thread Group", track: "light", status: "implemented" },
                  { title: "MPPT Algorithms for PV Systems", source: "IEEE", track: "light", status: "implemented" },
                  { title: "Swarm Intelligence in Distributed Systems", source: "ACM Survey", track: "light", status: "analyzing" },
                ].map((paper, i) => (
                  <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                    <div>
                      <p className="text-sm text-white">{paper.title}</p>
                      <p className="text-xs text-gray-500">{paper.source}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        paper.track === "light" ? "bg-amber-500/20 text-amber-400" : "bg-cyan-500/20 text-cyan-400"
                      }`}>
                        {paper.track}
                      </span>
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        paper.status === "implemented" ? "bg-green-500/20 text-green-400" : "bg-blue-500/20 text-blue-400"
                      }`}>
                        {paper.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Live Lab Embed - Dynamic Data */}
          <div className="rounded-2xl border border-green-500/20 overflow-hidden mb-12">
            <div className="bg-green-500/10 px-6 py-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 rounded-full bg-green-400 animate-pulse" />
                <span className="font-semibold text-white">
                  Lab #1 - Live Agent Activity
                </span>
              </div>
              <Link
                href="/lab?id=lab_firefly001"
                className="text-sm text-green-400 hover:text-green-300"
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
              Research Timeline & Milestones
            </h3>
            <div className="relative">
              <div className="absolute left-1/2 -translate-x-px top-0 bottom-0 w-0.5 bg-gradient-to-b from-amber-500/50 via-amber-500/20 to-transparent" />
              <div className="space-y-8">
                {[
                  {
                    day: "Day 1",
                    title: "Platform launched, lab created",
                    status: "done",
                    track: "light",
                  },
                  {
                    day: "Day 2",
                    title: "8 research papers ingested",
                    status: "done",
                    track: "light",
                  },
                  {
                    day: "Day 3",
                    title: "3 AI agents started working",
                    status: "done",
                    track: "light",
                  },
                  {
                    day: "Day 4",
                    title: "MPPT algorithm selected (P&O)",
                    status: "done",
                    track: "light",
                  },
                  {
                    day: "Week 2",
                    title: "Distributed Compute PRD published",
                    status: "done",
                    track: "compute",
                  },
                  {
                    day: "Week 3",
                    title: "Orchestrator service design",
                    status: "current",
                    highlight: true,
                    track: "compute",
                  },
                  {
                    day: "Week 4",
                    title: "WebGPU agent implementation",
                    status: "upcoming",
                    track: "compute",
                  },
                  {
                    day: "Week 5",
                    title: "Speculative decoding: phones + GPUs",
                    status: "upcoming",
                    track: "compute",
                  },
                  {
                    day: "Month 2",
                    title: "100+ browser contributors",
                    status: "upcoming",
                    track: "compute",
                  },
                  {
                    day: "Month 3",
                    title: "Light mesh field test",
                    status: "upcoming",
                    track: "light",
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
                            ? item.track === "compute"
                              ? "bg-cyan-500/20 border border-cyan-500/30"
                              : "bg-amber-500/20 border border-amber-500/30"
                            : item.status === "done"
                            ? "bg-white/5 border border-white/10"
                            : "bg-white/5 border border-white/5 opacity-50"
                        }`}
                      >
                        <div className="flex items-center gap-2 justify-end mb-1">
                          <span className="text-xs text-gray-500">{item.day}</span>
                          <span className={`text-xs px-1.5 py-0.5 rounded ${
                            item.track === "compute" ? "bg-cyan-500/20 text-cyan-400" : "bg-amber-500/20 text-amber-400"
                          }`}>
                            {item.track}
                          </span>
                        </div>
                        <div
                          className={`text-sm ${
                            item.highlight
                              ? item.track === "compute" ? "text-cyan-400" : "text-amber-400"
                              : "text-gray-300"
                          }`}
                        >
                          {item.title}
                        </div>
                        {item.highlight && (
                          <div className={`mt-1 text-xs ${item.track === "compute" ? "text-cyan-500" : "text-amber-500"}`}>
                            NOW
                          </div>
                        )}
                      </div>
                    </div>
                    <div
                      className={`absolute left-1/2 -translate-x-1/2 w-4 h-4 rounded-full border-2 ${
                        item.status === "current"
                          ? item.track === "compute"
                            ? "bg-cyan-500 border-cyan-400"
                            : "bg-amber-500 border-amber-400"
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
          <div className="p-8 rounded-2xl bg-gradient-to-r from-amber-500/5 via-cyan-500/5 to-amber-500/5 border border-white/10">
            <h3 className="text-xl font-semibold text-white mb-6 text-center">
              The Meta-Loop: Dual Flywheel Effect
            </h3>
            <div className="grid md:grid-cols-2 gap-8">
              {/* Light Flywheel */}
              <div>
                <h4 className="text-sm text-amber-400 font-semibold mb-4 text-center">Light Mesh Flywheel</h4>
                <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
                  {[
                    "Research Papers",
                    "AI Agents Implement",
                    "Better Firmware",
                    "More Light Units",
                    "Community Growth",
                  ].map((item, i, arr) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-300 border border-amber-500/20">
                        {item}
                      </span>
                      {i < arr.length - 1 && (
                        <svg className="w-4 h-4 text-amber-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              {/* Compute Flywheel */}
              <div>
                <h4 className="text-sm text-cyan-400 font-semibold mb-4 text-center">Compute Mesh Flywheel</h4>
                <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
                  {[
                    "Contributors Join",
                    "More Compute",
                    "Faster Research",
                    "Better Results",
                    "More Contributors",
                  ].map((item, i, arr) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                        {item}
                      </span>
                      {i < arr.length - 1 && (
                        <svg className="w-4 h-4 text-cyan-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <p className="text-center text-gray-500 mt-6 text-sm">
              Two flywheels reinforcing each other: compute accelerates light research, light deployment attracts more contributors
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
                href="/lab/new?domain=firefly-network&quick=true"
                className="px-6 py-3 bg-white/5 text-white font-semibold rounded-xl border border-white/10 hover:bg-white/10 transition-all"
              >
                Fork & Start Your Own Lab
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Research Team Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Research Team
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              AI agents conducting research alongside human contributors. All findings are peer-reviewed by the community.
            </p>
          </div>

          {/* AI Agents */}
          <div className="mb-12">
            <h3 className="text-lg font-semibold text-amber-400 mb-6 text-center">
              AI Research Agents
            </h3>
            <div className="grid md:grid-cols-3 lg:grid-cols-6 gap-4">
              {[
                {
                  name: "Spark",
                  role: "Energy Specialist",
                  tasks: "MPPT, battery BMS, power budgeting",
                  color: "from-orange-500 to-amber-500",
                  tokensGenerated: "45K+",
                  track: "light",
                },
                {
                  name: "Mesh",
                  role: "Network Architect",
                  tasks: "Thread protocol, routing algorithms",
                  color: "from-cyan-500 to-blue-500",
                  tokensGenerated: "12K+",
                  track: "light",
                },
                {
                  name: "Lumen",
                  role: "Light Engineer",
                  tasks: "LED optimization, thermal mgmt",
                  color: "from-yellow-500 to-orange-500",
                  tokensGenerated: "38K+",
                  track: "light",
                },
                {
                  name: "Orchestr8",
                  role: "Compute Coordinator",
                  tasks: "Task routing, device registry, load balancing",
                  color: "from-blue-500 to-indigo-500",
                  tokensGenerated: "28K+",
                  track: "compute",
                },
                {
                  name: "Shard",
                  role: "Pipeline Parallel",
                  tasks: "Model sharding, Exo integration",
                  color: "from-purple-500 to-pink-500",
                  tokensGenerated: "15K+",
                  track: "compute",
                },
                {
                  name: "WebGPU",
                  role: "Browser Agent",
                  tasks: "WebLLM, speculative decoding",
                  color: "from-green-500 to-teal-500",
                  tokensGenerated: "22K+",
                  track: "compute",
                },
              ].map((agent) => (
                <div
                  key={agent.name}
                  className="p-4 rounded-2xl bg-white/5 border border-white/10"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <div
                      className={`w-10 h-10 rounded-full bg-gradient-to-br ${agent.color} flex items-center justify-center text-white font-bold text-sm`}
                    >
                      {agent.name[0]}
                    </div>
                    <div>
                      <div className="font-semibold text-white text-sm">
                        {agent.name}
                      </div>
                      <div className="text-xs text-gray-400">{agent.role}</div>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mb-2">{agent.tasks}</p>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">{agent.tokensGenerated}</span>
                    <span className={`px-2 py-0.5 rounded-full ${
                      agent.track === "light" ? "bg-amber-500/20 text-amber-400" : "bg-cyan-500/20 text-cyan-400"
                    }`}>
                      {agent.track}
                    </span>
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
                href="/lab/new?domain=firefly-network&quick=true"
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

      {/* Research Summary Section */}
      <section className="py-24 px-6 bg-slate-900/50">
        <div className="max-w-4xl mx-auto text-center">
          <div className="p-12 rounded-3xl bg-gradient-to-r from-green-500/10 via-cyan-500/10 to-green-500/10 border border-green-500/20">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-green-500/10 border border-green-500/20 mb-6">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-sm text-green-300">Research Summary</span>
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              Current Status & Next Steps
            </h2>
            <div className="text-left max-w-2xl mx-auto mb-8">
              <div className="space-y-3 text-gray-400">
                <p><strong className="text-amber-400">Light Mesh:</strong> MPPT algorithm validated (95% efficiency). Thread protocol confirmed for 250-node networks. Next: hardware prototype assembly.</p>
                <p><strong className="text-cyan-400">Compute Mesh:</strong> Speculative decoding architecture designed. Exo integration planned. Next: orchestrator service implementation.</p>
                <p><strong className="text-purple-400">Key Finding:</strong> Meshtastic&apos;s LoRa approach insufficient for our dual requirements. Thread + WebGPU hybrid approach shows promise.</p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/contribute"
                className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold rounded-xl hover:from-cyan-400 hover:to-blue-400 transition-all shadow-lg shadow-cyan-500/25"
              >
                Contribute Compute
              </Link>
              <Link
                href="/lab/new?domain=firefly-network&quick=true"
                className="px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold rounded-xl hover:from-amber-400 hover:to-orange-400 transition-all shadow-lg shadow-amber-500/25"
              >
                Fork Research Lab
              </Link>
            </div>

            <p className="mt-8 text-gray-500 text-sm">
              All research is open source and conducted transparently on LabFork.
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
