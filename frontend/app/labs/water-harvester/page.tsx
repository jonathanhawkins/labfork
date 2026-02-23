"use client";

import { useState, useMemo, useEffect } from "react";
import Link from "next/link";
import {
  Droplets,
  Sun,
  Thermometer,
  Wind,
  Leaf,
  Bug,
  FlaskConical,
  Printer,
  Calculator,
  FileText,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Zap,
  DollarSign,
  Clock,
  MapPin,
  Github,
  BookOpen,
  Lightbulb,
  Layers,
  Settings2,
  Play,
  Download,
  Share2,
  Star,
  GitFork,
  Loader2,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useSimulation, formatSimulationResults } from "@/lib/simulations/use-simulation";
import type { SimulationParams } from "@/lib/simulations/types";
import dynamic from "next/dynamic";

// Dynamic import for 3D scene (client-side only, no SSR)
// Using MOFHarvesterScene - physics-accurate MIT/Berkeley MOF-801 design
const MOFHarvesterScene = dynamic(
  () => import("@/components/MOFHarvesterScene"),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-background-card">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-foreground-muted">Loading MOF Water Harvester...</p>
        </div>
      </div>
    ),
  }
);

// CFD Results visualization (OpenFOAM simulation data)
const CFDResultsVisualization = dynamic(
  () => import("@/components/CFDResultsVisualization"),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-64 flex items-center justify-center bg-gray-900 rounded-xl">
        <div className="text-center">
          <div className="w-10 h-10 border-4 border-blue-400 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-gray-400">Loading CFD Results...</p>
        </div>
      </div>
    ),
  }
);

// Research papers data
const RESEARCH_PAPERS = [
  {
    title: "Water harvesting from air with metal-organic frameworks powered by natural sunlight",
    authors: "Kim et al.",
    journal: "Science",
    year: 2017,
    url: "https://www.science.org/doi/10.1126/science.aam8743",
    keyFinding: "MOFs can harvest water at 20% humidity using only solar heat",
    category: "mof",
  },
  {
    title: "Enhanced continuous atmospheric water harvesting with scalable hygroscopic gel",
    authors: "Nature Communications",
    journal: "Nature Communications",
    year: 2024,
    url: "https://www.nature.com/articles/s41467-024-52137-4",
    keyFinding: "3.5-8.9 L/m²/day with concentrated solar + hygroscopic gel",
    category: "sorbent",
  },
  {
    title: "Environmentally adaptive MOF-based device enables continuous self-optimizing atmospheric water harvesting",
    authors: "Nature Communications",
    journal: "Nature Communications",
    year: 2022,
    url: "https://www.nature.com/articles/s41467-022-32642-0",
    keyFinding: "169% increase in water production with adaptive harvesting",
    category: "mof",
  },
  {
    title: "Three-Dimensionally Structured Flexible Fog Harvesting Surfaces Inspired by Namib Desert Beetles",
    authors: "MDPI Micromachines",
    journal: "Micromachines",
    year: 2019,
    url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6470850/",
    keyFinding: "16x higher water collection with 3D bumpy beetle-inspired surfaces",
    category: "biomimicry",
  },
  {
    title: "Highly efficient fog harvesting on superhydrophobic microfibers through droplet oscillation and sweeping",
    authors: "Soft Matter",
    journal: "Soft Matter",
    year: 2018,
    url: "https://pubmed.ncbi.nlm.nih.gov/30320332/",
    keyFinding: "Vibration causes droplet coalescence, improving collection efficiency",
    category: "biomimicry",
  },
  {
    title: "High-performance solar-driven water harvesting from air with cheap and scalable hygroscopic salt modified MOF",
    authors: "Chemical Engineering Journal",
    journal: "Chemical Engineering Journal",
    year: 2023,
    url: "https://www.sciencedirect.com/science/article/abs/pii/S1385894723006861",
    keyFinding: "CaCl₂ + MOF-808 achieves 7x better uptake at 30% RH, cheap materials",
    category: "sorbent",
  },
  {
    title: "Solar-Driven Drum-Type Atmospheric Water Harvester Based on Bio-Based Gels",
    authors: "Advanced Materials",
    journal: "Advanced Materials",
    year: 2024,
    url: "https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.202403876",
    keyFinding: "Bio-based CAL gel: 1.74 kg/kg/h capture rate at 30% RH",
    category: "sorbent",
  },
];

// Bill of materials - Basic (Irrigation Grade)
const BILL_OF_MATERIALS_BASIC = [
  { item: "CaCl₂ ice melt - 10 lb bag", cost: [6, 10], source: "Hardware store", essential: true, note: "Cheapest option" },
  { item: "Silica gel beads - 2 lb", cost: [8, 12], source: "Amazon / Craft store", essential: true, note: "Cat litter crystals work too ($4)" },
  { item: "Bathroom mirrors (4x)", cost: [4, 8], source: "Dollar store", essential: true, note: "$1 each" },
  { item: "Clear plastic dome (salad bowl)", cost: [2, 5], source: "Dollar store", essential: true, note: "" },
  { item: "Aluminum foil on cardboard", cost: [0, 3], source: "Kitchen / recycling", essential: true, note: "FREE alternative to mirrors" },
  { item: "PVC pipe frame", cost: [8, 12], source: "Hardware store", essential: true, note: "Or use scrap wood - FREE" },
  { item: "2L bottles (collection)", cost: [0, 0], source: "Recycling - FREE", essential: true, note: "" },
  { item: "Tubing + funnel", cost: [3, 6], source: "Hardware store", essential: true, note: "" },
  { item: "Silicone sealant", cost: [4, 6], source: "Hardware store", essential: true, note: "" },
];

// Bill of materials - Food Safe (Drinking Grade)
const BILL_OF_MATERIALS_FOOD_SAFE = [
  { item: "Food-grade CaCl₂ (Pickle Crisp)", cost: [8, 12], source: "Grocery / Amazon", essential: true, note: "Ball brand or brewing supply" },
  { item: "Food-grade silica gel", cost: [10, 15], source: "Amazon", essential: true, note: "Look for 'FDA approved'" },
  { item: "Stainless steel bowl (condenser)", cost: [8, 15], source: "Kitchen store / thrift", essential: true, note: "Thrift stores = $3-5" },
  { item: "Glass jar collection (mason jars)", cost: [0, 8], source: "Recycling / Dollar store", essential: true, note: "FREE if reusing" },
  { item: "Food-grade silicone tubing", cost: [6, 10], source: "Amazon / brewing supply", essential: true, note: "" },
  { item: "Activated carbon filter", cost: [5, 10], source: "Pet store / Amazon", essential: true, note: "Aquarium filters work" },
  { item: "Cotton/mesh pre-filter", cost: [0, 3], source: "Fabric scraps - FREE", essential: true, note: "Coffee filters work" },
];

// Ultra-low-cost alternatives
const BUDGET_HACKS = [
  { standard: "Bathroom mirrors", hack: "Aluminum foil on cardboard", savings: "$4-8", note: "80% as effective" },
  { standard: "Silica gel beads", hack: "Crystal cat litter (silica)", savings: "$6-10", note: "Same material, bulk pricing" },
  { standard: "PVC frame", hack: "Scrap wood / bamboo", savings: "$8-12", note: "FREE from construction sites" },
  { standard: "Plastic dome", hack: "Cut 2L bottle top", savings: "$2-5", note: "FREE, smaller scale" },
  { standard: "Commercial sealant", hack: "Beeswax + oil", savings: "$4-6", note: "Food-safe, DIY" },
];

// Water treatment options
const TREATMENT_OPTIONS = [
  { method: "Boiling (1 min)", cost: [0, 0], removes: "Bacteria, viruses", note: "FREE - fuel cost only" },
  { method: "SODIS (sun disinfection)", cost: [0, 0], removes: "Pathogens", note: "FREE - clear bottle in sun 6hrs" },
  { method: "Activated carbon filter", cost: [5, 10], removes: "Chemicals, taste, some metals", note: "Aquarium filter carbon" },
  { method: "Cotton + sand filter", cost: [0, 2], removes: "Sediment, particles", note: "DIY from recycled materials" },
  { method: "UV-C LED module", cost: [8, 15], removes: "All pathogens", note: "Reusable, solar-powered options" },
];

// Combined for backward compatibility
const BILL_OF_MATERIALS = BILL_OF_MATERIALS_BASIC;

// Yield estimates by climate
const YIELD_ESTIMATES = [
  { climate: "Humid coastal", humidity: "60-80%", dailyYield: [0.5, 1.5], weeklyYield: [3.5, 10.5], color: "text-green-400" },
  { climate: "Moderate", humidity: "40-60%", dailyYield: [0.3, 0.8], weeklyYield: [2, 5.5], color: "text-blue-400" },
  { climate: "Semi-arid", humidity: "20-40%", dailyYield: [0.1, 0.4], weeklyYield: [0.7, 2.8], color: "text-yellow-400" },
  { climate: "Desert", humidity: "10-20%", dailyYield: [0.05, 0.2], weeklyYield: [0.35, 1.4], color: "text-orange-400" },
];

// Open source tools
const OPEN_SOURCE_TOOLS = [
  {
    name: "OpenJSCAD",
    description: "Browser-based parametric CAD - no install needed",
    url: "https://openjscad.xyz",
    category: "cad",
    browser: true,
  },
  {
    name: "JoltPhysics.js",
    description: "Physics engine with buoyancy/fluid support",
    url: "https://jrouwe.github.io/JoltPhysics.js/",
    category: "physics",
    browser: true,
  },
  {
    name: "CadQuery",
    description: "Python parametric CAD scripting",
    url: "https://cadquery.readthedocs.io/",
    category: "cad",
    browser: false,
  },
  {
    name: "OpenFOAM",
    description: "Industrial CFD for condensation simulation",
    url: "https://www.openfoam.com",
    category: "simulation",
    browser: false,
  },
  {
    name: "Three.js",
    description: "3D visualization in browser",
    url: "https://threejs.org",
    category: "visualization",
    browser: true,
  },
  {
    name: "FreeCAD",
    description: "Full parametric CAD with Python scripting",
    url: "https://freecad.org",
    category: "cad",
    browser: false,
  },
];

// Biomimicry inspirations
const BIOMIMICRY_SOURCES = [
  {
    name: "Namib Desert Beetle",
    mechanism: "Hydrophilic bumps on hydrophobic surface",
    benefit: "16x improvement in fog collection",
    icon: Bug,
  },
  {
    name: "Tillandsia (Air Plants)",
    mechanism: "Trichomes absorb water directly from air",
    benefit: "No roots needed, pure atmospheric capture",
    icon: Leaf,
  },
  {
    name: "Desert Moss",
    mechanism: "Nano-grooved awns condense dew",
    benefit: "Works at microscale for maximum surface area",
    icon: Leaf,
  },
  {
    name: "Bromeliad Tank",
    mechanism: "Rosette leaf structure channels water to center",
    benefit: "Can hold up to 10L in natural tanks",
    icon: Droplets,
  },
];

export default function WaterHarvesterLab() {
  const [activeTab, setActiveTab] = useState<"overview" | "science" | "build" | "simulate" | "research" | "community">("overview");
  const [expandedPaper, setExpandedPaper] = useState<string | null>(null);
  const [designParams, setDesignParams] = useState({
    sorbentWidth: 30,
    sorbentDepth: 25,
    mirrorCount: 4,
    mirrorAngle: 45,
    condenserType: "cone",
    surfacePattern: "beetle",
    humidity: 45,
  });

  // Simulation hook
  const {
    simulation,
    isLoading: isSimulating,
    isPolling,
    error: simError,
    runSimulation,
    cancelSimulation,
  } = useSimulation();

  // Run quick simulation when parameters change
  useEffect(() => {
    const params: SimulationParams = {
      type: 'water_harvester',
      parameters: {
        sorbent_width_cm: designParams.sorbentWidth,
        sorbent_depth_cm: designParams.sorbentDepth,
        mirror_count: designParams.mirrorCount,
        mirror_angle: designParams.mirrorAngle,
        humidity_percent: designParams.humidity,
        surface_pattern: designParams.surfacePattern as 'beetle' | 'flat',
        temperature_ambient_c: 25,
      },
      mode: 'quick',
    };
    runSimulation(params);
  }, [designParams]); // eslint-disable-line react-hooks/exhaustive-deps

  // Run full simulation on 4090
  const runFullSimulation = async () => {
    const params: SimulationParams = {
      type: 'water_harvester',
      parameters: {
        sorbent_width_cm: designParams.sorbentWidth,
        sorbent_depth_cm: designParams.sorbentDepth,
        mirror_count: designParams.mirrorCount,
        mirror_angle: designParams.mirrorAngle,
        humidity_percent: designParams.humidity,
        surface_pattern: designParams.surfacePattern as 'beetle' | 'flat',
        temperature_ambient_c: 25,
      },
      mode: 'full',
    };
    await runSimulation(params);
  };

  // Format simulation results for display
  const simResults = simulation?.results ? formatSimulationResults(simulation.results) : null;

  // Calculate costs
  const costs = useMemo(() => {
    const essential = BILL_OF_MATERIALS.filter(m => m.essential);
    const optional = BILL_OF_MATERIALS.filter(m => !m.essential);
    const essentialMin = essential.reduce((sum, m) => sum + m.cost[0], 0);
    const essentialMax = essential.reduce((sum, m) => sum + m.cost[1], 0);
    const optionalMin = optional.reduce((sum, m) => sum + m.cost[0], 0);
    const optionalMax = optional.reduce((sum, m) => sum + m.cost[1], 0);
    return {
      essentialMin,
      essentialMax,
      optionalMin,
      optionalMax,
      totalMin: essentialMin + optionalMin,
      totalMax: essentialMax + optionalMax,
    };
  }, []);

  const tabs = [
    { id: "overview", label: "Overview", icon: Droplets },
    { id: "science", label: "Science", icon: FlaskConical },
    { id: "build", label: "Build It", icon: Printer },
    { id: "simulate", label: "Simulate", icon: Settings2 },
    { id: "research", label: "Research Gap", icon: Lightbulb },
    { id: "community", label: "Community", icon: Share2 },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Header */}
      <div className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 via-transparent to-cyan-900/20" />
        <div className="relative max-w-7xl mx-auto px-4 py-12">
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 rounded-xl bg-blue-500/20 border border-blue-500/30">
                  <Droplets className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm text-foreground-muted">LabFork Research Lab</p>
                  <h1 className="text-3xl md:text-4xl font-bold text-foreground-bright">
                    Atmospheric Water Harvester
                  </h1>
                </div>
              </div>
              <p className="text-lg text-foreground max-w-2xl">
                Open-source solar-powered device that extracts water from air using biomimicry,
                cheap sorbents, and 3D-printable components. From research papers to working prototype.
              </p>
              <div className="flex flex-wrap gap-2 mt-4">
                <span className="px-3 py-1 rounded-full text-xs bg-blue-500/20 text-blue-400 border border-blue-500/30">
                  Biomimicry
                </span>
                <span className="px-3 py-1 rounded-full text-xs bg-green-500/20 text-green-400 border border-green-500/30">
                  Solar Passive
                </span>
                <span className="px-3 py-1 rounded-full text-xs bg-purple-500/20 text-purple-400 border border-purple-500/30">
                  3D Printable
                </span>
                <span className="px-3 py-1 rounded-full text-xs bg-orange-500/20 text-orange-400 border border-orange-500/30">
                  ~$60-120 Build Cost
                </span>
              </div>
            </div>
            <div className="hidden md:flex gap-2">
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:border-foreground-muted transition-colors">
                <Star className="w-4 h-4" />
                <span>Star</span>
              </button>
              <button className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:border-foreground-muted transition-colors">
                <GitFork className="w-4 h-4" />
                <span>Fork</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-border sticky top-0 bg-background z-10">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 -mb-px overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={cn(
                  "flex items-center gap-2 px-4 py-3 text-sm border-b-2 transition-colors whitespace-nowrap touch-target",
                  activeTab === tab.id
                    ? "border-blue-400 text-blue-400"
                    : "border-transparent text-foreground-muted hover:text-foreground"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tab Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="space-y-8">
            {/* How It Works */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">How It Works</h2>
              <div className="grid md:grid-cols-2 gap-6">
                {/* Night Cycle */}
                <div className="p-6 rounded-xl border border-border bg-background-card">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-lg bg-indigo-500/20">
                      <Wind className="w-5 h-5 text-indigo-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-foreground-bright">Night Cycle: Absorption</h3>
                  </div>
                  <p className="text-foreground-muted mb-4">
                    Sorbent bed (CaCl₂ + silica gel) is exposed to cool night air.
                    The hygroscopic material absorbs moisture from the atmosphere,
                    even at humidity as low as 10-20%.
                  </p>
                  <div className="font-mono text-sm text-foreground-muted bg-background p-4 rounded-lg">
                    <pre>{`    Cool night air (10-60% RH)
         ↓ ↓ ↓ ↓ ↓ ↓
    ┌─────────────────────┐
    │░░░ SORBENT BED ░░░░│
    │░░ CaCl₂ + Silica ░░│
    │░░░░░░░░░░░░░░░░░░░░│
    └─────────────────────┘
         ↑     ↑     ↑
        H₂O   H₂O   H₂O
       absorbed from air`}</pre>
                  </div>
                </div>

                {/* Day Cycle */}
                <div className="p-6 rounded-xl border border-border bg-background-card">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 rounded-lg bg-yellow-500/20">
                      <Sun className="w-5 h-5 text-yellow-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-foreground-bright">Day Cycle: Release + Collection</h3>
                  </div>
                  <p className="text-foreground-muted mb-4">
                    Mirrors concentrate sunlight to heat the sorbent to 80-120°C.
                    Water vapor releases, rises, and condenses on a cooler surface
                    with beetle-inspired texture. Droplets roll into collection bottle.
                  </p>
                  <div className="font-mono text-sm text-foreground-muted bg-background p-4 rounded-lg">
                    <pre>{`        ☀ SUN ☀
           │
    ╱╲     │     ╱╲  ← Mirrors
     ╲     │     ╱
      ╲    ▼    ╱
    ┌──────────────┐
    │▓▓ HEATED ▓▓▓│ → 80-120°C
    │▓▓ SORBENT ▓▓│
    └──────┬───────┘
           │ vapor ↑
    ┌──────┴───────┐
    │≋≋ CONDENSER ≋│ → Beetle surface
    └──────┬───────┘
           💧
        [Bottle]`}</pre>
                  </div>
                </div>
              </div>
            </section>

            {/* Key Stats */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">At A Glance</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-xl border border-border bg-background-card text-center">
                  <DollarSign className="w-8 h-8 mx-auto mb-2 text-green-400" />
                  <p className="text-2xl font-bold text-foreground-bright">${costs.essentialMin}-{costs.essentialMax}</p>
                  <p className="text-sm text-foreground-muted">Build Cost</p>
                </div>
                <div className="p-4 rounded-xl border border-border bg-background-card text-center">
                  <Droplets className="w-8 h-8 mx-auto mb-2 text-blue-400" />
                  <p className="text-2xl font-bold text-foreground-bright">0.1-1.5 L</p>
                  <p className="text-sm text-foreground-muted">Daily Yield</p>
                </div>
                <div className="p-4 rounded-xl border border-border bg-background-card text-center">
                  <Zap className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
                  <p className="text-2xl font-bold text-foreground-bright">0 W</p>
                  <p className="text-sm text-foreground-muted">Power Required</p>
                </div>
                <div className="p-4 rounded-xl border border-border bg-background-card text-center">
                  <Clock className="w-8 h-8 mx-auto mb-2 text-purple-400" />
                  <p className="text-2xl font-bold text-foreground-bright">~6 mo</p>
                  <p className="text-sm text-foreground-muted">ROI vs Bottled</p>
                </div>
              </div>
            </section>

            {/* Biomimicry Section */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Nature&apos;s Blueprints</h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                {BIOMIMICRY_SOURCES.map((source) => (
                  <div key={source.name} className="p-4 rounded-xl border border-border bg-background-card hover:border-foreground-muted transition-colors">
                    <source.icon className="w-8 h-8 mb-3 text-green-400" />
                    <h3 className="font-semibold text-foreground-bright mb-2">{source.name}</h3>
                    <p className="text-sm text-foreground-muted mb-2">{source.mechanism}</p>
                    <p className="text-xs text-green-400">{source.benefit}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Use Cases */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">When Does This Make Sense?</h2>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="p-6 rounded-xl border border-green-500/30 bg-green-500/5">
                  <h3 className="text-lg font-semibold text-green-400 mb-4">✅ Good Use Cases</h3>
                  <ul className="space-y-2 text-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">•</span>
                      <span>Emergency preparedness / survival kits</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">•</span>
                      <span>Off-grid cabins / remote camping</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">•</span>
                      <span>Developing regions without water access</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">•</span>
                      <span>Coastal fog-heavy regions (Chile, Peru, Morocco)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-green-400 mt-1">•</span>
                      <span>Educational STEM demonstrations</span>
                    </li>
                  </ul>
                </div>
                <div className="p-6 rounded-xl border border-red-500/30 bg-red-500/5">
                  <h3 className="text-lg font-semibold text-red-400 mb-4">❌ Not Recommended</h3>
                  <ul className="space-y-2 text-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-red-400 mt-1">•</span>
                      <span>Urban areas with tap water (250x more expensive)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-400 mt-1">•</span>
                      <span>Primary water source for a household</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-400 mt-1">•</span>
                      <span>Extreme desert (&lt;10% humidity)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-400 mt-1">•</span>
                      <span>&quot;Save money on water bill&quot; scenarios</span>
                    </li>
                  </ul>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* Science Tab */}
        {activeTab === "science" && (
          <div className="space-y-8">
            {/* Research Papers */}
            <section>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-foreground-bright">Research Papers</h2>
                <span className="text-sm text-foreground-muted">{RESEARCH_PAPERS.length} papers compiled</span>
              </div>
              <div className="space-y-4">
                {RESEARCH_PAPERS.map((paper) => (
                  <div
                    key={paper.title}
                    className="p-4 rounded-xl border border-border bg-background-card hover:border-foreground-muted transition-colors"
                  >
                    <div
                      className="flex items-start justify-between cursor-pointer"
                      onClick={() => setExpandedPaper(expandedPaper === paper.title ? null : paper.title)}
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={cn(
                            "px-2 py-0.5 rounded text-xs",
                            paper.category === "mof" && "bg-purple-500/20 text-purple-400",
                            paper.category === "sorbent" && "bg-blue-500/20 text-blue-400",
                            paper.category === "biomimicry" && "bg-green-500/20 text-green-400"
                          )}>
                            {paper.category === "mof" ? "MOF" : paper.category === "sorbent" ? "Sorbent" : "Biomimicry"}
                          </span>
                          <span className="text-xs text-foreground-muted">{paper.journal} • {paper.year}</span>
                        </div>
                        <h3 className="font-semibold text-foreground-bright">{paper.title}</h3>
                        <p className="text-sm text-green-400 mt-1">Key: {paper.keyFinding}</p>
                      </div>
                      <div className="flex items-center gap-2 ml-4">
                        <a
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="p-2 rounded-lg hover:bg-background transition-colors"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink className="w-4 h-4 text-foreground-muted" />
                        </a>
                        {expandedPaper === paper.title ? (
                          <ChevronUp className="w-4 h-4 text-foreground-muted" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-foreground-muted" />
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {/* The Physics */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">The Physics</h2>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="p-6 rounded-xl border border-border bg-background-card">
                  <Thermometer className="w-8 h-8 mb-4 text-red-400" />
                  <h3 className="font-semibold text-foreground-bright mb-2">Solar Concentration</h3>
                  <p className="text-sm text-foreground-muted">
                    Mirrors focus sunlight to heat sorbent to 80-120°C. This temperature
                    drives water release from hygroscopic materials. Research shows 1.8x
                    concentration is sufficient.
                  </p>
                </div>
                <div className="p-6 rounded-xl border border-border bg-background-card">
                  <FlaskConical className="w-8 h-8 mb-4 text-blue-400" />
                  <h3 className="font-semibold text-foreground-bright mb-2">Sorption-Desorption</h3>
                  <p className="text-sm text-foreground-muted">
                    CaCl₂ and MOFs absorb water at night when cool, release it when heated.
                    CaCl₂ can absorb 1-6x its weight in water. Combined with silica gel,
                    it prevents deliquescence.
                  </p>
                </div>
                <div className="p-6 rounded-xl border border-border bg-background-card">
                  <Bug className="w-8 h-8 mb-4 text-green-400" />
                  <h3 className="font-semibold text-foreground-bright mb-2">Biomimetic Surfaces</h3>
                  <p className="text-sm text-foreground-muted">
                    Namib beetle-inspired hydrophilic/hydrophobic patterning creates 16x
                    improvement in droplet collection. Vibration aids coalescence and
                    roll-off, improving efficiency by 30-50%.
                  </p>
                </div>
              </div>
            </section>

            {/* Yield Estimates */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Expected Yields by Climate</h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Climate</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Humidity</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Daily Yield</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Weekly Yield</th>
                    </tr>
                  </thead>
                  <tbody>
                    {YIELD_ESTIMATES.map((est) => (
                      <tr key={est.climate} className="border-b border-border">
                        <td className={cn("py-3 px-4 font-medium", est.color)}>{est.climate}</td>
                        <td className="py-3 px-4 text-foreground-muted">{est.humidity} RH</td>
                        <td className="py-3 px-4 text-foreground">{est.dailyYield[0]}-{est.dailyYield[1]} L</td>
                        <td className="py-3 px-4 text-foreground">{est.weeklyYield[0]}-{est.weeklyYield[1]} L</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-sm text-foreground-muted mt-4">
                * Based on ~0.25 m² sorbent bed with ~1 kg CaCl₂/silica mix. Human needs: 2-3 L/day minimum.
              </p>
            </section>
          </div>
        )}

        {/* Build Tab */}
        {activeTab === "build" && (
          <div className="space-y-8">
            {/* Build Version Selector */}
            <section>
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1 p-6 rounded-xl border-2 border-green-500/50 bg-green-500/5 cursor-pointer">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-bold text-green-400">Basic Build</h3>
                    <span className="text-2xl font-bold text-green-400">$25-45</span>
                  </div>
                  <p className="text-sm text-foreground-muted mb-3">Irrigation grade - cheapest possible</p>
                  <ul className="text-xs text-foreground-muted space-y-1">
                    <li>• Ice melt CaCl₂ + cat litter silica</li>
                    <li>• Aluminum foil reflectors</li>
                    <li>• Recycled bottles & containers</li>
                  </ul>
                </div>
                <div className="flex-1 p-6 rounded-xl border border-blue-500/30 bg-blue-500/5 cursor-pointer hover:border-blue-500/50 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-bold text-blue-400">Food-Safe Build</h3>
                    <span className="text-2xl font-bold text-blue-400">$45-70</span>
                  </div>
                  <p className="text-sm text-foreground-muted mb-3">Drinking grade - safe for consumption</p>
                  <ul className="text-xs text-foreground-muted space-y-1">
                    <li>• Food-grade CaCl₂ (Pickle Crisp)</li>
                    <li>• Stainless steel condenser</li>
                    <li>• Activated carbon filtration</li>
                  </ul>
                </div>
              </div>
            </section>

            {/* Basic Materials */}
            <section>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-foreground-bright">Basic Build (Irrigation)</h2>
                <span className="px-3 py-1 rounded-full text-sm bg-green-500/20 text-green-400">Lowest Cost</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Item</th>
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Cost</th>
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Source</th>
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Tip</th>
                    </tr>
                  </thead>
                  <tbody>
                    {BILL_OF_MATERIALS_BASIC.map((item) => (
                      <tr key={item.item} className="border-b border-border">
                        <td className="py-2 px-3 text-foreground">{item.item}</td>
                        <td className="py-2 px-3 text-foreground">
                          {item.cost[0] === 0 && item.cost[1] === 0 ? (
                            <span className="text-green-400 font-semibold">FREE</span>
                          ) : (
                            <span>${item.cost[0]}-{item.cost[1]}</span>
                          )}
                        </td>
                        <td className="py-2 px-3 text-foreground-muted">{item.source}</td>
                        <td className="py-2 px-3 text-green-400 text-xs">{item.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Budget Hacks */}
            <section>
              <h2 className="text-xl font-bold text-foreground-bright mb-4">Cost-Cutting Hacks</h2>
              <div className="grid md:grid-cols-2 gap-3">
                {BUDGET_HACKS.map((hack) => (
                  <div key={hack.standard} className="p-4 rounded-lg border border-border bg-background-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-foreground-muted line-through text-sm">{hack.standard}</span>
                      <span className="text-green-400 font-semibold">Save {hack.savings}</span>
                    </div>
                    <p className="text-foreground font-medium">{hack.hack}</p>
                    <p className="text-xs text-foreground-muted mt-1">{hack.note}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-4 rounded-lg border border-green-500/30 bg-green-500/5">
                <p className="text-green-400 font-semibold">Ultra-Budget Build: $15-25</p>
                <p className="text-sm text-foreground-muted">Using all hacks: foil reflectors, cat litter, scrap wood frame, recycled containers</p>
              </div>
            </section>

            {/* Food Safe Materials */}
            <section>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-foreground-bright">Food-Safe Build (Drinking)</h2>
                <span className="px-3 py-1 rounded-full text-sm bg-blue-500/20 text-blue-400">Potable Water</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Item</th>
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Cost</th>
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Source</th>
                      <th className="text-left py-2 px-3 text-foreground-muted font-medium">Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {BILL_OF_MATERIALS_FOOD_SAFE.map((item) => (
                      <tr key={item.item} className="border-b border-border">
                        <td className="py-2 px-3 text-foreground">{item.item}</td>
                        <td className="py-2 px-3 text-foreground">
                          {item.cost[0] === 0 && item.cost[1] === 0 ? (
                            <span className="text-green-400 font-semibold">FREE</span>
                          ) : (
                            <span>${item.cost[0]}-{item.cost[1]}</span>
                          )}
                        </td>
                        <td className="py-2 px-3 text-foreground-muted">{item.source}</td>
                        <td className="py-2 px-3 text-blue-400 text-xs">{item.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Water Treatment */}
            <section>
              <h2 className="text-xl font-bold text-foreground-bright mb-4">Water Treatment Options</h2>
              <p className="text-foreground-muted mb-4">Even with food-safe materials, treatment is recommended for drinking water.</p>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                {TREATMENT_OPTIONS.map((opt) => (
                  <div key={opt.method} className="p-4 rounded-lg border border-border bg-background-card">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-foreground-bright">{opt.method}</span>
                      {opt.cost[0] === 0 && opt.cost[1] === 0 ? (
                        <span className="text-green-400 font-bold">FREE</span>
                      ) : (
                        <span className="text-foreground">${opt.cost[0]}-{opt.cost[1]}</span>
                      )}
                    </div>
                    <p className="text-sm text-blue-400 mb-1">Removes: {opt.removes}</p>
                    <p className="text-xs text-foreground-muted">{opt.note}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-4 rounded-lg border border-yellow-500/30 bg-yellow-500/5">
                <p className="text-yellow-400 font-semibold">Recommended Combo: Boiling + Carbon Filter = $5-10 total</p>
                <p className="text-sm text-foreground-muted">Kills all pathogens + removes chemicals and improves taste</p>
              </div>
            </section>

            {/* 3D Printable Parts */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">3D Printable Components</h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  { name: "Condenser Cone v2.3", downloads: 1200, rating: 4.8 },
                  { name: "Bottle Adapter", downloads: 890, rating: 4.6 },
                  { name: "Beetle Surface Texture", downloads: 2100, rating: 4.9 },
                  { name: "Mirror Mount Bracket", downloads: 650, rating: 4.5 },
                ].map((part) => (
                  <div key={part.name} className="p-4 rounded-xl border border-border bg-background-card hover:border-foreground-muted transition-colors">
                    <div className="aspect-square bg-background rounded-lg mb-4 flex items-center justify-center">
                      <Printer className="w-12 h-12 text-foreground-muted" />
                    </div>
                    <h3 className="font-semibold text-foreground-bright mb-1">{part.name}</h3>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-foreground-muted flex items-center gap-1">
                        <Download className="w-3 h-3" />
                        {part.downloads.toLocaleString()}
                      </span>
                      <span className="text-yellow-400 flex items-center gap-1">
                        <Star className="w-3 h-3 fill-current" />
                        {part.rating}
                      </span>
                    </div>
                    <button className="w-full mt-3 px-3 py-2 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors text-sm">
                      Download STL
                    </button>
                  </div>
                ))}
              </div>
              <div className="mt-6 p-4 rounded-xl border border-border bg-background-card">
                <div className="flex items-center gap-3">
                  <Github className="w-6 h-6 text-foreground-muted" />
                  <div>
                    <p className="text-foreground-bright font-medium">All STL files on GitHub</p>
                    <p className="text-sm text-foreground-muted">github.com/labfork/water-harvester-stl</p>
                  </div>
                  <a
                    href="#"
                    className="ml-auto px-4 py-2 rounded-lg border border-border hover:border-foreground-muted transition-colors text-sm"
                  >
                    View Repository
                  </a>
                </div>
              </div>
            </section>

            {/* Open Source Tools */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Open Source Design Tools</h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {OPEN_SOURCE_TOOLS.map((tool) => (
                  <a
                    key={tool.name}
                    href={tool.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-4 rounded-xl border border-border bg-background-card hover:border-foreground-muted transition-colors group"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-semibold text-foreground-bright group-hover:text-blue-400 transition-colors">
                        {tool.name}
                      </h3>
                      {tool.browser && (
                        <span className="px-2 py-0.5 rounded text-xs bg-green-500/20 text-green-400">
                          Browser
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-foreground-muted">{tool.description}</p>
                    <div className="flex items-center gap-1 mt-3 text-xs text-foreground-muted">
                      <ExternalLink className="w-3 h-3" />
                      {tool.url.replace("https://", "")}
                    </div>
                  </a>
                ))}
              </div>
            </section>
          </div>
        )}

        {/* Simulate Tab */}
        {activeTab === "simulate" && (
          <div className="space-y-8">
            {/* OpenFOAM CFD Results */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">
                CFD Analysis
                <span className="ml-3 text-sm font-normal text-foreground-muted">(OpenFOAM Simulation)</span>
              </h2>
              <CFDResultsVisualization />
            </section>

            {/* Design Configurator */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Design Configurator</h2>
              <div className="grid lg:grid-cols-2 gap-8">
                {/* Controls */}
                <div className="space-y-6">
                  <div className="p-6 rounded-xl border border-border bg-background-card">
                    <h3 className="font-semibold text-foreground-bright mb-4">Sorbent Bed</h3>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <label className="text-foreground-muted">Width</label>
                          <span className="text-foreground">{designParams.sorbentWidth} cm</span>
                        </div>
                        <input
                          type="range"
                          min="10"
                          max="60"
                          value={designParams.sorbentWidth}
                          onChange={(e) => setDesignParams({ ...designParams, sorbentWidth: parseInt(e.target.value) })}
                          className="w-full accent-blue-400"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <label className="text-foreground-muted">Depth</label>
                          <span className="text-foreground">{designParams.sorbentDepth} cm</span>
                        </div>
                        <input
                          type="range"
                          min="10"
                          max="60"
                          value={designParams.sorbentDepth}
                          onChange={(e) => setDesignParams({ ...designParams, sorbentDepth: parseInt(e.target.value) })}
                          className="w-full accent-blue-400"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-6 rounded-xl border border-border bg-background-card">
                    <h3 className="font-semibold text-foreground-bright mb-4">Mirror Array</h3>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <label className="text-foreground-muted">Mirror Count</label>
                          <span className="text-foreground">{designParams.mirrorCount}</span>
                        </div>
                        <input
                          type="range"
                          min="2"
                          max="8"
                          value={designParams.mirrorCount}
                          onChange={(e) => setDesignParams({ ...designParams, mirrorCount: parseInt(e.target.value) })}
                          className="w-full accent-blue-400"
                        />
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <label className="text-foreground-muted">Angle</label>
                          <span className="text-foreground">{designParams.mirrorAngle}°</span>
                        </div>
                        <input
                          type="range"
                          min="20"
                          max="70"
                          value={designParams.mirrorAngle}
                          onChange={(e) => setDesignParams({ ...designParams, mirrorAngle: parseInt(e.target.value) })}
                          className="w-full accent-blue-400"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="p-6 rounded-xl border border-border bg-background-card">
                    <h3 className="font-semibold text-foreground-bright mb-4">Condenser Surface</h3>
                    <div className="space-y-4">
                      <div>
                        <label className="text-sm text-foreground-muted block mb-2">Surface Pattern</label>
                        <div className="grid grid-cols-2 gap-2">
                          {["beetle", "flat"].map((pattern) => (
                            <button
                              key={pattern}
                              onClick={() => setDesignParams({ ...designParams, surfacePattern: pattern })}
                              className={cn(
                                "px-4 py-2 rounded-lg text-sm transition-colors capitalize",
                                designParams.surfacePattern === pattern
                                  ? "bg-blue-500 text-white"
                                  : "bg-background border border-border hover:border-foreground-muted"
                              )}
                            >
                              {pattern}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="p-6 rounded-xl border border-border bg-background-card">
                    <h3 className="font-semibold text-foreground-bright mb-4">Environment</h3>
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <label className="text-foreground-muted">Humidity</label>
                        <span className="text-foreground">{designParams.humidity}% RH</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="80"
                        value={designParams.humidity}
                        onChange={(e) => setDesignParams({ ...designParams, humidity: parseInt(e.target.value) })}
                        className="w-full accent-blue-400"
                      />
                    </div>
                  </div>
                </div>

                {/* Preview & Results */}
                <div className="space-y-6">
                  {/* 3D Visualization - The Living Harvester */}
                  <div className="aspect-square rounded-xl border border-border bg-background-card relative overflow-hidden">
                    {/* Status indicator overlay */}
                    {simulation && (
                      <div className={cn(
                        "absolute top-4 right-4 px-3 py-1 rounded-full text-xs flex items-center gap-2 z-10",
                        simulation.status === 'completed' && "bg-green-500/20 text-green-400",
                        simulation.status === 'running' && "bg-blue-500/20 text-blue-400",
                        simulation.status === 'pending' && "bg-yellow-500/20 text-yellow-400",
                        simulation.status === 'failed' && "bg-red-500/20 text-red-400"
                      )}>
                        {simulation.status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
                        {simulation.status === 'completed' && <CheckCircle2 className="w-3 h-3" />}
                        {simulation.status === 'failed' && <AlertCircle className="w-3 h-3" />}
                        {simulation.status}
                      </div>
                    )}

                    {/* 3D Scene - Physics-accurate MOF Water Harvester */}
                    <MOFHarvesterScene
                      humidity={designParams.humidity}
                      dailyYield={simulation?.results ? (simulation.results as { daily_yield_liters?: number }).daily_yield_liters : undefined}
                      sorbentTemp={simulation?.results ? (simulation.results as { peak_temperature_c?: number }).peak_temperature_c : undefined}
                      className="absolute inset-0"
                    />

                    {/* Run full simulation button overlay */}
                    <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between z-10">
                      <span className="text-xs text-foreground-muted bg-background/80 px-2 py-1 rounded">
                        Drag to rotate | Pinch to zoom
                      </span>
                      <button
                        onClick={runFullSimulation}
                        disabled={isSimulating || isPolling}
                        className="px-3 py-1.5 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors text-sm disabled:opacity-50 flex items-center gap-2"
                      >
                        {isPolling ? (
                          <>
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Running...
                          </>
                        ) : (
                          <>
                            <Play className="w-3 h-3" />
                            Full Sim
                          </>
                        )}
                      </button>
                    </div>

                    {/* Error overlay */}
                    {simError && (
                      <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-20">
                        <div className="text-center p-6">
                          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-400" />
                          <p className="text-red-400 mb-2">Simulation Error</p>
                          <p className="text-sm text-foreground-muted mb-4">{simError}</p>
                          <button
                            onClick={runFullSimulation}
                            className="px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors text-sm"
                          >
                            <RefreshCw className="w-4 h-4 inline mr-2" />
                            Retry
                          </button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Live Simulation Results */}
                  <div className={cn(
                    "p-6 rounded-xl border",
                    simulation?.status === 'completed' ? "border-green-500/30 bg-green-500/5" : "border-border bg-background-card"
                  )}>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className={cn(
                        "font-semibold",
                        simulation?.status === 'completed' ? "text-green-400" : "text-foreground-bright"
                      )}>
                        {simulation?.params.mode === 'full' ? 'CFD Simulation Results' : 'Quick Estimate'}
                      </h3>
                      {simulation?.id && (
                        <span className="text-xs text-foreground-muted font-mono">
                          {simulation.id}
                        </span>
                      )}
                    </div>

                    {simResults ? (
                      <>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-3xl font-bold text-foreground-bright">
                              {simResults.dailyYield}
                            </p>
                            <p className="text-sm text-foreground-muted">Daily yield</p>
                          </div>
                          <div>
                            <p className="text-3xl font-bold text-foreground-bright">
                              {simResults.efficiency}
                            </p>
                            <p className="text-sm text-foreground-muted">Efficiency</p>
                          </div>
                          <div>
                            <p className="text-xl font-semibold text-foreground">
                              {simResults.collectionRate}
                            </p>
                            <p className="text-sm text-foreground-muted">Collection rate</p>
                          </div>
                          <div>
                            <p className="text-xl font-semibold text-foreground">
                              {simResults.peakTemp}
                            </p>
                            <p className="text-sm text-foreground-muted">Peak temperature</p>
                          </div>
                        </div>
                        <div className="mt-4 pt-4 border-t border-border">
                          <div className="flex items-center justify-between text-sm">
                            <p className="text-foreground-muted">
                              {(designParams.sorbentWidth * designParams.sorbentDepth / 100).toFixed(2)} m² sorbent area
                            </p>
                            {designParams.surfacePattern === "beetle" ? (
                              <span className="text-green-400">+50% beetle surface bonus</span>
                            ) : (
                              <span className="text-yellow-400">Try beetle surface for +50%</span>
                            )}
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-8">
                        <Loader2 className="w-8 h-8 mx-auto mb-2 text-foreground-muted animate-spin" />
                        <p className="text-foreground-muted">Calculating...</p>
                      </div>
                    )}
                  </div>

                  {/* Export Actions */}
                  <div className="flex gap-4">
                    <button className="flex-1 px-4 py-3 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors flex items-center justify-center gap-2">
                      <Download className="w-4 h-4" />
                      Export STL
                    </button>
                    <button className="flex-1 px-4 py-3 rounded-lg border border-border hover:border-foreground-muted transition-colors flex items-center justify-center gap-2">
                      <Share2 className="w-4 h-4" />
                      Share Design
                    </button>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* Research Gap Tab */}
        {activeTab === "research" && (
          <div className="space-y-8">
            {/* Honest Assessment */}
            <section>
              <div className="p-6 rounded-xl border border-yellow-500/30 bg-yellow-500/5">
                <h2 className="text-2xl font-bold text-yellow-400 mb-4">Honest Assessment</h2>
                <p className="text-foreground mb-4">
                  We believe in transparency. This technology is real and works, but it&apos;s not a silver bullet. Here&apos;s the truth:
                </p>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-semibold text-foreground-bright mb-2">Current Limitations</h3>
                    <ul className="space-y-2 text-sm text-foreground-muted">
                      <li className="flex items-start gap-2">
                        <span className="text-red-400">•</span>
                        <span><strong>1 unit = 0.5-1.5 L/day</strong> — A person needs 2-3 L/day minimum</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-400">•</span>
                        <span><strong>Humidity catch-22</strong> — Driest places need water most, but yield least</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-400">•</span>
                        <span><strong>Maintenance required</strong> — Sorbent degrades, mirrors need cleaning</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-400">•</span>
                        <span><strong>No long-term field data</strong> — DIY durability is unproven</span>
                      </li>
                    </ul>
                  </div>
                  <div>
                    <h3 className="font-semibold text-foreground-bright mb-2">Where It Works</h3>
                    <ul className="space-y-2 text-sm text-foreground-muted">
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Coastal fog regions</strong> — Peru, Chile, Morocco, Namibia (proven)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Emergency backup</strong> — When infrastructure fails</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Supplementing rainwater</strong> — Fills dry season gaps</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Small-scale gardens</strong> — 0.5 L/day waters seedlings</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>

            {/* The Gap */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">The Research Gap</h2>
              <div className="p-6 rounded-xl border border-border bg-background-card">
                <div className="grid md:grid-cols-3 gap-6 text-center">
                  <div>
                    <p className="text-3xl font-bold text-purple-400">7 L/kg/day</p>
                    <p className="text-sm text-foreground-muted">Lab MOFs (best case)</p>
                    <p className="text-xs text-foreground-muted mt-1">$100+/kg materials</p>
                  </div>
                  <div className="flex items-center justify-center">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-yellow-400">5-10x GAP</p>
                      <p className="text-sm text-foreground-muted">This is what we&apos;re trying to close</p>
                    </div>
                  </div>
                  <div>
                    <p className="text-3xl font-bold text-green-400">0.5-1.5 L/day</p>
                    <p className="text-sm text-foreground-muted">DIY Reality</p>
                    <p className="text-xs text-foreground-muted mt-1">$5-10/kg materials</p>
                  </div>
                </div>
              </div>
            </section>

            {/* Research Priorities */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Research Priorities</h2>
              <div className="space-y-4">
                <div className="p-6 rounded-xl border border-purple-500/30 bg-purple-500/5">
                  <div className="flex items-start gap-4">
                    <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400 font-bold">P1</div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-purple-400 mb-2">Cheap Sorbent Discovery</h3>
                      <p className="text-sm text-foreground-muted mb-3">
                        Find material that costs &lt;$5/kg and performs &gt;3 L/kg/day
                      </p>
                      <div className="grid md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-foreground-bright mb-1">AI Approach:</p>
                          <ul className="text-foreground-muted space-y-1">
                            <li>• Screen molecular structures for water affinity</li>
                            <li>• Cross-reference with cheap raw materials</li>
                            <li>• Predict synthesis from common chemicals</li>
                          </ul>
                        </div>
                        <div>
                          <p className="text-foreground-bright mb-1">Candidates:</p>
                          <ul className="text-foreground-muted space-y-1">
                            <li>• Agricultural waste + salt composites</li>
                            <li>• Clay-based (bentonite, zeolite)</li>
                            <li>• Bio-based hydrogels (algae, cellulose)</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 rounded-xl border border-blue-500/30 bg-blue-500/5">
                  <div className="flex items-start gap-4">
                    <div className="p-2 rounded-lg bg-blue-500/20 text-blue-400 font-bold">P2</div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-blue-400 mb-2">Surface Optimization</h3>
                      <p className="text-sm text-foreground-muted mb-3">
                        3D-printable surface achieving 80% of beetle efficiency
                      </p>
                      <div className="grid md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-foreground-bright mb-1">AI Approach:</p>
                          <ul className="text-foreground-muted space-y-1">
                            <li>• Generative design for bump patterns</li>
                            <li>• CFD simulation of droplet behavior</li>
                            <li>• Optimize for printability + performance</li>
                          </ul>
                        </div>
                        <div>
                          <p className="text-foreground-bright mb-1">Target:</p>
                          <ul className="text-foreground-muted space-y-1">
                            <li>• Current: flat surfaces = baseline</li>
                            <li>• Beetle pattern: 16x improvement (lab)</li>
                            <li>• Goal: 8-10x with printable design</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-6 rounded-xl border border-green-500/30 bg-green-500/5">
                  <div className="flex items-start gap-4">
                    <div className="p-2 rounded-lg bg-green-500/20 text-green-400 font-bold">P3</div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-green-400 mb-2">Community Data Pipeline</h3>
                      <p className="text-sm text-foreground-muted mb-3">
                        1000 real-world tests across 50 climates
                      </p>
                      <div className="grid md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-foreground-bright mb-1">AI Approach:</p>
                          <ul className="text-foreground-muted space-y-1">
                            <li>• Anomaly detection (which designs outperform?)</li>
                            <li>• Climate-design matching algorithms</li>
                            <li>• Automated hypothesis generation</li>
                          </ul>
                        </div>
                        <div>
                          <p className="text-foreground-bright mb-1">What We Need:</p>
                          <ul className="text-foreground-muted space-y-1">
                            <li>• Standardized test protocol</li>
                            <li>• Location + weather data</li>
                            <li>• Daily yield measurements</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* AI Timeline */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">What&apos;s Realistic With AI</h2>
              <div className="space-y-4">
                <div className="p-4 rounded-lg border border-green-500/30 bg-green-500/5">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-2 py-1 rounded text-xs bg-green-500/20 text-green-400">NOW</span>
                    <span className="font-semibold text-foreground-bright">Available Today</span>
                  </div>
                  <ul className="text-sm text-foreground-muted grid md:grid-cols-2 gap-1">
                    <li>• Literature synthesis (done)</li>
                    <li>• Parametric design tools (built)</li>
                    <li>• Community data collection (ready)</li>
                    <li>• Basic parameter optimization</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg border border-blue-500/30 bg-blue-500/5">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-2 py-1 rounded text-xs bg-blue-500/20 text-blue-400">3-6 MONTHS</span>
                    <span className="font-semibold text-foreground-bright">With Focused Effort</span>
                  </div>
                  <ul className="text-sm text-foreground-muted grid md:grid-cols-2 gap-1">
                    <li>• Model trained on water harvesting papers</li>
                    <li>• Automated design suggestions by climate</li>
                    <li>• Pattern recognition on test results</li>
                    <li>• Predictive yield estimates</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg border border-purple-500/30 bg-purple-500/5">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-2 py-1 rounded text-xs bg-purple-500/20 text-purple-400">6-12 MONTHS</span>
                    <span className="font-semibold text-foreground-bright">Stretch Goals</span>
                  </div>
                  <ul className="text-sm text-foreground-muted grid md:grid-cols-2 gap-1">
                    <li>• AI-suggested sorbent formulations</li>
                    <li>• Generative 3D surface designs</li>
                    <li>• Location → yield predictor</li>
                    <li>• Automated design iteration</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg border border-orange-500/30 bg-orange-500/5">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="px-2 py-1 rounded text-xs bg-orange-500/20 text-orange-400">HARD</span>
                    <span className="font-semibold text-foreground-bright">Needs Breakthroughs</span>
                  </div>
                  <ul className="text-sm text-foreground-muted grid md:grid-cols-2 gap-1">
                    <li>• Novel material synthesis (AI suggests, humans test)</li>
                    <li>• Self-maintaining systems</li>
                    <li>• Community-scale infrastructure</li>
                    <li>• Closing the full 10x gap</li>
                  </ul>
                </div>
              </div>
            </section>

            {/* Call to Action */}
            <section>
              <div className="p-6 rounded-xl border border-foreground-muted bg-background-card text-center">
                <h3 className="text-xl font-bold text-foreground-bright mb-2">The Value of Open Source Research</h3>
                <p className="text-foreground-muted max-w-2xl mx-auto mb-4">
                  LabFork&apos;s value isn&apos;t that we&apos;ve solved water scarcity. It&apos;s that we&apos;ve made the research
                  accessible so thousands of people can iterate, test, and improve. One of those iterations might crack the code.
                </p>
                <p className="text-lg text-foreground-bright">
                  This won&apos;t give everyone clean water tomorrow.<br/>
                  But 10 years of open-source iteration might produce something that does.
                </p>
              </div>
            </section>
          </div>
        )}

        {/* Community Tab */}
        {activeTab === "community" && (
          <div className="space-y-8">
            {/* Community Results */}
            <section>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-foreground-bright">Real-World Test Results</h2>
                <button className="px-4 py-2 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors text-sm">
                  Submit Your Results
                </button>
              </div>

              {/* Placeholder for community data */}
              <div className="p-12 rounded-xl border border-dashed border-border text-center">
                <MapPin className="w-12 h-12 mx-auto mb-4 text-foreground-muted" />
                <h3 className="text-xl font-semibold text-foreground-bright mb-2">Be the First to Test!</h3>
                <p className="text-foreground-muted max-w-md mx-auto mb-6">
                  Build a prototype, measure your yields, and share your results with the community.
                  Help us validate these designs across different climates.
                </p>
                <div className="flex justify-center gap-4">
                  <button className="px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors">
                    Download Test Protocol
                  </button>
                  <button className="px-4 py-2 rounded-lg border border-border hover:border-foreground-muted transition-colors">
                    View Sample Data Format
                  </button>
                </div>
              </div>
            </section>

            {/* Leaderboard Preview */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Leaderboard (Coming Soon)</h2>
              <div className="overflow-x-auto">
                <table className="w-full opacity-50">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Builder</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Location</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Design</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Climate</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Yield</th>
                      <th className="text-left py-3 px-4 text-foreground-muted font-medium">Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-border">
                      <td className="py-3 px-4 text-foreground">@your_name</td>
                      <td className="py-3 px-4 text-foreground-muted">Your City</td>
                      <td className="py-3 px-4 text-foreground">v1.0</td>
                      <td className="py-3 px-4 text-foreground-muted">??% RH</td>
                      <td className="py-3 px-4 text-foreground">? L/day</td>
                      <td className="py-3 px-4 text-foreground">$??</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            {/* Discussion */}
            <section>
              <h2 className="text-2xl font-bold text-foreground-bright mb-6">Discussion</h2>
              <div className="p-6 rounded-xl border border-border bg-background-card">
                <div className="flex items-center gap-3 mb-4">
                  <Github className="w-6 h-6 text-foreground-muted" />
                  <p className="text-foreground">Join the discussion on GitHub</p>
                </div>
                <a
                  href="#"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-border hover:border-foreground-muted transition-colors text-sm"
                >
                  <ExternalLink className="w-4 h-4" />
                  Open GitHub Discussions
                </a>
              </div>
            </section>
          </div>
        )}
      </div>

      {/* Footer CTA */}
      <div className="border-t border-border bg-background-card">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <h3 className="text-xl font-bold text-foreground-bright">Ready to Build?</h3>
              <p className="text-foreground-muted">Start with a $50 minimal prototype and validate yields in your climate.</p>
            </div>
            <div className="flex gap-4">
              <Link
                href="#"
                className="px-6 py-3 rounded-lg bg-blue-500 text-white hover:bg-blue-600 transition-colors"
              >
                Download Build Guide
              </Link>
              <Link
                href="#"
                className="px-6 py-3 rounded-lg border border-border hover:border-foreground-muted transition-colors"
              >
                Fork This Lab
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
