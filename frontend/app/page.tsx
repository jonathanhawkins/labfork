"use client";

import Link from "next/link";
import { useState } from "react";
import {
  Mic,
  Sparkles,
  Radio,
  Film,
  GitCompare,
  GraduationCap,
  Scroll,
  FlaskConical,
  Bot,
  ChevronRight,
  ExternalLink,
  Circle,
} from "lucide-react";

// Collapsible Section Component
function Section({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-border">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between py-3 px-4 text-foreground-bright hover:text-foreground transition-colors"
      >
        <span className="text-sm">{title}</span>
        <span className="text-muted-foreground">{isOpen ? "-" : "+"}</span>
      </button>
      {isOpen && <div className="px-4 pb-4 animate-fade-in">{children}</div>}
    </div>
  );
}

// Navigation Item Component
function NavItem({
  href,
  icon: Icon,
  label,
  description,
  active = false,
}: {
  href: string;
  icon: React.ElementType;
  label: string;
  description: string;
  active?: boolean;
}) {
  return (
    <Link
      href={href}
      className="flex items-center gap-3 py-2 group transition-colors"
    >
      <Circle
        className={`w-2 h-2 ${
          active
            ? "fill-foreground-bright text-foreground-bright"
            : "fill-transparent text-muted-foreground group-hover:text-foreground"
        }`}
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <Icon
            className={`w-3.5 h-3.5 ${
              active
                ? "text-foreground-bright"
                : "text-muted-foreground group-hover:text-foreground"
            }`}
          />
          <span
            className={`text-sm ${
              active
                ? "text-foreground-bright"
                : "text-foreground group-hover:text-foreground-bright"
            }`}
          >
            {label}
          </span>
        </div>
        <span className="text-xs text-muted-foreground truncate block mt-0.5">
          {description}
        </span>
      </div>
      <ChevronRight className="w-3.5 h-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
    </Link>
  );
}

// Stat Row Component
function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

const tools = [
  {
    href: "/studio",
    icon: Mic,
    label: "Studio",
    description: "Record voice samples with auto prosody labeling",
    category: "collect",
  },
  {
    href: "/perform",
    icon: Scroll,
    label: "Perform",
    description: "Script-based emotional recording",
    category: "collect",
  },
  {
    href: "/generate",
    icon: Sparkles,
    label: "Generate",
    description: "Create speech with prosody control",
    category: "create",
  },
  {
    href: "/author",
    icon: Film,
    label: "Author",
    description: "Keyframe timeline for emotion transitions",
    category: "create",
  },
  {
    href: "/live",
    icon: Radio,
    label: "Live",
    description: "Real-time voice transformation",
    category: "create",
  },
  {
    href: "/compare",
    icon: GitCompare,
    label: "Compare",
    description: "A/B test base vs fine-tuned models",
    category: "analyze",
  },
  {
    href: "/training",
    icon: GraduationCap,
    label: "Training",
    description: "Real-time dashboard with metrics",
    category: "analyze",
  },
  {
    href: "/evaluate",
    icon: FlaskConical,
    label: "A/B Test",
    description: "Evaluate model outputs",
    category: "analyze",
  },
  {
    href: "/lab",
    icon: Bot,
    label: "Lab",
    description: "Research agent workspace",
    category: "analyze",
  },
];

const techStack = [
  { name: "Next.js 14", category: "Frontend" },
  { name: "Three.js", category: "3D" },
  { name: "FastAPI", category: "Backend" },
  { name: "PyTorch", category: "ML" },
  { name: "Whisper", category: "ASR" },
  { name: "Qwen2-Audio", category: "Emotion" },
  { name: "Parselmouth", category: "Acoustic" },
  { name: "CSM-1B", category: "Voice" },
  { name: "Pocket TTS", category: "Clone" },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Sidebar */}
      <aside className="w-[280px] flex-shrink-0 border-r border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Tools" defaultOpen>
          <div className="space-y-1">
            {tools
              .filter((t) => t.category === "collect")
              .map((tool) => (
                <NavItem key={tool.href} {...tool} />
              ))}
          </div>
          <div className="border-t border-border my-3" />
          <div className="space-y-1">
            {tools
              .filter((t) => t.category === "create")
              .map((tool) => (
                <NavItem key={tool.href} {...tool} />
              ))}
          </div>
          <div className="border-t border-border my-3" />
          <div className="space-y-1">
            {tools
              .filter((t) => t.category === "analyze")
              .map((tool) => (
                <NavItem key={tool.href} {...tool} />
              ))}
          </div>
        </Section>

        <Section title="Tech Stack">
          <div className="space-y-1">
            {techStack.map((tech) => (
              <div
                key={tech.name}
                className="flex items-center justify-between py-1"
              >
                <span className="text-xs text-muted-foreground">
                  {tech.category}
                </span>
                <span className="text-xs text-foreground">{tech.name}</span>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Links">
          <div className="space-y-2">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              GitHub Repository
            </a>
            <a
              href="#"
              className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Documentation
            </a>
          </div>
        </Section>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-border mt-auto">
          <div className="text-xs text-muted-foreground">
            Voice Clone Pipeline
          </div>
          <div className="text-xxs text-foreground-subtle mt-0.5">
            Prosody-controlled voice synthesis
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 h-[calc(100vh-48px)] overflow-y-auto">
        <div className="p-8 max-w-4xl">
          {/* Header */}
          <div className="mb-12">
            <h1 className="text-lg text-foreground-bright mb-2">
              Voice Clone Pipeline
            </h1>
            <p className="text-sm text-foreground leading-relaxed max-w-2xl">
              An exploration of emotional voice control through multi-layer
              prosody analysis. Testing whether explicit prosody conditioning
              helps with limited training data.
            </p>
          </div>

          {/* Research Question */}
          <div className="mb-12">
            <div className="text-xs text-muted-foreground mb-3">
              Research Question
            </div>
            <div className="text-foreground-bright text-base">
              Can explicit prosody labels improve voice cloning quality?
            </div>
          </div>

          {/* The Prosody Cube */}
          <div className="mb-12">
            <div className="text-xs text-muted-foreground mb-4">
              The Prosody Cube
            </div>
            <div className="grid grid-cols-2 gap-px border border-border rounded overflow-hidden bg-border">
              <div className="bg-background p-4">
                <div className="text-sm text-foreground-bright mb-1">
                  Semantic
                </div>
                <div className="text-xs text-muted-foreground">
                  Emotion, intent, tone via Qwen2-Audio
                </div>
              </div>
              <div className="bg-background p-4">
                <div className="text-sm text-foreground-bright mb-1">
                  Acoustic
                </div>
                <div className="text-xs text-muted-foreground">
                  Pitch, formants, HNR via Parselmouth
                </div>
              </div>
              <div className="bg-background p-4">
                <div className="text-sm text-foreground-bright mb-1">
                  Rhythm
                </div>
                <div className="text-xs text-muted-foreground">
                  Pauses, rate, syllables via librosa
                </div>
              </div>
              <div className="bg-background p-4">
                <div className="text-sm text-foreground-bright mb-1">
                  Contour
                </div>
                <div className="text-xs text-muted-foreground">
                  Pitch trajectory time-series
                </div>
              </div>
            </div>
          </div>

          {/* Results */}
          <div className="mb-12">
            <div className="text-xs text-muted-foreground mb-4">
              Research Results
            </div>
            <div className="border border-border rounded overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-background-elevated">
                    <th className="text-left py-2 px-4 text-muted-foreground font-normal">
                      Version
                    </th>
                    <th className="text-left py-2 px-4 text-muted-foreground font-normal">
                      Description
                    </th>
                    <th className="text-right py-2 px-4 text-muted-foreground font-normal">
                      F0 Corr
                    </th>
                    <th className="text-right py-2 px-4 text-muted-foreground font-normal">
                      Emotion
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border">
                    <td className="py-2 px-4 text-foreground">v1</td>
                    <td className="py-2 px-4 text-muted-foreground">
                      No prosody conditioning
                    </td>
                    <td className="py-2 px-4 text-right text-foreground">
                      -0.006
                    </td>
                    <td className="py-2 px-4 text-right text-muted-foreground">
                      N/A
                    </td>
                  </tr>
                  <tr className="border-b border-border">
                    <td className="py-2 px-4 text-foreground">v2</td>
                    <td className="py-2 px-4 text-muted-foreground">
                      + Prosody encoder
                    </td>
                    <td className="py-2 px-4 text-right text-foreground">
                      0.328
                    </td>
                    <td className="py-2 px-4 text-right text-foreground">0%</td>
                  </tr>
                  <tr className="bg-background-elevated">
                    <td className="py-2 px-4 text-foreground-bright">v3</td>
                    <td className="py-2 px-4 text-foreground">
                      + Intensity fix, + Energy predictor
                    </td>
                    <td className="py-2 px-4 text-right text-foreground-bright">
                      0.328
                    </td>
                    <td className="py-2 px-4 text-right text-foreground-bright">
                      50%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Key Achievement */}
          <div className="mb-12">
            <div className="text-xs text-muted-foreground mb-4">
              Key Achievement
            </div>
            <div className="border border-border rounded p-4">
              <div className="text-sm text-foreground-bright mb-3">
                Fixed Inverted Pitch Patterns
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-muted-foreground mb-1">
                    Happy Pitch
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground line-through">
                      144 Hz
                    </span>
                    <span className="text-muted-foreground mx-2">-{">"}</span>
                    <span className="text-foreground-bright">211 Hz</span>
                    <span className="text-xs text-muted-foreground ml-2">
                      (now highest)
                    </span>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground mb-1">
                    Sad Pitch
                  </div>
                  <div className="text-sm">
                    <span className="text-muted-foreground line-through">
                      274 Hz
                    </span>
                    <span className="text-muted-foreground mx-2">-{">"}</span>
                    <span className="text-foreground-bright">167 Hz</span>
                    <span className="text-xs text-muted-foreground ml-2">
                      (now lowest)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Conclusion */}
          <div className="mb-12">
            <div className="text-xs text-muted-foreground mb-4">Conclusion</div>
            <div className="border border-border rounded p-4">
              <div className="text-sm text-foreground-bright mb-2">
                Hypothesis: Supported (with caveats)
              </div>
              <p className="text-sm text-foreground leading-relaxed">
                The energy predictor auxiliary loss and intensity mapping fix
                demonstrate that explicit prosody conditioning works. Pitch
                patterns now correctly differentiate emotions. Happy and sad are
                correctly detected; angry/neutral still need work.
              </p>
              <div className="flex items-center gap-6 mt-4 text-xs">
                <div className="flex items-center gap-2 text-foreground">
                  <Circle className="w-1.5 h-1.5 fill-foreground" />
                  Pipeline Complete
                </div>
                <div className="flex items-center gap-2 text-foreground">
                  <Circle className="w-1.5 h-1.5 fill-foreground" />
                  Evaluation Complete
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-border px-8 py-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Voice Clone Pipeline</span>
            <span>Built by Jonathan Hawkins | Aligned Tools</span>
          </div>
        </div>
      </main>

      {/* Right Panel - Quick Stats */}
      <aside className="w-[240px] flex-shrink-0 border-l border-border bg-background-elevated h-[calc(100vh-48px)] overflow-y-auto">
        <Section title="Status" defaultOpen>
          <StatRow label="Pipeline" value="Ready" />
          <StatRow label="Model" value="v3 Energy" />
          <StatRow label="Samples" value="42" />
        </Section>

        <Section title="Performance" defaultOpen>
          <StatRow label="F0 Correlation" value="0.328" />
          <StatRow label="Emotion Accuracy" value="50%" />
          <StatRow label="Pitch Pattern" value="Correct" />
        </Section>

        <Section title="Hardware">
          <StatRow label="Training" value="RTX 4090" />
          <StatRow label="Inference" value="M4 Pro" />
          <StatRow label="VRAM" value="24GB" />
        </Section>

        <Section title="Quick Actions">
          <div className="space-y-2">
            <Link
              href="/studio"
              className="block w-full text-center py-2 text-sm bg-foreground text-background rounded hover:bg-foreground-bright transition-colors"
            >
              Open Studio
            </Link>
            <Link
              href="/generate"
              className="block w-full text-center py-2 text-sm border border-border text-foreground rounded hover:bg-accent transition-colors"
            >
              Generate Speech
            </Link>
          </div>
        </Section>
      </aside>
    </div>
  );
}
