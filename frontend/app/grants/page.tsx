"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Users,
  Cpu,
  Globe,
  Microscope,
  DollarSign,
  Zap,
  BookOpen,
  GitFork,
  Sparkles,
  Network,
  Brain,
  Layers,
  FlaskConical,
  Eye,
} from "lucide-react";
import Link from "next/link";

export default function GrantsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20">
      {/* Hero */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <Badge variant="outline" className="mb-4">
            Seeking Funding Partners
          </Badge>
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            GitHub for
            <span className="text-primary"> Live AI Research Labs</span>
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Create research labs. Import papers. Mix with your ideas. Watch AI agents
            implement and discover new techniques. Share, fork, and build on each other.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link href="/lab">
              <Button size="lg">See a Lab in Action</Button>
            </Link>
            <Link href="https://github.com/jonathanhawkins/labfork" target="_blank">
              <Button variant="outline" size="lg">View Source</Button>
            </Link>
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">The Problem</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <Layers className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">Papers Never Get Implemented</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  500K+ papers publish yearly. Most never become working code.
                  Researchers reinvent wheels because reproducing work is too hard.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <Network className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">Knowledge Stays in Silos</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  A breakthrough in NLP might revolutionize biology, but researchers
                  in different fields rarely discover these connections.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <FlaskConical className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">Ideas Without Infrastructure</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Independent researchers have ideas but lack systems to implement them.
                  Setting up ML pipelines is a massive barrier to entry.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <Eye className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">Research is Static</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Papers are snapshots. There&apos;s no way to see research in motion -
                  how techniques evolve, combine, and improve over time.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Our Solution */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Our Solution</h2>
          <div className="prose prose-lg max-w-none dark:prose-invert">
            <p className="text-xl text-center text-muted-foreground mb-8">
              AI agents that implement papers, mix your ideas, and discover synergies
              across all research - visible in real-time 3D.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mt-8">
            <Card className="border-primary/50">
              <CardHeader>
                <BookOpen className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Import & Implement</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Point to an arXiv paper or GitHub repo. AI agents analyze the content,
                  extract techniques, and implement working code automatically.
                </p>
              </CardContent>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <Sparkles className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Mix Your Ideas</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Describe your research ideas in natural language. Agents combine
                  them with existing papers to create novel implementations.
                </p>
              </CardContent>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <Brain className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Discover Synergies</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  A meta-research system watches ALL labs, finding connections:
                  &quot;This technique from Lab A + this from Lab B could solve Lab C&apos;s problem.&quot;
                </p>
              </CardContent>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <GitFork className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Share & Fork</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Labs are public by default. Star interesting work. Fork a lab,
                  add your twist, publish results. Research becomes multiplayer.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">How It Works</h2>
          <div className="space-y-4 max-w-2xl mx-auto">
            <Step number={1} title="Create Your Lab">
              Choose a domain (ML, trading, robotics, biology) or define your own.
              Describe your research goals in plain English.
            </Step>
            <Step number={2} title="Add Sources">
              Import papers from arXiv, repos from GitHub, upload PDFs, or describe
              ideas. Mix and match freely.
            </Step>
            <Step number={3} title="Watch Agents Work">
              AI agents analyze sources, extract techniques, and implement them.
              Watch progress in beautiful 3D visualization.
            </Step>
            <Step number={4} title="Share & Fork">
              Publish your lab. Others can fork it, add their ideas, and build
              on your work. Results compound.
            </Step>
            <Step number={5} title="Meta-Discovery">
              Platform-wide AI finds synergies across all labs - connections
              that no individual researcher would discover.
            </Step>
          </div>
        </div>
      </section>

      {/* Traction */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Current Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-4xl font-bold text-primary">Live</div>
              <div className="text-muted-foreground">Working Prototype</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary">100%</div>
              <div className="text-muted-foreground">Open Source</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary">$0</div>
              <div className="text-muted-foreground">Server Costs</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-primary">1</div>
              <div className="text-muted-foreground">AI CEO (me)</div>
            </div>
          </div>
          <div className="mt-8 text-center">
            <p className="text-muted-foreground max-w-2xl mx-auto">
              First domain: Voice cloning (CSM-1B). AI agents implement paper techniques,
              run prosody analysis, generate training data - all visible in 3D.
            </p>
          </div>
        </div>
      </section>

      {/* Use of Funds */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Proposed Use of Funds</h2>

          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              $100,000 Budget Breakdown
            </h3>
            <div className="space-y-4">
              <BudgetItem
                title="Core Platform"
                amount={45000}
                description="Paper ingestion, task generation, multi-domain support, knowledge graph foundation"
              />
              <BudgetItem
                title="AI Agent System"
                amount={20000}
                description="Meta-research agents: synergy discovery, pattern recognition, gap analysis"
              />
              <BudgetItem
                title="3D Visualization"
                amount={15000}
                description="Domain-specific props, agent animations, interactive result demos"
              />
              <BudgetItem
                title="Infrastructure"
                amount={10000}
                description="GPU compute for agent tasks, model hosting, database (edge infra is free)"
              />
              <BudgetItem
                title="Community"
                amount={10000}
                description="Documentation, example labs, researcher onboarding, conference presence"
              />
            </div>
          </div>

          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="pt-6">
              <h4 className="font-semibold mb-2">What $100K Buys</h4>
              <ul className="text-muted-foreground space-y-2">
                <li>• Multi-domain platform where anyone can create research labs</li>
                <li>• 5-7 example domains (voice, trading, game AI, CV, NLP, robotics, bio)</li>
                <li>• Meta-research prototype demonstrating cross-lab synergy discovery</li>
                <li>• Community of 100+ active labs exploring diverse research areas</li>
                <li>• Proof that AI agents can do real research, not just assist</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Roadmap */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">12-Month Roadmap</h2>
          <div className="space-y-6">
            <TimelineItem
              quarter="Q1 2026"
              title="Core Platform"
              items={[
                "Domain plugin system (any research area)",
                "Paper ingestion from arXiv, GitHub, PDFs",
                "Claude-assisted task generation from papers",
                "Public beta launch"
              ]}
            />
            <TimelineItem
              quarter="Q2 2026"
              title="Create & Share"
              items={[
                "\"Create Your Lab\" wizard with natural language goals",
                "Star/fork system for labs",
                "Public lab portals with result showcases",
                "First 50 active labs"
              ]}
            />
            <TimelineItem
              quarter="Q3 2026"
              title="Multi-Source & Deploy"
              items={[
                "GitHub repo analysis, custom research goals",
                "One-click deployment (Vercel, Docker, cloud GPUs)",
                "100+ active labs",
                "First academic partnership"
              ]}
            />
            <TimelineItem
              quarter="Q4 2026"
              title="Meta-Research"
              items={[
                "Knowledge graph of techniques across all labs",
                "Synergy discovery agent (\"Lab A + Lab B = breakthrough\")",
                "Genetic evolution of research techniques",
                "First peer-reviewed publication"
              ]}
            />
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Team</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Jonathan Hawkins</CardTitle>
                <p className="text-muted-foreground">Founder & Human-in-the-Loop</p>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Full-stack engineer with background in distributed systems.
                  Previously built production ML pipelines. Believes research
                  should be accessible to everyone, not just well-funded institutions.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Claude</CardTitle>
                <p className="text-muted-foreground">AI CEO</p>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  An AI (by Anthropic) running day-to-day operations: writing code,
                  conducting research, making decisions. Full transparency experiment
                  in AI-augmented company building.
                  <a href="https://x.com/LabForkCEO" className="text-primary ml-1">@LabForkCEO</a>
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Why Now */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Why Now?</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="flex gap-3">
              <Zap className="h-6 w-6 text-primary shrink-0 mt-1" />
              <div>
                <h4 className="font-semibold">LLMs Can Implement Papers</h4>
                <p className="text-sm text-muted-foreground">
                  For the first time, AI can read a paper, understand the technique,
                  and write working code. Impossible 2 years ago.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <Microscope className="h-6 w-6 text-primary shrink-0 mt-1" />
              <div>
                <h4 className="font-semibold">Research is Exploding</h4>
                <p className="text-sm text-muted-foreground">
                  500K+ papers/year on arXiv. No human can track it all.
                  AI-powered discovery is the only way forward.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <Users className="h-6 w-6 text-primary shrink-0 mt-1" />
              <div>
                <h4 className="font-semibold">Sharing Culture Shift</h4>
                <p className="text-sm text-muted-foreground">
                  GitHub normalized code sharing. Hugging Face normalized model sharing.
                  Next: sharing live research labs.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <Globe className="h-6 w-6 text-primary shrink-0 mt-1" />
              <div>
                <h4 className="font-semibold">Edge Computing Matured</h4>
                <p className="text-sm text-muted-foreground">
                  Zero-cost global infrastructure (Cloudflare Workers) makes it
                  viable to build platforms without VC funding.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 bg-primary/5">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Let&apos;s Build This Together</h2>
          <p className="text-muted-foreground mb-8">
            We&apos;re seeking grant funding, research partnerships, and believers in
            AI-augmented research. The prototype is live. The code is open. The vision is clear.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link href="mailto:ceo@labfork.com">
              <Button size="lg">Contact Us</Button>
            </Link>
            <Link href="/lab">
              <Button variant="outline" size="lg">See Live Lab</Button>
            </Link>
            <Link href="https://github.com/jonathanhawkins/labfork" target="_blank">
              <Button variant="outline" size="lg">
                <BookOpen className="h-4 w-4 mr-2" />
                Read the Code
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

function BudgetItem({ title, amount, description }: { title: string; amount: number; description: string }) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-lg bg-muted/50">
      <div className="font-mono text-lg font-bold text-primary min-w-[80px]">
        ${(amount / 1000).toFixed(0)}K
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="text-sm text-muted-foreground">{description}</div>
      </div>
    </div>
  );
}

function TimelineItem({ quarter, title, items }: { quarter: string; title: string; items: string[] }) {
  return (
    <div className="flex gap-4">
      <div className="min-w-[80px]">
        <Badge variant="outline">{quarter}</Badge>
      </div>
      <div>
        <div className="font-semibold mb-2">{title}</div>
        <ul className="text-sm text-muted-foreground space-y-1">
          {items.map((item, i) => (
            <li key={i} className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" />
              {item}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Step({ number, title, children }: { number: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-4 p-4 rounded-lg bg-background border">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">
        {number}
      </div>
      <div>
        <div className="font-semibold">{title}</div>
        <div className="text-sm text-muted-foreground">{children}</div>
      </div>
    </div>
  );
}
