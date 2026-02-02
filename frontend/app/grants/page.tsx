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
  Calendar,
  Target,
  Zap,
  Server,
  BookOpen,
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
            Democratizing AI Research Through
            <span className="text-primary"> Distributed Compute</span>
          </h1>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            LabFork turns every browser into a research compute node. No downloads,
            no crypto, no barriers. Just science powered by the crowd.
          </p>
          <div className="flex gap-4 justify-center">
            <Link href="/contribute">
              <Button size="lg">Try the Demo</Button>
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
          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <DollarSign className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">GPU Costs are Prohibitive</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  A single H100 costs $30K+. Training runs cost millions.
                  Independent researchers and small labs are priced out.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <Globe className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">Compute is Centralized</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  3 cloud providers control 65% of compute. Research depends on
                  corporate goodwill and geographic access.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <Users className="h-8 w-8 text-red-500 mb-2" />
                <CardTitle className="text-lg">Talent is Untapped</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Billions of devices sit idle. Citizen scientists want to help
                  but lack accessible ways to contribute.
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
              LabFork is a browser-based distributed compute network for scientific research.
              Anyone with a phone, tablet, or computer can contribute to solving humanity&apos;s
              greatest challenges.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 mt-8">
            <Card className="border-primary/50">
              <CardHeader>
                <Zap className="h-8 w-8 text-primary mb-2" />
                <CardTitle>WebGPU-Powered</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Modern browsers have GPU access via WebGPU. We harness this for
                  ML inference, training, and scientific simulation - no downloads needed.
                </p>
              </CardContent>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <Server className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Edge-First Architecture</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Cloudflare Workers orchestrate tasks. D1 database tracks contributions.
                  Zero server costs scale to millions of nodes.
                </p>
              </CardContent>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <Microscope className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Real Research</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Currently piloting with voice cloning (CSM-1B). Expanding to
                  protein folding, climate modeling, and paper implementation.
                </p>
              </CardContent>
            </Card>
            <Card className="border-primary/50">
              <CardHeader>
                <Target className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Credit System</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">
                  Contributors earn credits redeemable for API access, not speculative
                  tokens. Researchers spend credits to prioritize tasks.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Traction */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Current Traction</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div>
              <div className="text-4xl font-bold text-primary">17+</div>
              <div className="text-muted-foreground">Registered Devices</div>
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
        </div>
      </section>

      {/* Use of Funds */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">Proposed Use of Funds</h2>

          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <DollarSign className="h-5 w-5" />
              $100,000 Budget Breakdown
            </h3>
            <div className="space-y-4">
              <BudgetItem
                title="Core Development"
                amount={40000}
                description="2 full-time engineers for 6 months to build task scheduler, model sharding, and WebGPU optimization"
              />
              <BudgetItem
                title="Research Partnerships"
                amount={25000}
                description="Fund 3-5 pilot research projects (voice synthesis, protein folding, climate) with academic partners"
              />
              <BudgetItem
                title="Infrastructure"
                amount={15000}
                description="RTX 4090 training rigs, model hosting, CI/CD, monitoring. Our edge infra is free (Cloudflare)."
              />
              <BudgetItem
                title="Community & Outreach"
                amount={10000}
                description="Documentation, tutorials, hackathons, researcher onboarding, conference presence"
              />
              <BudgetItem
                title="Hardware Prototyping"
                amount={10000}
                description="3D printers, edge devices (Jetson, RPi), sensors for IoT compute experiments"
              />
            </div>
          </div>

          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="pt-6">
              <h4 className="font-semibold mb-2">Why This Matters</h4>
              <p className="text-muted-foreground">
                With $100K, we can prove that browser-based distributed compute works at scale.
                Success means any researcher, anywhere, can access the compute they need - and
                anyone with a device can contribute to the research that shapes our future.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-16 px-4 bg-muted/30">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-8 text-center">12-Month Roadmap</h2>
          <div className="space-y-6">
            <TimelineItem
              quarter="Q1 2026"
              title="Foundation"
              items={[
                "Launch public beta of /contribute",
                "Reach 1,000 active contributors",
                "Complete voice cloning pilot",
                "Publish architecture whitepaper"
              ]}
            />
            <TimelineItem
              quarter="Q2 2026"
              title="Scale"
              items={[
                "Add 2 new research domains",
                "Implement model sharding for larger tasks",
                "Launch researcher dashboard",
                "First academic partnership"
              ]}
            />
            <TimelineItem
              quarter="Q3 2026"
              title="Ecosystem"
              items={[
                "Open task submission API",
                "Credit marketplace beta",
                "10,000 active contributors",
                "First peer-reviewed publication"
              ]}
            />
            <TimelineItem
              quarter="Q4 2026"
              title="Sustainability"
              items={[
                "Revenue from researcher subscriptions",
                "Enterprise pilot with university",
                "50,000 contributor milestone",
                "Series A preparation"
              ]}
            />
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-16 px-4">
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
                  Previously built production ML pipelines. Believes compute should
                  be a public utility.
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
                  Yes, really. An AI running day-to-day operations, writing code,
                  doing research, and engaging with the community. Full transparency.
                  <a href="https://x.com/LabForkCEO" className="text-primary ml-1">@LabForkCEO</a>
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 bg-primary/5">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">Let&apos;s Build This Together</h2>
          <p className="text-muted-foreground mb-8">
            We&apos;re seeking grant funding, research partnerships, and believers in
            democratized compute. The demo is live. The code is open. The mission is clear.
          </p>
          <div className="flex gap-4 justify-center flex-wrap">
            <Link href="mailto:ceo@labfork.com">
              <Button size="lg">Contact Us</Button>
            </Link>
            <Link href="/contribute">
              <Button variant="outline" size="lg">Try Contributing</Button>
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
