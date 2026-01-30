/**
 * Platform Metrics API
 *
 * GET /api/metrics/platform - Get platform-wide metrics
 * POST /api/metrics/platform - Trigger metrics aggregation
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createMetricsAggregator,
  aggregateMetrics,
  getCurrentMetrics,
  getMetricsTrend,
  getSnapshots,
  takeSnapshot,
  generateSummary,
  MetricsPeriod,
  AggregationSource,
} from "@/lib/meta/community";

// Singleton metrics aggregator
let metricsAggregator = createMetricsAggregator();

// Initialize with demo metrics
initializeDemoMetrics();

function initializeDemoMetrics() {
  const demoSource: AggregationSource = {
    labs: generateDemoLabs(150),
    papers: generateDemoPapers(500),
    techniques: generateDemoTechniques(1200),
    tasks: generateDemoTasks(800),
    collaborations: generateDemoCollaborations(120),
    discoveries: generateDemoDiscoveries(300),
    apiCalls: 50000,
    computeHours: 120,
    storageGB: 45,
  };

  aggregateMetrics(metricsAggregator, demoSource, "daily");
  takeSnapshot(metricsAggregator, "daily");
}

function generateDemoLabs(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: `lab-${i}`,
    isActive: Math.random() > 0.2,
    createdAt: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
    techniquesCount: Math.floor(Math.random() * 50),
    collaborationsCount: Math.floor(Math.random() * 10),
    domains: ["TTS", "Voice Cloning", "Speech Recognition", "Prosody"].slice(
      0,
      Math.floor(Math.random() * 3) + 1
    ),
  }));
}

function generateDemoPapers(count: number) {
  const statuses = ["pending", "processed", "failed"] as const;
  const sources = ["arxiv", "semanticscholar", "github", "huggingface"];

  return Array.from({ length: count }, (_, i) => ({
    id: `paper-${i}`,
    status: (Math.random() > 0.1 ? "processed" : Math.random() > 0.5 ? "pending" : "failed") as (typeof statuses)[number],
    source: sources[Math.floor(Math.random() * sources.length)],
    processingTimeMs: Math.floor(Math.random() * 5000) + 1000,
    createdAt: new Date(Date.now() - Math.random() * 180 * 24 * 60 * 60 * 1000).toISOString(),
  }));
}

function generateDemoTechniques(count: number) {
  const statuses: ("active" | "deprecated" | "experimental")[] = [
    "active",
    "deprecated",
    "experimental",
  ];
  const categories = [
    "prosody",
    "codec",
    "attention",
    "vocoder",
    "encoder",
    "decoder",
    "embedding",
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `tech-${i}`,
    status: (Math.random() > 0.3 ? "active" : Math.random() > 0.5 ? "experimental" : "deprecated") as "active" | "deprecated" | "experimental",
    category: categories[Math.floor(Math.random() * categories.length)],
    usageCount: Math.floor(Math.random() * 100),
    generation: Math.floor(Math.random() * 5) + 1,
    createdAt: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
  }));
}

function generateDemoTasks(count: number) {
  const statuses: ("pending" | "in_progress" | "completed" | "failed")[] = [
    "pending",
    "in_progress",
    "completed",
    "failed",
  ];

  return Array.from({ length: count }, (_, i) => ({
    id: `task-${i}`,
    status: statuses[Math.floor(Math.random() * statuses.length)],
    completionTimeMs:
      Math.random() > 0.3 ? Math.floor(Math.random() * 3600000) : undefined,
    createdAt: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
  }));
}

function generateDemoCollaborations(count: number) {
  const statuses: ("active" | "completed" | "failed")[] = ["active", "completed", "failed"];

  return Array.from({ length: count }, (_, i) => ({
    id: `collab-${i}`,
    status: statuses[Math.floor(Math.random() * statuses.length)],
    participantCount: Math.floor(Math.random() * 6) + 2,
    durationDays: Math.random() > 0.5 ? Math.floor(Math.random() * 60) + 7 : undefined,
    participants: Array.from(
      { length: Math.floor(Math.random() * 4) + 2 },
      (_, j) => ({
        labId: `lab-${Math.floor(Math.random() * 150)}`,
        labName: `Lab ${Math.floor(Math.random() * 150)}`,
      })
    ),
    createdAt: new Date(Date.now() - Math.random() * 180 * 24 * 60 * 60 * 1000).toISOString(),
  }));
}

function generateDemoDiscoveries(count: number) {
  const types = ["synergy", "pattern", "gap", "evolution", "transfer"];

  return Array.from({ length: count }, (_, i) => ({
    id: `disc-${i}`,
    type: types[Math.floor(Math.random() * types.length)],
    createdAt: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
  }));
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const period = (searchParams.get("period") as MetricsPeriod) || "daily";
    const includeTrends = searchParams.get("includeTrends") === "true";
    const includeSnapshots = searchParams.get("includeSnapshots") === "true";
    const includeSummary = searchParams.get("includeSummary") === "true";
    const trendMetric = searchParams.get("trendMetric") || "labs.total";
    const snapshotLimit = parseInt(searchParams.get("snapshotLimit") || "7", 10);

    const metrics = getCurrentMetrics(metricsAggregator);

    const response: {
      metrics: typeof metrics;
      trends?: ReturnType<typeof getMetricsTrend>;
      snapshots?: ReturnType<typeof getSnapshots>;
      summary?: ReturnType<typeof generateSummary>;
    } = { metrics };

    if (includeTrends) {
      response.trends = getMetricsTrend(metricsAggregator, trendMetric, 30);
    }

    if (includeSnapshots) {
      response.snapshots = getSnapshots(metricsAggregator, period, snapshotLimit);
    }

    if (includeSummary) {
      response.summary = generateSummary(metrics);
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error("Error fetching platform metrics:", error);
    return NextResponse.json(
      { error: "Failed to fetch platform metrics" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { period = "daily", takeSnapshotFlag = false, source } = body as {
      period?: MetricsPeriod;
      takeSnapshotFlag?: boolean;
      source?: AggregationSource;
    };

    // Use provided source or regenerate demo data
    const aggregationSource = source || {
      labs: generateDemoLabs(150),
      papers: generateDemoPapers(500),
      techniques: generateDemoTechniques(1200),
      tasks: generateDemoTasks(800),
      collaborations: generateDemoCollaborations(120),
      discoveries: generateDemoDiscoveries(300),
      apiCalls: 50000 + Math.floor(Math.random() * 5000),
      computeHours: 120 + Math.floor(Math.random() * 20),
      storageGB: 45 + Math.floor(Math.random() * 5),
    };

    const metrics = aggregateMetrics(metricsAggregator, aggregationSource, period);

    if (takeSnapshotFlag) {
      takeSnapshot(metricsAggregator, period);
    }

    return NextResponse.json(metrics, { status: 201 });
  } catch (error) {
    console.error("Error aggregating metrics:", error);
    return NextResponse.json(
      { error: "Failed to aggregate metrics" },
      { status: 500 }
    );
  }
}
