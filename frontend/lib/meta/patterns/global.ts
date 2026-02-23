/**
 * Global Pattern Recognition Instance
 *
 * Provides a singleton pattern recognition engine for the application.
 */

import { getGlobalGraph } from "../knowledge-graph";
import { PatternRecognition } from "./recognition";
import { PatternReport, ResearchTrend, ArchitecturePattern, TechniqueAdoption, CrossDomainTransfer, PatternRecognitionConfig } from "./types";

let globalPatternRecognition: PatternRecognition | null = null;
let lastReport: PatternReport | null = null;

/**
 * Get the global pattern recognition instance
 */
export function getGlobalPatternRecognition(): PatternRecognition {
  if (!globalPatternRecognition) {
    const graph = getGlobalGraph();
    globalPatternRecognition = new PatternRecognition(graph);
  }
  return globalPatternRecognition;
}

/**
 * Reset the global pattern recognition instance
 */
export function resetGlobalPatternRecognition(): void {
  globalPatternRecognition = null;
  lastReport = null;
}

/**
 * Update pattern recognition configuration
 */
export function updatePatternConfig(
  config: Partial<PatternRecognitionConfig>
): void {
  const recognition = getGlobalPatternRecognition();
  recognition.updateConfig(config);
}

/**
 * Run pattern analysis and get report
 */
export function analyzePatterns(): PatternReport {
  const recognition = getGlobalPatternRecognition();
  lastReport = recognition.analyze();
  return lastReport;
}

/**
 * Get the last generated report
 */
export function getLastReport(): PatternReport | null {
  return lastReport;
}

/**
 * Get current trends
 */
export function getCurrentTrends(): ResearchTrend[] {
  const recognition = getGlobalPatternRecognition();
  return recognition.getTrends();
}

/**
 * Get emerging trends
 */
export function getEmergingTrends(limit?: number): ResearchTrend[] {
  const recognition = getGlobalPatternRecognition();
  return recognition.getEmergingTrends(limit);
}

/**
 * Get detected patterns
 */
export function getDetectedPatterns(): ArchitecturePattern[] {
  const recognition = getGlobalPatternRecognition();
  return recognition.getPatterns();
}

/**
 * Get adoption metrics
 */
export function getAdoptionMetrics(): TechniqueAdoption[] {
  const recognition = getGlobalPatternRecognition();
  return recognition.getAdoptionMetrics();
}

/**
 * Get cross-domain transfers
 */
export function getCrossDomainTransfers(): CrossDomainTransfer[] {
  const recognition = getGlobalPatternRecognition();
  return recognition.getCrossDomainTransfers();
}
