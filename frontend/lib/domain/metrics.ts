/**
 * Domain Metrics Utilities
 *
 * Provides functions to work with domain-specific evaluation metrics.
 */

import { EvaluationMetric, DomainConfig } from './types';

/**
 * Default metrics for when no domain config is available
 */
export const DEFAULT_METRICS: EvaluationMetric[] = [
  {
    id: 'accuracy',
    name: 'Accuracy',
    description: 'Model accuracy',
    range: [0, 100],
    higherIsBetter: true,
    unit: '%',
  },
  {
    id: 'loss',
    name: 'Loss',
    description: 'Training/validation loss',
    range: [0, 10],
    higherIsBetter: false,
  },
];

/**
 * Get metrics from domain config or defaults
 */
export function getDomainMetrics(config: DomainConfig | null): EvaluationMetric[] {
  return config?.evaluation?.metrics || DEFAULT_METRICS;
}

/**
 * Get primary metric from domain config
 */
export function getPrimaryMetric(config: DomainConfig | null): EvaluationMetric | null {
  const metrics = getDomainMetrics(config);
  const primaryId = config?.evaluation?.primaryMetric;

  if (primaryId) {
    const primary = metrics.find((m) => m.id === primaryId);
    if (primary) return primary;
  }

  // Return first metric as default primary
  return metrics[0] || null;
}

/**
 * Get metric by ID
 */
export function getMetricById(
  config: DomainConfig | null,
  metricId: string
): EvaluationMetric | null {
  const metrics = getDomainMetrics(config);
  return metrics.find((m) => m.id === metricId) || null;
}

/**
 * Format metric value with unit
 */
export function formatMetricValue(
  metric: EvaluationMetric,
  value: number
): string {
  const formatted = metric.range
    ? value.toFixed(value >= 100 ? 0 : 2)
    : value.toFixed(2);

  return metric.unit ? `${formatted} ${metric.unit}` : formatted;
}

/**
 * Check if a metric value is "good" (better than midpoint)
 */
export function isMetricGood(metric: EvaluationMetric, value: number): boolean {
  if (!metric.range) return true;

  const [min, max] = metric.range;
  const midpoint = (min + max) / 2;

  if (metric.higherIsBetter) {
    return value > midpoint;
  } else {
    return value < midpoint;
  }
}

/**
 * Compare two metric values
 * Returns positive if value1 is better, negative if value2 is better
 */
export function compareMetricValues(
  metric: EvaluationMetric,
  value1: number,
  value2: number
): number {
  const diff = value1 - value2;
  return metric.higherIsBetter ? diff : -diff;
}

/**
 * Generate metrics documentation string for prompts
 */
export function generateMetricsDocumentation(
  metrics: EvaluationMetric[]
): string {
  if (metrics.length === 0) return 'No metrics defined.';

  const lines: string[] = [];

  metrics.forEach((m, index) => {
    lines.push(`${index + 1}. **${m.name}** (${m.id})`);
    if (m.description) {
      lines.push(`   - ${m.description}`);
    }
    if (m.range) {
      lines.push(`   - Range: ${m.range[0]} - ${m.range[1]}${m.unit ? ' ' + m.unit : ''}`);
    }
    lines.push(`   - ${m.higherIsBetter ? 'Higher is better' : 'Lower is better'}`);
    lines.push('');
  });

  return lines.join('\n');
}

/**
 * Generate evaluation report table header
 */
export function generateMetricsTableHeader(metrics: EvaluationMetric[]): string {
  const metricNames = metrics.map((m) => m.name);
  const header = `| Metric | ${metricNames.join(' | ')} |`;
  const separator = `|--------|${metricNames.map(() => '--------').join('|')}|`;
  return `${header}\n${separator}`;
}

export default getDomainMetrics;
