import { describe, it, expect } from 'vitest';
import {
  getDomainMetrics,
  getPrimaryMetric,
  getMetricById,
  formatMetricValue,
  isMetricGood,
  compareMetricValues,
  generateMetricsDocumentation,
  DEFAULT_METRICS,
} from '@/lib/domain/metrics';
import { DomainConfig, EvaluationMetric } from '@/lib/domain/types';

describe('Domain Metrics', () => {
  const mockConfig: DomainConfig = {
    name: 'Test',
    slug: 'test',
    description: 'Test domain',
    branding: {
      primaryColor: '#000',
      accentColor: '#fff',
      backgroundStyle: 'sky',
    },
    scene: { props: [] },
    research: { arxivCategories: [], keywords: [] },
    evaluation: {
      primaryMetric: 'mos',
      metrics: [
        {
          id: 'mos',
          name: 'Mean Opinion Score',
          description: 'Naturalness rating',
          range: [1, 5],
          higherIsBetter: true,
        },
        {
          id: 'f0_rmse',
          name: 'Pitch RMSE',
          description: 'Pitch accuracy',
          range: [0, 100],
          higherIsBetter: false,
          unit: 'Hz',
        },
      ],
    },
  };

  describe('getDomainMetrics', () => {
    it('returns metrics from config', () => {
      const metrics = getDomainMetrics(mockConfig);
      expect(metrics).toHaveLength(2);
      expect(metrics[0].id).toBe('mos');
    });

    it('returns default metrics when config is null', () => {
      const metrics = getDomainMetrics(null);
      expect(metrics).toBe(DEFAULT_METRICS);
    });

    it('returns default metrics when no evaluation config', () => {
      const configWithoutEval: DomainConfig = {
        ...mockConfig,
        evaluation: undefined,
      };
      const metrics = getDomainMetrics(configWithoutEval);
      expect(metrics).toBe(DEFAULT_METRICS);
    });
  });

  describe('getPrimaryMetric', () => {
    it('returns primary metric from config', () => {
      const primary = getPrimaryMetric(mockConfig);
      expect(primary?.id).toBe('mos');
    });

    it('returns first metric if primary not specified', () => {
      const configNoPrimary: DomainConfig = {
        ...mockConfig,
        evaluation: {
          primaryMetric: 'nonexistent',
          metrics: mockConfig.evaluation!.metrics,
        },
      };
      const primary = getPrimaryMetric(configNoPrimary);
      expect(primary?.id).toBe('mos');
    });

    it('returns null for null config', () => {
      const primary = getPrimaryMetric(null);
      expect(primary).not.toBeNull(); // Should return first default metric
    });
  });

  describe('getMetricById', () => {
    it('returns metric by id', () => {
      const metric = getMetricById(mockConfig, 'mos');
      expect(metric?.id).toBe('mos');
      expect(metric?.name).toBe('Mean Opinion Score');
    });

    it('returns null for unknown id', () => {
      const metric = getMetricById(mockConfig, 'unknown');
      expect(metric).toBeNull();
    });
  });

  describe('formatMetricValue', () => {
    const mosMetric: EvaluationMetric = {
      id: 'mos',
      name: 'MOS',
      range: [1, 5],
      higherIsBetter: true,
    };

    const rmseMetric: EvaluationMetric = {
      id: 'rmse',
      name: 'RMSE',
      range: [0, 100],
      higherIsBetter: false,
      unit: 'Hz',
    };

    it('formats value with unit', () => {
      expect(formatMetricValue(rmseMetric, 15.5)).toBe('15.50 Hz');
    });

    it('formats value without unit', () => {
      expect(formatMetricValue(mosMetric, 3.75)).toBe('3.75');
    });

    it('formats large values with no decimals', () => {
      expect(formatMetricValue(rmseMetric, 150)).toBe('150 Hz');
    });
  });

  describe('isMetricGood', () => {
    const higherBetter: EvaluationMetric = {
      id: 'test',
      name: 'Test',
      range: [0, 100],
      higherIsBetter: true,
    };

    const lowerBetter: EvaluationMetric = {
      id: 'test',
      name: 'Test',
      range: [0, 100],
      higherIsBetter: false,
    };

    it('returns true for high value when higher is better', () => {
      expect(isMetricGood(higherBetter, 75)).toBe(true);
    });

    it('returns false for low value when higher is better', () => {
      expect(isMetricGood(higherBetter, 25)).toBe(false);
    });

    it('returns true for low value when lower is better', () => {
      expect(isMetricGood(lowerBetter, 25)).toBe(true);
    });

    it('returns false for high value when lower is better', () => {
      expect(isMetricGood(lowerBetter, 75)).toBe(false);
    });
  });

  describe('compareMetricValues', () => {
    const higherBetter: EvaluationMetric = {
      id: 'test',
      name: 'Test',
      higherIsBetter: true,
    };

    const lowerBetter: EvaluationMetric = {
      id: 'test',
      name: 'Test',
      higherIsBetter: false,
    };

    it('returns positive when value1 is better (higher is better)', () => {
      expect(compareMetricValues(higherBetter, 80, 60)).toBeGreaterThan(0);
    });

    it('returns negative when value1 is worse (higher is better)', () => {
      expect(compareMetricValues(higherBetter, 60, 80)).toBeLessThan(0);
    });

    it('returns positive when value1 is better (lower is better)', () => {
      expect(compareMetricValues(lowerBetter, 20, 40)).toBeGreaterThan(0);
    });

    it('returns negative when value1 is worse (lower is better)', () => {
      expect(compareMetricValues(lowerBetter, 40, 20)).toBeLessThan(0);
    });
  });

  describe('generateMetricsDocumentation', () => {
    it('generates documentation string', () => {
      const metrics = mockConfig.evaluation!.metrics;
      const docs = generateMetricsDocumentation(metrics);

      expect(docs).toContain('Mean Opinion Score');
      expect(docs).toContain('Pitch RMSE');
      expect(docs).toContain('Higher is better');
      expect(docs).toContain('Lower is better');
    });

    it('returns message for empty metrics', () => {
      const docs = generateMetricsDocumentation([]);
      expect(docs).toBe('No metrics defined.');
    });
  });
});
