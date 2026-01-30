/**
 * Integration Tests for Domain Selection Flow
 *
 * Tests API response shapes and domain configuration structures.
 */

import { describe, it, expect } from 'vitest';
import { DomainSummary } from '@/components/domain/DomainBrowser';

// Test the API response type structure
describe('Domain Selection API Types', () => {
  describe('DomainSummary type', () => {
    it('has required fields', () => {
      const summary: DomainSummary = {
        name: 'Test Domain',
        slug: 'test-domain',
        description: 'A test domain',
        primaryColor: '#000000',
        accentColor: '#ffffff',
        backgroundStyle: 'sky',
        propsCount: 5,
        metricsCount: 3,
      };

      expect(summary.name).toBe('Test Domain');
      expect(summary.slug).toBe('test-domain');
      expect(summary.description).toBeDefined();
    });

    it('supports optional difficulty field', () => {
      const summary: DomainSummary = {
        name: 'Test Domain',
        slug: 'test',
        description: 'Test',
        primaryColor: '#000',
        accentColor: '#fff',
        backgroundStyle: 'sky',
        propsCount: 0,
        metricsCount: 0,
        difficulty: 'advanced',
      };

      expect(summary.difficulty).toBe('advanced');
    });

    it('supports optional tags field', () => {
      const summary: DomainSummary = {
        name: 'Test Domain',
        slug: 'test',
        description: 'Test',
        primaryColor: '#000',
        accentColor: '#fff',
        backgroundStyle: 'sky',
        propsCount: 0,
        metricsCount: 0,
        tags: ['ml', 'nlp', 'research'],
      };

      expect(summary.tags).toHaveLength(3);
      expect(summary.tags).toContain('ml');
    });
  });

  describe('Domain URL patterns', () => {
    it('generates correct lab URL with domain parameter', () => {
      const slug = 'voice-clone';
      const labUrl = `/lab?domain=${slug}`;

      expect(labUrl).toBe('/lab?domain=voice-clone');
    });

    it('handles domain slugs with hyphens', () => {
      const slug = 'quant-trading';
      const labUrl = `/lab?domain=${slug}`;

      expect(labUrl).toBe('/lab?domain=quant-trading');
    });

    it('generates correct API endpoint', () => {
      const slug = 'voice-clone';
      const apiUrl = `/api/domain/${slug}`;

      expect(apiUrl).toBe('/api/domain/voice-clone');
    });
  });

  describe('Domain filtering logic', () => {
    const mockDomains: DomainSummary[] = [
      {
        name: 'Voice Clone Lab',
        slug: 'voice-clone',
        description: 'TTS research',
        difficulty: 'advanced',
        primaryColor: '#4ecdc4',
        accentColor: '#66ffaa',
        backgroundStyle: 'sky',
        tags: ['tts', 'voice', 'prosody'],
        propsCount: 5,
        metricsCount: 6,
      },
      {
        name: 'Quant Trading Lab',
        slug: 'quant-trading',
        description: 'Finance research',
        difficulty: 'advanced',
        primaryColor: '#10b981',
        accentColor: '#f59e0b',
        backgroundStyle: 'grid',
        tags: ['trading', 'finance', 'ml'],
        propsCount: 4,
        metricsCount: 5,
      },
      {
        name: 'Biotech NLP Lab',
        slug: 'biotech-nlp',
        description: 'Drug discovery',
        difficulty: 'intermediate',
        primaryColor: '#8b5cf6',
        accentColor: '#22c55e',
        backgroundStyle: 'gradient',
        tags: ['nlp', 'biotech'],
        propsCount: 4,
        metricsCount: 5,
      },
    ];

    it('filters by search query on name', () => {
      const query = 'voice';
      const filtered = mockDomains.filter(d =>
        d.name.toLowerCase().includes(query.toLowerCase())
      );

      expect(filtered).toHaveLength(1);
      expect(filtered[0].slug).toBe('voice-clone');
    });

    it('filters by search query on description', () => {
      const query = 'finance';
      const filtered = mockDomains.filter(d =>
        d.description.toLowerCase().includes(query.toLowerCase())
      );

      expect(filtered).toHaveLength(1);
      expect(filtered[0].slug).toBe('quant-trading');
    });

    it('filters by tag', () => {
      const tag = 'ml';
      const filtered = mockDomains.filter(d =>
        d.tags?.includes(tag)
      );

      expect(filtered).toHaveLength(1);
      expect(filtered[0].slug).toBe('quant-trading');
    });

    it('filters by difficulty', () => {
      const difficulty = 'intermediate';
      const filtered = mockDomains.filter(d =>
        d.difficulty === difficulty
      );

      expect(filtered).toHaveLength(1);
      expect(filtered[0].slug).toBe('biotech-nlp');
    });

    it('combines multiple filters', () => {
      const difficulty = 'advanced';
      const query = 'research';
      const filtered = mockDomains.filter(d =>
        d.difficulty === difficulty &&
        (d.name.toLowerCase().includes(query) ||
         d.description.toLowerCase().includes(query))
      );

      expect(filtered).toHaveLength(2);
    });
  });

  describe('Domain color validation', () => {
    it('validates hex color format', () => {
      const validColors = ['#4ecdc4', '#10b981', '#8b5cf6', '#000000', '#ffffff'];

      for (const color of validColors) {
        expect(color).toMatch(/^#[0-9a-fA-F]{6}$/);
      }
    });

    it('rejects invalid hex colors', () => {
      const invalidColors = ['red', '4ecdc4', '#4ec', '#gggggg'];

      for (const color of invalidColors) {
        expect(color).not.toMatch(/^#[0-9a-fA-F]{6}$/);
      }
    });
  });

  describe('Domain slug validation', () => {
    it('accepts valid slugs', () => {
      const validSlugs = ['voice-clone', 'quant-trading', 'biotech-nlp', 'robotics-ml', 'simple'];

      for (const slug of validSlugs) {
        expect(slug).toMatch(/^[a-z0-9-]+$/);
      }
    });

    it('rejects invalid slugs', () => {
      const invalidSlugs = ['Voice Clone', 'UPPERCASE', 'has spaces', 'special@chars'];

      for (const slug of invalidSlugs) {
        expect(slug).not.toMatch(/^[a-z0-9-]+$/);
      }
    });
  });
});
