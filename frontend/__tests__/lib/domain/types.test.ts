import { describe, it, expect } from 'vitest';
import {
  DomainConfig,
  isDomainConfig,
  mergeWithDefaults,
  hexToNumber,
  numberToHex,
  DEFAULT_DOMAIN_CONFIG,
} from '@/lib/domain/types';

describe('Domain Types', () => {
  describe('hexToNumber', () => {
    it('converts hex string to number', () => {
      expect(hexToNumber('#ffffff')).toBe(0xffffff);
      expect(hexToNumber('#000000')).toBe(0x000000);
      expect(hexToNumber('#4ecdc4')).toBe(0x4ecdc4);
      expect(hexToNumber('#3b82f6')).toBe(0x3b82f6);
    });

    it('handles lowercase hex', () => {
      expect(hexToNumber('#aabbcc')).toBe(0xaabbcc);
    });

    it('handles uppercase hex', () => {
      expect(hexToNumber('#AABBCC')).toBe(0xaabbcc);
    });
  });

  describe('numberToHex', () => {
    it('converts number to hex string', () => {
      expect(numberToHex(0xffffff)).toBe('#ffffff');
      expect(numberToHex(0x000000)).toBe('#000000');
      expect(numberToHex(0x4ecdc4)).toBe('#4ecdc4');
    });

    it('pads short numbers', () => {
      expect(numberToHex(0x00ff00)).toBe('#00ff00');
      expect(numberToHex(0x000001)).toBe('#000001');
    });
  });

  describe('isDomainConfig', () => {
    it('returns true for valid config', () => {
      const config: DomainConfig = {
        name: 'Test Lab',
        slug: 'test-lab',
        description: 'A test lab',
        branding: {
          primaryColor: '#3b82f6',
          accentColor: '#22c55e',
          backgroundStyle: 'sky',
        },
        scene: {
          props: [],
        },
        research: {
          arxivCategories: ['cs.LG'],
          keywords: ['test'],
        },
      };
      expect(isDomainConfig(config)).toBe(true);
    });

    it('returns false for null', () => {
      expect(isDomainConfig(null)).toBe(false);
    });

    it('returns false for undefined', () => {
      expect(isDomainConfig(undefined)).toBe(false);
    });

    it('returns false for missing name', () => {
      const config = {
        slug: 'test',
        description: 'test',
        branding: { primaryColor: '#000', accentColor: '#fff' },
        scene: { props: [] },
        research: { arxivCategories: [], keywords: [] },
      };
      expect(isDomainConfig(config)).toBe(false);
    });

    it('returns false for missing branding', () => {
      const config = {
        name: 'Test',
        slug: 'test',
        description: 'test',
        scene: { props: [] },
        research: { arxivCategories: [], keywords: [] },
      };
      expect(isDomainConfig(config)).toBe(false);
    });

    it('returns false for missing scene', () => {
      const config = {
        name: 'Test',
        slug: 'test',
        description: 'test',
        branding: { primaryColor: '#000', accentColor: '#fff' },
        research: { arxivCategories: [], keywords: [] },
      };
      expect(isDomainConfig(config)).toBe(false);
    });

    it('returns false for missing research', () => {
      const config = {
        name: 'Test',
        slug: 'test',
        description: 'test',
        branding: { primaryColor: '#000', accentColor: '#fff' },
        scene: { props: [] },
      };
      expect(isDomainConfig(config)).toBe(false);
    });
  });

  describe('mergeWithDefaults', () => {
    it('fills in default values', () => {
      const partial = {
        name: 'My Lab',
        slug: 'my-lab',
        description: 'My research lab',
      };
      const merged = mergeWithDefaults(partial);

      expect(merged.name).toBe('My Lab');
      expect(merged.slug).toBe('my-lab');
      expect(merged.description).toBe('My research lab');
      expect(merged.difficulty).toBe('intermediate');
      expect(merged.branding.backgroundStyle).toBe('sky');
      expect(merged.scene.decorations?.plants).toBe(true);
    });

    it('preserves provided values', () => {
      const partial = {
        name: 'My Lab',
        slug: 'my-lab',
        description: 'My lab',
        difficulty: 'advanced' as const,
        branding: {
          primaryColor: '#ff0000',
          accentColor: '#00ff00',
          backgroundStyle: 'space' as const,
        },
      };
      const merged = mergeWithDefaults(partial);

      expect(merged.difficulty).toBe('advanced');
      expect(merged.branding.primaryColor).toBe('#ff0000');
      expect(merged.branding.accentColor).toBe('#00ff00');
      expect(merged.branding.backgroundStyle).toBe('space');
    });

    it('provides default name and slug if missing', () => {
      const merged = mergeWithDefaults({});

      expect(merged.name).toBe('Unnamed Lab');
      expect(merged.slug).toBe('unnamed');
      expect(merged.description).toBe('A research lab');
    });
  });

  describe('DEFAULT_DOMAIN_CONFIG', () => {
    it('has expected default values', () => {
      expect(DEFAULT_DOMAIN_CONFIG.difficulty).toBe('intermediate');
      expect(DEFAULT_DOMAIN_CONFIG.branding?.primaryColor).toBe('#3b82f6');
      expect(DEFAULT_DOMAIN_CONFIG.branding?.accentColor).toBe('#22c55e');
      expect(DEFAULT_DOMAIN_CONFIG.branding?.backgroundStyle).toBe('sky');
      expect(DEFAULT_DOMAIN_CONFIG.research?.arxivCategories).toEqual(['cs.LG']);
    });
  });
});
