/**
 * Integration Tests for Domain Plugin Loading
 *
 * Tests the full flow of loading domain configurations from YAML files.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

// Mock the fs module
vi.mock('fs', () => ({
  existsSync: vi.fn(),
  readFileSync: vi.fn(),
  readdirSync: vi.fn(),
  statSync: vi.fn(),
}));

vi.mock('path', async () => {
  const actual = await vi.importActual('path');
  return {
    ...actual,
    join: (...args: string[]) => args.join('/'),
  };
});

describe('Domain Loading Integration', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  describe('Loading voice-clone domain', () => {
    const mockDomainYaml = `
name: "Voice Clone Research"
slug: "voice-clone"
description: "Research prosody and emotion conditioning in TTS"

branding:
  primaryColor: "#4ecdc4"
  accentColor: "#66ffaa"
  backgroundStyle: "sky"

scene:
  props:
    - id: supercomputer
      type: supercomputer
      position: [-6, 0, -5]
    - id: microphone
      type: microphone
      position: [-6, 0, 5]

research:
  arxivCategories:
    - cs.SD
    - cs.CL
  keywords:
    - prosody
    - emotion TTS

evaluation:
  primaryMetric: mos
  metrics:
    - id: mos
      name: Mean Opinion Score
      range: [1, 5]
      higherIsBetter: true
`;

    it('should load and parse domain YAML correctly', async () => {
      // Setup mocks
      (fs.existsSync as any).mockReturnValue(true);
      (fs.readFileSync as any).mockReturnValue(mockDomainYaml);

      // Import the loader dynamically to get fresh mock state
      const { loadDomainYaml, validateDomainConfig } = await import('@/lib/domain/loader');

      const config = loadDomainYaml('voice-clone');

      expect(config).toBeDefined();
      expect(config.name).toBe('Voice Clone Research');
      expect(config.slug).toBe('voice-clone');
      expect(config.branding.primaryColor).toBe('#4ecdc4');
    });

    it('should validate required fields', async () => {
      const { validateDomainConfig } = await import('@/lib/domain/loader');

      const validConfig = {
        name: 'Test',
        slug: 'test',
        description: 'Test domain',
        branding: {
          primaryColor: '#000',
          accentColor: '#fff',
        },
        scene: {
          props: [],
        },
        research: {
          arxivCategories: ['cs.LG'],
          keywords: ['test'],
        },
      };

      const result = validateDomainConfig(validConfig);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect invalid slug format', async () => {
      const { validateDomainConfig } = await import('@/lib/domain/loader');

      const invalidConfig = {
        name: 'Test',
        slug: 'Invalid Slug With Spaces',
        description: 'Test',
        branding: { primaryColor: '#000', accentColor: '#fff' },
        scene: { props: [] },
        research: { arxivCategories: [], keywords: [] },
      };

      const result = validateDomainConfig(invalidConfig);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain(
        'slug must contain only lowercase letters, numbers, and hyphens'
      );
    });

    it('should detect missing required sections', async () => {
      const { validateDomainConfig } = await import('@/lib/domain/loader');

      const incompleteConfig = {
        name: 'Test',
        slug: 'test',
        description: 'Test',
        // Missing branding, scene, research
      };

      const result = validateDomainConfig(incompleteConfig);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });
  });

  describe('Domain listing', () => {
    it('should list available domains', async () => {
      (fs.existsSync as any).mockReturnValue(true);
      (fs.readdirSync as any).mockReturnValue(['voice-clone', 'quant-trading', '.hidden', '_template']);
      (fs.existsSync as any).mockImplementation((path: string) => {
        // Return true for domain.yaml files
        return path.includes('domain.yaml') && !path.includes('.hidden') && !path.includes('_template');
      });

      const { listDomains } = await import('@/lib/domain/loader');

      // Mock needs to return true for domains dir check and domain.yaml checks
      vi.mocked(fs.existsSync).mockImplementation((p: any) => {
        if (p.toString().includes('.hidden') || p.toString().includes('_template')) {
          return false;
        }
        return true;
      });

      const domains = listDomains();

      // Should filter out hidden and template directories
      expect(domains).not.toContain('.hidden');
      expect(domains).not.toContain('_template');
    });
  });

  describe('Activity loading from domain', () => {
    const mockTrainingActivity = `
id: training
name: Model Training
description: Fine-tuning the voice model
icon: Brain

visualization:
  animation: focused
  particles: data-flow
  color: 0x4ade80
  workLocation: supercomputer

agentBehavior:
  defaultStatus: working
  faceProp: true
`;

    it('should load activities from domain directory', async () => {
      (fs.existsSync as any).mockReturnValue(true);
      (fs.readdirSync as any).mockReturnValue(['training.yaml', 'recording.yaml']);
      (fs.readFileSync as any).mockReturnValue(mockTrainingActivity);

      const { loadDomainActivities } = await import('@/lib/activities/registry');

      // Clear any cached activities
      const { clearActivityCache } = await import('@/lib/activities/registry');
      clearActivityCache();

      const activities = loadDomainActivities('voice-clone');

      expect(activities).toBeDefined();
      expect(Array.isArray(activities)).toBe(true);
    });
  });

  describe('Edge cases', () => {
    it('should handle domain not found gracefully', async () => {
      (fs.existsSync as any).mockReturnValue(false);

      const { loadDomainConfigSafe } = await import('@/lib/domain/loader');

      const config = loadDomainConfigSafe('nonexistent-domain');
      expect(config).toBeNull();
    });

    it('should handle malformed YAML', async () => {
      (fs.existsSync as any).mockReturnValue(true);
      (fs.readFileSync as any).mockReturnValue('this is: not: valid: yaml: {{{}}}');

      const { loadDomainConfigSafe } = await import('@/lib/domain/loader');

      const config = loadDomainConfigSafe('malformed');
      expect(config).toBeNull();
    });
  });
});
