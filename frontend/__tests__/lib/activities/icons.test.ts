import { describe, it, expect } from 'vitest';
import {
  getActivityIcon,
  isValidIconName,
  getAvailableIcons,
  ICON_MAP,
  DEFAULT_ACTIVITY_ICONS,
} from '@/lib/activities/icons';
import { Activity } from 'lucide-react';

describe('Activity Icons', () => {
  describe('ICON_MAP', () => {
    it('contains Brain icon', () => {
      expect(ICON_MAP.Brain).toBeDefined();
    });

    it('contains Mic icon', () => {
      expect(ICON_MAP.Mic).toBeDefined();
    });

    it('contains Code icon', () => {
      expect(ICON_MAP.Code).toBeDefined();
    });

    it('contains common icons', () => {
      const expectedIcons = ['Brain', 'Mic', 'Sparkles', 'ChartBar', 'Search', 'Code', 'Pause'];
      expectedIcons.forEach(icon => {
        expect(ICON_MAP[icon]).toBeDefined();
      });
    });
  });

  describe('DEFAULT_ACTIVITY_ICONS', () => {
    it('maps training to Brain', () => {
      expect(DEFAULT_ACTIVITY_ICONS.training).toBe('Brain');
    });

    it('maps recording to Mic', () => {
      expect(DEFAULT_ACTIVITY_ICONS.recording).toBe('Mic');
    });

    it('maps generation to Sparkles', () => {
      expect(DEFAULT_ACTIVITY_ICONS.generation).toBe('Sparkles');
    });

    it('maps idle to Pause', () => {
      expect(DEFAULT_ACTIVITY_ICONS.idle).toBe('Pause');
    });
  });

  describe('getActivityIcon', () => {
    it('returns icon by explicit name', () => {
      const icon = getActivityIcon('Brain');
      expect(icon).toBe(ICON_MAP.Brain);
    });

    it('returns icon by activity id', () => {
      const icon = getActivityIcon(undefined, 'training');
      expect(icon).toBe(ICON_MAP.Brain);
    });

    it('prefers explicit name over activity id', () => {
      const icon = getActivityIcon('Mic', 'training');
      expect(icon).toBe(ICON_MAP.Mic);
    });

    it('returns Activity as fallback', () => {
      const icon = getActivityIcon(undefined, 'unknown-activity');
      expect(icon).toBe(Activity);
    });

    it('returns Activity for invalid icon name', () => {
      const icon = getActivityIcon('NonexistentIcon');
      expect(icon).toBe(Activity);
    });
  });

  describe('isValidIconName', () => {
    it('returns true for valid icon names', () => {
      expect(isValidIconName('Brain')).toBe(true);
      expect(isValidIconName('Mic')).toBe(true);
      expect(isValidIconName('Code')).toBe(true);
    });

    it('returns false for invalid icon names', () => {
      expect(isValidIconName('NotAnIcon')).toBe(false);
      expect(isValidIconName('')).toBe(false);
    });
  });

  describe('getAvailableIcons', () => {
    it('returns array of icon names', () => {
      const icons = getAvailableIcons();
      expect(Array.isArray(icons)).toBe(true);
      expect(icons.length).toBeGreaterThan(0);
    });

    it('includes expected icons', () => {
      const icons = getAvailableIcons();
      expect(icons).toContain('Brain');
      expect(icons).toContain('Mic');
      expect(icons).toContain('Code');
    });
  });
});
