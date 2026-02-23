import { describe, it, expect } from 'vitest';
import {
  ActivityConfig,
  BUILTIN_ACTIVITIES,
  getBuiltinActivity,
  mergeActivityDefaults,
} from '@/lib/activities/types';

describe('Activity Types', () => {
  describe('BUILTIN_ACTIVITIES', () => {
    it('includes training activity', () => {
      const training = BUILTIN_ACTIVITIES.find(a => a.id === 'training');
      expect(training).toBeDefined();
      expect(training?.name).toBe('Model Training');
    });

    it('includes recording activity', () => {
      const recording = BUILTIN_ACTIVITIES.find(a => a.id === 'recording');
      expect(recording).toBeDefined();
      expect(recording?.name).toBe('Voice Recording');
    });

    it('includes generation activity', () => {
      const generation = BUILTIN_ACTIVITIES.find(a => a.id === 'generation');
      expect(generation).toBeDefined();
      expect(generation?.name).toBe('Speech Generation');
    });

    it('includes idle activity', () => {
      const idle = BUILTIN_ACTIVITIES.find(a => a.id === 'idle');
      expect(idle).toBeDefined();
      expect(idle?.name).toBe('Idle');
    });

    it('all activities have required fields', () => {
      BUILTIN_ACTIVITIES.forEach(activity => {
        expect(activity.id).toBeDefined();
        expect(activity.name).toBeDefined();
        expect(activity.description).toBeDefined();
        expect(activity.visualization).toBeDefined();
        expect(activity.visualization.animation).toBeDefined();
        expect(activity.visualization.particles).toBeDefined();
        expect(activity.visualization.color).toBeDefined();
        expect(activity.agentBehavior).toBeDefined();
        expect(activity.agentBehavior.defaultStatus).toBeDefined();
      });
    });
  });

  describe('getBuiltinActivity', () => {
    it('returns activity by id', () => {
      const training = getBuiltinActivity('training');
      expect(training).toBeDefined();
      expect(training?.id).toBe('training');
    });

    it('returns undefined for unknown id', () => {
      const unknown = getBuiltinActivity('nonexistent');
      expect(unknown).toBeUndefined();
    });
  });

  describe('mergeActivityDefaults', () => {
    it('fills in default values', () => {
      const partial = {
        id: 'custom',
        name: 'Custom Activity',
      };
      const merged = mergeActivityDefaults(partial);

      expect(merged.id).toBe('custom');
      expect(merged.name).toBe('Custom Activity');
      expect(merged.description).toBe('');
      expect(merged.visualization).toBeDefined();
      expect(merged.agentBehavior).toBeDefined();
    });

    it('preserves provided values', () => {
      const partial: Partial<ActivityConfig> = {
        id: 'custom',
        name: 'Custom',
        description: 'A custom activity',
        visualization: {
          animation: 'typing',
          particles: 'sparks',
          color: 0xff0000,
        },
      };
      const merged = mergeActivityDefaults(partial);

      expect(merged.visualization.animation).toBe('typing');
      expect(merged.visualization.particles).toBe('sparks');
      expect(merged.visualization.color).toBe(0xff0000);
    });

    it('provides default id if missing', () => {
      const merged = mergeActivityDefaults({});
      expect(merged.id).toBe('unknown');
    });
  });
});
