/**
 * Onboarding state management for distributed compute contributors
 *
 * Tracks whether users have completed the onboarding wizard
 * and provides utilities for managing onboarding state.
 */

const ONBOARDING_KEY = 'labfork-compute-onboarding-complete';
const ONBOARDING_VERSION = '1.0';

export interface OnboardingState {
  completed: boolean;
  version: string;
  completedAt?: number;
}

/**
 * Check if user has completed onboarding
 */
export function hasCompletedOnboarding(): boolean {
  if (typeof window === 'undefined') return false;

  try {
    const state = localStorage.getItem(ONBOARDING_KEY);
    if (!state) return false;

    const parsed: OnboardingState = JSON.parse(state);
    return parsed.completed && parsed.version === ONBOARDING_VERSION;
  } catch {
    return false;
  }
}

/**
 * Mark onboarding as complete
 */
export function markOnboardingComplete(): void {
  if (typeof window === 'undefined') return;

  const state: OnboardingState = {
    completed: true,
    version: ONBOARDING_VERSION,
    completedAt: Date.now(),
  };

  localStorage.setItem(ONBOARDING_KEY, JSON.stringify(state));
}

/**
 * Reset onboarding state (for testing or replay)
 */
export function resetOnboarding(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(ONBOARDING_KEY);
}

/**
 * Get full onboarding state
 */
export function getOnboardingState(): OnboardingState | null {
  if (typeof window === 'undefined') return null;

  try {
    const state = localStorage.getItem(ONBOARDING_KEY);
    if (!state) return null;
    return JSON.parse(state);
  } catch {
    return null;
  }
}
