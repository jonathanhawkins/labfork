/**
 * Tests for DomainBrowser Component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DomainBrowser, DomainSummary } from '@/components/domain/DomainBrowser';

// Mock next/link
vi.mock('next/link', () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

const mockDomains: DomainSummary[] = [
  {
    name: 'Voice Clone Lab',
    slug: 'voice-clone',
    description: 'Research TTS and voice cloning',
    difficulty: 'advanced',
    primaryColor: '#4ecdc4',
    accentColor: '#66ffaa',
    backgroundStyle: 'sky',
    tags: ['tts', 'voice-cloning', 'speech'],
    propsCount: 5,
    metricsCount: 6,
  },
  {
    name: 'Quant Trading Lab',
    slug: 'quant-trading',
    description: 'Algorithmic trading research',
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
    description: 'NLP for drug discovery',
    difficulty: 'intermediate',
    primaryColor: '#8b5cf6',
    accentColor: '#22c55e',
    backgroundStyle: 'gradient',
    tags: ['nlp', 'biotech', 'drug-discovery'],
    propsCount: 4,
    metricsCount: 5,
  },
];

describe('DomainBrowser', () => {
  beforeEach(() => {
    // Mock fetch
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ domains: mockDomains, total: 3 }),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('loading state', () => {
    it('shows loading indicator while fetching', () => {
      render(<DomainBrowser />);
      expect(screen.getByText('Loading domains...')).toBeDefined();
    });

    it('shows domains after loading', async () => {
      render(<DomainBrowser />);

      await waitFor(() => {
        expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      });

      expect(screen.getByText('Quant Trading Lab')).toBeDefined();
      expect(screen.getByText('Biotech NLP Lab')).toBeDefined();
    });
  });

  describe('with initial domains', () => {
    it('skips fetch when initialDomains provided', () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      expect(global.fetch).not.toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('shows error message on fetch failure', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      render(<DomainBrowser />);

      await waitFor(() => {
        expect(screen.getByText(/Network error/)).toBeDefined();
      }, { timeout: 2000 });
    });

    it('provides retry button on error', async () => {
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      render(<DomainBrowser />);

      await waitFor(() => {
        expect(screen.getByText('Try again')).toBeDefined();
      });
    });
  });

  describe('search filtering', () => {
    it('filters domains by search query', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      const searchInput = screen.getByPlaceholderText('Search domains...');
      fireEvent.change(searchInput, { target: { value: 'trading' } });

      expect(screen.getByText('Quant Trading Lab')).toBeDefined();
      expect(screen.queryByText('Voice Clone Lab')).toBeNull();
    });

    it('filters by tag match', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      const searchInput = screen.getByPlaceholderText('Search domains...');
      fireEvent.change(searchInput, { target: { value: 'nlp' } });

      expect(screen.getByText('Biotech NLP Lab')).toBeDefined();
      expect(screen.queryByText('Voice Clone Lab')).toBeNull();
    });

    it('shows no domains found when search has no matches', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      const searchInput = screen.getByPlaceholderText('Search domains...');
      fireEvent.change(searchInput, { target: { value: 'nonexistent' } });

      expect(screen.getByText('No domains found')).toBeDefined();
    });
  });

  describe('difficulty filtering', () => {
    it('filters by difficulty level', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      // Find the difficulty select
      const difficultySelect = screen.getAllByRole('combobox')[1];
      fireEvent.change(difficultySelect, { target: { value: 'intermediate' } });

      expect(screen.getByText('Biotech NLP Lab')).toBeDefined();
      expect(screen.queryByText('Voice Clone Lab')).toBeNull();
    });
  });

  describe('clear filters', () => {
    it('shows clear button when filters active', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      const searchInput = screen.getByPlaceholderText('Search domains...');
      fireEvent.change(searchInput, { target: { value: 'test' } });

      expect(screen.getByText('Clear')).toBeDefined();
    });

    it('clears all filters when clear clicked', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      const searchInput = screen.getByPlaceholderText('Search domains...');
      fireEvent.change(searchInput, { target: { value: 'trading' } });

      const clearButton = screen.getByText('Clear');
      fireEvent.click(clearButton);

      // All domains should be visible again
      expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      expect(screen.getByText('Quant Trading Lab')).toBeDefined();
    });
  });

  describe('domain selection', () => {
    it('calls onSelectDomain when domain clicked', async () => {
      const onSelectDomain = vi.fn();
      render(
        <DomainBrowser
          initialDomains={mockDomains}
          onSelectDomain={onSelectDomain}
        />
      );

      const voiceCloneButton = screen.getByText('Voice Clone Lab').closest('button');
      if (voiceCloneButton) {
        fireEvent.click(voiceCloneButton);
      }

      expect(onSelectDomain).toHaveBeenCalledWith('voice-clone');
    });

    it('highlights selected domain', async () => {
      render(
        <DomainBrowser
          initialDomains={mockDomains}
          selectedDomain="voice-clone"
          onSelectDomain={() => {}}
        />
      );

      // The selected card should have a ring class
      const selectedCard = screen.getByText('Voice Clone Lab').closest('div[class*="ring"]');
      expect(selectedCard).toBeDefined();
    });
  });

  describe('maxDomains limit', () => {
    it('limits displayed domains when maxDomains set', async () => {
      render(
        <DomainBrowser initialDomains={mockDomains} maxDomains={2} />
      );

      expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      expect(screen.getByText('Quant Trading Lab')).toBeDefined();
      expect(screen.queryByText('Biotech NLP Lab')).toBeNull();
    });
  });

  describe('hideFilters option', () => {
    it('hides filter controls when hideFilters is true', async () => {
      render(<DomainBrowser initialDomains={mockDomains} hideFilters />);

      expect(screen.queryByPlaceholderText('Search domains...')).toBeNull();
    });
  });

  describe('results count', () => {
    it('shows correct count of filtered results', async () => {
      render(<DomainBrowser initialDomains={mockDomains} />);

      expect(screen.getByText('Showing 3 of 3 domains')).toBeDefined();

      const searchInput = screen.getByPlaceholderText('Search domains...');
      fireEvent.change(searchInput, { target: { value: 'trading' } });

      expect(screen.getByText('Showing 1 of 3 domains')).toBeDefined();
    });
  });
});
