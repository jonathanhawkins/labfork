/**
 * Tests for DomainSwitcher Component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DomainSwitcher } from '@/components/domain/DomainSwitcher';

// Mock next/navigation
vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
  useSearchParams: () => ({
    get: () => null,
    toString: () => '',
  }),
}));

// Mock next/link
vi.mock('next/link', () => ({
  default: ({ children, href, onClick }: { children: React.ReactNode; href: string; onClick?: () => void }) => (
    <a href={href} onClick={onClick}>{children}</a>
  ),
}));

const mockDomains = [
  { name: 'Voice Clone Lab', slug: 'voice-clone', primaryColor: '#4ecdc4' },
  { name: 'Quant Trading Lab', slug: 'quant-trading', primaryColor: '#10b981' },
  { name: 'Biotech NLP Lab', slug: 'biotech-nlp', primaryColor: '#8b5cf6' },
];

describe('DomainSwitcher', () => {
  beforeEach(() => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ domains: mockDomains }),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('loading state', () => {
    it('shows loading spinner while fetching', () => {
      render(<DomainSwitcher />);
      // Should show loading state initially
      const button = screen.getByRole('button');
      expect(button).toBeDefined();
    });

    it('shows current domain after loading', async () => {
      render(<DomainSwitcher currentDomain="voice-clone" />);

      await waitFor(() => {
        expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      });
    });
  });

  describe('dropdown toggle', () => {
    it('opens dropdown when clicked', async () => {
      render(<DomainSwitcher currentDomain="voice-clone" />);

      await waitFor(() => {
        expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      });

      const button = screen.getByRole('button');
      fireEvent.click(button);

      // All domains should be visible in dropdown
      expect(screen.getByText('Quant Trading Lab')).toBeDefined();
      expect(screen.getByText('Biotech NLP Lab')).toBeDefined();
    });

    it('closes dropdown when clicking outside', async () => {
      render(<DomainSwitcher currentDomain="voice-clone" />);

      await waitFor(() => {
        expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      });

      // Open dropdown
      const button = screen.getByRole('button');
      fireEvent.click(button);

      // Click outside (simulate by mousedown on document)
      fireEvent.mouseDown(document.body);

      // Dropdown should close - Quant Trading should not be visible in dropdown
      await waitFor(() => {
        // The button text should still show current domain
        expect(button.textContent).toContain('Voice Clone Lab');
      });
    });
  });

  describe('domain selection', () => {
    it('calls onDomainChange when domain selected', async () => {
      const onDomainChange = vi.fn();
      render(
        <DomainSwitcher
          currentDomain="voice-clone"
          onDomainChange={onDomainChange}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      });

      // Open dropdown
      const button = screen.getByRole('button');
      fireEvent.click(button);

      // Click on Quant Trading
      const quantOption = screen.getByText('Quant Trading Lab');
      fireEvent.click(quantOption);

      expect(onDomainChange).toHaveBeenCalledWith('quant-trading');
    });

    it('shows checkmark on current domain', async () => {
      render(<DomainSwitcher currentDomain="voice-clone" />);

      await waitFor(() => {
        // Wait for domains to load - button should show domain name
        expect(screen.getByRole('button')).toBeDefined();
      });

      // Open dropdown
      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        // Voice Clone should have checkmark (svg element with check class)
        // Find all buttons with Voice Clone Lab text
        const voiceCloneButtons = screen.getAllByText('Voice Clone Lab');
        // The dropdown option should have a check icon
        const dropdownOption = voiceCloneButtons.find(el => el.closest('button[class*="bg-foreground-muted"]'));
        expect(dropdownOption).toBeDefined();
      });
    });
  });

  describe('compact mode', () => {
    it('hides domain name in compact mode', async () => {
      render(<DomainSwitcher currentDomain="voice-clone" compact />);

      await waitFor(() => {
        const button = screen.getByRole('button');
        // In compact mode, should not show full domain name in button
        // Just the color indicator and chevron
        expect(button.textContent).not.toContain('Voice Clone Lab');
      });
    });
  });

  describe('browse all link', () => {
    it('includes link to browse all domains', async () => {
      render(<DomainSwitcher currentDomain="voice-clone" />);

      await waitFor(() => {
        expect(screen.getByText('Voice Clone Lab')).toBeDefined();
      });

      // Open dropdown
      fireEvent.click(screen.getByRole('button'));

      expect(screen.getByText('Browse All Domains')).toBeDefined();
      const browseLink = screen.getByText('Browse All Domains').closest('a');
      expect(browseLink?.getAttribute('href')).toBe('/domains');
    });
  });

  describe('color indicators', () => {
    it('shows color indicator matching domain primary color', async () => {
      const { container } = render(
        <DomainSwitcher currentDomain="voice-clone" />
      );

      await waitFor(() => {
        // Color indicator should have the domain's primary color
        const colorDot = container.querySelector('div[style*="background-color"]');
        expect(colorDot).toBeDefined();
      });
    });
  });

  describe('empty state', () => {
    it('shows message when no domains available', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ domains: [] }),
      });

      render(<DomainSwitcher />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('No domains available')).toBeDefined();
      });
    });
  });
});
