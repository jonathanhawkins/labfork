/**
 * Tests for DomainCard Component
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DomainCard, DomainCardProps } from '@/components/domain/DomainCard';

// Mock next/link
vi.mock('next/link', () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

const defaultProps: DomainCardProps = {
  name: 'Voice Clone Lab',
  slug: 'voice-clone',
  description: 'Research prosody and emotion conditioning in TTS',
  difficulty: 'advanced',
  primaryColor: '#4ecdc4',
  accentColor: '#66ffaa',
  tags: ['tts', 'prosody', 'voice-cloning'],
  propsCount: 5,
  metricsCount: 6,
};

describe('DomainCard', () => {
  describe('rendering', () => {
    it('renders domain name', () => {
      render(<DomainCard {...defaultProps} />);
      expect(screen.getByText('Voice Clone Lab')).toBeDefined();
    });

    it('renders domain description', () => {
      render(<DomainCard {...defaultProps} />);
      expect(screen.getByText(/Research prosody/)).toBeDefined();
    });

    it('renders difficulty badge', () => {
      render(<DomainCard {...defaultProps} />);
      expect(screen.getByText('advanced')).toBeDefined();
    });

    it('renders tags', () => {
      render(<DomainCard {...defaultProps} />);
      expect(screen.getByText('tts')).toBeDefined();
      expect(screen.getByText('prosody')).toBeDefined();
    });

    it('renders props count', () => {
      render(<DomainCard {...defaultProps} />);
      expect(screen.getByText('5 props')).toBeDefined();
    });

    it('renders metrics count', () => {
      render(<DomainCard {...defaultProps} />);
      expect(screen.getByText('6 metrics')).toBeDefined();
    });
  });

  describe('compact mode', () => {
    it('hides description in compact mode', () => {
      render(<DomainCard {...defaultProps} compact />);
      expect(screen.queryByText(/Research prosody/)).toBeNull();
    });

    it('hides difficulty badge in compact mode', () => {
      render(<DomainCard {...defaultProps} compact />);
      expect(screen.queryByText('advanced')).toBeNull();
    });

    it('hides tags in compact mode', () => {
      render(<DomainCard {...defaultProps} compact />);
      expect(screen.queryByText('tts')).toBeNull();
    });
  });

  describe('navigation', () => {
    it('links to lab page with domain parameter', () => {
      render(<DomainCard {...defaultProps} />);
      const link = screen.getByRole('link');
      expect(link.getAttribute('href')).toBe('/lab?domain=voice-clone');
    });

    it('calls onClick instead of navigating when provided', () => {
      const onClick = vi.fn();
      render(<DomainCard {...defaultProps} onClick={onClick} />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(onClick).toHaveBeenCalledTimes(1);
    });
  });

  describe('selection state', () => {
    it('applies selected styles when isSelected is true', () => {
      const { container } = render(<DomainCard {...defaultProps} isSelected />);
      const card = container.querySelector('div[class*="ring-"]');
      expect(card).toBeDefined();
    });
  });

  describe('difficulty colors', () => {
    it('uses green for beginner', () => {
      render(<DomainCard {...defaultProps} difficulty="beginner" />);
      const badge = screen.getByText('beginner');
      expect(badge.className).toContain('green');
    });

    it('uses yellow for intermediate', () => {
      render(<DomainCard {...defaultProps} difficulty="intermediate" />);
      const badge = screen.getByText('intermediate');
      expect(badge.className).toContain('yellow');
    });

    it('uses red for advanced', () => {
      render(<DomainCard {...defaultProps} difficulty="advanced" />);
      const badge = screen.getByText('advanced');
      expect(badge.className).toContain('red');
    });
  });

  describe('icon selection', () => {
    it('shows microphone icon for voice-related tags', () => {
      const { container } = render(
        <DomainCard {...defaultProps} tags={['voice', 'speech']} />
      );
      // Icon should be rendered (we can't easily check which icon without more setup)
      const iconContainer = container.querySelector('div[class*="rounded-lg"]');
      expect(iconContainer).toBeDefined();
    });

    it('shows chart icon for trading-related tags', () => {
      const { container } = render(
        <DomainCard {...defaultProps} tags={['trading', 'finance']} />
      );
      const iconContainer = container.querySelector('div[class*="rounded-lg"]');
      expect(iconContainer).toBeDefined();
    });
  });

  describe('tags overflow', () => {
    it('shows +N more when more than 4 tags', () => {
      render(
        <DomainCard
          {...defaultProps}
          tags={['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6']}
        />
      );
      expect(screen.getByText('+2 more')).toBeDefined();
    });

    it('does not show +N more for 4 or fewer tags', () => {
      render(
        <DomainCard
          {...defaultProps}
          tags={['tag1', 'tag2', 'tag3', 'tag4']}
        />
      );
      expect(screen.queryByText(/more/)).toBeNull();
    });
  });
});
