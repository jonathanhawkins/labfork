/**
 * API Route: List All Domains
 *
 * GET /api/domains - Returns list of all available domains with summary info
 * GET /api/domains?category=ml - Filter by tag/category
 * GET /api/domains?difficulty=beginner - Filter by difficulty
 */

import { NextRequest, NextResponse } from 'next/server';

// Force dynamic rendering since we read from filesystem and use query params
export const dynamic = 'force-dynamic';
import { listDomains, loadDomainConfigSafe } from '@/lib/domain/loader';
import { DomainConfig } from '@/lib/domain/types';

/**
 * Summary info returned for each domain (lighter than full config)
 */
export interface DomainSummary {
  name: string;
  slug: string;
  description: string;
  difficulty?: 'beginner' | 'intermediate' | 'advanced';
  primaryColor: string;
  accentColor: string;
  backgroundStyle: string;
  tags?: string[];
  propsCount: number;
  metricsCount: number;
}

/**
 * Convert full domain config to summary
 */
function toSummary(config: DomainConfig): DomainSummary {
  return {
    name: config.name,
    slug: config.slug,
    description: config.description,
    difficulty: config.difficulty,
    primaryColor: config.branding.primaryColor,
    accentColor: config.branding.accentColor,
    backgroundStyle: config.branding.backgroundStyle,
    tags: config.tags,
    propsCount: config.scene.props.length,
    metricsCount: config.evaluation?.metrics.length ?? 0,
  };
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get('category');
    const difficulty = searchParams.get('difficulty');
    const search = searchParams.get('search');

    // Get all domain slugs
    const slugs = listDomains();

    // Load each domain config
    const domains: DomainSummary[] = [];

    for (const slug of slugs) {
      const config = loadDomainConfigSafe(slug);
      if (!config) continue;

      // Apply filters
      if (category) {
        const hasCategory = config.tags?.some(
          (tag) => tag.toLowerCase().includes(category.toLowerCase())
        );
        if (!hasCategory) continue;
      }

      if (difficulty && config.difficulty !== difficulty) {
        continue;
      }

      if (search) {
        const searchLower = search.toLowerCase();
        const matchesSearch =
          config.name.toLowerCase().includes(searchLower) ||
          config.description.toLowerCase().includes(searchLower) ||
          config.tags?.some((tag) => tag.toLowerCase().includes(searchLower));
        if (!matchesSearch) continue;
      }

      domains.push(toSummary(config));
    }

    // Sort by name
    domains.sort((a, b) => a.name.localeCompare(b.name));

    return NextResponse.json({
      domains,
      total: domains.length,
      filters: {
        category,
        difficulty,
        search,
      },
    });
  } catch (error) {
    console.error('Error listing domains:', error);
    return NextResponse.json(
      { error: 'Failed to list domains' },
      { status: 500 }
    );
  }
}
