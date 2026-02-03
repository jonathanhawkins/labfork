import { NextRequest, NextResponse } from 'next/server';
import {
  listSuggestions,
  createSuggestion,
} from '@/lib/social/suggestions';
import type { CreateSuggestionInput, SuggestionAuthor } from '@/lib/social/suggestions/types';

/**
 * Community Suggestions API
 *
 * Uses the social suggestions system which works on all platforms.
 */

// Default lab ID for community suggestions
const COMMUNITY_LAB_ID = 'community';

// Rate limiting in-memory store (resets on deploy)
const rateLimits = new Map<string, { count: number; resetAt: number }>();

const RATE_LIMIT_MAX = 5;
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000; // 1 hour

/**
 * Check rate limit for an IP
 */
function checkRateLimit(ip: string): { allowed: boolean; remaining: number; resetIn: number } {
  const now = Date.now();
  const entry = rateLimits.get(ip);

  if (!entry || entry.resetAt < now) {
    rateLimits.set(ip, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return { allowed: true, remaining: RATE_LIMIT_MAX - 1, resetIn: RATE_LIMIT_WINDOW_MS };
  }

  if (entry.count >= RATE_LIMIT_MAX) {
    return { allowed: false, remaining: 0, resetIn: entry.resetAt - now };
  }

  entry.count++;
  return { allowed: true, remaining: RATE_LIMIT_MAX - entry.count, resetIn: entry.resetAt - now };
}

/**
 * Get client IP from request headers
 */
function getClientIp(request: NextRequest): string {
  const forwarded = request.headers.get('x-forwarded-for');
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  return request.headers.get('x-real-ip') || 'unknown';
}

/**
 * Sanitize user input
 */
function sanitizeInput(input: string, maxLength: number): string {
  return input
    .trim()
    .slice(0, maxLength)
    .replace(/<[^>]*>/g, '')
    .replace(/[^\w\s.,!?'"-]/g, '');
}

/**
 * Map category from legacy format to social suggestions format
 */
function mapCategory(category: string): CreateSuggestionInput['category'] {
  const mapping: Record<string, CreateSuggestionInput['category']> = {
    feature: 'feature_request',
    improvement: 'improvement',
    bug: 'bug_report',
  };
  return mapping[category] || 'feature_request';
}

/**
 * GET /api/suggestions
 * List public suggestions
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '50');
    const category = searchParams.get('category');

    const result = await listSuggestions({
      labId: COMMUNITY_LAB_ID,
      category: category ? mapCategory(category) : undefined,
      status: ['open', 'under_review', 'planned', 'in_progress', 'completed'],
      sortBy: 'votes',
      limit,
    });

    // Transform to legacy format for backward compatibility
    const suggestions = result.suggestions.map(s => ({
      id: s.id,
      title: s.title,
      description: s.description,
      category: s.category === 'feature_request' ? 'feature' :
                s.category === 'bug_report' ? 'bug' : s.category,
      status: s.status === 'open' ? 'pending' :
              s.status === 'under_review' ? 'approved' :
              s.status === 'in_progress' ? 'in-progress' :
              s.status === 'completed' ? 'done' : s.status,
      votes: s.stats.upvotes - s.stats.downvotes,
      submittedAt: s.createdAt,
    }));

    return NextResponse.json({
      suggestions,
      total: result.total,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error) {
    console.error('Error fetching suggestions:', error);
    return NextResponse.json(
      { error: 'Failed to fetch suggestions' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/suggestions
 * Submit a new suggestion
 */
export async function POST(request: NextRequest) {
  try {
    const ip = getClientIp(request);

    // Check rate limit
    const rateLimit = checkRateLimit(ip);
    if (!rateLimit.allowed) {
      const resetMinutes = Math.ceil(rateLimit.resetIn / 60000);
      return NextResponse.json(
        {
          error: 'Rate limit exceeded',
          message: `You can submit again in ${resetMinutes} minute(s)`,
          resetIn: rateLimit.resetIn,
        },
        {
          status: 429,
          headers: {
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': String(Date.now() + rateLimit.resetIn),
          },
        }
      );
    }

    const body = await request.json();
    const { title, description, category } = body;

    // Validation
    if (!title || typeof title !== 'string' || title.trim().length < 5) {
      return NextResponse.json(
        { error: 'Title must be at least 5 characters' },
        { status: 400 }
      );
    }

    if (!description || typeof description !== 'string' || description.trim().length < 10) {
      return NextResponse.json(
        { error: 'Description must be at least 10 characters' },
        { status: 400 }
      );
    }

    const validCategories = ['feature', 'improvement', 'bug'];
    if (!validCategories.includes(category)) {
      return NextResponse.json(
        { error: 'Category must be one of: feature, improvement, bug' },
        { status: 400 }
      );
    }

    // Create anonymous author
    const author: SuggestionAuthor = {
      id: `anon_${ip.replace(/\./g, '_').slice(0, 8)}`,
      username: 'anonymous',
      displayName: 'Community Member',
    };

    // Create suggestion using the social suggestions system
    const input: CreateSuggestionInput = {
      labId: COMMUNITY_LAB_ID,
      title: sanitizeInput(title, 100),
      description: sanitizeInput(description, 1000),
      category: mapCategory(category),
      priority: 'medium',
      tags: [],
    };

    const suggestion = await createSuggestion(input, author);

    return NextResponse.json({
      success: true,
      suggestion: {
        id: suggestion.id,
        title: suggestion.title,
        description: suggestion.description,
        category: category,
        status: 'pending',
        votes: 0,
        submittedAt: suggestion.createdAt,
      },
      remaining: rateLimit.remaining,
    }, {
      headers: {
        'X-RateLimit-Remaining': String(rateLimit.remaining),
      },
    });
  } catch (error) {
    console.error('Error creating suggestion:', error);
    return NextResponse.json(
      { error: 'Failed to create suggestion' },
      { status: 500 }
    );
  }
}
