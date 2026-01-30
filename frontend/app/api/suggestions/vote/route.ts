import { NextRequest, NextResponse } from 'next/server';
import {
  readFileSync,
  writeFileSync,
  existsSync,
} from 'fs';
import { join } from 'path';

/**
 * Vote API for Community Suggestions
 *
 * Allows users to upvote suggestions (one vote per IP per suggestion)
 */

interface Suggestion {
  id: string;
  title: string;
  description: string;
  category: 'feature' | 'improvement' | 'bug';
  status: 'pending' | 'approved' | 'in-progress' | 'done' | 'rejected';
  votes: number;
  submittedAt: string;
  voters?: string[]; // IPs that have voted (hashed)
}

interface SuggestionsStore {
  suggestions: Suggestion[];
  lastUpdated: string;
}

// Storage paths
const DATA_DIR = join(process.cwd(), '..', 'data');
const SUGGESTIONS_FILE = join(DATA_DIR, 'suggestions.json');

/**
 * Simple hash function for IP (for privacy)
 */
function hashIp(ip: string): string {
  let hash = 0;
  for (let i = 0; i < ip.length; i++) {
    const char = ip.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
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
 * Read suggestions from file
 */
function readSuggestions(): SuggestionsStore {
  try {
    if (!existsSync(SUGGESTIONS_FILE)) {
      return { suggestions: [], lastUpdated: new Date().toISOString() };
    }
    const content = readFileSync(SUGGESTIONS_FILE, 'utf-8');
    return JSON.parse(content);
  } catch {
    return { suggestions: [], lastUpdated: new Date().toISOString() };
  }
}

/**
 * Write suggestions to file
 */
function writeSuggestions(store: SuggestionsStore) {
  store.lastUpdated = new Date().toISOString();
  writeFileSync(SUGGESTIONS_FILE, JSON.stringify(store, null, 2));
}

/**
 * POST /api/suggestions/vote
 * Vote on a suggestion
 */
export async function POST(request: NextRequest) {
  try {
    const ip = getClientIp(request);
    const ipHash = hashIp(ip);

    const body = await request.json();
    const { id } = body;

    if (!id) {
      return NextResponse.json(
        { error: 'Suggestion ID required' },
        { status: 400 }
      );
    }

    const store = readSuggestions();
    const suggestion = store.suggestions.find(s => s.id === id);

    if (!suggestion) {
      return NextResponse.json(
        { error: 'Suggestion not found' },
        { status: 404 }
      );
    }

    // Initialize voters array if not present
    if (!suggestion.voters) {
      suggestion.voters = [];
    }

    // Check if already voted
    if (suggestion.voters.includes(ipHash)) {
      return NextResponse.json(
        { error: 'You have already voted on this suggestion' },
        { status: 400 }
      );
    }

    // Add vote
    suggestion.votes++;
    suggestion.voters.push(ipHash);
    writeSuggestions(store);

    return NextResponse.json({
      success: true,
      votes: suggestion.votes,
    });
  } catch (error) {
    console.error('Error voting:', error);
    return NextResponse.json(
      { error: 'Failed to record vote' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/suggestions/vote
 * Remove vote from a suggestion
 */
export async function DELETE(request: NextRequest) {
  try {
    const ip = getClientIp(request);
    const ipHash = hashIp(ip);

    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json(
        { error: 'Suggestion ID required' },
        { status: 400 }
      );
    }

    const store = readSuggestions();
    const suggestion = store.suggestions.find(s => s.id === id);

    if (!suggestion) {
      return NextResponse.json(
        { error: 'Suggestion not found' },
        { status: 404 }
      );
    }

    // Check if has voted
    if (!suggestion.voters || !suggestion.voters.includes(ipHash)) {
      return NextResponse.json(
        { error: 'You have not voted on this suggestion' },
        { status: 400 }
      );
    }

    // Remove vote
    suggestion.votes = Math.max(0, suggestion.votes - 1);
    suggestion.voters = suggestion.voters.filter(v => v !== ipHash);
    writeSuggestions(store);

    return NextResponse.json({
      success: true,
      votes: suggestion.votes,
    });
  } catch (error) {
    console.error('Error removing vote:', error);
    return NextResponse.json(
      { error: 'Failed to remove vote' },
      { status: 500 }
    );
  }
}
