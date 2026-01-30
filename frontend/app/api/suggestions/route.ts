import { NextRequest, NextResponse } from 'next/server';
import {
  readFileSync,
  writeFileSync,
  existsSync,
  mkdirSync,
} from 'fs';
import { join } from 'path';

/**
 * Community Suggestions API
 *
 * Stores suggestions in a local JSON file for simplicity.
 * In production, this would use a proper database.
 */

// Types
interface Suggestion {
  id: string;
  title: string;
  description: string;
  category: 'feature' | 'improvement' | 'bug';
  status: 'pending' | 'approved' | 'in-progress' | 'done' | 'rejected';
  votes: number;
  submittedAt: string;
  submitterIp?: string; // For rate limiting only, not exposed publicly
  taskId?: string; // Link to task list when approved
  adminNote?: string;
}

interface SuggestionsStore {
  suggestions: Suggestion[];
  lastUpdated: string;
}

interface RateLimitEntry {
  count: number;
  resetAt: number;
}

// Storage paths
const DATA_DIR = join(process.cwd(), '..', 'data');
const SUGGESTIONS_FILE = join(DATA_DIR, 'suggestions.json');
const RATE_LIMIT_FILE = join(DATA_DIR, 'rate-limits.json');

// Rate limiting: 5 submissions per hour per IP
const RATE_LIMIT_MAX = 5;
const RATE_LIMIT_WINDOW_MS = 60 * 60 * 1000; // 1 hour

/**
 * Ensure data directory and files exist
 */
function ensureDataFiles() {
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true });
  }

  if (!existsSync(SUGGESTIONS_FILE)) {
    const initial: SuggestionsStore = {
      suggestions: [],
      lastUpdated: new Date().toISOString(),
    };
    writeFileSync(SUGGESTIONS_FILE, JSON.stringify(initial, null, 2));
  }

  if (!existsSync(RATE_LIMIT_FILE)) {
    writeFileSync(RATE_LIMIT_FILE, JSON.stringify({}, null, 2));
  }
}

/**
 * Read suggestions from file
 */
function readSuggestions(): SuggestionsStore {
  ensureDataFiles();
  try {
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
  ensureDataFiles();
  store.lastUpdated = new Date().toISOString();
  writeFileSync(SUGGESTIONS_FILE, JSON.stringify(store, null, 2));
}

/**
 * Read rate limits
 */
function readRateLimits(): Record<string, RateLimitEntry> {
  ensureDataFiles();
  try {
    const content = readFileSync(RATE_LIMIT_FILE, 'utf-8');
    return JSON.parse(content);
  } catch {
    return {};
  }
}

/**
 * Write rate limits
 */
function writeRateLimits(limits: Record<string, RateLimitEntry>) {
  ensureDataFiles();
  writeFileSync(RATE_LIMIT_FILE, JSON.stringify(limits, null, 2));
}

/**
 * Check and update rate limit for an IP
 * Returns true if allowed, false if rate limited
 */
function checkRateLimit(ip: string): { allowed: boolean; remaining: number; resetIn: number } {
  const limits = readRateLimits();
  const now = Date.now();

  // Clean up expired entries
  for (const [key, entry] of Object.entries(limits)) {
    if (entry.resetAt < now) {
      delete limits[key];
    }
  }

  const entry = limits[ip];

  if (!entry || entry.resetAt < now) {
    // New window
    limits[ip] = {
      count: 1,
      resetAt: now + RATE_LIMIT_WINDOW_MS,
    };
    writeRateLimits(limits);
    return {
      allowed: true,
      remaining: RATE_LIMIT_MAX - 1,
      resetIn: RATE_LIMIT_WINDOW_MS,
    };
  }

  if (entry.count >= RATE_LIMIT_MAX) {
    return {
      allowed: false,
      remaining: 0,
      resetIn: entry.resetAt - now,
    };
  }

  // Increment count
  entry.count++;
  writeRateLimits(limits);

  return {
    allowed: true,
    remaining: RATE_LIMIT_MAX - entry.count,
    resetIn: entry.resetAt - now,
  };
}

/**
 * Generate a unique ID
 */
function generateId(): string {
  return `sug_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 8)}`;
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
    .replace(/<[^>]*>/g, '') // Remove HTML tags
    .replace(/[^\w\s.,!?'-]/g, ''); // Keep only safe characters
}

// ============== Demo Data for Vercel ==============

const DEMO_SUGGESTIONS: Suggestion[] = [
  {
    id: 'demo_1',
    title: 'Add real-time voice preview during recording',
    description: 'Would be helpful to hear a preview of how the cloned voice sounds while recording new samples, so users can adjust their speaking style.',
    category: 'feature',
    status: 'approved',
    votes: 42,
    submittedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'demo_2',
    title: 'Support for emotional intensity slider',
    description: 'Allow users to control how strongly emotions are expressed in generated speech - from subtle to exaggerated.',
    category: 'improvement',
    status: 'in-progress',
    votes: 38,
    submittedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'demo_3',
    title: 'Export to multiple audio formats',
    description: 'Currently only WAV is supported. Would like MP3, FLAC, and OGG export options.',
    category: 'feature',
    status: 'approved',
    votes: 27,
    submittedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'demo_4',
    title: 'Batch processing for multiple text inputs',
    description: 'Ability to queue up multiple text snippets and generate all of them in one go.',
    category: 'feature',
    status: 'approved',
    votes: 19,
    submittedAt: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: 'demo_5',
    title: 'Improve prosody for questions',
    description: 'Questions dont always have the right rising intonation at the end. Could be improved.',
    category: 'bug',
    status: 'done',
    votes: 15,
    submittedAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

// ============== API Handlers ==============

/**
 * GET /api/suggestions
 * List public suggestions
 */
export async function GET(request: NextRequest) {
  // On Vercel, return demo suggestions (no filesystem access)
  const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;

  if (isVercel) {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '50');

    return NextResponse.json({
      suggestions: DEMO_SUGGESTIONS.slice(0, limit),
      total: DEMO_SUGGESTIONS.length,
      lastUpdated: new Date().toISOString(),
      demo: true,
    });
  }

  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get('status');
    const category = searchParams.get('category');
    const limit = parseInt(searchParams.get('limit') || '50');

    const store = readSuggestions();
    let suggestions = store.suggestions;

    // Filter by status (only show approved/in-progress/done to public)
    if (status) {
      suggestions = suggestions.filter(s => s.status === status);
    } else {
      // By default, don't show pending or rejected to public
      suggestions = suggestions.filter(s =>
        ['approved', 'in-progress', 'done'].includes(s.status)
      );
    }

    // Filter by category
    if (category) {
      suggestions = suggestions.filter(s => s.category === category);
    }

    // Sort by votes (descending) then by date (newest first)
    suggestions.sort((a, b) => {
      if (b.votes !== a.votes) return b.votes - a.votes;
      return new Date(b.submittedAt).getTime() - new Date(a.submittedAt).getTime();
    });

    // Apply limit
    suggestions = suggestions.slice(0, limit);

    // Remove sensitive fields before returning
    const publicSuggestions = suggestions.map(s => ({
      id: s.id,
      title: s.title,
      description: s.description,
      category: s.category,
      status: s.status,
      votes: s.votes,
      submittedAt: s.submittedAt,
      taskId: s.taskId,
    }));

    return NextResponse.json({
      suggestions: publicSuggestions,
      total: publicSuggestions.length,
      lastUpdated: store.lastUpdated,
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
  // On Vercel demo mode, simulate successful submission
  const isVercel = process.env.VERCEL === '1' || process.env.VERCEL_ENV;

  if (isVercel) {
    const body = await request.json();
    return NextResponse.json({
      success: true,
      demo: true,
      message: 'Thanks for your suggestion! (Demo mode - submissions are simulated)',
      suggestion: {
        id: `demo_${Date.now()}`,
        title: body.title?.slice(0, 100) || 'Untitled',
        description: body.description?.slice(0, 500) || '',
        category: body.category || 'feature',
        status: 'pending',
        votes: 0,
        submittedAt: new Date().toISOString(),
      },
    });
  }

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

    // Create suggestion
    const suggestion: Suggestion = {
      id: generateId(),
      title: sanitizeInput(title, 100),
      description: sanitizeInput(description, 1000),
      category,
      status: 'pending',
      votes: 0,
      submittedAt: new Date().toISOString(),
      submitterIp: ip, // For rate limiting, not exposed publicly
    };

    // Save
    const store = readSuggestions();
    store.suggestions.push(suggestion);
    writeSuggestions(store);

    // Return public version
    return NextResponse.json({
      success: true,
      suggestion: {
        id: suggestion.id,
        title: suggestion.title,
        description: suggestion.description,
        category: suggestion.category,
        status: suggestion.status,
        votes: suggestion.votes,
        submittedAt: suggestion.submittedAt,
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

/**
 * PATCH /api/suggestions
 * Admin: Update suggestion status
 * Requires admin token (simple auth for now)
 */
export async function PATCH(request: NextRequest) {
  try {
    // Simple admin auth check
    const authHeader = request.headers.get('authorization');
    const adminToken = process.env.ADMIN_TOKEN || 'voice-clone-admin-2024';

    if (authHeader !== `Bearer ${adminToken}`) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const body = await request.json();
    const { id, status, adminNote, taskId } = body;

    if (!id) {
      return NextResponse.json(
        { error: 'Suggestion ID required' },
        { status: 400 }
      );
    }

    const validStatuses = ['pending', 'approved', 'in-progress', 'done', 'rejected'];
    if (status && !validStatuses.includes(status)) {
      return NextResponse.json(
        { error: 'Invalid status' },
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

    // Update fields
    if (status) suggestion.status = status;
    if (adminNote !== undefined) suggestion.adminNote = adminNote;
    if (taskId !== undefined) suggestion.taskId = taskId;

    writeSuggestions(store);

    return NextResponse.json({
      success: true,
      suggestion: {
        id: suggestion.id,
        title: suggestion.title,
        status: suggestion.status,
        taskId: suggestion.taskId,
      },
    });
  } catch (error) {
    console.error('Error updating suggestion:', error);
    return NextResponse.json(
      { error: 'Failed to update suggestion' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/suggestions
 * Admin: Delete a suggestion
 */
export async function DELETE(request: NextRequest) {
  try {
    // Simple admin auth check
    const authHeader = request.headers.get('authorization');
    const adminToken = process.env.ADMIN_TOKEN || 'voice-clone-admin-2024';

    if (authHeader !== `Bearer ${adminToken}`) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json(
        { error: 'Suggestion ID required' },
        { status: 400 }
      );
    }

    const store = readSuggestions();
    const index = store.suggestions.findIndex(s => s.id === id);

    if (index === -1) {
      return NextResponse.json(
        { error: 'Suggestion not found' },
        { status: 404 }
      );
    }

    store.suggestions.splice(index, 1);
    writeSuggestions(store);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting suggestion:', error);
    return NextResponse.json(
      { error: 'Failed to delete suggestion' },
      { status: 500 }
    );
  }
}
