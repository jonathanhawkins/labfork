/**
 * Supabase Client Configuration
 *
 * Creates and exports Supabase client instances for browser and server contexts.
 */

import { createClient } from "@supabase/supabase-js";
import type { Database } from "./types";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";

/**
 * Check if Supabase is properly configured (not placeholder values)
 */
function isValidConfiguration(url: string, key: string): boolean {
  // Check if values exist and are not placeholder strings
  if (!url || !key) return false;
  if (url.includes("your_supabase") || url.includes("placeholder")) return false;
  if (key.includes("your_supabase") || key.includes("placeholder")) return false;
  return true;
}

// Validate configuration
const isConfigured = isValidConfiguration(supabaseUrl, supabaseAnonKey);
if (!isConfigured && process.env.NODE_ENV === "production") {
  console.warn(
    "Supabase not configured. Credit system features will be unavailable. " +
    "Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY to enable."
  );
}

/**
 * Export configuration status for API routes
 */
export const isSupabaseConfigured = isConfigured;

/**
 * Browser-side Supabase client
 * Use this in client components and API routes
 * Returns null if Supabase is not configured
 */
let supabaseInstance: ReturnType<typeof createClient<Database>> | null = null;
if (isConfigured) {
  try {
    supabaseInstance = createClient<Database>(supabaseUrl, supabaseAnonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
      },
    });
  } catch (error) {
    console.error("Failed to create Supabase client:", error);
  }
}
export const supabase = supabaseInstance;

/**
 * Get Supabase client with custom configuration
 * Returns null if Supabase is not configured
 */
export function createSupabaseClient() {
  if (!isConfigured) return null;
  try {
    return createClient<Database>(supabaseUrl, supabaseAnonKey);
  } catch (error) {
    console.error("Failed to create Supabase client:", error);
    return null;
  }
}

/**
 * Server-side Supabase client
 * Use this in server components and server actions
 * Returns null if service role key is not configured
 */
export function createServerClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  // Check for valid configuration (not placeholders)
  if (!isValidConfiguration(supabaseUrl || "", supabaseServiceKey || "")) {
    if (process.env.NODE_ENV === "development") {
      // Silent in development to avoid log spam
      return null;
    }
    console.warn(
      "Supabase server credentials not configured. Server-side credit operations will be unavailable."
    );
    return null;
  }

  try {
    return createClient<Database>(supabaseUrl!, supabaseServiceKey!, {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    });
  } catch (error) {
    console.error("Failed to create Supabase server client:", error);
    return null;
  }
}
