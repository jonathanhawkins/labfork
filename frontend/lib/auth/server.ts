/**
 * Server-side Authentication
 *
 * This file contains server-only authentication logic using Clerk.
 * Import this ONLY from server components and API routes.
 */

import "server-only";
import type { User } from "./mock-user";
import { DEFAULT_USER } from "./mock-user";

/**
 * Get current user from server-side (API routes and Server Components)
 *
 * Uses Clerk authentication when available, falls back to mock user in development.
 */
export async function getServerUser(): Promise<User | null> {
  // Try Clerk authentication first
  try {
    const { currentUser } = await import("@clerk/nextjs/server");
    const clerkUser = await currentUser();

    if (clerkUser) {
      // Convert Clerk user to our User interface
      return {
        id: clerkUser.id,
        username:
          clerkUser.username ||
          clerkUser.emailAddresses[0]?.emailAddress.split("@")[0] ||
          "user",
        displayName:
          `${clerkUser.firstName || ""} ${clerkUser.lastName || ""}`.trim() ||
          clerkUser.username ||
          "User",
        avatar: clerkUser.imageUrl,
        email: clerkUser.emailAddresses[0]?.emailAddress,
        createdAt: new Date(clerkUser.createdAt).toISOString(),
      };
    }
  } catch (error) {
    // Clerk not configured or import failed - this is expected in development
    if (process.env.NODE_ENV === "development") {
      console.log("Clerk not configured, using mock user for development");
    }
  }

  // Check if we're in production and Clerk keys are set but no user found
  const hasClerkKeys =
    process.env.CLERK_SECRET_KEY ||
    process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;
  if (process.env.NODE_ENV === "production" && hasClerkKeys) {
    // In production with Clerk configured, no user means not authenticated
    return null;
  }

  // Fall back to mock user in development or when Clerk is not configured
  if (process.env.NODE_ENV === "development" || !hasClerkKeys) {
    return DEFAULT_USER;
  }

  // No authentication available
  return null;
}
