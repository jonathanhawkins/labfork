/**
 * Client-side Authentication
 *
 * This file contains client-only authentication logic.
 * Import this from client components ("use client").
 */

import { getCurrentUser, DEFAULT_USER } from "./mock-user";
import type { User } from "./mock-user";

/**
 * Get current user for client components
 *
 * This function works with both Clerk and mock authentication:
 * - When Clerk is configured, use the `useAuth()` hook from @clerk/nextjs
 * - When Clerk is not configured, falls back to mock user
 *
 * @example
 * ```tsx
 * import { useAuth } from "@clerk/nextjs";
 * import { getClientUser } from "@/lib/auth/client";
 *
 * function MyComponent() {
 *   const { user: clerkUser, isLoaded, isSignedIn } = useAuth();
 *   const user = getClientUser(clerkUser);
 *
 *   if (!isLoaded) return <Loading />;
 *   // user will be the Clerk user or mock user
 * }
 * ```
 */
export function getClientUser(clerkUser?: any): User | null {
  // If we have a Clerk user, convert it to our User interface
  if (clerkUser) {
    return {
      id: clerkUser.id,
      username:
        clerkUser.username ||
        clerkUser.emailAddresses?.[0]?.emailAddress.split("@")[0] ||
        "user",
      displayName:
        `${clerkUser.firstName || ""} ${clerkUser.lastName || ""}`.trim() ||
        clerkUser.username ||
        "User",
      avatar: clerkUser.imageUrl,
      email: clerkUser.emailAddresses?.[0]?.emailAddress,
      createdAt: clerkUser.createdAt
        ? new Date(clerkUser.createdAt).toISOString()
        : new Date().toISOString(),
    };
  }

  // Check if Clerk is configured (client-side check)
  const hasClerkKeys =
    typeof window !== "undefined" &&
    process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

  // If Clerk is configured but no user, return null (not authenticated)
  if (hasClerkKeys && process.env.NODE_ENV === "production") {
    return null;
  }

  // Fall back to mock user in development or when Clerk is not configured
  if (process.env.NODE_ENV === "development" || !hasClerkKeys) {
    return getCurrentUser();
  }

  return null;
}

/**
 * Utility to convert Clerk user to our User interface
 * Used in client components that receive Clerk user data
 */
export function convertClerkUser(clerkUser: any): User | null {
  if (!clerkUser) return null;

  return {
    id: clerkUser.id,
    username:
      clerkUser.username ||
      clerkUser.emailAddresses?.[0]?.emailAddress.split("@")[0] ||
      "user",
    displayName:
      `${clerkUser.firstName || ""} ${clerkUser.lastName || ""}`.trim() ||
      clerkUser.username ||
      "User",
    avatar: clerkUser.imageUrl,
    email: clerkUser.emailAddresses?.[0]?.emailAddress,
    createdAt: clerkUser.createdAt
      ? new Date(clerkUser.createdAt).toISOString()
      : new Date().toISOString(),
  };
}
