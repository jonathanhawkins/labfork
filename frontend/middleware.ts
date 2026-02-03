/**
 * Middleware for authentication and route protection
 *
 * Uses Clerk middleware which gracefully handles unconfigured state.
 * When Clerk keys are not set, it allows all requests through.
 */

import { clerkMiddleware } from "@clerk/nextjs/server";

// Clerk middleware automatically handles auth state
// Works in development without configuration
export default clerkMiddleware();

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    // Always run for API routes
    "/(api|trpc)(.*)",
  ],
};
