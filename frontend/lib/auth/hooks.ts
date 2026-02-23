"use client";

/**
 * Unified auth hook that works with or without Clerk.
 *
 * Uses a React context provided by AuthProvider in providers.tsx.
 * Pages should use useAuth() instead of Clerk's useUser() directly.
 */

import { createContext, useContext } from "react";
import type { User } from "./mock-user";

export interface AuthContextValue {
  user: User | null;
  isLoaded: boolean;
}

export const AuthContext = createContext<AuthContextValue>({
  user: null,
  isLoaded: true,
});

/**
 * Get current authenticated user.
 * Works in both Clerk (production) and mock (development) modes.
 */
export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}
