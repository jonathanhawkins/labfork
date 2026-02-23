/**
 * Mock User Authentication
 *
 * Simple mock user system for development.
 * Uses localStorage for persistence in the browser.
 */

import type { LabOwner } from "@/lib/labs/types";

/**
 * User interface
 */
export interface User {
  id: string;
  username: string;
  displayName: string;
  avatar?: string;
  email?: string;
  bio?: string;
  createdAt: string;
}

/**
 * Mock users for development
 */
export const MOCK_USERS: User[] = [
  {
    id: "user_1",
    username: "researcher",
    displayName: "Alex Researcher",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=researcher",
    email: "alex@research.lab",
    bio: "AI researcher focused on speech synthesis and prosody control.",
    createdAt: "2024-01-15T00:00:00Z",
  },
  {
    id: "user_2",
    username: "scientist",
    displayName: "Sam Scientist",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=scientist",
    email: "sam@science.lab",
    bio: "Machine learning scientist exploring neural network architectures.",
    createdAt: "2024-02-20T00:00:00Z",
  },
  {
    id: "user_3",
    username: "developer",
    displayName: "Dev Developer",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=developer",
    email: "dev@code.lab",
    bio: "Full-stack developer building AI-powered applications.",
    createdAt: "2024-03-10T00:00:00Z",
  },
];

/**
 * Default mock user (used when no user is selected)
 */
export const DEFAULT_USER: User = MOCK_USERS[0];

/**
 * Storage key for current user
 */
const CURRENT_USER_KEY = "lab_current_user";

/**
 * Check if we're in browser environment
 */
function isBrowser(): boolean {
  return typeof window !== "undefined";
}

/**
 * Get current user from localStorage
 */
export function getCurrentUser(): User | null {
  if (!isBrowser()) {
    return DEFAULT_USER;
  }

  try {
    const stored = localStorage.getItem(CURRENT_USER_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error("Failed to get current user:", error);
  }

  // Return default user if none set
  return DEFAULT_USER;
}

/**
 * Set current user
 */
export function setCurrentUser(user: User | null): void {
  if (!isBrowser()) return;

  try {
    if (user) {
      localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
    } else {
      localStorage.removeItem(CURRENT_USER_KEY);
    }
  } catch (error) {
    console.error("Failed to set current user:", error);
  }
}

/**
 * Get user by ID
 */
export function getUserById(id: string): User | null {
  return MOCK_USERS.find((u) => u.id === id) || null;
}

/**
 * Get user by username
 */
export function getUserByUsername(username: string): User | null {
  return MOCK_USERS.find((u) => u.username === username) || null;
}

/**
 * Get all mock users
 */
export function getAllUsers(): User[] {
  return MOCK_USERS;
}

/**
 * Convert User to LabOwner
 */
export function userToLabOwner(user: User): LabOwner {
  return {
    id: user.id,
    username: user.username,
    displayName: user.displayName,
    avatar: user.avatar,
  };
}

/**
 * Get current user as LabOwner
 */
export function getCurrentUserAsOwner(): LabOwner {
  const user = getCurrentUser() || DEFAULT_USER;
  return userToLabOwner(user);
}

/**
 * Check if user is logged in
 */
export function isLoggedIn(): boolean {
  return getCurrentUser() !== null;
}

/**
 * Switch to a different user (for development)
 */
export function switchUser(userId: string): User | null {
  const user = getUserById(userId);
  if (user) {
    setCurrentUser(user);
  }
  return user;
}

/**
 * Login as a mock user (for development)
 */
export function loginAs(username: string): User | null {
  const user = getUserByUsername(username);
  if (user) {
    setCurrentUser(user);
  }
  return user;
}

/**
 * Logout
 */
export function logout(): void {
  setCurrentUser(null);
}

/**
 * Generate avatar URL from username
 */
export function generateAvatarUrl(username: string): string {
  return `https://api.dicebear.com/7.x/avataaars/svg?seed=${encodeURIComponent(username)}`;
}

/**
 * Create a new mock user (for development)
 */
export function createMockUser(
  username: string,
  displayName: string
): User {
  const user: User = {
    id: `user_${Date.now().toString(36)}`,
    username,
    displayName,
    avatar: generateAvatarUrl(username),
    createdAt: new Date().toISOString(),
  };

  // Add to mock users (in memory only)
  MOCK_USERS.push(user);

  return user;
}

/**
 * DEPRECATED: Use getServerUser from @/lib/auth/server instead
 *
 * This function is kept for backward compatibility but will log a warning.
 * It returns the DEFAULT_USER for client-side usage.
 *
 * For server-side authentication in API routes, use:
 * import { getServerUser } from "@/lib/auth/server"
 */
export async function getServerUser(): Promise<User> {
  console.warn(
    "getServerUser from mock-user.ts is deprecated. Use getServerUser from @/lib/auth/server for API routes."
  );
  return DEFAULT_USER;
}

/**
 * Validate user has access to a resource
 */
export function hasAccess(resourceOwnerId: string, userId?: string): boolean {
  if (!userId) return false;
  return resourceOwnerId === userId;
}
