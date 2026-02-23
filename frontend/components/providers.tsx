"use client";

import { ThemeProvider } from "next-themes";
import { AuthContext } from "@/lib/auth/hooks";
import { getClientUser } from "@/lib/auth/client";

const clerkPubKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

/**
 * When Clerk IS configured, this inner component safely calls useUser
 * because it's guaranteed to be inside ClerkProvider.
 */
function ClerkAuthProvider({ children }: { children: React.ReactNode }) {
  // eslint-disable-next-line
  const { useUser } = require("@clerk/nextjs");
  const { user: clerkUser, isLoaded } = useUser();
  const user = getClientUser(clerkUser);

  return (
    <AuthContext.Provider value={{ user, isLoaded }}>
      {children}
    </AuthContext.Provider>
  );
}

/**
 * When Clerk is NOT configured, use mock user (development mode).
 */
function MockAuthProvider({ children }: { children: React.ReactNode }) {
  const user = getClientUser(null);

  return (
    <AuthContext.Provider value={{ user, isLoaded: true }}>
      {children}
    </AuthContext.Provider>
  );
}

export function Providers({ children }: { children: React.ReactNode }) {
  const themed = (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      {children}
    </ThemeProvider>
  );

  if (clerkPubKey) {
    // eslint-disable-next-line
    const { ClerkProvider } = require("@clerk/nextjs");

    return (
      <ClerkProvider
        publishableKey={clerkPubKey}
        appearance={{
          variables: { colorPrimary: "#3b82f6" },
        }}
      >
        <ClerkAuthProvider>{themed}</ClerkAuthProvider>
      </ClerkProvider>
    );
  }

  return <MockAuthProvider>{themed}</MockAuthProvider>;
}
