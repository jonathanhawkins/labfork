"use client";

/**
 * Error Boundary for Contribute Page
 *
 * Catches and handles errors in the /contribute route.
 * Provides user-friendly error UI with recovery options.
 */

import { useEffect } from "react";
import { motion } from "framer-motion";
import { AlertCircle, RefreshCw, Home } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ContributeError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log error to console in development
    if (process.env.NODE_ENV === "development") {
      console.error("Contribute page error:", error);
    }
  }, [error]);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className="max-w-md w-full"
      >
        <div className="bg-card border border-destructive/20 rounded-xl p-6">
          {/* Error Icon */}
          <div className="w-12 h-12 rounded-lg bg-destructive/10 flex items-center justify-center mb-4">
            <AlertCircle className="w-6 h-6 text-destructive" />
          </div>

          {/* Error Message */}
          <h1 className="text-lg font-semibold text-foreground mb-2">
            Failed to Load Contribute Page
          </h1>
          <p className="text-sm text-muted-foreground mb-6">
            {error.message ||
              "An unexpected error occurred while loading the compute contribution page. Please try again."}
          </p>

          {/* Error Details (Development Only) */}
          {process.env.NODE_ENV === "development" && (
            <details className="mb-6">
              <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground transition-colors mb-2">
                View error details
              </summary>
              <pre className="text-xs bg-muted/50 rounded-lg p-3 overflow-x-auto border border-border">
                {error.stack || error.message}
              </pre>
              {error.digest && (
                <p className="text-xs text-muted-foreground mt-2">
                  Error ID: {error.digest}
                </p>
              )}
            </details>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3">
            <Button
              onClick={reset}
              className="flex-1 min-h-[44px]"
              size="default"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Try Again
            </Button>

            <Button
              onClick={() => (window.location.href = "/")}
              variant="outline"
              className="flex-1 min-h-[44px]"
              size="default"
            >
              <Home className="w-4 h-4 mr-2" />
              Go Home
            </Button>
          </div>

          {/* Help Text */}
          <p className="text-xs text-muted-foreground mt-4 text-center">
            If this problem persists, please{" "}
            <a
              href="https://github.com/jonathanhawkins/labfork/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              report an issue
            </a>
            .
          </p>
        </div>
      </motion.div>
    </div>
  );
}
