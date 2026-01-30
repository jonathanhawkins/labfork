"use client";

/**
 * Compute Error Boundary Component
 *
 * Provides user-friendly error handling for compute pages.
 * Displays error message with recovery options.
 */

import { Component, ReactNode } from "react";
import { motion } from "framer-motion";
import { AlertCircle, RefreshCw, Bug } from "lucide-react";
import { Button } from "@/components/ui/button";

/**
 * Props for ComputeErrorBoundary
 */
export interface ComputeErrorBoundaryProps {
  /** Child components to render */
  children: ReactNode;
  /** Optional fallback component */
  fallback?: ReactNode;
  /** Optional error callback */
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  /** Show report issue link */
  showReportLink?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * State for error boundary
 */
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

/**
 * Compute Error Boundary Component
 *
 * React error boundary that catches errors in compute components
 * and displays a user-friendly error UI with recovery options.
 */
export class ComputeErrorBoundary extends Component<
  ComputeErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ComputeErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({
      error,
      errorInfo,
    });

    // Call optional error callback
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error to console in development
    if (process.env.NODE_ENV === "development") {
      console.error("ComputeErrorBoundary caught error:", error, errorInfo);
    }
  }

  /**
   * Reset error state and try again
   */
  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  /**
   * Report issue to GitHub
   */
  handleReportIssue = () => {
    const { error } = this.state;
    const title = encodeURIComponent(
      `Error: ${error?.message || "Unknown error in compute component"}`
    );
    const body = encodeURIComponent(
      `**Error Message:**\n${error?.message || "Unknown error"}\n\n**Stack Trace:**\n\`\`\`\n${error?.stack || "No stack trace available"}\n\`\`\`\n\n**Browser:**\n${navigator.userAgent}\n\n**URL:**\n${window.location.href}`
    );
    const issueUrl = `https://github.com/jonathanhawkins/labfork/issues/new?title=${title}&body=${body}`;
    window.open(issueUrl, "_blank", "noopener,noreferrer");
  };

  render() {
    const { hasError, error } = this.state;
    const { children, fallback, showReportLink = true, className } = this.props;

    if (hasError) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Default error UI
      return (
        <div className={className}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="min-h-[400px] flex items-center justify-center p-4"
          >
            <div className="max-w-md w-full">
              <div className="bg-card border border-destructive/20 rounded-xl p-6">
                {/* Error Icon */}
                <div className="w-12 h-12 rounded-lg bg-destructive/10 flex items-center justify-center mb-4">
                  <AlertCircle className="w-6 h-6 text-destructive" />
                </div>

                {/* Error Message */}
                <h2 className="text-lg font-semibold text-foreground mb-2">
                  Something went wrong
                </h2>
                <p className="text-sm text-muted-foreground mb-6">
                  {error?.message ||
                    "An unexpected error occurred while loading this page. Please try again."}
                </p>

                {/* Error Details (Development Only) */}
                {process.env.NODE_ENV === "development" && error?.stack && (
                  <details className="mb-6">
                    <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground transition-colors mb-2">
                      View error details
                    </summary>
                    <pre className="text-xs bg-muted/50 rounded-lg p-3 overflow-x-auto border border-border">
                      {error.stack}
                    </pre>
                  </details>
                )}

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-3">
                  <Button
                    onClick={this.handleReset}
                    className="flex-1 min-h-[44px]"
                    size="default"
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Try Again
                  </Button>

                  {showReportLink && (
                    <Button
                      onClick={this.handleReportIssue}
                      variant="outline"
                      className="flex-1 min-h-[44px]"
                      size="default"
                    >
                      <Bug className="w-4 h-4 mr-2" />
                      Report Issue
                    </Button>
                  )}
                </div>

                {/* Help Text */}
                <p className="text-xs text-muted-foreground mt-4 text-center">
                  If this problem persists, please contact support or{" "}
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
            </div>
          </motion.div>
        </div>
      );
    }

    return children;
  }
}

/**
 * Functional wrapper for error boundary
 * Use this as a convenient wrapper for functional components
 */
export function withComputeErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ComputeErrorBoundaryProps, "children">
) {
  return function WithErrorBoundary(props: P) {
    return (
      <ComputeErrorBoundary {...errorBoundaryProps}>
        <Component {...props} />
      </ComputeErrorBoundary>
    );
  };
}
