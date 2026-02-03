"use client";

/**
 * Service Worker Registration Component
 *
 * Handles automatic service worker registration for PWA support.
 * Critical for enabling offline functionality in low-connectivity areas.
 *
 * Features:
 * - Auto-registers service worker on mount
 * - Handles SW updates gracefully
 * - Shows update notification when new version available
 * - Tracks online/offline status
 */

import { useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { RefreshCw, Wifi, WifiOff, X } from "lucide-react";

export function ServiceWorkerRegistration() {
  const [registration, setRegistration] = useState<globalThis.ServiceWorkerRegistration | null>(null);
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [isOffline, setIsOffline] = useState(false);
  const [showOfflineBanner, setShowOfflineBanner] = useState(false);

  // Register service worker
  useEffect(() => {
    if (typeof window === "undefined" || !("serviceWorker" in navigator)) {
      return;
    }

    // Check initial online status
    setIsOffline(!navigator.onLine);

    const registerSW = async () => {
      try {
        const reg = await navigator.serviceWorker.register("/sw.js", {
          scope: "/",
        });

        setRegistration(reg);

        // Check for updates
        reg.addEventListener("updatefound", () => {
          const newWorker = reg.installing;
          if (newWorker) {
            newWorker.addEventListener("statechange", () => {
              if (
                newWorker.state === "installed" &&
                navigator.serviceWorker.controller
              ) {
                // New version available
                setUpdateAvailable(true);
              }
            });
          }
        });

        // Handle controller change (new SW activated)
        navigator.serviceWorker.addEventListener("controllerchange", () => {
          // New service worker took over
        });
      } catch (error) {
        // Service worker registration failed - silently continue
        // Error won't prevent app from working, just no PWA features
      }
    };

    registerSW();

    // Listen for online/offline events
    const handleOnline = () => {
      setIsOffline(false);
      setShowOfflineBanner(false);
    };

    const handleOffline = () => {
      setIsOffline(true);
      setShowOfflineBanner(true);
    };

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  // Apply update
  const applyUpdate = useCallback(() => {
    if (registration?.waiting) {
      // Tell waiting SW to skip waiting and become active
      registration.waiting.postMessage({ type: "SKIP_WAITING" });
      // Reload to get new version
      window.location.reload();
    }
  }, [registration]);

  // Dismiss offline banner
  const dismissOfflineBanner = useCallback(() => {
    setShowOfflineBanner(false);
  }, []);

  return (
    <>
      {/* Update Available Banner */}
      {updateAvailable && (
        <div
          className={cn(
            "fixed bottom-4 left-4 right-4 z-50",
            "sm:left-auto sm:right-4 sm:max-w-sm",
            "bg-blue-600 text-white rounded-lg shadow-lg",
            "p-4 flex items-center gap-3",
            "animate-in slide-in-from-bottom-4"
          )}
        >
          <RefreshCw className="w-5 h-5 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium">Update available</p>
            <p className="text-xs opacity-90">
              Refresh to get the latest features
            </p>
          </div>
          <button
            onClick={applyUpdate}
            className={cn(
              "px-3 py-1.5 rounded-md text-sm font-medium",
              "bg-white/20 hover:bg-white/30 transition-colors",
              "min-h-[44px] min-w-[44px]" // Touch-friendly
            )}
          >
            Update
          </button>
        </div>
      )}

      {/* Offline Banner */}
      {showOfflineBanner && (
        <div
          className={cn(
            "fixed top-0 left-0 right-0 z-50",
            "bg-amber-600 text-white",
            "px-4 py-2 flex items-center justify-center gap-2",
            "text-sm",
            "animate-in slide-in-from-top-2"
          )}
        >
          <WifiOff className="w-4 h-4" />
          <span>You're offline. Some features may be limited.</span>
          <button
            onClick={dismissOfflineBanner}
            className="ml-2 p-1 rounded hover:bg-white/20 min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Dismiss"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </>
  );
}

export default ServiceWorkerRegistration;
