"use client";

/**
 * InstallPrompt Component - PWA Installation Prompt for Mobile Contributors
 *
 * Features:
 * - Detects if app can be installed
 * - Shows install button on mobile devices
 * - Handles beforeinstallprompt event
 * - Mobile-first design with touch targets >= 44px
 * - Auto-dismisses on install or after user declines
 */

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, X, Smartphone, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: 'accepted' | 'dismissed';
    platform: string;
  }>;
  prompt(): Promise<void>;
}

interface InstallPromptProps {
  className?: string;
  variant?: 'banner' | 'card';
  autoShow?: boolean;
  onInstall?: () => void;
  onDismiss?: () => void;
}

export function InstallPrompt({
  className,
  variant = 'banner',
  autoShow = true,
  onInstall,
  onDismiss,
}: InstallPromptProps) {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [showPrompt, setShowPrompt] = useState(false);
  const [isInstalling, setIsInstalling] = useState(false);
  const [isIOS, setIsIOS] = useState(false);

  useEffect(() => {
    // Check if iOS
    const iOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !(window as any).MSStream;
    setIsIOS(iOS);

    // Check if already installed
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches ||
                         (window.navigator as any).standalone === true;

    if (isStandalone) {
      // Already installed, don't show prompt
      return;
    }

    // Check if user previously dismissed
    const dismissed = localStorage.getItem('pwa-install-dismissed');
    if (dismissed && Date.now() - parseInt(dismissed) < 7 * 24 * 60 * 60 * 1000) {
      // Don't show again for 7 days
      return;
    }

    // Listen for beforeinstallprompt event
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      const promptEvent = e as BeforeInstallPromptEvent;
      setDeferredPrompt(promptEvent);

      // Store on window for canInstallPWA check
      (window as any).__pwaInstallPrompt = promptEvent;

      if (autoShow) {
        setShowPrompt(true);
      }
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    // Listen for app installed event
    const handleAppInstalled = () => {
      console.log('[PWA] App installed');
      setDeferredPrompt(null);
      setShowPrompt(false);
      onInstall?.();
    };

    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, [autoShow, onInstall]);

  const handleInstallClick = async () => {
    if (!deferredPrompt) {
      console.warn('[PWA] No install prompt available');
      return;
    }

    setIsInstalling(true);

    try {
      // Show the install prompt
      await deferredPrompt.prompt();

      // Wait for user choice
      const { outcome } = await deferredPrompt.userChoice;

      console.log('[PWA] User choice:', outcome);

      if (outcome === 'accepted') {
        console.log('[PWA] User accepted install');
        onInstall?.();
      }

      // Clean up
      setDeferredPrompt(null);
      setShowPrompt(false);
      (window as any).__pwaInstallPrompt = undefined;
    } catch (error) {
      console.error('[PWA] Install prompt failed:', error);
    } finally {
      setIsInstalling(false);
    }
  };

  const handleDismiss = () => {
    setShowPrompt(false);
    localStorage.setItem('pwa-install-dismissed', Date.now().toString());
    onDismiss?.();
  };

  // Don't render if no prompt available and not iOS
  if (!deferredPrompt && !isIOS) {
    return null;
  }

  // Don't render if not showing
  if (!showPrompt) {
    return null;
  }

  if (variant === 'banner') {
    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          className={cn(
            "fixed bottom-4 left-4 right-4 z-50 md:left-auto md:right-4 md:max-w-md",
            className
          )}
        >
          <div className="bg-background-card border border-border rounded-xl p-4 shadow-lg">
            <div className="flex items-start gap-3">
              <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
                <Download className="w-5 h-5 text-blue-400" />
              </div>

              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-foreground-bright mb-1">
                  Install LabFork
                </h3>
                <p className="text-xs text-foreground-muted mb-3">
                  {isIOS
                    ? 'Add to Home Screen for full PWA experience'
                    : 'Install the app for offline access and background compute'}
                </p>

                {isIOS ? (
                  <div className="space-y-2 text-xs text-foreground-muted">
                    <p className="flex items-center gap-2">
                      <span>1. Tap the share button</span>
                      <Smartphone className="w-4 h-4" />
                    </p>
                    <p>2. Select "Add to Home Screen"</p>
                  </div>
                ) : (
                  <button
                    onClick={handleInstallClick}
                    disabled={isInstalling}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-foreground-bright text-background hover:bg-white active:bg-foreground-muted transition-colors font-medium min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isInstalling ? (
                      <>
                        <Zap className="w-4 h-4 animate-pulse" />
                        Installing...
                      </>
                    ) : (
                      <>
                        <Download className="w-4 h-4" />
                        Install Now
                      </>
                    )}
                  </button>
                )}
              </div>

              <button
                onClick={handleDismiss}
                className="w-11 h-11 flex items-center justify-center rounded-lg hover:bg-foreground-muted/10 active:bg-foreground-muted/20 transition-colors flex-shrink-0"
                aria-label="Dismiss install prompt"
              >
                <X className="w-5 h-5 text-foreground-muted" />
              </button>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    );
  }

  // Card variant
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.3, ease: "easeOut" }}
        className={cn("bg-background-card border border-border rounded-xl p-6", className)}
      >
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
            <Download className="w-6 h-6 text-blue-400" />
          </div>

          <div className="flex-1 min-w-0">
            <h2 className="text-lg font-medium text-foreground-bright mb-1">
              Install LabFork
            </h2>
            <p className="text-sm text-foreground-muted mb-4">
              {isIOS
                ? 'Add to your Home Screen for the best mobile experience'
                : 'Install the progressive web app for offline access, push notifications, and background compute'}
            </p>

            {isIOS ? (
              <div className="space-y-2 p-3 rounded-lg bg-foreground-muted/10 border border-border">
                <p className="text-sm font-medium text-foreground-bright mb-2">
                  Installation Steps:
                </p>
                <div className="space-y-2 text-sm text-foreground-muted">
                  <p className="flex items-center gap-2">
                    <Smartphone className="w-4 h-4" />
                    <span>1. Tap the share button in Safari</span>
                  </p>
                  <p className="pl-6">2. Scroll and select "Add to Home Screen"</p>
                  <p className="pl-6">3. Tap "Add" to install</p>
                </div>
              </div>
            ) : (
              <div className="flex gap-3">
                <button
                  onClick={handleInstallClick}
                  disabled={isInstalling}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-foreground-bright text-background hover:bg-white active:bg-foreground-muted transition-colors font-medium min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isInstalling ? (
                    <>
                      <Zap className="w-4 h-4 animate-pulse" />
                      Installing...
                    </>
                  ) : (
                    <>
                      <Download className="w-4 h-4" />
                      Install
                    </>
                  )}
                </button>

                <button
                  onClick={handleDismiss}
                  className="px-4 py-2.5 rounded-lg border border-border hover:bg-foreground-muted/10 active:bg-foreground-muted/20 transition-colors text-foreground-muted text-sm font-medium min-h-[44px]"
                >
                  Not Now
                </button>
              </div>
            )}
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
