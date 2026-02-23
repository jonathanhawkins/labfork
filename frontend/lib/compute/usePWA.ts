"use client";

/**
 * usePWA Hook - React hook for PWA functionality
 *
 * Provides:
 * - PWA installation status
 * - Battery monitoring
 * - Service worker registration
 * - Notification permissions
 */

import { useState, useEffect } from "react";
import {
  getPWAStatus,
  getBatteryStatus,
  monitorBattery,
  registerServiceWorker,
  requestNotificationPermission,
  type PWAStatus,
  type BatteryStatus,
} from "./pwa-utils";

export interface UsePWAOptions {
  autoRegisterServiceWorker?: boolean;
  monitorBatteryStatus?: boolean;
  lowBatteryThreshold?: number;
  onLowBattery?: () => void;
}

export interface UsePWAResult {
  pwaStatus: PWAStatus | null;
  batteryStatus: BatteryStatus | null;
  isLoading: boolean;
  requestNotifications: () => Promise<NotificationPermission>;
  registerSW: () => Promise<void>;
}

export function usePWA(options: UsePWAOptions = {}): UsePWAResult {
  const {
    autoRegisterServiceWorker = true,
    monitorBatteryStatus = true,
    lowBatteryThreshold = 0.2,
    onLowBattery,
  } = options;

  const [pwaStatus, setPWAStatus] = useState<PWAStatus | null>(null);
  const [batteryStatus, setBatteryStatus] = useState<BatteryStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize PWA features
  useEffect(() => {
    async function initialize() {
      try {
        // Get PWA status
        const status = await getPWAStatus();
        setPWAStatus(status);

        // Auto-register service worker
        if (autoRegisterServiceWorker && !status.serviceWorkerRegistered) {
          await registerServiceWorker();
          // Update status after registration
          const updatedStatus = await getPWAStatus();
          setPWAStatus(updatedStatus);
        }

        // Get initial battery status
        if (monitorBatteryStatus && status.batterySupported) {
          const battery = await getBatteryStatus();
          if (battery) {
            setBatteryStatus(battery);
          }
        }
      } catch (error) {
        console.error('[usePWA] Initialization failed:', error);
      } finally {
        setIsLoading(false);
      }
    }

    initialize();
  }, [autoRegisterServiceWorker, monitorBatteryStatus]);

  // Monitor battery changes
  useEffect(() => {
    if (!monitorBatteryStatus || !pwaStatus?.batterySupported) {
      return;
    }

    const cleanup = monitorBattery(
      (status) => {
        setBatteryStatus(status);
      },
      onLowBattery,
      lowBatteryThreshold
    );

    return cleanup;
  }, [monitorBatteryStatus, pwaStatus?.batterySupported, onLowBattery, lowBatteryThreshold]);

  // Request notification permission
  const requestNotifications = async (): Promise<NotificationPermission> => {
    const permission = await requestNotificationPermission();

    // Update PWA status
    const updatedStatus = await getPWAStatus();
    setPWAStatus(updatedStatus);

    return permission;
  };

  // Register service worker manually
  const registerSW = async (): Promise<void> => {
    await registerServiceWorker();

    // Update PWA status
    const updatedStatus = await getPWAStatus();
    setPWAStatus(updatedStatus);
  };

  return {
    pwaStatus,
    batteryStatus,
    isLoading,
    requestNotifications,
    registerSW,
  };
}
