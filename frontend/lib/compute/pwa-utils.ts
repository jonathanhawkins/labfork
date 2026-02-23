/**
 * PWA Utilities for Mobile Compute Contributors
 *
 * Provides utilities for:
 * - Detecting if running as installed PWA
 * - Managing service worker registration
 * - Requesting notification permissions
 * - Battery status monitoring
 * - Background task coordination
 */

export interface BatteryStatus {
  level: number; // 0-1
  charging: boolean;
  chargingTime: number | null;
  dischargingTime: number | null;
}

export interface PWAStatus {
  isInstalled: boolean;
  isStandalone: boolean;
  canInstall: boolean;
  serviceWorkerRegistered: boolean;
  notificationsEnabled: boolean;
  batterySupported: boolean;
}

/**
 * Check if app is running as installed PWA
 */
export function isPWAInstalled(): boolean {
  if (typeof window === 'undefined') return false;

  // Check if running in standalone mode
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches;

  // iOS Safari check
  const isIOSStandalone = (window.navigator as any).standalone === true;

  return isStandalone || isIOSStandalone;
}

/**
 * Check if app can be installed (beforeinstallprompt available)
 */
export function canInstallPWA(): boolean {
  if (typeof window === 'undefined') return false;

  // This will be set by the beforeinstallprompt event
  return (window as any).__pwaInstallPrompt !== undefined;
}

/**
 * Get comprehensive PWA status
 */
export async function getPWAStatus(): Promise<PWAStatus> {
  if (typeof window === 'undefined') {
    return {
      isInstalled: false,
      isStandalone: false,
      canInstall: false,
      serviceWorkerRegistered: false,
      notificationsEnabled: false,
      batterySupported: false,
    };
  }

  const isInstalled = isPWAInstalled();
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
  const canInstall = canInstallPWA();
  const serviceWorkerRegistered = 'serviceWorker' in navigator && !!(await navigator.serviceWorker.getRegistration());
  const notificationsEnabled = 'Notification' in window && Notification.permission === 'granted';
  const batterySupported = 'getBattery' in navigator;

  return {
    isInstalled,
    isStandalone,
    canInstall,
    serviceWorkerRegistered,
    notificationsEnabled,
    batterySupported,
  };
}

/**
 * Register service worker
 */
export async function registerServiceWorker(): Promise<ServiceWorkerRegistration | null> {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    console.warn('[PWA] Service workers not supported');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });

    console.log('[PWA] Service worker registered:', registration.scope);

    // Handle updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      if (newWorker) {
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            console.log('[PWA] New service worker available, reload to update');
            // Optionally notify user about update
          }
        });
      }
    });

    return registration;
  } catch (error) {
    console.error('[PWA] Service worker registration failed:', error);
    return null;
  }
}

/**
 * Unregister service worker
 */
export async function unregisterServiceWorker(): Promise<boolean> {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.getRegistration();
    if (registration) {
      const success = await registration.unregister();
      console.log('[PWA] Service worker unregistered:', success);
      return success;
    }
    return false;
  } catch (error) {
    console.error('[PWA] Service worker unregistration failed:', error);
    return false;
  }
}

/**
 * Request notification permission
 */
export async function requestNotificationPermission(): Promise<NotificationPermission> {
  if (typeof window === 'undefined' || !('Notification' in window)) {
    console.warn('[PWA] Notifications not supported');
    return 'denied';
  }

  try {
    const permission = await Notification.requestPermission();
    console.log('[PWA] Notification permission:', permission);
    return permission;
  } catch (error) {
    console.error('[PWA] Notification permission request failed:', error);
    return 'denied';
  }
}

/**
 * Get battery status
 */
export async function getBatteryStatus(): Promise<BatteryStatus | null> {
  if (typeof window === 'undefined' || !('getBattery' in navigator)) {
    console.warn('[PWA] Battery API not supported');
    return null;
  }

  try {
    const battery = await (navigator as any).getBattery();

    return {
      level: battery.level,
      charging: battery.charging,
      chargingTime: battery.chargingTime === Infinity ? null : battery.chargingTime,
      dischargingTime: battery.dischargingTime === Infinity ? null : battery.dischargingTime,
    };
  } catch (error) {
    console.error('[PWA] Failed to get battery status:', error);
    return null;
  }
}

/**
 * Monitor battery status changes
 */
export function monitorBattery(
  onUpdate: (status: BatteryStatus) => void,
  onLowBattery?: () => void,
  lowBatteryThreshold: number = 0.2
): () => void {
  if (typeof window === 'undefined' || !('getBattery' in navigator)) {
    console.warn('[PWA] Battery API not supported');
    return () => {};
  }

  let battery: any = null;

  const updateStatus = () => {
    if (battery) {
      const status: BatteryStatus = {
        level: battery.level,
        charging: battery.charging,
        chargingTime: battery.chargingTime === Infinity ? null : battery.chargingTime,
        dischargingTime: battery.dischargingTime === Infinity ? null : battery.dischargingTime,
      };

      onUpdate(status);

      // Check for low battery
      if (!battery.charging && battery.level <= lowBatteryThreshold && onLowBattery) {
        onLowBattery();
      }
    }
  };

  // Initialize battery monitoring
  (navigator as any).getBattery().then((bat: any) => {
    battery = bat;

    // Set up event listeners
    battery.addEventListener('levelchange', updateStatus);
    battery.addEventListener('chargingchange', updateStatus);
    battery.addEventListener('chargingtimechange', updateStatus);
    battery.addEventListener('dischargingtimechange', updateStatus);

    // Initial update
    updateStatus();
  }).catch((error: Error) => {
    console.error('[PWA] Battery monitoring failed:', error);
  });

  // Return cleanup function
  return () => {
    if (battery) {
      battery.removeEventListener('levelchange', updateStatus);
      battery.removeEventListener('chargingchange', updateStatus);
      battery.removeEventListener('chargingtimechange', updateStatus);
      battery.removeEventListener('dischargingtimechange', updateStatus);
    }
  };
}

/**
 * Check if device should contribute compute (battery-aware)
 */
export async function shouldContributeCompute(
  requireCharging: boolean = false,
  minBatteryLevel: number = 0.3
): Promise<boolean> {
  const battery = await getBatteryStatus();

  if (!battery) {
    // If battery API not available, allow contribution
    return true;
  }

  // Don't contribute if battery is low
  if (battery.level < minBatteryLevel) {
    return false;
  }

  // Check charging requirement
  if (requireCharging && !battery.charging) {
    return false;
  }

  return true;
}

/**
 * Subscribe to push notifications
 */
export async function subscribeToPushNotifications(
  vapidPublicKey: string
): Promise<PushSubscription | null> {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    console.warn('[PWA] Service workers not supported');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.ready;

    // Check if already subscribed
    let subscription = await registration.pushManager.getSubscription();

    if (!subscription) {
      // Subscribe to push notifications
      subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(vapidPublicKey),
      });

      console.log('[PWA] Subscribed to push notifications');
    }

    return subscription;
  } catch (error) {
    console.error('[PWA] Push subscription failed:', error);
    return null;
  }
}

/**
 * Unsubscribe from push notifications
 */
export async function unsubscribeFromPushNotifications(): Promise<boolean> {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();

    if (subscription) {
      const success = await subscription.unsubscribe();
      console.log('[PWA] Unsubscribed from push notifications:', success);
      return success;
    }

    return false;
  } catch (error) {
    console.error('[PWA] Push unsubscription failed:', error);
    return false;
  }
}

/**
 * Queue task for background sync
 */
export async function queueTaskForSync(task: any): Promise<void> {
  if (typeof window === 'undefined') return;

  try {
    // Store task in IndexedDB
    const db = await openDatabase();
    await addTask(db, task);

    // Register sync if service worker available
    if ('serviceWorker' in navigator && 'sync' in (navigator.serviceWorker as any)) {
      const registration = await navigator.serviceWorker.ready;
      await (registration as any).sync.register('sync-compute-tasks');
      console.log('[PWA] Task queued for background sync');
    }
  } catch (error) {
    console.error('[PWA] Failed to queue task for sync:', error);
  }
}

// Helper functions

function urlBase64ToUint8Array(base64String: string): Uint8Array<ArrayBuffer> {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');

  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray as Uint8Array<ArrayBuffer>;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('labfork-tasks', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains('pending')) {
        db.createObjectStore('pending', { keyPath: 'id' });
      }
    };
  });
}

function addTask(db: IDBDatabase, task: any): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['pending'], 'readwrite');
    const store = transaction.objectStore('pending');
    const request = store.add({ ...task, id: task.id || Date.now() });

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}
