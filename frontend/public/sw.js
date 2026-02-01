/**
 * LabFork Service Worker - Progressive Web App Support
 *
 * Features:
 * - Offline caching for core assets
 * - Offline fallback page for low-connectivity areas
 * - Background sync for compute task submissions
 * - Push notifications for task assignments
 * - Battery-aware compute throttling
 *
 * Mission: Enable anyone with a phone to contribute to AI research,
 * even in areas with spotty internet connectivity.
 */

const CACHE_VERSION = 'v3';
const CACHE_NAME = `labfork-${CACHE_VERSION}`;
const RUNTIME_CACHE = `labfork-runtime-${CACHE_VERSION}`;
const OFFLINE_PAGE = '/offline.html';

// Core assets to cache on install - these work offline
const CORE_ASSETS = [
  '/',
  '/offline.html',
  '/contribute',
  '/explore',
  '/labs',
  '/watch',
  '/manifest.json',
  '/icon-192.svg',
  '/icon-512.svg',
  '/icon-maskable-192.svg',
  '/icon-maskable-512.svg',
];

// Install event - cache core assets including offline page
self.addEventListener('install', (event) => {
  console.log('[SW] Installing service worker...');

  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching core assets for offline use');
        // Cache offline page first (critical)
        return cache.add(OFFLINE_PAGE)
          .then(() => {
            // Then cache other assets (non-blocking)
            return Promise.allSettled(
              CORE_ASSETS.filter(url => url !== OFFLINE_PAGE)
                .map(url => cache.add(url).catch(err => {
                  console.warn('[SW] Failed to cache:', url, err);
                }))
            );
          });
      })
      .then(() => {
        console.log('[SW] Installation complete');
        return self.skipWaiting();
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating service worker...');

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => {
              // Delete old version caches
              return name.startsWith('labfork-') &&
                     name !== CACHE_NAME &&
                     name !== RUNTIME_CACHE;
            })
            .map((name) => {
              console.log('[SW] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        console.log('[SW] Activation complete');
        return self.clients.claim();
      })
  );
});

// Fetch event - network first with offline fallback
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }

  // NEVER intercept API calls - let them go directly to the server
  // This prevents SSE, POST requests, and streaming from being broken
  if (url.pathname.startsWith('/api/')) {
    return;
  }

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip Next.js HMR and internal routes in development
  if (url.pathname.startsWith('/_next/webpack-hmr') ||
      url.pathname.includes('hot-update')) {
    return;
  }

  // For navigation requests (HTML pages), use network-first with offline fallback
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          // Cache successful navigation for future offline use
          if (response.ok) {
            const responseClone = response.clone();
            caches.open(RUNTIME_CACHE).then((cache) => {
              cache.put(request, responseClone);
            });
          }
          return response;
        })
        .catch(async () => {
          // Network failed - try cache first
          const cachedResponse = await caches.match(request);
          if (cachedResponse) {
            return cachedResponse;
          }

          // No cache - show offline page
          const offlineResponse = await caches.match(OFFLINE_PAGE);
          if (offlineResponse) {
            return offlineResponse;
          }

          // Last resort - return a basic offline response
          return new Response(
            '<html><body><h1>Offline</h1><p>No internet connection.</p></body></html>',
            { headers: { 'Content-Type': 'text/html' } }
          );
        })
    );
    return;
  }

  // For static assets, use cache-first with network fallback
  if (url.pathname.startsWith('/_next/static/') ||
      url.pathname.match(/\.(js|css|png|jpg|jpeg|svg|gif|woff|woff2)$/)) {
    event.respondWith(
      caches.match(request)
        .then((cachedResponse) => {
          if (cachedResponse) {
            // Refresh cache in background
            fetch(request).then((response) => {
              if (response.ok) {
                caches.open(RUNTIME_CACHE).then((cache) => {
                  cache.put(request, response);
                });
              }
            }).catch(() => {});
            return cachedResponse;
          }

          // Not cached, fetch from network
          return fetch(request)
            .then((response) => {
              if (response.ok) {
                const responseClone = response.clone();
                caches.open(RUNTIME_CACHE).then((cache) => {
                  cache.put(request, responseClone);
                });
              }
              return response;
            });
        })
    );
    return;
  }

  // For other requests, try network first, then cache
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const responseClone = response.clone();
          caches.open(RUNTIME_CACHE).then((cache) => {
            cache.put(request, responseClone);
          });
        }
        return response;
      })
      .catch(async () => {
        const cachedResponse = await caches.match(request);
        return cachedResponse || new Response('Offline', { status: 503 });
      })
  );
});

// Background Sync - for compute task submissions when offline
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync triggered:', event.tag);

  if (event.tag === 'sync-compute-tasks') {
    event.waitUntil(syncComputeTasks());
  }
});

async function syncComputeTasks() {
  try {
    const db = await openDatabase();
    const pendingTasks = await getPendingTasks(db);

    console.log('[SW] Syncing', pendingTasks.length, 'pending compute tasks');

    for (const task of pendingTasks) {
      try {
        const response = await fetch('/api/compute/submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(task),
        });

        if (response.ok) {
          await removeTask(db, task.id);
          console.log('[SW] Task synced:', task.id);
        }
      } catch (error) {
        console.error('[SW] Failed to sync task:', task.id, error);
      }
    }
  } catch (error) {
    console.error('[SW] Background sync failed:', error);
  }
}

// Push Notifications - for task assignments
self.addEventListener('push', (event) => {
  console.log('[SW] Push notification received');

  let data = {
    title: 'LabFork',
    body: 'New compute task available',
    icon: '/icon-192.svg',
    badge: '/icon-192.svg',
    tag: 'compute-notification',
  };

  if (event.data) {
    try {
      const payload = event.data.json();
      data = { ...data, ...payload };
    } catch (error) {
      console.error('[SW] Failed to parse push data:', error);
    }
  }

  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      icon: data.icon,
      badge: data.badge,
      tag: data.tag,
      requireInteraction: false,
      vibrate: [200, 100, 200],
      actions: [
        { action: 'view', title: 'View' },
        { action: 'dismiss', title: 'Dismiss' }
      ]
    })
  );
});

// Notification click - open app
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification clicked:', event.action);

  event.notification.close();

  if (event.action === 'dismiss') {
    return;
  }

  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((clientList) => {
        // Focus existing window if available
        for (const client of clientList) {
          if (client.url.includes('/contribute') && 'focus' in client) {
            return client.focus();
          }
        }
        // Open new window
        if (clients.openWindow) {
          return clients.openWindow('/contribute');
        }
      })
  );
});

// Message handler - for PWA communication
self.addEventListener('message', (event) => {
  console.log('[SW] Message received:', event.data);

  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }

  if (event.data && event.data.type === 'CACHE_URLS') {
    // Allow app to request caching of specific URLs
    const urls = event.data.urls || [];
    caches.open(RUNTIME_CACHE).then((cache) => {
      urls.forEach(url => {
        cache.add(url).catch(err => {
          console.warn('[SW] Failed to cache requested URL:', url, err);
        });
      });
    });
  }

  if (event.data && event.data.type === 'GET_CACHE_STATUS') {
    // Return cache status to app
    Promise.all([
      caches.open(CACHE_NAME).then(c => c.keys()),
      caches.open(RUNTIME_CACHE).then(c => c.keys())
    ]).then(([coreKeys, runtimeKeys]) => {
      event.ports[0].postMessage({
        type: 'CACHE_STATUS',
        core: coreKeys.map(r => r.url),
        runtime: runtimeKeys.map(r => r.url),
        version: CACHE_VERSION
      });
    });
  }
});

// Periodic background sync - keep cache fresh (requires permission)
self.addEventListener('periodicsync', (event) => {
  if (event.tag === 'refresh-cache') {
    event.waitUntil(refreshCoreAssets());
  }
});

async function refreshCoreAssets() {
  const cache = await caches.open(CACHE_NAME);

  for (const url of CORE_ASSETS) {
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (response.ok) {
        await cache.put(url, response);
      }
    } catch (error) {
      console.warn('[SW] Failed to refresh:', url);
    }
  }
}

// Helper functions for IndexedDB
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('labfork-tasks', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending')) {
        db.createObjectStore('pending', { keyPath: 'id' });
      }
    };
  });
}

function getPendingTasks(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['pending'], 'readonly');
    const store = transaction.objectStore('pending');
    const request = store.getAll();

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || []);
  });
}

function removeTask(db, taskId) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['pending'], 'readwrite');
    const store = transaction.objectStore('pending');
    const request = store.delete(taskId);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}
