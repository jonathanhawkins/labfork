# PWA Setup for Mobile Contributors

This document describes the Progressive Web App (PWA) implementation for LabFork, enabling mobile users to install the app and contribute compute power even when backgrounded.

## Overview

LabFork implements a full-featured PWA that allows mobile contributors to:
- Install the app to their home screen
- Contribute compute power in the background
- Receive push notifications for task assignments
- Monitor battery status and auto-pause on low battery
- Work offline with cached assets
- Sync tasks when connection is restored

## Files Created

### Manifest and Icons
- `/public/manifest.json` - PWA manifest with app metadata
- `/public/icon-192.svg` - App icon (192x192)
- `/public/icon-512.svg` - App icon (512x512)
- `/public/icon-maskable-192.svg` - Maskable icon for Android (192x192)
- `/public/icon-maskable-512.svg` - Maskable icon for Android (512x512)

### Service Worker
- `/public/sw.js` - Service worker for offline caching, background sync, and push notifications

### PWA Utilities
- `/lib/compute/pwa-utils.ts` - Core PWA utility functions
  - `getPWAStatus()` - Get comprehensive PWA status
  - `registerServiceWorker()` - Register service worker
  - `getBatteryStatus()` - Get battery level and charging state
  - `monitorBattery()` - Monitor battery changes
  - `shouldContributeCompute()` - Battery-aware compute decisions
  - `requestNotificationPermission()` - Request push notification access
  - `subscribeToPushNotifications()` - Subscribe to push
  - `queueTaskForSync()` - Queue tasks for background sync

### React Hooks
- `/lib/compute/usePWA.ts` - React hook for PWA functionality
  - Auto-registers service worker
  - Monitors battery status
  - Provides notification management
  - Returns PWA and battery status

### Components
- `/components/compute/InstallPrompt.tsx` - PWA installation prompt
  - Detects if app can be installed
  - Handles beforeinstallprompt event
  - Shows iOS-specific instructions
  - Mobile-first design (44px touch targets)
  - Auto-dismisses after 7 days

### Updated Files
- `/app/layout.tsx` - Added manifest link and meta tags
- `/app/contribute/page.tsx` - Integrated PWA features and battery monitoring

## Features

### 1. Offline Support
The service worker caches core assets for offline access:
- Static pages (`/`, `/contribute`, `/labs`)
- App manifest and icons
- Runtime caching for API responses

### 2. Background Sync
When offline, compute tasks are queued in IndexedDB and automatically synced when connection is restored:
```typescript
await queueTaskForSync(task);
```

### 3. Push Notifications
Contributors receive notifications for new task assignments:
```typescript
const subscription = await subscribeToPushNotifications(vapidPublicKey);
```

### 4. Battery Awareness
The app monitors battery status and automatically pauses compute on low battery:
```typescript
const { batteryStatus } = usePWA({
  onLowBattery: () => {
    console.log('Low battery, pausing compute');
  }
});
```

### 5. Installation Experience
Users are prompted to install the app with platform-specific instructions:
- **Android/Chrome**: One-click install button
- **iOS Safari**: Step-by-step instructions for "Add to Home Screen"

## Usage

### Basic PWA Hook
```typescript
import { usePWA } from '@/lib/compute';

function MyComponent() {
  const {
    pwaStatus,
    batteryStatus,
    requestNotifications,
    registerSW,
  } = usePWA({
    autoRegisterServiceWorker: true,
    monitorBatteryStatus: true,
    lowBatteryThreshold: 0.2,
    onLowBattery: () => {
      // Handle low battery
    },
  });

  return (
    <div>
      <p>Installed: {pwaStatus?.isInstalled ? 'Yes' : 'No'}</p>
      <p>Battery: {Math.round((batteryStatus?.level ?? 0) * 100)}%</p>
    </div>
  );
}
```

### Installation Prompt
```typescript
import { InstallPrompt } from '@/components/compute/InstallPrompt';

function MyPage() {
  return (
    <div>
      {/* Banner variant (floating) */}
      <InstallPrompt variant="banner" />

      {/* Card variant (embedded) */}
      <InstallPrompt
        variant="card"
        onInstall={() => console.log('App installed!')}
        onDismiss={() => console.log('Prompt dismissed')}
      />
    </div>
  );
}
```

### Battery Monitoring
```typescript
import { monitorBattery } from '@/lib/compute';

useEffect(() => {
  const cleanup = monitorBattery(
    (status) => {
      console.log('Battery:', status.level);
      console.log('Charging:', status.charging);
    },
    () => {
      console.log('Low battery alert!');
    },
    0.2 // 20% threshold
  );

  return cleanup;
}, []);
```

### Check if Should Contribute
```typescript
import { shouldContributeCompute } from '@/lib/compute';

const canContribute = await shouldContributeCompute(
  true,  // requireCharging
  0.3    // minBatteryLevel
);

if (canContribute) {
  // Start compute tasks
}
```

## Testing

### Desktop Browser
1. Open Chrome DevTools
2. Go to Application > Manifest
3. Click "Add to home screen"
4. Check Application > Service Workers for registration status

### Mobile Device
1. Visit the app on your mobile device
2. Look for the install banner/prompt
3. Install the app to home screen
4. Open from home screen to test standalone mode
5. Go offline to test cached assets

### Battery Monitoring
1. Open DevTools > More tools > Sensors
2. Set battery level and charging status
3. Observe battery status updates in the UI
4. Test low battery threshold triggers

## Service Worker Updates

The service worker automatically updates when new versions are deployed:
1. New SW is downloaded in the background
2. Old SW continues serving until all tabs are closed
3. On next app open, new SW activates

To force update manually:
```typescript
const registration = await navigator.serviceWorker.ready;
await registration.update();
```

## iOS Considerations

iOS Safari has limitations:
- No `beforeinstallprompt` event
- Must manually "Add to Home Screen"
- Battery API not supported on all iOS versions
- Push notifications require iOS 16.4+

The InstallPrompt component automatically detects iOS and shows appropriate instructions.

## Security

### HTTPS Required
PWAs require HTTPS in production. Development on `localhost` works without HTTPS.

### Service Worker Scope
The service worker is registered at root (`/`) to cache all app routes.

### Push Notification Security
Push subscriptions use VAPID keys for secure server-to-client communication.

## Performance

### Cache Strategy
- **Core assets**: Cache-first with background update
- **API calls**: Network-first with cache fallback
- **Runtime cache**: Automatic caching of visited pages

### Background Sync
Tasks are queued in IndexedDB and synced when:
- Network connection is restored
- App is reopened
- Service worker wakes up

## Troubleshooting

### Service Worker Not Registering
1. Check browser console for errors
2. Verify HTTPS or localhost
3. Clear site data and reload
4. Check `navigator.serviceWorker` exists

### Install Prompt Not Showing
1. Verify manifest.json is valid
2. Check service worker is registered
3. Ensure HTTPS is active
4. Clear browser install dismissal (chrome://apps)

### Battery API Not Working
1. Check browser support (Chrome, Edge)
2. Verify HTTPS
3. Test on actual mobile device (not all simulators support it)

### Push Notifications Not Working
1. Request notification permission
2. Subscribe to push with valid VAPID key
3. Test with Chrome DevTools > Application > Push Messaging

## Browser Support

| Feature | Chrome | Edge | Safari | Firefox |
|---------|--------|------|--------|---------|
| Service Workers | ✅ | ✅ | ✅ | ✅ |
| Web App Manifest | ✅ | ✅ | ✅ | ✅ |
| Install Prompt | ✅ | ✅ | ❌* | ❌ |
| Battery API | ✅ | ✅ | ❌ | ❌ |
| Push Notifications | ✅ | ✅ | ✅** | ✅ |
| Background Sync | ✅ | ✅ | ❌ | ❌ |

\* iOS requires manual "Add to Home Screen"
\** iOS 16.4+ only

## Next Steps

1. **Add VAPID Keys**: Generate VAPID keys for push notifications
2. **Backend Integration**: Connect push subscription to backend
3. **Analytics**: Track PWA install rate and usage
4. **Optimization**: Fine-tune cache strategies based on usage patterns
5. **Testing**: Comprehensive testing on real mobile devices

## Resources

- [MDN: Progressive Web Apps](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
- [web.dev: PWA](https://web.dev/progressive-web-apps/)
- [Chrome DevTools: PWA Testing](https://developer.chrome.com/docs/devtools/progressive-web-apps/)
- [Battery Status API](https://developer.mozilla.org/en-US/docs/Web/API/Battery_Status_API)
- [Push API](https://developer.mozilla.org/en-US/docs/Web/API/Push_API)
