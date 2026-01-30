# PWA Quick Start Guide

## Installation

The PWA is already configured and ready to use. No additional dependencies needed.

## Quick Test

1. **Start the development server** (if not already running)
   ```bash
   npm run dev
   ```

2. **Navigate to `/contribute`**
   - You should see the install prompt (if supported by your browser)
   - Battery status will display (if supported)
   - Service worker will register automatically

3. **Test PWA Installation**
   - **Desktop Chrome**: Click install button or use DevTools
   - **Mobile**: Look for "Add to Home Screen" prompt
   - **iOS**: Tap Share → Add to Home Screen

## Key Files

```
frontend/
├── public/
│   ├── manifest.json           # PWA manifest
│   ├── sw.js                   # Service worker
│   └── icon-*.svg              # App icons
│
├── lib/compute/
│   ├── pwa-utils.ts            # Core PWA utilities
│   └── usePWA.ts               # React hook
│
├── components/compute/
│   └── InstallPrompt.tsx       # Install prompt component
│
└── app/
    ├── layout.tsx              # Manifest link + meta tags
    └── contribute/page.tsx     # PWA integration example
```

## Example Usage

### Simple Battery Monitor
```typescript
import { usePWA } from '@/lib/compute';

function MyComponent() {
  const { batteryStatus } = usePWA();

  if (!batteryStatus) return <p>Battery info unavailable</p>;

  return (
    <div>
      <p>Battery: {Math.round(batteryStatus.level * 100)}%</p>
      <p>Status: {batteryStatus.charging ? 'Charging' : 'Discharging'}</p>
    </div>
  );
}
```

### Check PWA Installation
```typescript
import { usePWA } from '@/lib/compute';

function MyComponent() {
  const { pwaStatus } = usePWA();

  return (
    <div>
      <p>Installed: {pwaStatus?.isInstalled ? 'Yes' : 'No'}</p>
      <p>Standalone: {pwaStatus?.isStandalone ? 'Yes' : 'No'}</p>
      <p>Can Install: {pwaStatus?.canInstall ? 'Yes' : 'No'}</p>
    </div>
  );
}
```

### Show Install Prompt
```typescript
import { InstallPrompt } from '@/components/compute/InstallPrompt';

function MyPage() {
  return (
    <div>
      <h1>My Page</h1>
      <InstallPrompt variant="card" />
    </div>
  );
}
```

### Battery-Aware Compute
```typescript
import { shouldContributeCompute } from '@/lib/compute';

async function startCompute() {
  // Check if device should contribute based on battery
  const canContribute = await shouldContributeCompute(
    true,  // require charging
    0.3    // minimum 30% battery
  );

  if (canContribute) {
    // Start compute tasks
  } else {
    console.log('Battery too low or not charging');
  }
}
```

## Testing Checklist

- [ ] Service worker registers on page load
- [ ] Manifest is valid (check DevTools > Application > Manifest)
- [ ] Install prompt appears (Chrome/Edge)
- [ ] Battery status displays (if supported)
- [ ] Low battery triggers pause (test with DevTools Sensors)
- [ ] Offline mode works (disable network in DevTools)
- [ ] App installs to home screen
- [ ] Standalone mode works (open from home screen)

## Browser DevTools

### Chrome/Edge
1. Open DevTools (F12)
2. Go to **Application** tab
3. Check:
   - **Manifest**: Verify manifest.json loads correctly
   - **Service Workers**: Verify registration
   - **Storage**: Check IndexedDB for queued tasks
   - **Cache Storage**: Verify cached assets

### Testing Offline
1. Open DevTools > Network
2. Check "Offline" checkbox
3. Reload page - should work from cache
4. Submit task - should queue for background sync

### Testing Battery
1. Open DevTools > More tools > Sensors
2. Set battery level (e.g., 15%)
3. Toggle charging state
4. Observe UI updates

## Common Issues

### Service Worker Not Registering
- Check console for errors
- Verify you're on HTTPS or localhost
- Try clearing site data (Application > Storage > Clear site data)

### Install Prompt Not Showing
- Only shows on HTTPS
- Won't show if already installed
- Some browsers don't support (Safari, Firefox)
- May be dismissed (clear chrome://apps)

### Battery API Not Working
- Only supported in Chrome/Edge
- Not available on all mobile devices
- Check `navigator.getBattery` exists

## Mobile Testing

### Android
1. Visit the app on Chrome/Edge
2. Install prompt should appear automatically
3. Tap "Install" or use browser menu "Add to Home screen"
4. Open from home screen to test standalone mode

### iOS
1. Visit the app in Safari
2. Tap Share button
3. Select "Add to Home Screen"
4. Tap "Add"
5. Open from home screen to test

**Note**: iOS has more limitations:
- No automatic install prompt
- Battery API may not work on older iOS versions
- Background sync not supported
- Push notifications require iOS 16.4+

## Production Deployment

Before deploying to production:

1. **Generate VAPID Keys** (for push notifications)
   ```bash
   npx web-push generate-vapid-keys
   ```

2. **Update Manifest** (if needed)
   - Change `start_url` if different
   - Update icon paths if serving from CDN
   - Set correct `theme_color`

3. **Configure Service Worker**
   - Update cache names for versioning
   - Adjust cache strategies if needed
   - Add any additional routes to cache

4. **Test on Real Devices**
   - Test on actual Android phones
   - Test on actual iPhones
   - Test on tablets
   - Test on different network conditions

## Next Steps

See [PWA_SETUP.md](./PWA_SETUP.md) for:
- Detailed API documentation
- Advanced usage patterns
- Backend integration guide
- Security considerations
- Performance optimization
