# ContributorDashboard Usage Examples

## Basic Usage

```tsx
import { ContributorDashboard } from "@/components/compute";

export default function ComputePage() {
  return (
    <div className="container mx-auto py-8">
      <ContributorDashboard />
    </div>
  );
}
```

## With Settings Handler

```tsx
"use client";

import { useState } from "react";
import { ContributorDashboard } from "@/components/compute";

export default function ComputePage() {
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <div className="container mx-auto py-8">
      <ContributorDashboard
        onSettingsClick={() => setSettingsOpen(true)}
      />

      {/* Settings dialog would go here */}
    </div>
  );
}
```

## Individual Component Usage

### TaskProgress Component

```tsx
import { TaskProgress } from "@/components/compute";

<TaskProgress
  taskName="Training Voice Model"
  progress={45}
  timeRemaining="5m 30s"
  isActive={true}
/>
```

### NetworkStats Component

```tsx
import { NetworkStats } from "@/components/compute";

<NetworkStats
  activeContributors={2500}
  networkComputePower={1234.5}
  yourRank={15}
  contributionPercent={2.3}
/>
```

### ContributionChart Component

```tsx
import { ContributionChart } from "@/components/compute";

const weekData = [
  { day: "Mon", tasksCompleted: 25, creditsEarned: 500 },
  { day: "Tue", tasksCompleted: 30, creditsEarned: 600 },
  { day: "Wed", tasksCompleted: 28, creditsEarned: 560 },
  { day: "Thu", tasksCompleted: 32, creditsEarned: 640 },
  { day: "Fri", tasksCompleted: 27, creditsEarned: 540 },
  { day: "Sat", tasksCompleted: 35, creditsEarned: 700 },
  { day: "Sun", tasksCompleted: 29, creditsEarned: 580 },
];

<ContributionChart data={weekData} />
```

## Real-time Integration Example

```tsx
"use client";

import { useState, useEffect } from "react";
import { ContributorDashboard } from "@/components/compute";

export default function ComputePage() {
  const [wsConnected, setWsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket
    const ws = new WebSocket("ws://localhost:8003/ws/compute");

    ws.onopen = () => {
      setWsConnected(true);
      console.log("Connected to compute WebSocket");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Update dashboard with real data
      console.log("Received:", data);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setWsConnected(false);
    };

    return () => ws.close();
  }, []);

  return (
    <div className="container mx-auto py-8">
      {!wsConnected && (
        <div className="mb-4 p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg">
          <p className="text-sm text-amber-400">
            Connecting to compute network...
          </p>
        </div>
      )}

      <ContributorDashboard />
    </div>
  );
}
```

## Responsive Layout Example

```tsx
import { ContributorDashboard } from "@/components/compute";

export default function ComputePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Mobile: Full width with padding */}
      <div className="px-4 py-6 sm:px-6 lg:px-8">
        {/* Desktop: Centered with max-width */}
        <div className="max-w-7xl mx-auto">
          <ContributorDashboard />
        </div>
      </div>
    </div>
  );
}
```

## With Navigation

```tsx
import { ContributorDashboard } from "@/components/compute";
import Link from "next/link";

export default function ComputePage() {
  return (
    <div className="min-h-screen bg-background">
      <nav className="border-b border-border p-4">
        <div className="max-w-7xl mx-auto flex items-center gap-4">
          <Link href="/" className="text-sm text-muted-foreground hover:text-foreground">
            Home
          </Link>
          <Link href="/compute" className="text-sm text-foreground-bright">
            Compute
          </Link>
          <Link href="/rewards" className="text-sm text-muted-foreground hover:text-foreground">
            Rewards
          </Link>
        </div>
      </nav>

      <main className="py-8">
        <ContributorDashboard />
      </main>
    </div>
  );
}
```

## Conditional Rendering

```tsx
"use client";

import { useState } from "react";
import { ContributorDashboard } from "@/components/compute";

export default function ComputePage() {
  const [isOptedIn, setIsOptedIn] = useState(false);

  if (!isOptedIn) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="max-w-md p-6 bg-background-card border border-border rounded-lg">
          <h2 className="text-xl font-semibold text-foreground-bright mb-4">
            Contribute Compute Power
          </h2>
          <p className="text-sm text-muted-foreground mb-6">
            Help advance AI research by contributing your device's unused compute power.
            Earn credits for every task completed.
          </p>
          <button
            onClick={() => setIsOptedIn(true)}
            className="w-full min-h-[44px] px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
          >
            Start Contributing
          </button>
        </div>
      </div>
    );
  }

  return <ContributorDashboard />;
}
```

## Testing

```tsx
import { render, screen } from "@testing-library/react";
import { ContributorDashboard } from "@/components/compute";

describe("ContributorDashboard", () => {
  it("renders with initial state", () => {
    render(<ContributorDashboard />);
    expect(screen.getByText("Compute Contribution")).toBeInTheDocument();
    expect(screen.getByText("Contributing")).toBeInTheDocument();
  });

  it("shows pause button when active", () => {
    render(<ContributorDashboard />);
    const pauseButton = screen.getByLabelText("Pause contribution");
    expect(pauseButton).toBeInTheDocument();
  });
});
```
