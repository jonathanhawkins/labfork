/**
 * SocialPreview
 *
 * Preview of social features including stars, forks, comments, and feeds.
 */

"use client";

import React, { useState, useEffect, useCallback } from "react";

interface ActivityItem {
  id: string;
  type: "star" | "fork" | "comment" | "discovery" | "collaboration";
  user: string;
  avatar: string;
  target: string;
  time: string;
  details?: string;
}


// Map activity types from API to display types
function mapActivityType(apiType: string): ActivityItem["type"] {
  const typeMap: Record<string, ActivityItem["type"]> = {
    lab_starred: "star",
    lab_forked: "fork",
    comment_added: "comment",
    result_published: "discovery",
    collaboration_started: "collaboration",
    lab_created: "discovery",
    task_completed: "discovery",
  };
  return typeMap[apiType] || "discovery";
}

// Format relative time
function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins} min ago`;

  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

const typeIcons: Record<string, React.ReactNode> = {
  star: (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
    </svg>
  ),
  fork: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
    </svg>
  ),
  comment: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  ),
  discovery: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  ),
  collaboration: (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
    </svg>
  ),
};

const typeColors: Record<string, string> = {
  star: "#f59e0b",
  fork: "#8b5cf6",
  comment: "#3b82f6",
  discovery: "#10b981",
  collaboration: "#ec4899",
};

function ActivityCard({ item }: { item: ActivityItem }) {
  return (
    <div className="flex items-start gap-3 p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
      {/* Avatar */}
      <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold">
        {item.avatar}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium text-white">{item.user}</span>
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs"
            style={{
              backgroundColor: `${typeColors[item.type]}20`,
              color: typeColors[item.type],
            }}
          >
            {typeIcons[item.type]}
            {item.type}
          </span>
          <span className="text-gray-500 text-sm">{item.target}</span>
        </div>
        {item.details && (
          <p className="text-sm text-gray-400 mt-1 line-clamp-1">{item.details}</p>
        )}
        <span className="text-xs text-gray-600 mt-1 block">{item.time}</span>
      </div>
    </div>
  );
}

export function SocialPreview() {
  const [activities, setActivities] = useState<ActivityItem[]>([]);

  // Fetch real activity data
  const fetchActivities = useCallback(async () => {
    try {
      const response = await fetch("/api/activity?publicOnly=true&limit=10");
      if (!response.ok) return;

      const data = await response.json();
      if (data.activities && data.activities.length > 0) {
        const mapped: ActivityItem[] = data.activities.map((a: {
          id: string;
          type: string;
          actor?: { displayName?: string; username?: string };
          target?: { title?: string; name?: string };
          data?: { description?: string };
          createdAt: string;
        }) => ({
          id: a.id,
          type: mapActivityType(a.type),
          user: a.actor?.displayName || a.actor?.username || "Anonymous",
          avatar: (a.actor?.displayName || a.actor?.username || "A").charAt(0).toUpperCase(),
          target: a.target?.title || a.target?.name || "Research",
          time: formatRelativeTime(a.createdAt),
          details: a.data?.description,
        }));
        setActivities(mapped);
      }
    } catch {
      // Activity fetch failed — keep empty state
    }
  }, []);

  useEffect(() => {
    fetchActivities();
    // Refresh every 60 seconds
    const interval = setInterval(fetchActivities, 60000);
    return () => clearInterval(interval);
  }, [fetchActivities]);

  return (
    <section className="py-24 bg-gradient-to-b from-slate-900 to-slate-950">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-400 text-sm mb-4">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            Collaborative Research
          </div>
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Research is Social
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Star labs you admire, fork techniques to build upon, collaborate with
            researchers worldwide.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Features */}
          <div className="space-y-6">
            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-amber-500/10">
                <svg className="w-6 h-6 text-amber-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-white text-lg mb-1">Star & Follow</h3>
                <p className="text-gray-400">
                  Bookmark interesting labs and get notified of breakthroughs.
                  See what top researchers are working on.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-purple-500/10">
                <svg className="w-6 h-6 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-white text-lg mb-1">Fork & Extend</h3>
                <p className="text-gray-400">
                  Clone any public lab as a starting point. Build on others&apos;
                  work while giving proper credit.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-pink-500/10">
                <svg className="w-6 h-6 text-pink-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-white text-lg mb-1">Collaborate</h3>
                <p className="text-gray-400">
                  Invite others to your lab. Share compute, techniques, and
                  credit for joint discoveries.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="p-3 rounded-xl bg-blue-500/10">
                <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </div>
              <div>
                <h3 className="font-semibold text-white text-lg mb-1">Weekly Digest</h3>
                <p className="text-gray-400">
                  Get personalized updates on breakthroughs, trending techniques,
                  and new papers in your domains.
                </p>
              </div>
            </div>
          </div>

          {/* Live Activity Feed */}
          <div className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="font-medium text-white">Live Activity</span>
              </div>
              <span className="text-sm text-gray-500">Global Feed</span>
            </div>
            <div className="p-4 space-y-3 max-h-[400px] overflow-y-auto">
              {activities.length > 0 ? (
                activities.map((item) => (
                  <ActivityCard key={item.id} item={item} />
                ))
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <svg className="w-8 h-8 text-gray-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-sm text-gray-500">
                    Activity will appear here as research progresses
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default SocialPreview;
