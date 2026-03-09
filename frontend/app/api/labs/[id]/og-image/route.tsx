import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

// Use nodejs runtime instead of edge so we can use fs
export const runtime = "nodejs";

interface RouteParams {
  params: Promise<{ id: string }>;
}

// Import dynamically to ensure server-side only
async function getLabData(id: string) {
  const { getLabById } = await import("@/lib/labs/repository");
  return getLabById(id);
}

/**
 * Generate OG image for lab pages
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams
) {
  const { id } = await params;

  // Fetch lab data
  const lab = await getLabData(id);

  if (!lab) {
    return new Response("Lab not found", { status: 404 });
  }

  // Domain colors
  const domainColors: Record<string, string> = {
    "voice-clone": "#3B82F6",
    "quant-trading": "#22C55E",
    "robotics": "#A855F7",
    "biotech": "#EC4899",
  };

  const accentColor = domainColors[lab.domainSlug] || "#6B7280";

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "#0A0A0A",
          padding: "60px",
        }}
      >
        {/* Top bar with accent color */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "6px",
            background: `linear-gradient(90deg, ${accentColor}, ${accentColor}88)`,
          }}
        />

        {/* Domain badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            marginBottom: "24px",
          }}
        >
          <div
            style={{
              padding: "8px 16px",
              borderRadius: "8px",
              backgroundColor: `${accentColor}22`,
              color: accentColor,
              fontSize: "18px",
              fontWeight: 500,
            }}
          >
            {lab.domainName}
          </div>
          {lab.isFeatured && (
            <div
              style={{
                padding: "8px 16px",
                borderRadius: "8px",
                backgroundColor: "#FBBF2422",
                color: "#FBBF24",
                fontSize: "18px",
                fontWeight: 500,
              }}
            >
              Featured
            </div>
          )}
        </div>

        {/* Lab name */}
        <div
          style={{
            fontSize: "64px",
            fontWeight: 700,
            color: "#FAFAFA",
            marginBottom: "16px",
            lineHeight: 1.1,
          }}
        >
          {lab.name}
        </div>

        {/* Description */}
        {lab.description && (
          <div
            style={{
              fontSize: "24px",
              color: "#A1A1AA",
              marginBottom: "32px",
              lineHeight: 1.4,
              maxWidth: "800px",
            }}
          >
            {lab.description.length > 150
              ? lab.description.slice(0, 150) + "..."
              : lab.description}
          </div>
        )}

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Footer with owner and stats */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          {/* Owner */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
            }}
          >
            <div
              style={{
                width: "48px",
                height: "48px",
                borderRadius: "50%",
                backgroundColor: "#27272A",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#A1A1AA",
                fontSize: "20px",
                fontWeight: 500,
              }}
            >
              {lab.owner.displayName.charAt(0).toUpperCase()}
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
              }}
            >
              <span style={{ color: "#FAFAFA", fontSize: "20px", fontWeight: 500 }}>
                {lab.owner.displayName}
              </span>
              <span style={{ color: "#71717A", fontSize: "16px" }}>
                @{lab.owner.username}
              </span>
            </div>
          </div>

          {/* Stats */}
          <div
            style={{
              display: "flex",
              gap: "32px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ color: "#FBBF24", fontSize: "24px" }}>★</span>
              <span style={{ color: "#FAFAFA", fontSize: "24px", fontWeight: 500 }}>
                {lab.stats.stars}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ color: "#71717A", fontSize: "24px" }}>⑂</span>
              <span style={{ color: "#FAFAFA", fontSize: "24px", fontWeight: 500 }}>
                {lab.stats.forks}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ color: "#71717A", fontSize: "24px" }}>◉</span>
              <span style={{ color: "#FAFAFA", fontSize: "24px", fontWeight: 500 }}>
                {lab.stats.tasks}
              </span>
            </div>
          </div>
        </div>

        {/* Branding */}
        <div
          style={{
            position: "absolute",
            bottom: "20px",
            right: "60px",
            color: "#52525B",
            fontSize: "16px",
          }}
        >
          labfork.com
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 630,
    }
  );
}
