import { ImageResponse } from "next/og";

export const runtime = "edge";

export const alt = "LabFork - Open Platform for AI Research";
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          backgroundColor: "#0f172a",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Gradient accent bar at top */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: "6px",
            background: "linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6)",
          }}
        />

        {/* Subtle gradient orbs for depth */}
        <div
          style={{
            position: "absolute",
            top: "-100px",
            right: "-100px",
            width: "500px",
            height: "500px",
            borderRadius: "50%",
            background:
              "radial-gradient(circle, rgba(139, 92, 246, 0.15), transparent 70%)",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: "-150px",
            left: "-100px",
            width: "600px",
            height: "600px",
            borderRadius: "50%",
            background:
              "radial-gradient(circle, rgba(59, 130, 246, 0.1), transparent 70%)",
          }}
        />

        {/* Main content */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            padding: "72px 80px",
            flex: 1,
            position: "relative",
          }}
        >
          {/* Logo and brand */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "20px",
              marginBottom: "48px",
            }}
          >
            {/* Fork icon */}
            <div
              style={{
                width: "56px",
                height: "56px",
                borderRadius: "14px",
                background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <svg
                width="32"
                height="32"
                viewBox="0 0 32 32"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M16 6 L16 14 M11 18 L16 14 L21 18 M11 18 L11 24 M21 18 L21 24"
                  stroke="white"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
                <circle cx="11" cy="25.5" r="2" fill="white" />
                <circle cx="21" cy="25.5" r="2" fill="white" />
              </svg>
            </div>
            <span
              style={{
                fontSize: "28px",
                fontWeight: 600,
                color: "#e2e8f0",
                letterSpacing: "-0.02em",
              }}
            >
              LabFork
            </span>
          </div>

          {/* Headline */}
          <div
            style={{
              fontSize: "64px",
              fontWeight: 700,
              color: "#f8fafc",
              lineHeight: 1.1,
              letterSpacing: "-0.03em",
              marginBottom: "24px",
              maxWidth: "900px",
            }}
          >
            Open Platform for
            <br />
            <span
              style={{
                background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                backgroundClip: "text",
                color: "transparent",
              }}
            >
              AI Research
            </span>
          </div>

          {/* Tagline */}
          <div
            style={{
              fontSize: "24px",
              color: "#94a3b8",
              lineHeight: 1.5,
              maxWidth: "700px",
            }}
          >
            Fork research labs, watch AI agents implement papers, discover
            synergies across domains.
          </div>

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Bottom domain pills */}
          <div
            style={{
              display: "flex",
              gap: "16px",
            }}
          >
            {[
              { label: "Voice Clone", color: "#3b82f6" },
              { label: "Quant Trading", color: "#22c55e" },
              { label: "Robotics", color: "#a855f7" },
              { label: "Biotech", color: "#ec4899" },
            ].map((domain) => (
              <div
                key={domain.label}
                style={{
                  padding: "10px 20px",
                  borderRadius: "10px",
                  backgroundColor: `${domain.color}18`,
                  border: `1px solid ${domain.color}40`,
                  color: domain.color,
                  fontSize: "18px",
                  fontWeight: 500,
                }}
              >
                {domain.label}
              </div>
            ))}
          </div>
        </div>

        {/* Bottom-right URL */}
        <div
          style={{
            position: "absolute",
            bottom: "28px",
            right: "80px",
            color: "#475569",
            fontSize: "18px",
            fontWeight: 500,
          }}
        >
          labfork.com
        </div>
      </div>
    ),
    {
      ...size,
    }
  );
}
