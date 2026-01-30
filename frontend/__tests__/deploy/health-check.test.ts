/**
 * Health Check API Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { NextRequest } from "next/server";
import { GET } from "@/app/api/health/route";

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe("Health Check API", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
  });

  describe("GET /api/health", () => {
    it("should return healthy status when all services are up", async () => {
      mockFetch.mockResolvedValue({
        ok: true,
      });

      const request = new NextRequest("http://localhost:3003/api/health");
      const response = await GET();
      const data = await response.json();

      expect(response.status).toBe(200);
      expect(data.status).toBe("healthy");
      expect(data.timestamp).toBeDefined();
      expect(data.uptime).toBeDefined();
    });

    it("should return degraded status when backend is down", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("8003")) {
          return Promise.reject(new Error("Connection refused"));
        }
        return Promise.resolve({ ok: true });
      });

      const response = await GET();
      const data = await response.json();

      expect(data.status).toBe("degraded");
      expect(data.checks.backend.status).toBe("down");
    });

    it("should return degraded status when ollama is down", async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes("11434")) {
          return Promise.reject(new Error("Connection refused"));
        }
        return Promise.resolve({ ok: true });
      });

      const response = await GET();
      const data = await response.json();

      expect(data.status).toBe("degraded");
      expect(data.checks.ollama.status).toBe("down");
    });

    it("should return unhealthy when all external services are down", async () => {
      mockFetch.mockRejectedValue(new Error("Connection refused"));

      const response = await GET();
      const data = await response.json();

      expect(data.status).toBe("unhealthy");
    });

    it("should include version information", async () => {
      mockFetch.mockResolvedValue({ ok: true });

      const response = await GET();
      const data = await response.json();

      expect(data.version).toBeDefined();
    });

    it("should include uptime in seconds", async () => {
      mockFetch.mockResolvedValue({ ok: true });

      const response = await GET();
      const data = await response.json();

      expect(typeof data.uptime).toBe("number");
      expect(data.uptime).toBeGreaterThanOrEqual(0);
    });

    it("should include latency for each service", async () => {
      mockFetch.mockResolvedValue({ ok: true });

      const response = await GET();
      const data = await response.json();

      expect(data.checks.frontend.latency).toBeDefined();
      expect(typeof data.checks.frontend.latency).toBe("number");
    });

    it("should handle timeout gracefully", async () => {
      mockFetch.mockImplementation(
        () =>
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error("Timeout")), 100)
          )
      );

      const response = await GET();
      const data = await response.json();

      expect(data).toBeDefined();
      expect(data.checks).toBeDefined();
    });
  });
});
