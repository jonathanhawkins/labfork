/**
 * SSH Tester Utility Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  testSSHConnection,
  detectRemoteGpu,
  getRemoteSystemInfo,
  formatSSHError,
  validateSSHConfig,
  formatSSHConnection,
  KNOWN_SSH_HOSTS,
  MOCK_SSH_CONNECTION_RESULT,
  MOCK_REMOTE_GPU,
} from "@/lib/lab-wizard/ssh-tester";
import type { SSHConfig } from "@/lib/lab-wizard/types";

describe("ssh-tester", () => {
  describe("testSSHConnection", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return success for valid connection", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, ...MOCK_SSH_CONNECTION_RESULT }),
      });

      const config: SSHConfig = {
        host: "192.0.2.100",
        port: 22,
        user: "doc",
      };

      const result = await testSSHConnection(config);

      expect(result.success).toBe(true);
    });

    it("should return failure for invalid connection", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ success: false, error: "Connection refused" }),
      });

      const config: SSHConfig = {
        host: "invalid-host",
        port: 22,
        user: "user",
      };

      const result = await testSSHConnection(config);

      expect(result.success).toBe(false);
      expect(result.error).toBe("Connection refused");
    });

    it("should handle network errors", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("Network error")
      );

      const config: SSHConfig = {
        host: "test-host",
        port: 22,
        user: "user",
      };

      const result = await testSSHConnection(config);

      expect(result.success).toBe(false);
      expect(result.error).toContain("Network error");
    });
  });

  describe("detectRemoteGpu", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return GPU info for remote machine", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: true, gpu: MOCK_REMOTE_GPU }),
      });

      const config: SSHConfig = {
        host: "192.0.2.100",
        port: 22,
        user: "doc",
      };

      const result = await detectRemoteGpu(config);

      expect(result).toBeDefined();
      expect(result?.name).toBe(MOCK_REMOTE_GPU.name);
      expect(result?.vram).toBe(MOCK_REMOTE_GPU.vram);
    });

    it("should return null when no GPU detected", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ success: false }),
      });

      const config: SSHConfig = {
        host: "test-host",
        port: 22,
        user: "user",
      };

      const result = await detectRemoteGpu(config);

      expect(result).toBeNull();
    });
  });

  describe("getRemoteSystemInfo", () => {
    beforeEach(() => {
      vi.stubGlobal("fetch", vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it("should return system info for remote machine", async () => {
      const mockSystemInfo = {
        os: "Linux",
        platform: "linux",
        arch: "x64",
        totalMemory: 64,
      };

      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({ success: true, systemInfo: mockSystemInfo }),
      });

      const config: SSHConfig = {
        host: "test-host",
        port: 22,
        user: "user",
      };

      const result = await getRemoteSystemInfo(config);

      expect(result).toBeDefined();
      expect(result?.os).toBe("Linux");
    });
  });

  describe("formatSSHError", () => {
    it("should format connection refused error", () => {
      const result = formatSSHError("ECONNREFUSED");

      expect(result).toContain("refused");
    });

    it("should format timeout error", () => {
      const result = formatSSHError("ETIMEDOUT");

      expect(result).toContain("timeout");
    });

    it("should format authentication error", () => {
      const result = formatSSHError("auth_failed: permission denied");

      expect(result.toLowerCase()).toContain("auth");
    });

    it("should return original error for unknown errors", () => {
      const result = formatSSHError("Some unknown error");

      expect(result).toBe("Some unknown error");
    });
  });

  describe("validateSSHConfig", () => {
    it("should return valid for complete config", () => {
      const config: SSHConfig = {
        host: "192.168.1.100",
        port: 22,
        user: "admin",
      };

      const result = validateSSHConfig(config);

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it("should return invalid when host is missing", () => {
      const config: SSHConfig = {
        host: "",
        port: 22,
        user: "admin",
      };

      const result = validateSSHConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain("Host is required");
    });

    it("should return invalid when user is missing", () => {
      const config: SSHConfig = {
        host: "192.168.1.100",
        port: 22,
        user: "",
      };

      const result = validateSSHConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain("Username is required");
    });

    it("should return invalid for invalid port", () => {
      const config: SSHConfig = {
        host: "192.168.1.100",
        port: -1,
        user: "admin",
      };

      const result = validateSSHConfig(config);

      expect(result.valid).toBe(false);
      expect(result.errors.some((e) => e.includes("port"))).toBe(true);
    });
  });

  describe("formatSSHConnection", () => {
    it("should format connection string correctly", () => {
      const config: SSHConfig = {
        host: "192.168.1.100",
        port: 22,
        user: "admin",
      };

      const result = formatSSHConnection(config);

      expect(result).toBe("admin@192.168.1.100:22");
    });

    it("should use default port 22 when not specified", () => {
      const config: SSHConfig = {
        host: "192.168.1.100",
        user: "admin",
      };

      const result = formatSSHConnection(config);

      expect(result).toContain(":22");
    });
  });

  describe("KNOWN_SSH_HOSTS", () => {
    it("should be an array (populated from env vars)", () => {
      // KNOWN_SSH_HOSTS is now dynamically populated from environment variables
      // In test environment without env vars, it will be empty
      expect(Array.isArray(KNOWN_SSH_HOSTS)).toBe(true);
    });

    it("should have valid configurations for all known hosts when populated", () => {
      // Only test if there are hosts configured via env vars
      if (KNOWN_SSH_HOSTS.length > 0) {
        KNOWN_SSH_HOSTS.forEach((host) => {
          expect(host.host).toBeTruthy();
          expect(host.user).toBeTruthy();
          expect(host.name).toBeTruthy();
        });
      }
    });
  });
});
