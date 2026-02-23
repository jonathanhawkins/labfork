/**
 * SSH Connection Tester
 *
 * Tests SSH connections and detects remote GPU hardware.
 */

import type { SSHConfig, GpuInfo } from "./types";

/**
 * SSH connection test result
 */
export interface SSHTestResult {
  success: boolean;
  /** Connection latency in ms */
  latency?: number;
  /** Remote hostname */
  hostname?: string;
  /** Remote OS */
  os?: string;
  /** Error message if failed */
  error?: string;
  /** Error code */
  errorCode?: "auth_failed" | "host_unreachable" | "timeout" | "unknown";
}

/**
 * Remote GPU detection result
 */
export interface RemoteGpuResult {
  success: boolean;
  gpu?: GpuInfo;
  error?: string;
}

/**
 * Complete remote system info
 */
export interface RemoteSystemInfo {
  success: boolean;
  info?: {
    hostname: string;
    os: string;
    kernel: string;
    gpu?: GpuInfo;
    ram?: number;
    cpuCores?: number;
    pythonVersion?: string;
    cudaAvailable?: boolean;
    torchVersion?: string;
    condaEnvs?: string[];
  };
  error?: string;
}

/**
 * Test SSH connection
 */
export async function testSSHConnection(
  config: SSHConfig
): Promise<SSHTestResult> {
  try {
    const response = await fetch("/api/lab/hardware/ssh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "test",
        host: config.host,
        port: config.port || 22,
        user: config.user,
        keyPath: config.keyPath,
      }),
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        latency: data.latency,
        hostname: data.hostname,
        os: data.os,
      };
    }

    return {
      success: false,
      error: data.error || "Connection failed",
      errorCode: data.errorCode || "unknown",
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Connection failed",
      errorCode: "unknown",
    };
  }
}

/**
 * Detect GPU on remote machine via SSH
 * Returns GpuInfo or null
 */
export async function detectRemoteGpu(
  config: SSHConfig
): Promise<GpuInfo | null> {
  try {
    const response = await fetch("/api/lab/hardware/ssh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "detect-gpu",
        host: config.host,
        port: config.port || 22,
        user: config.user,
        keyPath: config.keyPath,
      }),
    });

    const data = await response.json();

    if (data.success && data.gpu) {
      return data.gpu;
    }

    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Get complete system info from remote machine
 * Returns system info or null
 */
export async function getRemoteSystemInfo(
  config: SSHConfig
): Promise<RemoteSystemInfo["info"] | null> {
  try {
    const response = await fetch("/api/lab/hardware/ssh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "system-info",
        host: config.host,
        port: config.port || 22,
        user: config.user,
        keyPath: config.keyPath,
      }),
    });

    const data = await response.json();

    if (data.success && data.systemInfo) {
      return data.systemInfo;
    }

    return null;
  } catch (error) {
    return null;
  }
}

/**
 * SSH connection status
 */
export type SSHConnectionStatus =
  | "untested"
  | "testing"
  | "connected"
  | "failed";

/**
 * Get SSH status display info
 */
export function getSSHStatusInfo(status: SSHConnectionStatus): {
  label: string;
  color: string;
  bgColor: string;
} {
  switch (status) {
    case "untested":
      return {
        label: "Not tested",
        color: "text-foreground-muted",
        bgColor: "bg-foreground-muted/10",
      };
    case "testing":
      return {
        label: "Testing...",
        color: "text-yellow-400",
        bgColor: "bg-yellow-500/10",
      };
    case "connected":
      return {
        label: "Connected",
        color: "text-green-400",
        bgColor: "bg-green-500/10",
      };
    case "failed":
      return {
        label: "Failed",
        color: "text-red-400",
        bgColor: "bg-red-500/10",
      };
    default:
      return {
        label: "Unknown",
        color: "text-foreground-muted",
        bgColor: "bg-foreground-muted/10",
      };
  }
}

/**
 * Format SSH error for display
 */
export function formatSSHError(
  error: string
): string {
  const lowerError = error.toLowerCase();

  if (lowerError.includes("econnrefused") || lowerError.includes("refused")) {
    return "Connection refused. Check host and SSH service.";
  }
  if (lowerError.includes("etimedout") || lowerError.includes("timeout")) {
    return "Connection timeout. Check network and firewall.";
  }
  if (lowerError.includes("auth") || lowerError.includes("permission")) {
    return "Authentication failed. Check credentials and SSH key.";
  }

  return error;
}

/**
 * Validate SSH config before testing
 */
export function validateSSHConfig(config: Partial<SSHConfig>): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config.host?.trim()) {
    errors.push("Host is required");
  } else if (!/^[a-zA-Z0-9.-]+$/.test(config.host)) {
    errors.push("Invalid host format");
  }

  if (!config.user?.trim()) {
    errors.push("Username is required");
  }

  if (config.port !== undefined && (config.port < 1 || config.port > 65535)) {
    errors.push("Invalid port number (must be 1-65535)");
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Generate SSH connection string for display
 */
export function formatSSHConnection(config: SSHConfig): string {
  const port = config.port || 22;
  return `${config.user}@${config.host}:${port}`;
}

/**
 * Common SSH hosts from project config (populated from environment)
 */
export const KNOWN_SSH_HOSTS = (() => {
  const hosts = [];
  if (typeof process !== 'undefined' && process.env?.REMOTE_GPU_HOST) {
    hosts.push({
      name: "Remote GPU (from env)",
      host: process.env.REMOTE_GPU_HOST,
      user: process.env.REMOTE_GPU_USER || 'doc',
      description: "GPU training machine",
    });
  }
  return hosts;
})();

/**
 * Mock SSH test result for testing
 */
export const MOCK_SSH_RESULT: SSHTestResult = {
  success: true,
  latency: 45,
  hostname: "gpu-server",
  os: "Ubuntu 22.04",
};

// Alias for tests
export const MOCK_SSH_CONNECTION_RESULT = MOCK_SSH_RESULT;

/**
 * Mock remote GPU for testing
 */
export const MOCK_REMOTE_GPU: GpuInfo = {
  name: "NVIDIA GeForce RTX 4090",
  vram: 24,
  cudaVersion: "12.1",
  driverVersion: "535.104.05",
  available: true,
};

/**
 * Mock remote system info for testing
 */
export const MOCK_REMOTE_SYSTEM: RemoteSystemInfo["info"] = {
  hostname: "gpu-server",
  os: "Ubuntu 22.04 LTS",
  kernel: "5.15.0-88-generic",
  gpu: MOCK_REMOTE_GPU,
  ram: 64,
  cpuCores: 16,
  pythonVersion: "3.10.12",
  cudaAvailable: true,
  torchVersion: "2.1.0",
  condaEnvs: ["base", "voice"],
};
