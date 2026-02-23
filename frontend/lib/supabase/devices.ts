/**
 * Device Management Database Helpers
 *
 * Helper functions for managing compute devices in Supabase.
 */

import { supabase, createServerClient } from "./client";
import type { ComputeDevice } from "@/lib/compute/types";

/**
 * Register a new device
 */
export async function registerDevice(
  device: ComputeDevice
): Promise<ComputeDevice | null> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return null;
  }

  const { data, error } = await supabase
    .from("devices")
    .insert({
      id: device.id,
      user_id: device.userId,
      name: device.name,
      tier: device.tier,
      capabilities: device.capabilities as any,
      availability: device.availability as any,
      status: device.status,
      current_task_id: device.currentTaskId,
      stats: device.stats as any,
      registered_at: device.registeredAt,
      last_seen: device.lastHeartbeat,
    } as any)
    .select()
    .single();

  if (error || !data) {
    console.error("Error registering device:", error);
    return null;
  }

  return mapDeviceFromDb(data);
}

/**
 * Get device by ID
 */
export async function getDevice(
  deviceId: string
): Promise<ComputeDevice | null> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return null;
  }

  const { data, error } = await supabase
    .from("devices")
    .select("*")
    .eq("id", deviceId)
    .single();

  if (error || !data) {
    console.error("Error fetching device:", error);
    return null;
  }

  return mapDeviceFromDb(data);
}

/**
 * Get all devices for a user
 */
export async function getUserDevices(
  userId: string
): Promise<ComputeDevice[]> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return [];
  }

  const { data, error } = await supabase
    .from("devices")
    .select("*")
    .eq("user_id", userId)
    .order("last_seen", { ascending: false });

  if (error) {
    console.error("Error fetching user devices:", error);
    return [];
  }

  return data.map(mapDeviceFromDb);
}

/**
 * Update device status and heartbeat
 */
export async function updateDeviceHeartbeat(
  deviceId: string,
  status: ComputeDevice["status"],
  currentTaskId?: string
): Promise<boolean> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return false;
  }

  const updates: any = {
    status,
    last_seen: new Date().toISOString(),
  };

  if (currentTaskId !== undefined) {
    updates.current_task_id = currentTaskId;
  }

  const { error } = await supabase
    .from("devices")
    // @ts-ignore - Type inference issue with Supabase update
    .update(updates)
    .eq("id", deviceId);

  if (error) {
    console.error("Error updating device heartbeat:", error);
    return false;
  }

  return true;
}

/**
 * Update device statistics
 * Server-side only - requires service role
 */
export async function updateDeviceStats(
  deviceId: string,
  stats: {
    tasksCompleted?: number;
    creditsEarned?: number;
    totalComputeTime?: number;
  }
): Promise<boolean> {
  const client = createServerClient();
  if (!client) {
    console.error("Supabase server client not configured");
    return false;
  }

  // Get current stats
  const { data: device, error: fetchError } = await client
    .from("devices")
    .select("stats")
    .eq("id", deviceId)
    .single();

  if (fetchError || !device) {
    console.error("Error fetching device stats:", fetchError);
    return false;
  }

  const deviceRecord = device as any;
  const currentStats = deviceRecord.stats;
  const newStats = {
    tasksCompleted: stats.tasksCompleted ?? currentStats.tasksCompleted,
    creditsEarned: stats.creditsEarned ?? currentStats.creditsEarned,
    totalComputeTime: stats.totalComputeTime ?? currentStats.totalComputeTime,
  };

  const { error } = await client
    .from("devices")
    // @ts-ignore - Type inference issue with Supabase update
    .update({ stats: newStats })
    .eq("id", deviceId);

  if (error) {
    console.error("Error updating device stats:", error);
    return false;
  }

  return true;
}

/**
 * Get online devices (for network stats)
 */
export async function getOnlineDevices(): Promise<ComputeDevice[]> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return [];
  }

  const { data, error } = await supabase
    .from("devices")
    .select("*")
    .eq("status", "online")
    .order("last_seen", { ascending: false });

  if (error) {
    console.error("Error fetching online devices:", error);
    return [];
  }

  return data.map(mapDeviceFromDb);
}

/**
 * Get devices by tier
 */
export async function getDevicesByTier(
  tier: ComputeDevice["tier"]
): Promise<ComputeDevice[]> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return [];
  }

  const { data, error } = await supabase
    .from("devices")
    .select("*")
    .eq("tier", tier)
    .eq("status", "online")
    .order("last_seen", { ascending: false });

  if (error) {
    console.error("Error fetching devices by tier:", error);
    return [];
  }

  return data.map(mapDeviceFromDb);
}

/**
 * Mark stale devices as offline
 * Server-side only - requires service role
 */
export async function markStaleDevicesOffline(
  maxAgeMinutes = 5
): Promise<number> {
  const client = createServerClient();
  if (!client) {
    console.error("Supabase server client not configured");
    return 0;
  }

  const cutoffTime = new Date(Date.now() - maxAgeMinutes * 60 * 1000).toISOString();

  const { data, error } = await client
    .from("devices")
    // @ts-ignore - Type inference issue with Supabase update
    .update({ status: "offline" })
    .lt("last_seen", cutoffTime)
    .in("status", ["online", "busy"])
    .select("id");

  if (error) {
    console.error("Error marking stale devices offline:", error);
    return 0;
  }

  return data?.length || 0;
}

/**
 * Helper to map database row to ComputeDevice
 */
function mapDeviceFromDb(data: any): ComputeDevice {
  return {
    id: data.id,
    userId: data.user_id,
    name: data.name,
    tier: data.tier,
    capabilities: data.capabilities,
    availability: data.availability,
    status: data.status,
    currentTaskId: data.current_task_id,
    lastHeartbeat: data.last_seen,
    registeredAt: data.registered_at,
    stats: data.stats,
  };
}
