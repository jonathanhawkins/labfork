/**
 * Supabase Library
 *
 * Central exports for Supabase client and database helpers.
 */

export { supabase, createSupabaseClient, createServerClient, isSupabaseConfigured } from "./client";
export type { Database } from "./types";
export {
  getUserCredits,
  getCreditTransactions,
  recordEarnedCredits,
  recordSpentCredits,
  recordBonusCredits,
  recordRefundCredits,
} from "./credits";
export {
  registerDevice,
  getDevice,
  getUserDevices,
  updateDeviceHeartbeat,
  updateDeviceStats,
  getOnlineDevices,
  getDevicesByTier,
  markStaleDevicesOffline,
} from "./devices";
