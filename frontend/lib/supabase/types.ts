/**
 * Supabase Database Types
 *
 * Type definitions for database tables and queries.
 */

import type {
  DeviceCapabilities,
  DeviceAvailability,
  DeviceTier,
  DeviceStatus,
} from "@/lib/compute/types";

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export interface Database {
  public: {
    Tables: {
      contributor_profiles: {
        Row: {
          user_id: string;
          display_name: string;
          avatar_url: string | null;
          bio: string | null;
          badges: Json;
          joined_at: string;
          updated_at: string;
        };
        Insert: {
          user_id: string;
          display_name: string;
          avatar_url?: string | null;
          bio?: string | null;
          badges?: Json;
          joined_at?: string;
          updated_at?: string;
        };
        Update: {
          user_id?: string;
          display_name?: string;
          avatar_url?: string | null;
          bio?: string | null;
          badges?: Json;
          joined_at?: string;
          updated_at?: string;
        };
      };
      user_credits: {
        Row: {
          user_id: string;
          balance: number;
          total_earned: number;
          total_spent: number;
          updated_at: string;
        };
        Insert: {
          user_id: string;
          balance?: number;
          total_earned?: number;
          total_spent?: number;
          updated_at?: string;
        };
        Update: {
          user_id?: string;
          balance?: number;
          total_earned?: number;
          total_spent?: number;
          updated_at?: string;
        };
      };
      credit_transactions: {
        Row: {
          id: string;
          user_id: string;
          amount: number;
          type: "earn" | "spend" | "bonus" | "refund";
          task_id: string | null;
          description: string;
          created_at: string;
        };
        Insert: {
          id: string;
          user_id: string;
          amount: number;
          type: "earn" | "spend" | "bonus" | "refund";
          task_id?: string | null;
          description: string;
          created_at?: string;
        };
        Update: {
          id?: string;
          user_id?: string;
          amount?: number;
          type?: "earn" | "spend" | "bonus" | "refund";
          task_id?: string | null;
          description?: string;
          created_at?: string;
        };
      };
      devices: {
        Row: {
          id: string;
          user_id: string | null;
          name: string;
          tier: DeviceTier;
          capabilities: DeviceCapabilities;
          availability: DeviceAvailability;
          status: DeviceStatus;
          current_task_id: string | null;
          stats: {
            tasksCompleted: number;
            creditsEarned: number;
            totalComputeTime: number;
          };
          registered_at: string;
          last_seen: string;
        };
        Insert: {
          id: string;
          user_id?: string | null;
          name: string;
          tier: DeviceTier;
          capabilities: DeviceCapabilities;
          availability: DeviceAvailability;
          status: DeviceStatus;
          current_task_id?: string | null;
          stats?: {
            tasksCompleted: number;
            creditsEarned: number;
            totalComputeTime: number;
          };
          registered_at?: string;
          last_seen?: string;
        };
        Update: {
          id?: string;
          user_id?: string | null;
          name?: string;
          tier?: DeviceTier;
          capabilities?: DeviceCapabilities;
          availability?: DeviceAvailability;
          status?: DeviceStatus;
          current_task_id?: string | null;
          stats?: {
            tasksCompleted: number;
            creditsEarned: number;
            totalComputeTime: number;
          };
          registered_at?: string;
          last_seen?: string;
        };
      };
    };
    Views: Record<string, never>;
    Functions: Record<string, never>;
    Enums: Record<string, never>;
  };
}
