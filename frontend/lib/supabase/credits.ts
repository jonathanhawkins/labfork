/**
 * Credit System Database Helpers
 *
 * Helper functions for managing credits and transactions in Supabase.
 */

import { supabase, createServerClient } from "./client";
import type {
  UserCredits,
  CreditTransaction,
} from "@/lib/compute/types";
import { generateTransactionId } from "@/lib/compute/types";

/**
 * Get user credit balance and stats
 */
export async function getUserCredits(
  userId: string
): Promise<UserCredits | null> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return null;
  }

  const { data, error } = await supabase
    .from("user_credits")
    .select("*")
    .eq("user_id", userId)
    .single();

  if (error) {
    if (error.code === "PGRST116") {
      // No rows found - user has no credits yet
      return {
        userId,
        balance: 0,
        totalEarned: 0,
        totalSpent: 0,
        updatedAt: new Date().toISOString(),
      };
    }
    console.error("Error fetching user credits:", error);
    return null;
  }

  if (!data) {
    return null;
  }

  const record = data as any;
  return {
    userId: record.user_id,
    balance: record.balance,
    totalEarned: record.total_earned,
    totalSpent: record.total_spent,
    updatedAt: record.updated_at,
  };
}

/**
 * Get user transaction history
 */
export async function getCreditTransactions(
  userId: string,
  limit = 50,
  offset = 0
): Promise<CreditTransaction[]> {
  if (!supabase) {
    console.error("Supabase client not configured");
    return [];
  }

  const { data, error } = await supabase
    .from("credit_transactions")
    .select("*")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .range(offset, offset + limit - 1);

  if (error) {
    console.error("Error fetching credit transactions:", error);
    return [];
  }

  return data.map((tx: any) => ({
    id: tx.id,
    userId: tx.user_id,
    amount: tx.amount,
    type: tx.type,
    taskId: tx.task_id || undefined,
    description: tx.description,
    createdAt: tx.created_at,
  }));
}

/**
 * Record earned credits (from task completion)
 * Server-side only - requires service role
 */
export async function recordEarnedCredits(
  userId: string,
  amount: number,
  taskId: string,
  description: string
): Promise<CreditTransaction | null> {
  const client = createServerClient();
  if (!client) {
    console.error("Supabase server client not configured");
    return null;
  }

  const transaction: CreditTransaction = {
    id: generateTransactionId(),
    userId,
    amount,
    type: "earn",
    taskId,
    description,
    createdAt: new Date().toISOString(),
  };

  const { data, error } = await client
    .from("credit_transactions")
    .insert({
      id: transaction.id,
      user_id: transaction.userId,
      amount: transaction.amount,
      type: transaction.type,
      task_id: transaction.taskId,
      description: transaction.description,
    } as any)
    .select()
    .single();

  if (error || !data) {
    console.error("Error recording earned credits:", error);
    return null;
  }

  const record = data as any;
  return {
    id: record.id,
    userId: record.user_id,
    amount: record.amount,
    type: record.type,
    taskId: record.task_id || undefined,
    description: record.description,
    createdAt: record.created_at,
  };
}

/**
 * Record spent credits (for submitting tasks)
 * Server-side only - requires service role
 */
export async function recordSpentCredits(
  userId: string,
  amount: number,
  taskId: string,
  description: string
): Promise<CreditTransaction | null> {
  const client = createServerClient();
  if (!client) {
    console.error("Supabase server client not configured");
    return null;
  }

  // Check if user has sufficient balance
  const credits = await getUserCredits(userId);
  if (!credits || credits.balance < amount) {
    console.error("Insufficient balance:", { userId, balance: credits?.balance, required: amount });
    return null;
  }

  const transaction: CreditTransaction = {
    id: generateTransactionId(),
    userId,
    amount: -amount, // Negative for spending
    type: "spend",
    taskId,
    description,
    createdAt: new Date().toISOString(),
  };

  const { data, error } = await client
    .from("credit_transactions")
    .insert({
      id: transaction.id,
      user_id: transaction.userId,
      amount: transaction.amount,
      type: transaction.type,
      task_id: transaction.taskId,
      description: transaction.description,
    } as any)
    .select()
    .single();

  if (error || !data) {
    console.error("Error recording spent credits:", error);
    return null;
  }

  const record = data as any;
  return {
    id: record.id,
    userId: record.user_id,
    amount: record.amount,
    type: record.type,
    taskId: record.task_id || undefined,
    description: record.description,
    createdAt: record.created_at,
  };
}

/**
 * Record bonus credits (promotional or rewards)
 * Server-side only - requires service role
 */
export async function recordBonusCredits(
  userId: string,
  amount: number,
  description: string
): Promise<CreditTransaction | null> {
  const client = createServerClient();
  if (!client) {
    console.error("Supabase server client not configured");
    return null;
  }

  const transaction: CreditTransaction = {
    id: generateTransactionId(),
    userId,
    amount,
    type: "bonus",
    description,
    createdAt: new Date().toISOString(),
  };

  const { data, error } = await client
    .from("credit_transactions")
    .insert({
      id: transaction.id,
      user_id: transaction.userId,
      amount: transaction.amount,
      type: transaction.type,
      description: transaction.description,
    } as any)
    .select()
    .single();

  if (error || !data) {
    console.error("Error recording bonus credits:", error);
    return null;
  }

  const record = data as any;
  return {
    id: record.id,
    userId: record.user_id,
    amount: record.amount,
    type: record.type,
    taskId: record.task_id || undefined,
    description: record.description,
    createdAt: record.created_at,
  };
}

/**
 * Record refund credits (for failed tasks)
 * Server-side only - requires service role
 */
export async function recordRefundCredits(
  userId: string,
  amount: number,
  taskId: string,
  description: string
): Promise<CreditTransaction | null> {
  const client = createServerClient();
  if (!client) {
    console.error("Supabase server client not configured");
    return null;
  }

  const transaction: CreditTransaction = {
    id: generateTransactionId(),
    userId,
    amount,
    type: "refund",
    taskId,
    description,
    createdAt: new Date().toISOString(),
  };

  const { data, error } = await client
    .from("credit_transactions")
    .insert({
      id: transaction.id,
      user_id: transaction.userId,
      amount: transaction.amount,
      type: transaction.type,
      task_id: transaction.taskId,
      description: transaction.description,
    } as any)
    .select()
    .single();

  if (error || !data) {
    console.error("Error recording refund credits:", error);
    return null;
  }

  const record = data as any;
  return {
    id: record.id,
    userId: record.user_id,
    amount: record.amount,
    type: record.type,
    taskId: record.task_id || undefined,
    description: record.description,
    createdAt: record.created_at,
  };
}
