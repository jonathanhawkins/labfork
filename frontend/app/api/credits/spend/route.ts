/**
 * Spend Credits API Endpoint
 *
 * POST /api/credits/spend - Spend credits for submitting tasks
 */

import { NextRequest, NextResponse } from "next/server";
import { recordSpentCredits, getUserCredits, isSupabaseConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

interface SpendCreditsRequest {
  userId: string;
  amount: number;
  taskId: string;
  description: string;
}

export async function POST(request: NextRequest) {
  try {
    // Check if Supabase is configured
    if (!isSupabaseConfigured) {
      return NextResponse.json(
        {
          success: false,
          message: "Credits system not configured",
        },
        { status: 200 }
      );
    }

    const body: SpendCreditsRequest = await request.json();

    // Validate required fields
    if (!body.userId || !body.amount || !body.taskId || !body.description) {
      return NextResponse.json(
        {
          error: "Missing required fields: userId, amount, taskId, description",
        },
        { status: 400 }
      );
    }

    // Validate amount
    if (typeof body.amount !== "number" || body.amount <= 0) {
      return NextResponse.json(
        { error: "Invalid amount (must be positive number)" },
        { status: 400 }
      );
    }

    // Check if user has sufficient balance
    const credits = await getUserCredits(body.userId);
    if (!credits || credits.balance < body.amount) {
      return NextResponse.json(
        {
          error: "Insufficient balance",
          balance: credits?.balance || 0,
          required: body.amount,
        },
        { status: 402 } // Payment Required
      );
    }

    // Record the transaction
    const transaction = await recordSpentCredits(
      body.userId,
      body.amount,
      body.taskId,
      body.description
    );

    if (!transaction) {
      return NextResponse.json(
        { error: "Failed to record spent credits" },
        { status: 500 }
      );
    }

    // Get updated balance
    const updatedCredits = await getUserCredits(body.userId);

    return NextResponse.json(
      {
        success: true,
        transaction,
        balance: updatedCredits?.balance || 0,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Error in POST /api/credits/spend:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
