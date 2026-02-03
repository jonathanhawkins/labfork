/**
 * Earn Credits API Endpoint
 *
 * POST /api/credits/earn - Record earned credits from task completion
 */

import { NextRequest, NextResponse } from "next/server";
import { recordEarnedCredits, isSupabaseConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

interface EarnCreditsRequest {
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

    const body: EarnCreditsRequest = await request.json();

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

    // Record the transaction
    const transaction = await recordEarnedCredits(
      body.userId,
      body.amount,
      body.taskId,
      body.description
    );

    if (!transaction) {
      return NextResponse.json(
        { error: "Failed to record earned credits" },
        { status: 500 }
      );
    }

    return NextResponse.json(
      {
        success: true,
        transaction,
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Error in POST /api/credits/earn:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
