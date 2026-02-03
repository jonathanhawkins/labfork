/**
 * Credit History API Endpoint
 *
 * GET /api/credits/history - Get user transaction history
 */

import { NextRequest, NextResponse } from "next/server";
import { getCreditTransactions, isSupabaseConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    // Check if Supabase is configured
    if (!isSupabaseConfigured) {
      return NextResponse.json(
        {
          transactions: [],
          pagination: {
            limit: 50,
            offset: 0,
            count: 0,
          },
          message: "Credits system not configured",
        },
        { status: 200 }
      );
    }

    // Get parameters from query
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");
    const limitParam = searchParams.get("limit");
    const offsetParam = searchParams.get("offset");

    if (!userId) {
      return NextResponse.json(
        { error: "Missing userId parameter" },
        { status: 400 }
      );
    }

    const limit = limitParam ? parseInt(limitParam, 10) : 50;
    const offset = offsetParam ? parseInt(offsetParam, 10) : 0;

    // Validate parameters
    if (isNaN(limit) || limit < 1 || limit > 100) {
      return NextResponse.json(
        { error: "Invalid limit parameter (must be 1-100)" },
        { status: 400 }
      );
    }

    if (isNaN(offset) || offset < 0) {
      return NextResponse.json(
        { error: "Invalid offset parameter (must be >= 0)" },
        { status: 400 }
      );
    }

    const transactions = await getCreditTransactions(userId, limit, offset);

    return NextResponse.json(
      {
        transactions,
        pagination: {
          limit,
          offset,
          count: transactions.length,
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error in GET /api/credits/history:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
