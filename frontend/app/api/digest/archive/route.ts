/**
 * Digest Archive API
 *
 * GET /api/digest/archive - Get past weekly digests
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createDigestGenerator,
  getDigestArchive,
} from "@/lib/meta/community";

// Singleton digest generator
let digestGenerator = createDigestGenerator();

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get("limit") || "10", 10);

    const archive = getDigestArchive(digestGenerator, Math.min(limit, 52));

    return NextResponse.json({
      digests: archive,
      total: archive.length,
    });
  } catch (error) {
    console.error("Error fetching digest archive:", error);
    return NextResponse.json(
      { error: "Failed to fetch digest archive" },
      { status: 500 }
    );
  }
}
