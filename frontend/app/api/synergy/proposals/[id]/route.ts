/**
 * Single Synergy Proposal API
 *
 * GET /api/synergy/proposals/[id] - Get a specific proposal
 * PATCH /api/synergy/proposals/[id] - Update proposal (accept/reject)
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getProposal,
  updateProposalStatus,
  getGlobalSynergyDiscovery,
  isProposalStatus,
} from "@/lib/meta/synergy";

interface RouteParams {
  params: Promise<{ id: string }>;
}

export async function GET(request: NextRequest, { params }: RouteParams) {
  try {
    const { id } = await params;
    const proposal = getProposal(id);

    if (!proposal) {
      return NextResponse.json(
        { success: false, error: `Proposal not found: ${id}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      data: proposal,
    });
  } catch (error) {
    console.error("Error fetching proposal:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to fetch proposal",
      },
      { status: 500 }
    );
  }
}

export async function PATCH(request: NextRequest, { params }: RouteParams) {
  try {
    const { id } = await params;
    const body = await request.json();
    const { status, notes } = body;

    // Validate status
    if (!status || !isProposalStatus(status)) {
      return NextResponse.json(
        {
          success: false,
          error: `Invalid status: ${status}. Valid values: pending, accepted, rejected, exploring, validated, invalidated`,
        },
        { status: 400 }
      );
    }

    const proposal = getProposal(id);
    if (!proposal) {
      return NextResponse.json(
        { success: false, error: `Proposal not found: ${id}` },
        { status: 404 }
      );
    }

    // Update proposal
    const updated = updateProposalStatus(id, status, notes);

    // If accepting, mark as explored
    if (status === "accepted" || status === "exploring") {
      const discovery = getGlobalSynergyDiscovery();
      discovery.markExplored(proposal);
    }

    return NextResponse.json({
      success: true,
      data: updated,
    });
  } catch (error) {
    console.error("Error updating proposal:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error ? error.message : "Failed to update proposal",
      },
      { status: 500 }
    );
  }
}
