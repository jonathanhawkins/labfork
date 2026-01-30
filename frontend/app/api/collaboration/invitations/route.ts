/**
 * Invitations API
 *
 * GET /api/collaboration/invitations - List invitations
 * POST /api/collaboration/invitations - Send invitation
 * PATCH /api/collaboration/invitations - Respond to invitation
 */

import { NextRequest, NextResponse } from "next/server";
import {
  getGlobalMetaTaskManager,
  getGlobalWorkflowCoordinator,
  sendInvitation,
  respondToInvitation,
  getInvitationsForLab,
  getInvitationsForTask,
  getPendingInvitations,
  expireOldInvitations,
  MetaTaskId,
  InvitationId,
  ParticipantId,
  ParticipantRole,
  SendInvitationInput,
} from "@/lib/meta/collaboration";

/**
 * GET /api/collaboration/invitations
 * List invitations (for lab or task)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const coordinator = getGlobalWorkflowCoordinator();

    const labId = searchParams.get("labId");
    const metaTaskId = searchParams.get("metaTaskId") as MetaTaskId | null;
    const pendingOnly = searchParams.get("pending") === "true";

    // Expire old invitations first
    expireOldInvitations(coordinator);

    let invitations;

    if (labId) {
      invitations = getInvitationsForLab(coordinator, labId);
    } else if (metaTaskId) {
      invitations = getInvitationsForTask(coordinator, metaTaskId);
    } else if (pendingOnly) {
      invitations = getPendingInvitations(coordinator);
    } else {
      // Return all invitations (admin view)
      invitations = Array.from(coordinator.invitations.values());
    }

    // Sort by sent date (newest first)
    invitations.sort((a, b) =>
      new Date(b.sentAt).getTime() - new Date(a.sentAt).getTime()
    );

    return NextResponse.json({
      success: true,
      invitations,
      total: invitations.length,
    });
  } catch (error) {
    console.error("Failed to list invitations:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to list invitations",
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/collaboration/invitations
 * Send a new invitation
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const coordinator = getGlobalWorkflowCoordinator();
    const manager = getGlobalMetaTaskManager();

    // Validate required fields
    if (!body.metaTaskId) {
      return NextResponse.json(
        { success: false, error: "metaTaskId is required" },
        { status: 400 }
      );
    }

    if (!body.labId || !body.labName) {
      return NextResponse.json(
        { success: false, error: "labId and labName are required" },
        { status: 400 }
      );
    }

    if (!body.sentBy) {
      return NextResponse.json(
        { success: false, error: "sentBy (participant ID) is required" },
        { status: 400 }
      );
    }

    // Validate role
    const validRoles: ParticipantRole[] = [
      "co-lead",
      "contributor",
      "advisor",
      "reviewer",
      "observer",
    ];
    const role = (body.proposedRole as ParticipantRole) || "contributor";
    if (!validRoles.includes(role)) {
      return NextResponse.json(
        { success: false, error: `Invalid role: ${role}` },
        { status: 400 }
      );
    }

    const input: SendInvitationInput = {
      metaTaskId: body.metaTaskId as MetaTaskId,
      labId: body.labId,
      labName: body.labName,
      proposedRole: role,
      message: body.message || "",
      sentBy: body.sentBy as ParticipantId,
    };

    const invitation = sendInvitation(coordinator, manager, input);

    if (!invitation) {
      return NextResponse.json(
        { success: false, error: "Failed to send invitation" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      invitation,
    });
  } catch (error) {
    console.error("Failed to send invitation:", error);

    // Handle specific errors
    if (error instanceof Error) {
      if (error.message.includes("already a participant")) {
        return NextResponse.json(
          { success: false, error: "Lab is already a participant" },
          { status: 409 }
        );
      }
      if (error.message.includes("already pending")) {
        return NextResponse.json(
          { success: false, error: "Invitation already pending for this lab" },
          { status: 409 }
        );
      }
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to send invitation",
      },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/collaboration/invitations
 * Respond to an invitation
 */
export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const coordinator = getGlobalWorkflowCoordinator();
    const manager = getGlobalMetaTaskManager();

    // Validate required fields
    if (!body.invitationId) {
      return NextResponse.json(
        { success: false, error: "invitationId is required" },
        { status: 400 }
      );
    }

    if (body.accept === undefined) {
      return NextResponse.json(
        { success: false, error: "accept (boolean) is required" },
        { status: 400 }
      );
    }

    const invitationId = body.invitationId as InvitationId;
    const accept = body.accept as boolean;

    // If accepting, commitment and expertise are required
    if (accept) {
      if (!body.commitment) {
        return NextResponse.json(
          { success: false, error: "commitment is required when accepting" },
          { status: 400 }
        );
      }
    }

    const invitation = respondToInvitation(
      coordinator,
      manager,
      invitationId,
      accept,
      body.commitment,
      body.expertise
    );

    if (!invitation) {
      return NextResponse.json(
        { success: false, error: "Invitation not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      invitation,
      message: accept
        ? "Invitation accepted. You are now a participant."
        : "Invitation declined.",
    });
  } catch (error) {
    console.error("Failed to respond to invitation:", error);

    // Handle specific errors
    if (error instanceof Error) {
      if (error.message.includes("expired")) {
        return NextResponse.json(
          { success: false, error: "Invitation has expired" },
          { status: 400 }
        );
      }
      if (error.message.includes("cannot respond")) {
        return NextResponse.json(
          { success: false, error: error.message },
          { status: 400 }
        );
      }
    }

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to respond to invitation",
      },
      { status: 500 }
    );
  }
}
