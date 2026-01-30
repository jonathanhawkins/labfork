/**
 * Notifications API - Single Notification
 *
 * PATCH /api/notifications/[id] - Mark as read
 * DELETE /api/notifications/[id] - Archive notification
 */

import { NextRequest, NextResponse } from "next/server";
import { markAsRead, archiveNotification } from "@/lib/social/notifications";

interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * PATCH /api/notifications/[id]
 * Mark notification as read
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    const notification = await markAsRead(id);

    if (!notification) {
      return NextResponse.json(
        { error: "Notification not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(notification);
  } catch (error) {
    console.error("Error marking notification as read:", error);
    return NextResponse.json(
      { error: "Failed to mark as read" },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/notifications/[id]
 * Archive a notification
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams
) {
  try {
    const { id } = await params;

    const notification = await archiveNotification(id);

    if (!notification) {
      return NextResponse.json(
        { error: "Notification not found" },
        { status: 404 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error archiving notification:", error);
    return NextResponse.json(
      { error: "Failed to archive notification" },
      { status: 500 }
    );
  }
}
