/**
 * Notifications API
 *
 * GET /api/notifications - List notifications
 * POST /api/notifications/mark-read - Mark all as read
 */

import { NextRequest, NextResponse } from "next/server";
import {
  listNotifications,
  getUnreadCount,
  markAllAsRead,
  NotificationType,
  NotificationStatus,
} from "@/lib/social/notifications";

/**
 * GET /api/notifications
 * List notifications for a user
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    const userId = searchParams.get("userId");
    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    const status = searchParams.get("status") as NotificationStatus | null;
    const typesParam = searchParams.get("types");
    const types = typesParam
      ? (typesParam.split(",") as NotificationType[])
      : undefined;

    const options = {
      userId,
      status: status || undefined,
      types,
      page: parseInt(searchParams.get("page") || "1", 10),
      limit: parseInt(searchParams.get("limit") || "20", 10),
    };

    const result = await listNotifications(options);

    return NextResponse.json(result);
  } catch (error) {
    console.error("Error listing notifications:", error);
    return NextResponse.json(
      { error: "Failed to list notifications" },
      { status: 500 }
    );
  }
}

/**
 * POST /api/notifications
 * Mark all notifications as read
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId, action } = body as { userId?: string; action?: string };

    if (!userId) {
      return NextResponse.json(
        { error: "userId is required" },
        { status: 400 }
      );
    }

    if (action === "mark_all_read") {
      const count = await markAllAsRead(userId);
      return NextResponse.json({ success: true, markedCount: count });
    }

    return NextResponse.json(
      { error: "Invalid action" },
      { status: 400 }
    );
  } catch (error) {
    console.error("Error processing notification action:", error);
    return NextResponse.json(
      { error: "Failed to process action" },
      { status: 500 }
    );
  }
}
