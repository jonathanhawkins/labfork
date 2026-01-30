/**
 * Recent Alerts API
 *
 * GET /api/alerts/recent - Get recent alerts
 * POST /api/alerts/recent - Create a new alert (for testing/admin)
 */

import { NextRequest, NextResponse } from "next/server";
import {
  createAlertsSystem,
  getRecentAlerts,
  getAlertsForUser,
  detectAlert,
  markAlertRead,
  dismissAlert,
  getAlertStats,
  AlertType,
  SignificanceLevel,
} from "@/lib/meta/community";

// Singleton alerts system
let alertsSystem = createAlertsSystem();

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get("userId");
    const limit = parseInt(searchParams.get("limit") || "20", 10);
    const types = searchParams.get("types")?.split(",") as AlertType[] | undefined;
    const minSignificance = searchParams.get("minSignificance") as SignificanceLevel | undefined;
    const unreadOnly = searchParams.get("unreadOnly") === "true";
    const includeStats = searchParams.get("includeStats") === "true";

    let alerts;

    if (userId) {
      alerts = getAlertsForUser(alertsSystem, userId, limit);
    } else {
      alerts = getRecentAlerts(alertsSystem, {
        limit,
        types,
        minSignificance,
        unreadOnly,
      });
    }

    const response: {
      alerts: typeof alerts;
      stats?: ReturnType<typeof getAlertStats>;
    } = { alerts };

    if (includeStats) {
      response.stats = getAlertStats(alertsSystem);
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error("Error fetching recent alerts:", error);
    return NextResponse.json(
      { error: "Failed to fetch recent alerts" },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      type,
      title,
      description,
      relatedIds = [],
      sourceLabId,
      sourceLabName,
      techniqueIds,
      domain,
      metrics,
    } = body as {
      type: AlertType;
      title: string;
      description: string;
      relatedIds?: string[];
      sourceLabId?: string;
      sourceLabName?: string;
      techniqueIds?: string[];
      domain?: string;
      metrics?: Record<string, number>;
    };

    if (!type || !title || !description) {
      return NextResponse.json(
        { error: "type, title, and description are required" },
        { status: 400 }
      );
    }

    const alert = detectAlert(
      alertsSystem,
      {
        type,
        sourceLabId,
        sourceLabName,
        techniqueIds,
        domain,
        metrics,
      },
      title,
      description,
      relatedIds
    );

    return NextResponse.json(alert, { status: 201 });
  } catch (error) {
    console.error("Error creating alert:", error);
    return NextResponse.json(
      { error: "Failed to create alert" },
      { status: 500 }
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    const body = await request.json();
    const { alertId, action } = body as {
      alertId: string;
      action: "read" | "dismiss";
    };

    if (!alertId || !action) {
      return NextResponse.json(
        { error: "alertId and action are required" },
        { status: 400 }
      );
    }

    let alert;

    if (action === "read") {
      alert = markAlertRead(alertsSystem, alertId);
    } else if (action === "dismiss") {
      alert = dismissAlert(alertsSystem, alertId);
    } else {
      return NextResponse.json(
        { error: "Invalid action. Use 'read' or 'dismiss'" },
        { status: 400 }
      );
    }

    if (!alert) {
      return NextResponse.json(
        { error: "Alert not found" },
        { status: 404 }
      );
    }

    return NextResponse.json(alert);
  } catch (error) {
    console.error("Error updating alert:", error);
    return NextResponse.json(
      { error: "Failed to update alert" },
      { status: 500 }
    );
  }
}
