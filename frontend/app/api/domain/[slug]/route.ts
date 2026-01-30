import { NextResponse } from "next/server";
import {
  loadDomainConfig,
  domainExists,
  DomainConfigError,
} from "@/lib/domain/loader";

/**
 * GET /api/domain/[slug]
 *
 * Load domain configuration by slug
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params;

    // Validate slug format
    if (!slug || !/^[a-z0-9-]+$/.test(slug)) {
      return NextResponse.json(
        { error: "Invalid domain slug format" },
        { status: 400 }
      );
    }

    // Check if domain exists
    if (!domainExists(slug)) {
      return NextResponse.json(
        { error: `Domain not found: ${slug}` },
        { status: 404 }
      );
    }

    // Load and return config
    const config = loadDomainConfig(slug);

    return NextResponse.json(config);
  } catch (error) {
    console.error("Error loading domain config:", error);

    if (error instanceof DomainConfigError) {
      return NextResponse.json(
        { error: error.message, slug: error.slug },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
