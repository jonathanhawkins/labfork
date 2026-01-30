/**
 * PDF Upload and Analysis API
 *
 * POST - Upload and parse a PDF paper
 */

import { NextRequest, NextResponse } from "next/server";
import {
  parsePDFFromText,
  toPaperMetadata,
  fromManualEntry,
  PDFParseResult,
  ManualPDFMetadata,
} from "@/lib/research/pdf-parser";
import { createPaper } from "@/lib/papers/types";

export const dynamic = "force-dynamic";

// Maximum file size (50MB)
const MAX_FILE_SIZE = 50 * 1024 * 1024;

/**
 * POST /api/research/pdf - Upload and parse PDF
 *
 * Body: FormData with 'file' field, or JSON with 'text' and optional 'manual' fields
 * Returns: { success: boolean, parseResult?: PDFParseResult, paper?: Paper, error?: string }
 */
export async function POST(request: NextRequest) {
  try {
    const contentType = request.headers.get("content-type") || "";

    // Handle form data upload
    if (contentType.includes("multipart/form-data")) {
      const formData = await request.formData();
      const file = formData.get("file") as File | null;
      const manualData = formData.get("manual") as string | null;

      if (!file) {
        return NextResponse.json(
          { success: false, error: "PDF file is required" },
          { status: 400 }
        );
      }

      // Validate file type
      if (!file.type.includes("pdf") && !file.name.toLowerCase().endsWith(".pdf")) {
        return NextResponse.json(
          { success: false, error: "Only PDF files are allowed" },
          { status: 400 }
        );
      }

      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        return NextResponse.json(
          {
            success: false,
            error: `File too large. Maximum size is ${Math.round(MAX_FILE_SIZE / (1024 * 1024))}MB`,
          },
          { status: 400 }
        );
      }

      // Read file content
      const arrayBuffer = await file.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);

      // Check PDF magic bytes
      const pdfMagic = [0x25, 0x50, 0x44, 0x46]; // %PDF
      const isPDF = pdfMagic.every((byte, i) => bytes[i] === byte);

      if (!isPDF) {
        return NextResponse.json(
          { success: false, error: "Invalid PDF file" },
          { status: 400 }
        );
      }

      // For now, we extract text in a simplified way
      // In production, this would use a PDF parsing library like pdf-parse
      const textContent = extractTextFromPDF(bytes);
      const parseResult = parsePDFFromText(textContent, file.name);

      // If manual data provided, use it to supplement
      let paper = null;
      if (manualData) {
        try {
          const manual: ManualPDFMetadata = JSON.parse(manualData);
          const metadata = fromManualEntry(manual, file.name);
          paper = createPaper(metadata);
        } catch {
          // Ignore invalid manual data
        }
      }

      // Try to create paper from parse result
      if (!paper && parseResult.paperMetadata) {
        paper = createPaper(parseResult.paperMetadata);
      } else if (!paper && parseResult.metadata) {
        const metadata = toPaperMetadata(parseResult.metadata, file.name);
        paper = createPaper(metadata);
      }

      return NextResponse.json({
        success: parseResult.status === "success" || parseResult.status === "needs_manual",
        parseResult,
        paper,
        needsManualEntry: parseResult.needsManualEntry,
      });
    }

    // Handle JSON with text content
    const body = await request.json();
    const { text, manual, filename } = body;

    if (manual) {
      // Direct manual entry
      try {
        const metadata = fromManualEntry(manual as ManualPDFMetadata, filename || "uploaded.pdf");
        const paper = createPaper(metadata);

        return NextResponse.json({
          success: true,
          parseResult: {
            status: "success" as const,
            paperMetadata: metadata,
          },
          paper,
        });
      } catch (err) {
        return NextResponse.json(
          {
            success: false,
            error: err instanceof Error ? err.message : "Invalid manual metadata",
          },
          { status: 400 }
        );
      }
    }

    if (!text) {
      return NextResponse.json(
        { success: false, error: "Text content or manual metadata is required" },
        { status: 400 }
      );
    }

    // Parse text content
    const parseResult = parsePDFFromText(text, filename || "uploaded.pdf");

    let paper = null;
    if (parseResult.paperMetadata) {
      paper = createPaper(parseResult.paperMetadata);
    } else if (parseResult.metadata) {
      const metadata = toPaperMetadata(parseResult.metadata, filename || "uploaded.pdf");
      paper = createPaper(metadata);
    }

    return NextResponse.json({
      success: parseResult.status === "success" || parseResult.status === "needs_manual",
      parseResult,
      paper,
      needsManualEntry: parseResult.needsManualEntry,
    });
  } catch (error) {
    console.error("Error processing PDF:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to process PDF",
      },
      { status: 500 }
    );
  }
}

/**
 * Extract text from PDF bytes
 * This is a simplified extraction - in production use a proper PDF library
 */
function extractTextFromPDF(bytes: Uint8Array): string {
  // Convert bytes to string for basic text extraction
  // This is a simplified approach that looks for text streams
  const text = new TextDecoder("latin1").decode(bytes);

  // Look for text between stream markers
  const textParts: string[] = [];
  const streamRegex = /stream\s*([\s\S]*?)\s*endstream/g;
  let match;

  while ((match = streamRegex.exec(text)) !== null) {
    const streamContent = match[1];
    // Try to extract readable text
    const readable = streamContent
      .replace(/[^\x20-\x7E\n\r\t]/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    if (readable.length > 20) {
      textParts.push(readable);
    }
  }

  // Also try to extract text from object strings
  const stringRegex = /\(([^)]+)\)/g;
  while ((match = stringRegex.exec(text)) !== null) {
    const content = match[1];
    if (content.length > 10 && /^[\x20-\x7E]+$/.test(content)) {
      textParts.push(content);
    }
  }

  return textParts.join("\n");
}
