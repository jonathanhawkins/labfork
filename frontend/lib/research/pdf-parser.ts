/**
 * PDF Upload and Parsing Service
 *
 * Handles PDF paper uploads:
 * - File upload and validation
 * - Text extraction from PDFs
 * - Metadata extraction (title, authors, abstract, references)
 * - OCR fallback for scanned papers (placeholder)
 * - Manual metadata entry support
 */

import type { PaperMetadata, PaperAuthor } from "@/lib/papers/types";

// ============================================================================
// Types
// ============================================================================

/**
 * PDF parse status
 */
export type PDFParseStatus =
  | "pending"
  | "uploading"
  | "processing"
  | "extracting"
  | "success"
  | "needs_manual"
  | "error";

/**
 * Extracted PDF metadata
 */
export interface ExtractedPDFMetadata {
  /** Extracted title */
  title?: string;
  /** Extracted authors */
  authors: PaperAuthor[];
  /** Extracted abstract */
  abstract?: string;
  /** Extracted publication date */
  publishedDate?: string;
  /** Extracted DOI */
  doi?: string;
  /** Extracted arXiv ID */
  arxivId?: string;
  /** Extracted venue/conference */
  venue?: string;
  /** Extracted references */
  references: ExtractedReference[];
  /** Page count */
  pageCount?: number;
  /** Full text content */
  fullText?: string;
  /** Confidence score for extraction (0-1) */
  confidence: number;
}

/**
 * Extracted reference
 */
export interface ExtractedReference {
  /** Reference text */
  text: string;
  /** Parsed title if available */
  title?: string;
  /** Parsed authors if available */
  authors?: string[];
  /** Extracted identifiers */
  arxivId?: string;
  doi?: string;
  url?: string;
}

/**
 * PDF parse result
 */
export interface PDFParseResult {
  /** Parse status */
  status: PDFParseStatus;
  /** Extracted metadata */
  metadata?: ExtractedPDFMetadata;
  /** Converted paper metadata (if confident) */
  paperMetadata?: PaperMetadata;
  /** Error message if failed */
  error?: string;
  /** Fields needing manual entry */
  needsManualEntry?: string[];
  /** Processing time in ms */
  processingTime?: number;
}

/**
 * Manual metadata entry
 */
export interface ManualPDFMetadata {
  /** Paper title */
  title: string;
  /** Authors (comma-separated or array) */
  authors: string | string[];
  /** Abstract text */
  abstract?: string;
  /** Publication date */
  publishedDate?: string;
  /** DOI */
  doi?: string;
  /** arXiv ID */
  arxivId?: string;
  /** Venue/conference */
  venue?: string;
}

/**
 * PDF upload options
 */
export interface PDFUploadOptions {
  /** Maximum file size in bytes */
  maxSize?: number;
  /** Allowed MIME types */
  allowedTypes?: string[];
  /** Whether to extract full text */
  extractFullText?: boolean;
  /** Whether to attempt OCR for scanned PDFs */
  enableOCR?: boolean;
  /** Maximum pages to process */
  maxPages?: number;
}

// ============================================================================
// Constants
// ============================================================================

/** Default max file size (50MB) */
export const DEFAULT_MAX_SIZE = 50 * 1024 * 1024;

/** Allowed MIME types */
export const ALLOWED_PDF_TYPES = ["application/pdf"];

/** PDF header magic bytes */
export const PDF_MAGIC_BYTES = [0x25, 0x50, 0x44, 0x46]; // %PDF

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Validate PDF file
 */
export function validatePDFFile(
  file: File,
  options: PDFUploadOptions = {}
): { valid: boolean; error?: string } {
  const { maxSize = DEFAULT_MAX_SIZE, allowedTypes = ALLOWED_PDF_TYPES } = options;

  // Check file exists
  if (!file) {
    return { valid: false, error: "No file provided" };
  }

  // Check file type
  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: `Invalid file type: ${file.type}. Only PDF files are allowed.`,
    };
  }

  // Check file size
  if (file.size > maxSize) {
    const maxSizeMB = Math.round(maxSize / (1024 * 1024));
    const fileSizeMB = Math.round(file.size / (1024 * 1024));
    return {
      valid: false,
      error: `File too large: ${fileSizeMB}MB. Maximum size is ${maxSizeMB}MB.`,
    };
  }

  // Check file name
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    return {
      valid: false,
      error: "File must have .pdf extension",
    };
  }

  return { valid: true };
}

/**
 * Validate PDF bytes (check magic number)
 */
export async function validatePDFBytes(file: File): Promise<{ valid: boolean; error?: string }> {
  try {
    const buffer = await file.slice(0, 4).arrayBuffer();
    const bytes = new Uint8Array(buffer);

    // Check for PDF magic bytes
    const isPDF = PDF_MAGIC_BYTES.every((byte, i) => bytes[i] === byte);

    if (!isPDF) {
      return { valid: false, error: "File does not appear to be a valid PDF" };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: "Failed to read file" };
  }
}

// ============================================================================
// Text Extraction Functions
// ============================================================================

/**
 * Extract text from PDF (placeholder - would use pdf.js or server-side library)
 *
 * In production, this would:
 * 1. Use pdf.js in browser to extract text
 * 2. Or send to server endpoint with pdf-parse library
 */
export async function extractTextFromPDF(
  file: File,
  options: PDFUploadOptions = {}
): Promise<{ text: string; pageCount: number } | null> {
  // This is a placeholder. In production, implement with:
  // - Browser: pdf.js (pdfjs-dist)
  // - Server: pdf-parse, pdf2json, or pymupdf via API

  // For now, return null to indicate extraction not available
  console.warn("PDF text extraction not implemented. Use server-side processing.");
  return null;
}

// ============================================================================
// Metadata Extraction Functions
// ============================================================================

/**
 * Patterns for extracting metadata from PDF text
 */
const METADATA_PATTERNS = {
  // Title patterns (usually at the top, larger font would need PDF parsing)
  title: [
    /^([A-Z][^\n]{20,150})$/m, // Capitalized line at start
    /^Title:\s*(.+)$/im,
    /^#\s+(.+)$/m, // Markdown-style heading
  ],

  // Abstract patterns
  abstract: [
    /Abstract[:\s]*\n?([\s\S]{100,2000}?)(?=\n\n|\n1\.|Introduction|Keywords)/i,
    /ABSTRACT[:\s]*\n?([\s\S]{100,2000}?)(?=\n\n|1\.|Introduction)/,
    /Summary[:\s]*\n?([\s\S]{100,2000}?)(?=\n\n|1\.|Introduction)/i,
  ],

  // Author patterns
  authors: [
    /(?:Authors?|By)[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|Abstract|Keywords)/i,
    /^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)$/m,
  ],

  // DOI patterns
  doi: [
    /(?:DOI|doi)[:\s]*(10\.\d{4,}\/[^\s]+)/i,
    /(10\.\d{4,}\/[^\s\)]+)/,
  ],

  // arXiv patterns
  arxiv: [
    /arXiv[:\s]*(\d{4}\.\d{4,5})/i,
    /arxiv\.org\/abs\/(\d{4}\.\d{4,5})/i,
  ],

  // Date patterns
  date: [
    /(?:Published|Date|Submitted)[:\s]*(\d{1,2}\s+\w+\s+\d{4})/i,
    /(?:Published|Date|Submitted)[:\s]*(\w+\s+\d{1,2},?\s+\d{4})/i,
    /(\d{4})\s*©/,
    /(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}/i,
  ],

  // Venue/conference patterns
  venue: [
    /(?:Conference|Proceedings|Journal|Workshop)[:\s]*([^\n]+)/i,
    /(?:ICML|NeurIPS|ICLR|CVPR|ACL|EMNLP|NAACL|INTERSPEECH|ICASSP)\s*\d{4}/i,
  ],
};

/**
 * Extract title from PDF text
 */
export function extractTitle(text: string): string | undefined {
  // First, try to get title from first few lines
  const lines = text.split("\n").slice(0, 20).filter(l => l.trim().length > 10);

  // Look for a prominent line (longer, possibly all caps or title case)
  for (const line of lines) {
    const trimmed = line.trim();
    // Skip author-like lines, abstract, etc.
    if (/^(abstract|authors?|keywords|introduction)/i.test(trimmed)) continue;
    // Skip lines with email addresses or affiliations
    if (/@/.test(trimmed) || /university|institute|department/i.test(trimmed)) continue;

    // Good title candidate
    if (trimmed.length >= 20 && trimmed.length <= 200) {
      return trimmed;
    }
  }

  // Try patterns
  for (const pattern of METADATA_PATTERNS.title) {
    const match = text.match(pattern);
    if (match && match[1] && match[1].length >= 10 && match[1].length <= 200) {
      return match[1].trim();
    }
  }

  return undefined;
}

/**
 * Extract abstract from PDF text
 */
export function extractAbstract(text: string): string | undefined {
  for (const pattern of METADATA_PATTERNS.abstract) {
    const match = text.match(pattern);
    if (match && match[1]) {
      // Clean up the abstract
      let abstract = match[1]
        .replace(/\n/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      if (abstract.length >= 50) {
        return abstract;
      }
    }
  }

  return undefined;
}

/**
 * Extract authors from PDF text
 */
export function extractAuthors(text: string): PaperAuthor[] {
  const authors: PaperAuthor[] = [];

  for (const pattern of METADATA_PATTERNS.authors) {
    const match = text.match(pattern);
    if (match && match[1]) {
      // Split by common delimiters
      const authorStr = match[1].replace(/\n/g, ", ");
      const names = authorStr.split(/[,;]|(?:\s+and\s+)/i);

      for (const name of names) {
        const cleaned = name.trim().replace(/^\d+\.?\s*/, ""); // Remove numbering
        if (cleaned.length >= 3 && cleaned.length <= 50) {
          // Check it looks like a name (has space, not all caps acronym)
          if (/\s/.test(cleaned) && !/^[A-Z]{2,}$/.test(cleaned)) {
            authors.push({ name: cleaned });
          }
        }
      }

      if (authors.length > 0) break;
    }
  }

  return authors;
}

/**
 * Extract DOI from PDF text
 */
export function extractDOI(text: string): string | undefined {
  for (const pattern of METADATA_PATTERNS.doi) {
    const match = text.match(pattern);
    if (match && match[1]) {
      // Clean up DOI
      let doi = match[1].replace(/[.,;)\]]+$/, ""); // Remove trailing punctuation
      return doi;
    }
  }
  return undefined;
}

/**
 * Extract arXiv ID from PDF text
 */
export function extractArxivId(text: string): string | undefined {
  for (const pattern of METADATA_PATTERNS.arxiv) {
    const match = text.match(pattern);
    if (match && match[1]) {
      return match[1];
    }
  }
  return undefined;
}

/**
 * Extract publication date from PDF text
 */
export function extractDate(text: string): string | undefined {
  for (const pattern of METADATA_PATTERNS.date) {
    const match = text.match(pattern);
    if (match && match[1]) {
      try {
        const date = new Date(match[1]);
        if (!isNaN(date.getTime())) {
          return date.toISOString().split("T")[0];
        }
      } catch {
        // Return raw match if can't parse
        return match[1];
      }
    }
  }
  return undefined;
}

/**
 * Extract venue/conference from PDF text
 */
export function extractVenue(text: string): string | undefined {
  for (const pattern of METADATA_PATTERNS.venue) {
    const match = text.match(pattern);
    if (match) {
      return (match[1] || match[0]).trim();
    }
  }
  return undefined;
}

/**
 * Extract references from PDF text
 */
export function extractReferences(text: string): ExtractedReference[] {
  const references: ExtractedReference[] = [];

  // Find references section
  const refSectionMatch = text.match(
    /(?:References|Bibliography|Works Cited)\s*\n([\s\S]+?)(?:\n\n[A-Z]|\nAppendix|$)/i
  );

  if (!refSectionMatch) {
    return references;
  }

  const refSection = refSectionMatch[1];

  // Split by reference numbers or newlines
  const refLines = refSection.split(/\n\s*\[\d+\]|\n\s*\d+\.\s+/);

  for (const line of refLines) {
    const cleaned = line.trim().replace(/\s+/g, " ");
    if (cleaned.length < 20) continue;

    const ref: ExtractedReference = { text: cleaned };

    // Try to extract arXiv ID
    const arxivMatch = cleaned.match(/arXiv[:\s]*(\d{4}\.\d{4,5})/i);
    if (arxivMatch) {
      ref.arxivId = arxivMatch[1];
    }

    // Try to extract DOI
    const doiMatch = cleaned.match(/(10\.\d{4,}\/[^\s]+)/);
    if (doiMatch) {
      ref.doi = doiMatch[1].replace(/[.,;)\]]+$/, "");
    }

    // Try to extract URL
    const urlMatch = cleaned.match(/(https?:\/\/[^\s]+)/);
    if (urlMatch) {
      ref.url = urlMatch[1].replace(/[.,;)\]]+$/, "");
    }

    references.push(ref);
  }

  return references.slice(0, 100); // Limit to 100 references
}

/**
 * Full metadata extraction from PDF text
 */
export function extractMetadataFromText(text: string): ExtractedPDFMetadata {
  const title = extractTitle(text);
  const authors = extractAuthors(text);
  const abstract = extractAbstract(text);
  const publishedDate = extractDate(text);
  const doi = extractDOI(text);
  const arxivId = extractArxivId(text);
  const venue = extractVenue(text);
  const references = extractReferences(text);

  // Calculate confidence based on what we found
  let confidence = 0;
  if (title) confidence += 0.25;
  if (authors.length > 0) confidence += 0.25;
  if (abstract) confidence += 0.25;
  if (doi || arxivId) confidence += 0.15;
  if (publishedDate) confidence += 0.05;
  if (venue) confidence += 0.05;

  return {
    title,
    authors,
    abstract,
    publishedDate,
    doi,
    arxivId,
    venue,
    references,
    fullText: text,
    confidence,
  };
}

// ============================================================================
// Conversion Functions
// ============================================================================

/**
 * Convert extracted metadata to PaperMetadata
 */
export function toPaperMetadata(
  extracted: ExtractedPDFMetadata,
  pdfUrl: string
): PaperMetadata | null {
  // Need at least title to create paper metadata
  if (!extracted.title) {
    return null;
  }

  // Generate an ID based on available identifiers
  let id = "";
  if (extracted.arxivId) {
    id = `arxiv:${extracted.arxivId}`;
  } else if (extracted.doi) {
    id = `doi:${extracted.doi}`;
  } else {
    // Generate from title
    const slug = extracted.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .slice(0, 50);
    id = `pdf:${slug}-${Date.now().toString(36)}`;
  }

  return {
    id,
    title: extracted.title,
    authors: extracted.authors,
    abstract: extracted.abstract || "",
    source: "pdf",
    url: pdfUrl,
    pdfUrl,
    publishedDate: extracted.publishedDate,
    doi: extracted.doi,
    venue: extracted.venue,
    sourceMetadata: {
      arxivId: extracted.arxivId,
      extractionConfidence: extracted.confidence,
      referenceCount: extracted.references.length,
    },
  };
}

/**
 * Create PaperMetadata from manual entry
 */
export function fromManualEntry(
  manual: ManualPDFMetadata,
  pdfUrl: string
): PaperMetadata {
  // Parse authors
  const authorList = typeof manual.authors === "string"
    ? manual.authors.split(/[,;]/).map(a => a.trim()).filter(Boolean)
    : manual.authors;

  const authors: PaperAuthor[] = authorList.map(name => ({ name }));

  // Generate ID
  let id = "";
  if (manual.arxivId) {
    id = `arxiv:${manual.arxivId}`;
  } else if (manual.doi) {
    id = `doi:${manual.doi}`;
  } else {
    const slug = manual.title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .slice(0, 50);
    id = `pdf:${slug}-${Date.now().toString(36)}`;
  }

  return {
    id,
    title: manual.title,
    authors,
    abstract: manual.abstract || "",
    source: "pdf",
    url: pdfUrl,
    pdfUrl,
    publishedDate: manual.publishedDate,
    doi: manual.doi,
    venue: manual.venue,
    sourceMetadata: {
      arxivId: manual.arxivId,
      isManualEntry: true,
    },
  };
}

// ============================================================================
// Main Parse Function
// ============================================================================

/**
 * Parse PDF file and extract metadata
 *
 * This is a placeholder that works with pre-extracted text.
 * In production, integrate with pdf.js or server-side PDF parsing.
 */
export async function parsePDF(
  file: File,
  options: PDFUploadOptions = {}
): Promise<PDFParseResult> {
  const startTime = Date.now();

  // Validate file
  const validation = validatePDFFile(file, options);
  if (!validation.valid) {
    return {
      status: "error",
      error: validation.error,
      processingTime: Date.now() - startTime,
    };
  }

  // Validate PDF bytes
  const bytesValidation = await validatePDFBytes(file);
  if (!bytesValidation.valid) {
    return {
      status: "error",
      error: bytesValidation.error,
      processingTime: Date.now() - startTime,
    };
  }

  // Try to extract text (placeholder)
  const extractedText = await extractTextFromPDF(file, options);

  if (!extractedText) {
    // Text extraction not available, need manual entry
    return {
      status: "needs_manual",
      needsManualEntry: ["title", "authors", "abstract"],
      processingTime: Date.now() - startTime,
    };
  }

  // Extract metadata from text
  const metadata = extractMetadataFromText(extractedText.text);
  metadata.pageCount = extractedText.pageCount;

  // Determine what needs manual entry
  const needsManualEntry: string[] = [];
  if (!metadata.title) needsManualEntry.push("title");
  if (metadata.authors.length === 0) needsManualEntry.push("authors");
  if (!metadata.abstract) needsManualEntry.push("abstract");

  // Generate paper URL (would be actual upload URL in production)
  const pdfUrl = URL.createObjectURL(file);

  // Try to convert to paper metadata
  const paperMetadata = toPaperMetadata(metadata, pdfUrl);

  // Determine status
  let status: PDFParseStatus;
  if (needsManualEntry.length > 0) {
    status = "needs_manual";
  } else if (metadata.confidence >= 0.7) {
    status = "success";
  } else {
    status = "needs_manual";
    if (!needsManualEntry.includes("title")) needsManualEntry.push("title");
  }

  return {
    status,
    metadata,
    paperMetadata: paperMetadata || undefined,
    needsManualEntry: needsManualEntry.length > 0 ? needsManualEntry : undefined,
    processingTime: Date.now() - startTime,
  };
}

/**
 * Parse PDF from text content (for testing or pre-extracted text)
 */
export function parsePDFFromText(
  text: string,
  filename: string
): PDFParseResult {
  const startTime = Date.now();

  // Extract metadata
  const metadata = extractMetadataFromText(text);

  // Determine what needs manual entry
  const needsManualEntry: string[] = [];
  if (!metadata.title) needsManualEntry.push("title");
  if (metadata.authors.length === 0) needsManualEntry.push("authors");
  if (!metadata.abstract) needsManualEntry.push("abstract");

  // Generate PDF URL from filename
  const pdfUrl = `file://${filename}`;

  // Convert to paper metadata
  const paperMetadata = toPaperMetadata(metadata, pdfUrl);

  return {
    status: needsManualEntry.length > 0 ? "needs_manual" : "success",
    metadata,
    paperMetadata: paperMetadata || undefined,
    needsManualEntry: needsManualEntry.length > 0 ? needsManualEntry : undefined,
    processingTime: Date.now() - startTime,
  };
}

// ============================================================================
// Utility Exports
// ============================================================================

export const STATUS_LABELS: Record<PDFParseStatus, string> = {
  pending: "Pending",
  uploading: "Uploading",
  processing: "Processing",
  extracting: "Extracting Text",
  success: "Success",
  needs_manual: "Needs Manual Entry",
  error: "Error",
};

export const STATUS_COLORS: Record<PDFParseStatus, string> = {
  pending: "text-foreground-muted",
  uploading: "text-blue-400",
  processing: "text-yellow-400",
  extracting: "text-purple-400",
  success: "text-green-400",
  needs_manual: "text-orange-400",
  error: "text-red-400",
};
