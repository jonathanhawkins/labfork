/**
 * Tests for PDF Parser
 */

import { describe, it, expect, vi } from "vitest";
import {
  validatePDFFile,
  extractTitle,
  extractAbstract,
  extractAuthors,
  extractDOI,
  extractArxivId,
  extractDate,
  extractVenue,
  extractReferences,
  extractMetadataFromText,
  toPaperMetadata,
  fromManualEntry,
  parsePDFFromText,
  DEFAULT_MAX_SIZE,
  ALLOWED_PDF_TYPES,
  STATUS_LABELS,
  STATUS_COLORS,
} from "@/lib/research/pdf-parser";

// Mock File class for testing
function createMockFile(
  content: string,
  name: string,
  type: string,
  size?: number
): File {
  const blob = new Blob([content], { type });
  const file = new File([blob], name, { type });
  if (size !== undefined) {
    Object.defineProperty(file, "size", { value: size });
  }
  return file;
}

describe("PDF Parser", () => {
  describe("validatePDFFile", () => {
    it("validates a valid PDF file", () => {
      const file = createMockFile("PDF content", "paper.pdf", "application/pdf");
      const result = validatePDFFile(file);
      expect(result.valid).toBe(true);
    });

    it("rejects non-PDF files", () => {
      const file = createMockFile("content", "paper.txt", "text/plain");
      const result = validatePDFFile(file);
      expect(result.valid).toBe(false);
      expect(result.error).toContain("Invalid file type");
    });

    it("rejects files that are too large", () => {
      const file = createMockFile("content", "paper.pdf", "application/pdf", 100 * 1024 * 1024);
      const result = validatePDFFile(file);
      expect(result.valid).toBe(false);
      expect(result.error).toContain("too large");
    });

    it("accepts custom max size", () => {
      const file = createMockFile("content", "paper.pdf", "application/pdf", 10 * 1024 * 1024);
      const result = validatePDFFile(file, { maxSize: 5 * 1024 * 1024 });
      expect(result.valid).toBe(false);
    });

    it("rejects files without .pdf extension", () => {
      const file = createMockFile("content", "paper.doc", "application/pdf");
      const result = validatePDFFile(file);
      expect(result.valid).toBe(false);
      expect(result.error).toContain(".pdf extension");
    });
  });

  describe("extractTitle", () => {
    it("extracts title from first prominent line", () => {
      const text = `
Attention Is All You Need

Authors: Vaswani et al.

Abstract
We propose a new architecture...
      `;
      const title = extractTitle(text);
      expect(title).toBe("Attention Is All You Need");
    });

    it("skips abstract line", () => {
      const text = `
Abstract
This paper presents a novel approach to neural machine translation.

Introduction
      `;
      const title = extractTitle(text);
      expect(title).not.toBe("Abstract");
    });

    it("skips author/affiliation lines", () => {
      const text = `
john@university.edu
Stanford University

Deep Learning for Natural Language Processing
      `;
      const title = extractTitle(text);
      expect(title).toBe("Deep Learning for Natural Language Processing");
    });

    it("returns undefined for empty text", () => {
      const title = extractTitle("");
      expect(title).toBeUndefined();
    });
  });

  describe("extractAbstract", () => {
    it("extracts abstract section", () => {
      const text = `
Title Here

Abstract
This paper introduces a novel method for speech synthesis using neural networks.
We propose a new architecture that achieves state-of-the-art results.

1. Introduction
      `;
      const abstract = extractAbstract(text);
      expect(abstract).toContain("novel method");
      expect(abstract).toContain("speech synthesis");
    });

    it("extracts ABSTRACT in all caps", () => {
      const text = `
Title

ABSTRACT
Our contribution is a new framework for processing audio signals efficiently. This is an important addition to the field of audio processing.

1. Introduction
      `;
      const abstract = extractAbstract(text);
      expect(abstract).toBeDefined();
      expect(abstract).toContain("new framework");
    });

    it("handles multi-line abstracts", () => {
      const text = `
Abstract:
This is the first sentence of a long abstract.
This is the second sentence continuing the abstract.
The abstract provides important context.

Keywords: NLP, AI
      `;
      const abstract = extractAbstract(text);
      expect(abstract).toContain("first sentence");
      expect(abstract).toContain("second sentence");
    });

    it("returns undefined when no abstract found", () => {
      const text = "Just some random text without abstract section";
      const abstract = extractAbstract(text);
      expect(abstract).toBeUndefined();
    });
  });

  describe("extractAuthors", () => {
    it("extracts comma-separated authors", () => {
      const text = `
Authors: John Smith, Jane Doe, Bob Wilson

Abstract
      `;
      const authors = extractAuthors(text);
      expect(authors.length).toBe(3);
      expect(authors[0].name).toBe("John Smith");
      expect(authors[1].name).toBe("Jane Doe");
    });

    it("extracts authors with 'and' separator", () => {
      const text = `
By John Smith and Jane Doe

Abstract
      `;
      const authors = extractAuthors(text);
      expect(authors.length).toBeGreaterThanOrEqual(1);
    });

    it("filters out non-name entries", () => {
      const text = `
Authors: John Smith, ICML 2024, Abstract

Abstract
      `;
      const authors = extractAuthors(text);
      // Should not include "ICML 2024" or "Abstract"
      expect(authors.every(a => a.name.includes(" "))).toBe(true);
    });

    it("returns empty array when no authors found", () => {
      const text = "No author information here";
      const authors = extractAuthors(text);
      expect(authors).toHaveLength(0);
    });
  });

  describe("extractDOI", () => {
    it("extracts DOI with prefix", () => {
      const text = "DOI: 10.1234/example.paper.2024";
      const doi = extractDOI(text);
      expect(doi).toBe("10.1234/example.paper.2024");
    });

    it("extracts DOI without prefix", () => {
      const text = "Available at 10.5678/neural.network";
      const doi = extractDOI(text);
      expect(doi).toBe("10.5678/neural.network");
    });

    it("removes trailing punctuation", () => {
      const text = "DOI: 10.1234/paper.2024.";
      const doi = extractDOI(text);
      expect(doi).toBe("10.1234/paper.2024");
    });

    it("returns undefined when no DOI found", () => {
      const text = "No DOI in this text";
      const doi = extractDOI(text);
      expect(doi).toBeUndefined();
    });
  });

  describe("extractArxivId", () => {
    it("extracts arXiv ID with prefix", () => {
      const text = "arXiv:2401.12345";
      const arxivId = extractArxivId(text);
      expect(arxivId).toBe("2401.12345");
    });

    it("extracts arXiv ID from URL", () => {
      const text = "Available at arxiv.org/abs/2312.98765";
      const arxivId = extractArxivId(text);
      expect(arxivId).toBe("2312.98765");
    });

    it("returns undefined when no arXiv ID found", () => {
      const text = "No arXiv reference here";
      const arxivId = extractArxivId(text);
      expect(arxivId).toBeUndefined();
    });
  });

  describe("extractDate", () => {
    it("extracts date with 'Published' prefix", () => {
      const text = "Published: January 15, 2024";
      const date = extractDate(text);
      expect(date).toBeTruthy();
    });

    it("extracts date in various formats", () => {
      const texts = [
        "Date: 15 March 2024",
        "Submitted: March 15, 2024",
        "December 2023",
      ];

      for (const text of texts) {
        const date = extractDate(text);
        // At least some formats should be extractable
        if (text.includes("Date:") || text.includes("Submitted:")) {
          expect(date).toBeTruthy();
        }
      }
    });

    it("returns undefined when no date found", () => {
      const text = "No date information here";
      const date = extractDate(text);
      expect(date).toBeUndefined();
    });
  });

  describe("extractVenue", () => {
    it("extracts conference name", () => {
      const text = "Conference: International Conference on Machine Learning";
      const venue = extractVenue(text);
      expect(venue).toContain("Machine Learning");
    });

    it("extracts well-known venue abbreviations", () => {
      const texts = [
        "NeurIPS 2024",
        "ICML 2024",
        "CVPR 2023",
        "ACL 2024",
      ];

      for (const text of texts) {
        const venue = extractVenue(text);
        expect(venue).toBeTruthy();
      }
    });

    it("returns undefined when no venue found", () => {
      const text = "No venue information";
      const venue = extractVenue(text);
      expect(venue).toBeUndefined();
    });
  });

  describe("extractReferences", () => {
    it("extracts references section", () => {
      const text = `
Main content here.

References

[1] Smith, J. Neural Networks. arXiv:2401.12345. 2024.

[2] Doe, J. Deep Learning. doi: 10.1234/paper. 2023.

[3] Wilson, B. Machine Learning. https://example.com/paper.pdf

Appendix
      `;
      const refs = extractReferences(text);
      expect(refs.length).toBeGreaterThan(0);
    });

    it("extracts arXiv IDs from references", () => {
      const text = `
References

[1] Author. Title. arXiv:2401.12345.
      `;
      const refs = extractReferences(text);
      const refWithArxiv = refs.find(r => r.arxivId);
      expect(refWithArxiv?.arxivId).toBe("2401.12345");
    });

    it("extracts DOIs from references", () => {
      const text = `
References

[1] Author. Title. doi: 10.1234/example.
      `;
      const refs = extractReferences(text);
      const refWithDoi = refs.find(r => r.doi);
      expect(refWithDoi?.doi).toBe("10.1234/example");
    });

    it("extracts URLs from references", () => {
      const text = `
References

[1] Author. Title. https://example.com/paper.pdf
      `;
      const refs = extractReferences(text);
      const refWithUrl = refs.find(r => r.url);
      expect(refWithUrl?.url).toContain("example.com");
    });

    it("returns empty array when no references section", () => {
      const text = "Just some content without references";
      const refs = extractReferences(text);
      expect(refs).toHaveLength(0);
    });

    it("limits references to 100", () => {
      const refLines = Array(150).fill("[N] Author. Title. 2024.").join("\n");
      const text = `References\n\n${refLines}`;
      const refs = extractReferences(text);
      expect(refs.length).toBeLessThanOrEqual(100);
    });
  });

  describe("extractMetadataFromText", () => {
    it("extracts all available metadata", () => {
      const text = `
Attention Is All You Need

Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar

Abstract
We propose the Transformer, a model architecture eschewing recurrence and relying entirely on attention mechanisms to draw global dependencies between input and output.

DOI: 10.5555/3295222.3295349
arXiv: 1706.03762
Published: June 2017
Conference: NeurIPS 2017

References
[1] Bahdanau et al. Neural Machine Translation. arXiv:1409.0473.

Introduction
      `;
      const metadata = extractMetadataFromText(text);

      expect(metadata.title).toBe("Attention Is All You Need");
      expect(metadata.authors.length).toBeGreaterThan(0);
      expect(metadata.abstract).toContain("Transformer");
      expect(metadata.doi).toBe("10.5555/3295222.3295349");
      expect(metadata.arxivId).toBe("1706.03762");
      expect(metadata.venue).toContain("NeurIPS");
      expect(metadata.references.length).toBeGreaterThan(0);
      expect(metadata.confidence).toBeGreaterThan(0.5);
    });

    it("calculates confidence based on extracted fields", () => {
      const fullText = `
Title Here

Authors: John Doe

Abstract
This is the abstract.

DOI: 10.1234/paper
      `;
      const fullMetadata = extractMetadataFromText(fullText);

      const partialText = "Just some random text";
      const partialMetadata = extractMetadataFromText(partialText);

      expect(fullMetadata.confidence).toBeGreaterThan(partialMetadata.confidence);
    });

    it("stores full text in metadata", () => {
      const text = "Some paper content here";
      const metadata = extractMetadataFromText(text);
      expect(metadata.fullText).toBe(text);
    });
  });

  describe("toPaperMetadata", () => {
    it("converts extracted metadata to PaperMetadata", () => {
      const extracted = {
        title: "Test Paper",
        authors: [{ name: "John Doe" }],
        abstract: "Test abstract",
        doi: "10.1234/test",
        arxivId: "2401.12345",
        venue: "ICML 2024",
        references: [],
        confidence: 0.8,
      };

      const paperMetadata = toPaperMetadata(extracted, "file://paper.pdf");

      expect(paperMetadata).not.toBeNull();
      expect(paperMetadata?.title).toBe("Test Paper");
      expect(paperMetadata?.authors).toHaveLength(1);
      expect(paperMetadata?.source).toBe("pdf");
      expect(paperMetadata?.id).toContain("arxiv:");
    });

    it("uses DOI for ID when no arXiv ID", () => {
      const extracted = {
        title: "Test Paper",
        authors: [],
        doi: "10.1234/test",
        references: [],
        confidence: 0.5,
      };

      const paperMetadata = toPaperMetadata(extracted, "file://paper.pdf");
      expect(paperMetadata?.id).toContain("doi:");
    });

    it("generates slug-based ID when no identifiers", () => {
      const extracted = {
        title: "My Test Paper Title",
        authors: [],
        references: [],
        confidence: 0.5,
      };

      const paperMetadata = toPaperMetadata(extracted, "file://paper.pdf");
      expect(paperMetadata?.id).toContain("pdf:");
      expect(paperMetadata?.id).toContain("my-test-paper");
    });

    it("returns null when no title", () => {
      const extracted = {
        authors: [{ name: "John Doe" }],
        references: [],
        confidence: 0.2,
      };

      const paperMetadata = toPaperMetadata(extracted, "file://paper.pdf");
      expect(paperMetadata).toBeNull();
    });
  });

  describe("fromManualEntry", () => {
    it("creates PaperMetadata from manual entry", () => {
      const manual = {
        title: "Manually Entered Paper",
        authors: "John Doe, Jane Smith",
        abstract: "This is the abstract",
        doi: "10.1234/manual",
      };

      const paperMetadata = fromManualEntry(manual, "file://paper.pdf");

      expect(paperMetadata.title).toBe("Manually Entered Paper");
      expect(paperMetadata.authors).toHaveLength(2);
      expect(paperMetadata.authors[0].name).toBe("John Doe");
      expect(paperMetadata.authors[1].name).toBe("Jane Smith");
      expect(paperMetadata.sourceMetadata?.isManualEntry).toBe(true);
    });

    it("handles authors as array", () => {
      const manual = {
        title: "Paper",
        authors: ["John Doe", "Jane Smith"],
      };

      const paperMetadata = fromManualEntry(manual, "file://paper.pdf");
      expect(paperMetadata.authors).toHaveLength(2);
    });

    it("uses arXiv ID for ID when provided", () => {
      const manual = {
        title: "Paper",
        authors: "Author",
        arxivId: "2401.12345",
      };

      const paperMetadata = fromManualEntry(manual, "file://paper.pdf");
      expect(paperMetadata.id).toBe("arxiv:2401.12345");
    });
  });

  describe("parsePDFFromText", () => {
    it("parses text and returns result", () => {
      const text = `
A Novel Approach to Machine Learning

Authors: John Doe, Jane Smith

Abstract
This is a test abstract for the paper that provides important context about the research and methodology used in this study.

1. Introduction
      `;

      const result = parsePDFFromText(text, "paper.pdf");

      expect(result.metadata).toBeDefined();
      expect(result.metadata?.title).toBeDefined();
      // May need manual entry depending on extraction confidence
      expect(["success", "needs_manual"]).toContain(result.status);
    });

    it("returns needs_manual when missing fields", () => {
      const text = "x"; // Very short content that can't have valid title

      const result = parsePDFFromText(text, "paper.pdf");

      expect(result.status).toBe("needs_manual");
      expect(result.needsManualEntry).toBeDefined();
      expect(result.needsManualEntry!.length).toBeGreaterThan(0);
    });

    it("includes processing time", () => {
      const text = "Some content";
      const result = parsePDFFromText(text, "paper.pdf");
      expect(result.processingTime).toBeDefined();
      expect(result.processingTime).toBeGreaterThanOrEqual(0);
    });
  });

  describe("Constants", () => {
    it("has default max size set", () => {
      expect(DEFAULT_MAX_SIZE).toBe(50 * 1024 * 1024);
    });

    it("has allowed PDF types", () => {
      expect(ALLOWED_PDF_TYPES).toContain("application/pdf");
    });

    it("has status labels", () => {
      expect(STATUS_LABELS.success).toBe("Success");
      expect(STATUS_LABELS.error).toBe("Error");
      expect(STATUS_LABELS.needs_manual).toBe("Needs Manual Entry");
    });

    it("has status colors", () => {
      expect(STATUS_COLORS.success).toContain("green");
      expect(STATUS_COLORS.error).toContain("red");
    });
  });
});
