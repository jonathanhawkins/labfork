import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  isArxivId,
  parseArxivInput,
  getArxivUrl,
  getArxivPdfUrl,
} from "@/lib/papers/arxiv";

describe("arXiv API Client", () => {
  describe("isArxivId", () => {
    it("should match new format IDs (YYMM.NNNNN)", () => {
      expect(isArxivId("2401.12345")).toBe(true);
      expect(isArxivId("2312.00001")).toBe(true);
      expect(isArxivId("1901.12345")).toBe(true);
      // Note: Implementation uses \d{4,5} so 6-digit is not matched
      expect(isArxivId("2401.1234")).toBe(true); // 4-digit version
    });

    it("should match new format IDs with version suffix", () => {
      expect(isArxivId("2401.12345v1")).toBe(true);
      expect(isArxivId("2401.12345v12")).toBe(true);
    });

    it("should match old format IDs (category/YYMMNNN)", () => {
      expect(isArxivId("cs/0601001")).toBe(true);
      expect(isArxivId("hep-th/9901001")).toBe(true);
      expect(isArxivId("math.AG/0601001")).toBe(true);
    });

    it("should reject invalid IDs", () => {
      expect(isArxivId("not-an-id")).toBe(false);
      expect(isArxivId("12345")).toBe(false);
      expect(isArxivId("2401")).toBe(false);
      expect(isArxivId("")).toBe(false);
      expect(isArxivId("10.1234/example")).toBe(false); // DOI
    });
  });

  describe("parseArxivInput", () => {
    it("should extract ID from bare arXiv ID", () => {
      expect(parseArxivInput("2401.12345")).toBe("2401.12345");
      expect(parseArxivInput("  2401.12345  ")).toBe("2401.12345");
    });

    it("should extract ID from arXiv abs URL", () => {
      expect(parseArxivInput("https://arxiv.org/abs/2401.12345")).toBe(
        "2401.12345"
      );
      // Implementation strips version suffix
      expect(parseArxivInput("http://arxiv.org/abs/2401.12345v2")).toBe(
        "2401.12345"
      );
    });

    it("should extract ID from arXiv pdf URL", () => {
      // Implementation uses URL regex that captures without .pdf extension
      expect(parseArxivInput("https://arxiv.org/pdf/2401.12345.pdf")).toBe(
        "2401.12345"
      );
    });

    it("should handle old format IDs", () => {
      expect(parseArxivInput("cs/0601001")).toBe("cs/0601001");
      expect(parseArxivInput("https://arxiv.org/abs/hep-th/9901001")).toBe(
        "hep-th/9901001"
      );
    });

    it("should return null for invalid input", () => {
      expect(parseArxivInput("not-an-id")).toBeNull();
      expect(parseArxivInput("https://google.com")).toBeNull();
      expect(parseArxivInput("")).toBeNull();
    });
  });

  describe("getArxivUrl", () => {
    it("should return correct abstract URL", () => {
      expect(getArxivUrl("2401.12345")).toBe("https://arxiv.org/abs/2401.12345");
      expect(getArxivUrl("cs/0601001")).toBe("https://arxiv.org/abs/cs/0601001");
    });
  });

  describe("getArxivPdfUrl", () => {
    it("should return correct PDF URL", () => {
      expect(getArxivPdfUrl("2401.12345")).toBe(
        "https://arxiv.org/pdf/2401.12345.pdf"
      );
      expect(getArxivPdfUrl("cs/0601001")).toBe(
        "https://arxiv.org/pdf/cs/0601001.pdf"
      );
    });
  });
});
