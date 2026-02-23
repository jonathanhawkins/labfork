import { describe, it, expect } from "vitest";
import {
  isDOI,
  parseDOI,
  isSemanticScholarUrl,
  parseSemanticScholarUrl,
} from "@/lib/papers/semantic-scholar";

describe("Semantic Scholar API Client", () => {
  describe("isDOI", () => {
    it("should match valid DOIs", () => {
      expect(isDOI("10.1234/example")).toBe(true);
      expect(isDOI("10.1000/xyz123")).toBe(true);
      expect(isDOI("10.1038/nature12373")).toBe(true);
      expect(isDOI("10.1109/TASLP.2023.1234567")).toBe(true);
    });

    it("should match DOI URLs", () => {
      expect(isDOI("https://doi.org/10.1234/example")).toBe(true);
      expect(isDOI("http://dx.doi.org/10.1234/example")).toBe(true);
    });

    it("should reject invalid DOIs", () => {
      expect(isDOI("not-a-doi")).toBe(false);
      expect(isDOI("11.1234/example")).toBe(false);
      expect(isDOI("2401.12345")).toBe(false); // arXiv ID
      expect(isDOI("")).toBe(false);
    });
  });

  describe("parseDOI", () => {
    it("should extract DOI from bare DOI", () => {
      expect(parseDOI("10.1234/example")).toBe("10.1234/example");
      expect(parseDOI("  10.1234/example  ")).toBe("10.1234/example");
    });

    it("should extract DOI from DOI URLs", () => {
      expect(parseDOI("https://doi.org/10.1234/example")).toBe(
        "10.1234/example"
      );
      expect(parseDOI("http://dx.doi.org/10.1234/example")).toBe(
        "10.1234/example"
      );
    });

    it("should return null for invalid input", () => {
      expect(parseDOI("not-a-doi")).toBeNull();
      expect(parseDOI("2401.12345")).toBeNull();
      expect(parseDOI("")).toBeNull();
    });
  });

  describe("isSemanticScholarUrl", () => {
    it("should match Semantic Scholar paper URLs", () => {
      // S2 URLs have 40-character hex paper IDs
      expect(
        isSemanticScholarUrl(
          "https://www.semanticscholar.org/paper/Title-Author/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        )
      ).toBe(true);
      expect(
        isSemanticScholarUrl(
          "https://semanticscholar.org/paper/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        )
      ).toBe(true);
    });

    it("should reject non-S2 URLs", () => {
      expect(isSemanticScholarUrl("https://arxiv.org/abs/2401.12345")).toBe(
        false
      );
      expect(isSemanticScholarUrl("https://google.com")).toBe(false);
      expect(isSemanticScholarUrl("not-a-url")).toBe(false);
    });
  });

  describe("parseSemanticScholarUrl", () => {
    it("should extract paper ID from S2 URLs", () => {
      // S2 paper IDs are 40-character hex strings
      expect(
        parseSemanticScholarUrl(
          "https://www.semanticscholar.org/paper/Title-Author/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
        )
      ).toBe("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0");
    });

    it("should return null for invalid URLs", () => {
      expect(parseSemanticScholarUrl("https://arxiv.org/abs/2401.12345")).toBeNull();
      expect(parseSemanticScholarUrl("not-a-url")).toBeNull();
    });
  });
});
