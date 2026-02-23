import { describe, it, expect } from "vitest";
import {
  detectInputSource,
  isPdfUrl,
  isPapersWithCodeUrl,
  isHuggingFacePapersUrl,
  validateInput,
  getInputSuggestions,
} from "@/lib/papers/parser";

describe("Paper Input Parser", () => {
  describe("detectInputSource", () => {
    it("should detect arXiv IDs", () => {
      const result = detectInputSource("2401.12345");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("arxiv");
      expect(result?.identifier).toBe("2401.12345");
      expect(result?.confidence).toBe(1);
    });

    it("should detect arXiv URLs", () => {
      const result = detectInputSource("https://arxiv.org/abs/2401.12345");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("arxiv");
      expect(result?.identifier).toBe("2401.12345");
    });

    it("should detect DOIs", () => {
      const result = detectInputSource("10.1234/example");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("doi");
      expect(result?.identifier).toBe("10.1234/example");
    });

    it("should detect DOI URLs", () => {
      const result = detectInputSource("https://doi.org/10.1234/example");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("doi");
      expect(result?.identifier).toBe("10.1234/example");
    });

    it("should detect Semantic Scholar URLs", () => {
      // S2 URLs must have 40-char hex IDs
      const result = detectInputSource(
        "https://www.semanticscholar.org/paper/Title/a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
      );
      expect(result).not.toBeNull();
      expect(result?.source).toBe("semantic-scholar");
    });

    it("should detect GitHub URLs", () => {
      const result = detectInputSource("https://github.com/user/repo");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("github");
    });

    it("should detect PDF URLs", () => {
      const result = detectInputSource("https://example.com/paper.pdf");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("pdf");
    });

    it("should detect Papers With Code URLs", () => {
      const result = detectInputSource(
        "https://paperswithcode.com/paper/some-paper"
      );
      expect(result).not.toBeNull();
      expect(result?.source).toBe("papers-with-code");
    });

    it("should detect Hugging Face Papers URLs", () => {
      const result = detectInputSource("https://huggingface.co/papers/2401.12345");
      expect(result).not.toBeNull();
      expect(result?.source).toBe("arxiv");
    });

    it("should return null for unrecognized input", () => {
      expect(detectInputSource("random text")).toBeNull();
      expect(detectInputSource("")).toBeNull();
      expect(detectInputSource("https://google.com")).toBeNull();
    });
  });

  describe("isPdfUrl", () => {
    it("should match PDF URLs", () => {
      expect(isPdfUrl("https://example.com/paper.pdf")).toBe(true);
      expect(isPdfUrl("http://site.org/file.PDF")).toBe(true);
    });

    it("should reject non-PDF URLs", () => {
      expect(isPdfUrl("https://example.com/paper.html")).toBe(false);
      expect(isPdfUrl("https://arxiv.org/abs/2401.12345")).toBe(false);
    });
  });

  describe("isPapersWithCodeUrl", () => {
    it("should match Papers With Code URLs", () => {
      expect(
        isPapersWithCodeUrl("https://paperswithcode.com/paper/some-paper")
      ).toBe(true);
      expect(
        isPapersWithCodeUrl("https://www.paperswithcode.com/paper/title")
      ).toBe(true);
    });

    it("should reject non-PWC URLs", () => {
      expect(isPapersWithCodeUrl("https://arxiv.org/abs/2401.12345")).toBe(
        false
      );
      expect(isPapersWithCodeUrl("not-a-url")).toBe(false);
    });
  });

  describe("isHuggingFacePapersUrl", () => {
    it("should match Hugging Face Papers URLs", () => {
      expect(
        isHuggingFacePapersUrl("https://huggingface.co/papers/2401.12345")
      ).toBe(true);
    });

    it("should reject non-HF Papers URLs", () => {
      expect(
        isHuggingFacePapersUrl("https://huggingface.co/models/bert")
      ).toBe(false);
      expect(isHuggingFacePapersUrl("https://arxiv.org/abs/2401.12345")).toBe(
        false
      );
    });
  });

  describe("validateInput", () => {
    it("should validate supported inputs", () => {
      expect(validateInput("2401.12345").valid).toBe(true);
      expect(validateInput("https://arxiv.org/abs/2401.12345").valid).toBe(true);
      expect(validateInput("10.1234/example").valid).toBe(true);
      expect(validateInput("https://github.com/user/repo").valid).toBe(true);
    });

    it("should invalidate unsupported inputs", () => {
      const result = validateInput("random text");
      expect(result.valid).toBe(false);
      expect(result.error).toBeDefined();
    });

    it("should invalidate empty input", () => {
      const result = validateInput("");
      expect(result.valid).toBe(false);
    });
  });

  describe("getInputSuggestions", () => {
    it("should suggest corrections for common mistakes", () => {
      // Partial arXiv ID - should have suggestions
      const suggestions = getInputSuggestions("2401");
      expect(suggestions.length).toBeGreaterThan(0);
    });

    it("should provide helpful hints for unrecognized input", () => {
      const suggestions = getInputSuggestions("some random text");
      // Should still provide some suggestions for unrecognized input
      expect(suggestions).toBeDefined();
    });
  });
});
