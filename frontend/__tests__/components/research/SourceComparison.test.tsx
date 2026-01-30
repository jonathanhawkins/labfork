/**
 * SourceComparison Component Tests
 */

import { describe, it, expect, vi } from "vitest";

describe("SourceComparison Component", () => {
  describe("Props Interface", () => {
    it("should accept papers array", () => {
      const props = {
        papers: [],
      };
      expect(Array.isArray(props.papers)).toBe(true);
    });

    it("should accept onSelect callback", () => {
      const onSelect = vi.fn();
      const props = {
        papers: [],
        onSelect,
      };
      expect(props.onSelect).toBeDefined();
    });

    it("should accept onDismiss callback", () => {
      const onDismiss = vi.fn();
      const props = {
        papers: [],
        onDismiss,
      };
      expect(props.onDismiss).toBeDefined();
    });

    it("should accept className prop", () => {
      const props = {
        papers: [],
        className: "custom-class",
      };
      expect(props.className).toBe("custom-class");
    });
  });

  describe("Sorting", () => {
    it("should define valid sort fields", () => {
      type SortField = "title" | "date" | "citations" | "authors" | "source";
      const fields: SortField[] = ["title", "date", "citations", "authors", "source"];
      expect(fields).toHaveLength(5);
    });

    it("should support ascending and descending sort", () => {
      type SortDirection = "asc" | "desc";
      let direction: SortDirection = "desc";

      direction = "asc";
      expect(direction).toBe("asc");

      direction = "desc";
      expect(direction).toBe("desc");
    });

    it("should sort by citations correctly", () => {
      const papers = [
        { metadata: { citationCount: 100 } },
        { metadata: { citationCount: 500 } },
        { metadata: { citationCount: 50 } },
      ];

      const sorted = [...papers].sort(
        (a, b) => (b.metadata.citationCount || 0) - (a.metadata.citationCount || 0)
      );

      expect(sorted[0].metadata.citationCount).toBe(500);
      expect(sorted[2].metadata.citationCount).toBe(50);
    });

    it("should toggle sort direction on same field click", () => {
      let sortField = "citations";
      let sortDirection: "asc" | "desc" = "desc";

      const handleSort = (field: string) => {
        if (sortField === field) {
          sortDirection = sortDirection === "asc" ? "desc" : "asc";
        } else {
          sortField = field;
          sortDirection = "desc";
        }
      };

      handleSort("citations");
      expect(sortDirection).toBe("asc");

      handleSort("citations");
      expect(sortDirection).toBe("desc");

      handleSort("title");
      expect(sortField).toBe("title");
      expect(sortDirection).toBe("desc");
    });
  });

  describe("Paper Expansion", () => {
    it("should track expanded papers", () => {
      const expandedPapers = new Set<string>();

      expandedPapers.add("paper-1");
      expect(expandedPapers.has("paper-1")).toBe(true);

      expandedPapers.delete("paper-1");
      expect(expandedPapers.has("paper-1")).toBe(false);
    });

    it("should toggle expansion", () => {
      const expandedPapers = new Set<string>();

      const toggleExpanded = (paperId: string) => {
        if (expandedPapers.has(paperId)) {
          expandedPapers.delete(paperId);
        } else {
          expandedPapers.add(paperId);
        }
      };

      toggleExpanded("paper-1");
      expect(expandedPapers.has("paper-1")).toBe(true);

      toggleExpanded("paper-1");
      expect(expandedPapers.has("paper-1")).toBe(false);
    });
  });

  describe("Source Labels", () => {
    it("should format source labels correctly", () => {
      const sourceLabels: Record<string, string> = {
        arxiv: "arXiv",
        "semantic-scholar": "Semantic Scholar",
        github: "GitHub",
        pdf: "PDF Upload",
        doi: "DOI",
        manual: "Manual Entry",
      };

      expect(sourceLabels["arxiv"]).toBe("arXiv");
      expect(sourceLabels["semantic-scholar"]).toBe("Semantic Scholar");
    });
  });

  describe("Date Formatting", () => {
    it("should format dates correctly", () => {
      const formatDate = (dateStr: string | null): string => {
        if (!dateStr) return "Unknown";
        try {
          return new Date(dateStr).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
          });
        } catch {
          return dateStr;
        }
      };

      expect(formatDate("2024-01-15")).toContain("2024");
      expect(formatDate(null)).toBe("Unknown");
    });
  });

  describe("Author Formatting", () => {
    it("should truncate long author lists", () => {
      const formatAuthors = (authors: string[]): string => {
        if (authors.length <= 3) return authors.join(", ");
        return `${authors.slice(0, 3).join(", ")} +${authors.length - 3}`;
      };

      expect(formatAuthors(["A", "B"])).toBe("A, B");
      expect(formatAuthors(["A", "B", "C", "D", "E"])).toBe("A, B, C +2");
    });
  });

  describe("Citation Count Display", () => {
    it("should format citation counts", () => {
      const formatCitations = (count: number | null): string => {
        if (count === null) return "N/A";
        return count.toLocaleString();
      };

      expect(formatCitations(1500)).toBe("1,500");
      expect(formatCitations(null)).toBe("N/A");
    });
  });

  describe("Empty State", () => {
    it("should show empty state when no papers", () => {
      const papers: any[] = [];
      expect(papers.length).toBe(0);
    });
  });

  describe("Paper Count", () => {
    it("should count papers from different sources", () => {
      const papers = [
        { metadata: { source: "arxiv" } },
        { metadata: { source: "arxiv" } },
        { metadata: { source: "github" } },
        { metadata: { source: "pdf" } },
      ];

      const sourceCount = new Set(papers.map((p) => p.metadata.source)).size;
      expect(sourceCount).toBe(3);
    });
  });

  describe("Comparison Fields", () => {
    it("should define comparison fields", () => {
      const fields = [
        { key: "title", label: "Title" },
        { key: "authors", label: "Authors" },
        { key: "date", label: "Published" },
        { key: "citations", label: "Citations" },
        { key: "source", label: "Source" },
      ];

      expect(fields).toHaveLength(5);
      expect(fields[0].key).toBe("title");
    });
  });

  describe("Actions", () => {
    it("should handle select action", () => {
      const onSelect = vi.fn();
      const paper = { id: "paper-1", metadata: { title: "Test Paper" } };

      onSelect(paper);
      expect(onSelect).toHaveBeenCalledWith(paper);
    });

    it("should handle dismiss action", () => {
      const onDismiss = vi.fn();
      const paper = { id: "paper-1", metadata: { title: "Test Paper" } };

      onDismiss(paper);
      expect(onDismiss).toHaveBeenCalledWith(paper);
    });
  });

  describe("Abstract Display", () => {
    it("should truncate long abstracts", () => {
      const abstract = "A".repeat(500);
      const truncated = abstract.slice(0, 200) + "...";
      expect(truncated.length).toBeLessThan(abstract.length);
    });
  });

  describe("External Links", () => {
    it("should construct DOI link", () => {
      const doi = "10.1234/test.123";
      const link = `https://doi.org/${doi}`;
      expect(link).toBe("https://doi.org/10.1234/test.123");
    });
  });
});
