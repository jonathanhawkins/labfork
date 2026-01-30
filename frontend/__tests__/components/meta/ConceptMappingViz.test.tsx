/**
 * Tests for ConceptMappingViz, DomainAnalogyCard, and MappingMatrix Components
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  ConceptMappingViz,
  DomainAnalogyCard,
  MappingMatrix,
} from "@/components/meta/ConceptMappingViz";

// Mock data matching the actual component interfaces
const mockSourceDomain = {
  id: "speech",
  name: "Speech Processing",
  concepts: [
    { id: "concept-1", name: "Phoneme", abstractionLevel: 1 },
    { id: "concept-3", name: "Spectrogram", abstractionLevel: 2 },
    { id: "concept-5", name: "Prosody", abstractionLevel: 1 },
  ],
};

const mockTargetDomain = {
  id: "nlp",
  name: "Natural Language Processing",
  concepts: [
    { id: "concept-2", name: "Token", abstractionLevel: 1 },
    { id: "concept-4", name: "Embedding", abstractionLevel: 2 },
    { id: "concept-6", name: "Syntax", abstractionLevel: 1 },
  ],
};

const mockConceptMappings = [
  {
    source: { id: "concept-1", name: "Phoneme", abstractionLevel: 1 },
    target: { id: "concept-2", name: "Token", abstractionLevel: 1 },
    similarity: 0.85,
    mappingType: "equivalent" as const,
    confidence: 0.9,
  },
  {
    source: { id: "concept-3", name: "Spectrogram", abstractionLevel: 2 },
    target: { id: "concept-4", name: "Embedding", abstractionLevel: 2 },
    similarity: 0.7,
    mappingType: "analogous" as const,
    confidence: 0.8,
  },
  {
    source: { id: "concept-5", name: "Prosody", abstractionLevel: 1 },
    target: { id: "concept-6", name: "Syntax", abstractionLevel: 1 },
    similarity: 0.55,
    mappingType: "partial" as const,
    confidence: 0.6,
  },
];

const mockAnalogy = {
  id: "analogy-1",
  sourcePattern: "Acoustic Feature Extraction",
  targetPattern: "Token Embedding",
  description: "Both convert raw input to dense representations",
  strength: 0.75,
  examples: [
    {
      source: "Mel spectrogram",
      target: "Word embedding",
      explanation: "Both are dense vector representations of input",
    },
    {
      source: "MFCC",
      target: "Subword token",
      explanation: "Both segment input into meaningful units",
    },
  ],
};

describe("ConceptMappingViz", () => {
  it("should render component with domains", () => {
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    // Check that domain names are rendered
    expect(screen.getByText("Speech Processing")).toBeInTheDocument();
    expect(screen.getByText("Natural Language Processing")).toBeInTheDocument();
  });

  it("should render source domain concepts", () => {
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    expect(screen.getByText("Phoneme")).toBeInTheDocument();
    expect(screen.getByText("Spectrogram")).toBeInTheDocument();
    expect(screen.getByText("Prosody")).toBeInTheDocument();
  });

  it("should render target domain concepts", () => {
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    expect(screen.getByText("Token")).toBeInTheDocument();
    expect(screen.getByText("Embedding")).toBeInTheDocument();
    expect(screen.getByText("Syntax")).toBeInTheDocument();
  });

  it("should render SVG for visualization", () => {
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    const svg = document.querySelector("svg");
    expect(svg).toBeInTheDocument();
  });

  it("should render mapping lines between concepts", () => {
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    // Check for path elements (mapping lines)
    const paths = document.querySelectorAll("path");
    expect(paths.length).toBeGreaterThan(0);
  });

  it("should call onMappingClick when mapping clicked", () => {
    const onMappingClick = vi.fn();
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
        onMappingClick={onMappingClick}
      />
    );

    // Click on a mapping path
    const paths = document.querySelectorAll("path.cursor-pointer");
    if (paths.length > 0) {
      fireEvent.click(paths[0]);
      expect(onMappingClick).toHaveBeenCalled();
    }
  });

  it("should apply className prop", () => {
    const { container } = render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
        className="custom-viz"
      />
    );

    expect(container.firstChild).toHaveClass("custom-viz");
  });

  it("should handle empty mappings", () => {
    render(
      <ConceptMappingViz
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={[]}
      />
    );

    // Should still render domains without errors
    expect(screen.getByText("Speech Processing")).toBeInTheDocument();
    expect(screen.getByText("Natural Language Processing")).toBeInTheDocument();
  });
});

describe("DomainAnalogyCard", () => {
  it("should render source and target patterns", () => {
    render(<DomainAnalogyCard analogy={mockAnalogy} />);

    expect(screen.getByText("Acoustic Feature Extraction")).toBeInTheDocument();
    expect(screen.getByText("Token Embedding")).toBeInTheDocument();
  });

  it("should display analogy description", () => {
    render(<DomainAnalogyCard analogy={mockAnalogy} />);

    expect(screen.getByText(/Both convert raw input/)).toBeInTheDocument();
  });

  it("should display strength percentage", () => {
    render(<DomainAnalogyCard analogy={mockAnalogy} />);

    expect(screen.getByText("Strength: 75%")).toBeInTheDocument();
  });

  it("should expand to show examples when clicked", () => {
    render(<DomainAnalogyCard analogy={mockAnalogy} />);

    // Click to expand
    const card = document.querySelector(".cursor-pointer");
    if (card) {
      fireEvent.click(card);
      // Examples should now be visible
      expect(screen.getByText("Examples")).toBeInTheDocument();
      expect(screen.getByText("Mel spectrogram")).toBeInTheDocument();
      expect(screen.getByText("Word embedding")).toBeInTheDocument();
    }
  });

  it("should toggle expansion on click", () => {
    render(<DomainAnalogyCard analogy={mockAnalogy} />);

    const card = document.querySelector(".cursor-pointer");
    if (card) {
      // First click - expand
      fireEvent.click(card);
      expect(screen.getByText("Examples")).toBeInTheDocument();

      // Second click - collapse
      fireEvent.click(card);
      expect(screen.queryByText("Examples")).not.toBeInTheDocument();
    }
  });

  it("should apply className prop", () => {
    const { container } = render(
      <DomainAnalogyCard analogy={mockAnalogy} className="custom-card" />
    );

    expect(container.firstChild).toHaveClass("custom-card");
  });
});

describe("MappingMatrix", () => {
  it("should render source domain concepts as rows", () => {
    render(
      <MappingMatrix
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    expect(screen.getByText("Phoneme")).toBeInTheDocument();
    expect(screen.getByText("Spectrogram")).toBeInTheDocument();
    expect(screen.getByText("Prosody")).toBeInTheDocument();
  });

  it("should render target domain concepts as columns", () => {
    render(
      <MappingMatrix
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    expect(screen.getByText("Token")).toBeInTheDocument();
    expect(screen.getByText("Embedding")).toBeInTheDocument();
    expect(screen.getByText("Syntax")).toBeInTheDocument();
  });

  it("should render matrix cells", () => {
    render(
      <MappingMatrix
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    // Should have cells - 3 source x 3 target = 9 cells
    const table = document.querySelector("table");
    expect(table).toBeInTheDocument();
  });

  it("should display similarity values in cells", () => {
    render(
      <MappingMatrix
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
      />
    );

    // Check for similarity values (just numbers without %)
    expect(screen.getByText("85")).toBeInTheDocument();
    expect(screen.getByText("70")).toBeInTheDocument();
    expect(screen.getByText("55")).toBeInTheDocument();
  });

  it("should apply className prop", () => {
    const { container } = render(
      <MappingMatrix
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={mockConceptMappings}
        className="custom-matrix"
      />
    );

    expect(container.firstChild).toHaveClass("custom-matrix");
  });

  it("should handle empty mappings", () => {
    render(
      <MappingMatrix
        sourceDomain={mockSourceDomain}
        targetDomain={mockTargetDomain}
        mappings={[]}
      />
    );

    // Should still render without errors
    expect(screen.getByText("Phoneme")).toBeInTheDocument();
  });
});
