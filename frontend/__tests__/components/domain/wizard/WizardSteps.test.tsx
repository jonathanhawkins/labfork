/**
 * Tests for Domain Wizard Step Components
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import {
  WizardStepTemplate,
  WizardStepBranding,
  WizardStepResearch,
  WizardStepScene,
  TEMPLATES,
  ARXIV_CATEGORIES,
} from "@/components/domain/wizard";

describe("WizardStepTemplate", () => {
  describe("rendering", () => {
    it("renders header and description", () => {
      render(
        <WizardStepTemplate
          selectedTemplate={null}
          onSelectTemplate={() => {}}
        />
      );

      expect(screen.getByText("Choose a Template")).toBeDefined();
      expect(
        screen.getByText("Start with a template or create from scratch")
      ).toBeDefined();
    });

    it("renders all templates", () => {
      render(
        <WizardStepTemplate
          selectedTemplate={null}
          onSelectTemplate={() => {}}
        />
      );

      for (const template of TEMPLATES) {
        expect(screen.getByText(template.name)).toBeDefined();
      }
    });

    it("shows Blank Slate option", () => {
      render(
        <WizardStepTemplate
          selectedTemplate={null}
          onSelectTemplate={() => {}}
        />
      );

      expect(screen.getByText("Blank Slate")).toBeDefined();
      expect(
        screen.getByText("Start from scratch with minimal configuration")
      ).toBeDefined();
    });
  });

  describe("selection", () => {
    it("calls onSelectTemplate when template clicked", () => {
      const onSelect = vi.fn();
      render(
        <WizardStepTemplate
          selectedTemplate={null}
          onSelectTemplate={onSelect}
        />
      );

      fireEvent.click(screen.getByText("Voice Research"));
      expect(onSelect).toHaveBeenCalledWith("voice-research");
    });

    it("highlights selected template", () => {
      render(
        <WizardStepTemplate
          selectedTemplate="voice-research"
          onSelectTemplate={() => {}}
        />
      );

      // Find the container with Voice Research
      const voiceButton = screen.getByText("Voice Research").closest("button");
      expect(voiceButton?.className).toContain("ring");
    });
  });

  describe("template data", () => {
    it("templates have required fields", () => {
      for (const template of TEMPLATES) {
        expect(template.id).toBeDefined();
        expect(template.name).toBeDefined();
        expect(template.description).toBeDefined();
        expect(template.primaryColor).toMatch(/^#[0-9a-fA-F]{6}$/);
        expect(template.accentColor).toMatch(/^#[0-9a-fA-F]{6}$/);
        expect(["beginner", "intermediate", "advanced"]).toContain(
          template.difficulty
        );
      }
    });
  });
});

describe("WizardStepBranding", () => {
  const defaultBranding = {
    name: "",
    slug: "",
    description: "",
    primaryColor: "#4ecdc4",
    accentColor: "#66ffaa",
    difficulty: "intermediate" as const,
  };

  describe("rendering", () => {
    it("renders all input fields", () => {
      render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={() => {}}
        />
      );

      expect(screen.getByPlaceholderText("My Research Lab")).toBeDefined();
      expect(screen.getByPlaceholderText("my-research-lab")).toBeDefined();
      expect(
        screen.getByPlaceholderText(/brief description/)
      ).toBeDefined();
    });

    it("renders difficulty buttons", () => {
      render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={() => {}}
        />
      );

      expect(screen.getByText("Beginner")).toBeDefined();
      expect(screen.getByText("Intermediate")).toBeDefined();
      expect(screen.getByText("Advanced")).toBeDefined();
    });

    it("renders color presets", () => {
      const { container } = render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={() => {}}
        />
      );

      // Should have 8 color preset buttons
      const colorButtons = container.querySelectorAll('button[title]');
      expect(colorButtons.length).toBeGreaterThanOrEqual(8);
    });
  });

  describe("input handling", () => {
    it("updates name and auto-generates slug", () => {
      const onChange = vi.fn();
      render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={onChange}
        />
      );

      const nameInput = screen.getByPlaceholderText("My Research Lab");
      fireEvent.change(nameInput, { target: { value: "Test Domain" } });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Test Domain",
          slug: "test-domain",
        })
      );
    });

    it("updates description", () => {
      const onChange = vi.fn();
      render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={onChange}
        />
      );

      const descInput = screen.getByPlaceholderText(/brief description/);
      fireEvent.change(descInput, {
        target: { value: "A test description" },
      });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "A test description",
        })
      );
    });

    it("updates difficulty", () => {
      const onChange = vi.fn();
      render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={onChange}
        />
      );

      fireEvent.click(screen.getByText("Advanced"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          difficulty: "advanced",
        })
      );
    });
  });

  describe("validation", () => {
    it("displays error messages", () => {
      render(
        <WizardStepBranding
          branding={defaultBranding}
          onBrandingChange={() => {}}
          errors={{
            name: "Name is required",
            slug: "Invalid slug",
          }}
        />
      );

      expect(screen.getByText("Name is required")).toBeDefined();
      expect(screen.getByText("Invalid slug")).toBeDefined();
    });
  });
});

describe("WizardStepResearch", () => {
  const defaultResearch = {
    arxivCategories: [],
    keywords: [],
    tags: [],
  };

  describe("rendering", () => {
    it("renders arXiv categories", () => {
      render(
        <WizardStepResearch
          research={defaultResearch}
          onResearchChange={() => {}}
        />
      );

      for (const cat of ARXIV_CATEGORIES.slice(0, 3)) {
        expect(screen.getByText(cat.id)).toBeDefined();
      }
    });

    it("renders keyword input", () => {
      render(
        <WizardStepResearch
          research={defaultResearch}
          onResearchChange={() => {}}
        />
      );

      expect(screen.getByPlaceholderText("Add a keyword...")).toBeDefined();
    });

    it("renders suggested tags", () => {
      render(
        <WizardStepResearch
          research={defaultResearch}
          onResearchChange={() => {}}
        />
      );

      expect(screen.getByText("machine-learning")).toBeDefined();
      expect(screen.getByText("nlp")).toBeDefined();
    });
  });

  describe("category selection", () => {
    it("toggles arXiv category", () => {
      const onChange = vi.fn();
      render(
        <WizardStepResearch
          research={defaultResearch}
          onResearchChange={onChange}
        />
      );

      fireEvent.click(screen.getByText("cs.LG"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          arxivCategories: ["cs.LG"],
        })
      );
    });

    it("removes category when clicked again", () => {
      const onChange = vi.fn();
      render(
        <WizardStepResearch
          research={{ ...defaultResearch, arxivCategories: ["cs.LG"] }}
          onResearchChange={onChange}
        />
      );

      fireEvent.click(screen.getByText("cs.LG"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          arxivCategories: [],
        })
      );
    });
  });

  describe("keyword management", () => {
    it("adds keyword on Enter", () => {
      const onChange = vi.fn();
      render(
        <WizardStepResearch
          research={defaultResearch}
          onResearchChange={onChange}
        />
      );

      const input = screen.getByPlaceholderText("Add a keyword...");
      fireEvent.change(input, { target: { value: "test keyword" } });
      fireEvent.keyDown(input, { key: "Enter" });

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          keywords: ["test keyword"],
        })
      );
    });
  });

  describe("tag selection", () => {
    it("toggles tag", () => {
      const onChange = vi.fn();
      render(
        <WizardStepResearch
          research={defaultResearch}
          onResearchChange={onChange}
        />
      );

      fireEvent.click(screen.getByText("nlp"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          tags: ["nlp"],
        })
      );
    });
  });
});

describe("WizardStepScene", () => {
  const defaultScene = {
    backgroundStyle: "sky" as const,
    props: [],
    cameraAngle: "isometric" as const,
  };

  describe("rendering", () => {
    it("renders background style options", () => {
      render(
        <WizardStepScene scene={defaultScene} onSceneChange={() => {}} />
      );

      expect(screen.getByText("Gradient Sky")).toBeDefined();
      expect(screen.getByText("Grid Floor")).toBeDefined();
      expect(screen.getByText("Minimal")).toBeDefined();
    });

    it("renders camera angle options", () => {
      render(
        <WizardStepScene scene={defaultScene} onSceneChange={() => {}} />
      );

      expect(screen.getByText("Isometric")).toBeDefined();
      expect(screen.getByText("Front")).toBeDefined();
      expect(screen.getByText("Top Down")).toBeDefined();
      expect(screen.getByText("Orbit")).toBeDefined();
    });

    it("renders props by category", () => {
      render(
        <WizardStepScene scene={defaultScene} onSceneChange={() => {}} />
      );

      expect(screen.getByText("Microphone")).toBeDefined();
      expect(screen.getByText("Server Rack")).toBeDefined();
      expect(screen.getByText("Monitor")).toBeDefined();
    });
  });

  describe("background style selection", () => {
    it("changes background style", () => {
      const onChange = vi.fn();
      render(
        <WizardStepScene scene={defaultScene} onSceneChange={onChange} />
      );

      fireEvent.click(screen.getByText("Grid Floor"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          backgroundStyle: "grid",
        })
      );
    });
  });

  describe("camera angle selection", () => {
    it("changes camera angle", () => {
      const onChange = vi.fn();
      render(
        <WizardStepScene scene={defaultScene} onSceneChange={onChange} />
      );

      fireEvent.click(screen.getByText("Front"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          cameraAngle: "front",
        })
      );
    });
  });

  describe("prop selection", () => {
    it("toggles prop selection", () => {
      const onChange = vi.fn();
      render(
        <WizardStepScene scene={defaultScene} onSceneChange={onChange} />
      );

      fireEvent.click(screen.getByText("Microphone"));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          props: ["microphone"],
        })
      );
    });

    it("shows selected count", () => {
      render(
        <WizardStepScene
          scene={{ ...defaultScene, props: ["microphone", "gpu"] }}
          onSceneChange={() => {}}
        />
      );

      expect(screen.getByText("2 selected")).toBeDefined();
    });
  });
});
