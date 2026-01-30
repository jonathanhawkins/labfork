/**
 * Domain Configuration Tests
 *
 * Tests that all domain configurations are valid and complete.
 */

import { describe, it, expect, vi } from "vitest";

// Use require for Node.js modules to avoid vitest mocking issues
// eslint-disable-next-line @typescript-eslint/no-require-imports
const fs = require("fs");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const path = require("path");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const yaml = require("js-yaml");

// Domain slugs to test
const DOMAIN_SLUGS = [
  "voice-clone",
  "quant-trading",
  "game-ai",
  "robotics-ml",
  "drug-discovery",
  "climate-modeling",
  "nlp-research",
  "computer-vision",
  "biotech-nlp",
];

// Required fields in domain.yaml
const REQUIRED_FIELDS = [
  "name",
  "slug",
  "description",
  "difficulty",
  "branding",
  "scene",
  "research",
  "evaluation",
  "hardware",
];

const REQUIRED_BRANDING_FIELDS = [
  "primaryColor",
  "accentColor",
  "backgroundStyle",
];

const REQUIRED_RESEARCH_FIELDS = [
  "arxivCategories",
  "keywords",
];

const REQUIRED_EVALUATION_FIELDS = [
  "primaryMetric",
  "metrics",
];

const REQUIRED_HARDWARE_FIELDS = [
  "gpuRequired",
  "minGpuVram",
  "platforms",
];

// Valid difficulty levels
const VALID_DIFFICULTIES = ["beginner", "intermediate", "advanced", "expert"];

// Valid background styles
const VALID_BACKGROUND_STYLES = ["sky", "grid", "gradient", "dark", "space", "minimal"];

// Valid platforms
const VALID_PLATFORMS = ["darwin", "linux", "win32"];

function loadDomainConfig(slug: string): Record<string, unknown> | null {
  // Use process.cwd() to get the project root (frontend dir)
  // Then go up one level to the project root where .domains is
  const projectRoot = path.resolve(process.cwd(), "..");
  const domainsPath = path.join(projectRoot, ".domains");
  const configPath = path.join(domainsPath, slug, "domain.yaml");

  if (!fs.existsSync(configPath)) {
    return null;
  }

  const content = fs.readFileSync(configPath, "utf-8");
  return yaml.load(content) as Record<string, unknown>;
}

describe("Domain Configurations", () => {
  describe("All domains exist", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      it(`should have domain.yaml for ${slug}`, () => {
        const config = loadDomainConfig(slug);
        expect(config).not.toBeNull();
      });
    });
  });

  describe("Required fields", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      describe(slug, () => {
        const config = loadDomainConfig(slug);

        if (!config) {
          it.skip("domain not found", () => {});
          return;
        }

        REQUIRED_FIELDS.forEach((field) => {
          it(`should have ${field} field`, () => {
            expect(config).toHaveProperty(field);
          });
        });

        it("should have valid slug matching directory", () => {
          expect(config.slug).toBe(slug);
        });

        it("should have non-empty name", () => {
          expect(typeof config.name).toBe("string");
          expect((config.name as string).length).toBeGreaterThan(0);
        });

        it("should have non-empty description", () => {
          expect(typeof config.description).toBe("string");
          expect((config.description as string).length).toBeGreaterThan(0);
        });

        it("should have valid difficulty", () => {
          expect(VALID_DIFFICULTIES).toContain(config.difficulty);
        });
      });
    });
  });

  describe("Branding configuration", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      describe(slug, () => {
        const config = loadDomainConfig(slug);

        if (!config || !config.branding) {
          it.skip("branding not found", () => {});
          return;
        }

        const branding = config.branding as Record<string, unknown>;

        REQUIRED_BRANDING_FIELDS.forEach((field) => {
          it(`should have ${field} in branding`, () => {
            expect(branding).toHaveProperty(field);
          });
        });

        it("should have valid hex color for primaryColor", () => {
          const color = branding.primaryColor as string;
          expect(color).toMatch(/^#[0-9a-fA-F]{6}$/);
        });

        it("should have valid hex color for accentColor", () => {
          const color = branding.accentColor as string;
          expect(color).toMatch(/^#[0-9a-fA-F]{6}$/);
        });

        it("should have valid backgroundStyle", () => {
          expect(VALID_BACKGROUND_STYLES).toContain(branding.backgroundStyle);
        });
      });
    });
  });

  describe("Research configuration", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      describe(slug, () => {
        const config = loadDomainConfig(slug);

        if (!config || !config.research) {
          it.skip("research not found", () => {});
          return;
        }

        const research = config.research as Record<string, unknown>;

        REQUIRED_RESEARCH_FIELDS.forEach((field) => {
          it(`should have ${field} in research`, () => {
            expect(research).toHaveProperty(field);
          });
        });

        it("should have at least one arXiv category", () => {
          const categories = research.arxivCategories as string[];
          expect(Array.isArray(categories)).toBe(true);
          expect(categories.length).toBeGreaterThan(0);
        });

        it("should have at least 3 keywords", () => {
          const keywords = research.keywords as string[];
          expect(Array.isArray(keywords)).toBe(true);
          expect(keywords.length).toBeGreaterThanOrEqual(3);
        });

        it("should have valid arXiv category format", () => {
          const categories = research.arxivCategories as string[];
          categories.forEach((cat) => {
            // arXiv categories are like: cs.AI, q-fin.PM, physics.ao-ph, physics.chem-ph
            // Format: category.subcategory where subcategory can be 2+ chars with optional hyphen
            expect(cat).toMatch(/^[a-z-]+\.[a-zA-Z]{2,}(-[a-z]+)?$/);
          });
        });
      });
    });
  });

  describe("Evaluation configuration", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      describe(slug, () => {
        const config = loadDomainConfig(slug);

        if (!config || !config.evaluation) {
          it.skip("evaluation not found", () => {});
          return;
        }

        const evaluation = config.evaluation as Record<string, unknown>;

        REQUIRED_EVALUATION_FIELDS.forEach((field) => {
          it(`should have ${field} in evaluation`, () => {
            expect(evaluation).toHaveProperty(field);
          });
        });

        it("should have at least one metric", () => {
          const metrics = evaluation.metrics as unknown[];
          expect(Array.isArray(metrics)).toBe(true);
          expect(metrics.length).toBeGreaterThan(0);
        });

        it("should have primaryMetric matching a metric id", () => {
          const metrics = evaluation.metrics as Array<{ id: string }>;
          const metricIds = metrics.map((m) => m.id);
          expect(metricIds).toContain(evaluation.primaryMetric);
        });

        it("should have valid metric structure", () => {
          const metrics = evaluation.metrics as Array<Record<string, unknown>>;
          metrics.forEach((metric) => {
            expect(metric).toHaveProperty("id");
            expect(metric).toHaveProperty("name");
            expect(metric).toHaveProperty("description");
            expect(metric).toHaveProperty("range");
            expect(metric).toHaveProperty("higherIsBetter");
          });
        });
      });
    });
  });

  describe("Hardware configuration", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      describe(slug, () => {
        const config = loadDomainConfig(slug);

        if (!config || !config.hardware) {
          it.skip("hardware not found", () => {});
          return;
        }

        const hardware = config.hardware as Record<string, unknown>;

        REQUIRED_HARDWARE_FIELDS.forEach((field) => {
          it(`should have ${field} in hardware`, () => {
            expect(hardware).toHaveProperty(field);
          });
        });

        it("should have boolean gpuRequired", () => {
          expect(typeof hardware.gpuRequired).toBe("boolean");
        });

        it("should have positive minGpuVram", () => {
          const vram = hardware.minGpuVram as number;
          expect(typeof vram).toBe("number");
          expect(vram).toBeGreaterThan(0);
        });

        it("should have valid platforms", () => {
          const platforms = hardware.platforms as string[];
          expect(Array.isArray(platforms)).toBe(true);
          platforms.forEach((platform) => {
            expect(VALID_PLATFORMS).toContain(platform);
          });
        });
      });
    });
  });

  describe("Scene configuration", () => {
    DOMAIN_SLUGS.forEach((slug) => {
      describe(slug, () => {
        const config = loadDomainConfig(slug);

        if (!config || !config.scene) {
          it.skip("scene not found", () => {});
          return;
        }

        const scene = config.scene as Record<string, unknown>;

        it("should have props array", () => {
          expect(scene).toHaveProperty("props");
          expect(Array.isArray(scene.props)).toBe(true);
        });

        it("should have at least one prop", () => {
          const props = scene.props as unknown[];
          expect(props.length).toBeGreaterThan(0);
        });

        it("should have valid prop structure", () => {
          const props = scene.props as Array<Record<string, unknown>>;
          props.forEach((prop) => {
            expect(prop).toHaveProperty("id");
            expect(prop).toHaveProperty("type");
            expect(prop).toHaveProperty("position");
            expect(Array.isArray(prop.position)).toBe(true);
            expect((prop.position as number[]).length).toBe(3);
          });
        });
      });
    });
  });

  describe("Domain count", () => {
    it("should have exactly 9 domains configured", () => {
      const validDomains = DOMAIN_SLUGS.filter((slug) => {
        const config = loadDomainConfig(slug);
        return config !== null;
      });
      expect(validDomains.length).toBe(9);
    });
  });
});
