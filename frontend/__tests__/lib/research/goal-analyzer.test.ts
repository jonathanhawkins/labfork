/**
 * Tests for Research Goal Analyzer
 */

import { describe, it, expect } from "vitest";
import {
  parseGoal,
  matchDomains,
  suggestPapers,
  generatePlan,
  estimateResources,
  assessGoalComplexity,
  extractGoalTechniques,
  calculateConfidence,
  analyzeGoal,
  RESEARCH_DOMAINS,
  STEP_TYPE_LABELS,
  RESOURCE_TYPE_LABELS,
  PAPER_CATEGORY_LABELS,
} from "@/lib/research/goal-analyzer";

describe("Goal Analyzer", () => {
  describe("parseGoal", () => {
    it("extracts objective from goal", () => {
      const goal = "I want to build a voice cloning system that preserves emotion";
      const result = parseGoal(goal);
      expect(result.objective).toBeTruthy();
      expect(result.objective.length).toBeGreaterThan(5);
    });

    it("extracts techniques mentioned", () => {
      const goal = "Build a TTS system using transformers and attention mechanisms";
      const result = parseGoal(goal);
      expect(result.techniques.length).toBeGreaterThan(0);
    });

    it("identifies domain keywords", () => {
      const goal = "Create a voice cloning model for speaker adaptation";
      const result = parseGoal(goal);
      expect(result.domain).toBe("voice-clone");
    });

    it("extracts outcomes", () => {
      const goal = "Improve prosody quality to achieve more natural speech";
      const result = parseGoal(goal);
      expect(result.outcomes.length).toBeGreaterThan(0);
    });

    it("extracts constraints", () => {
      const goal = "Build a TTS system without requiring large datasets";
      const result = parseGoal(goal);
      expect(result.constraints.length).toBeGreaterThan(0);
    });

    it("filters stop words from keywords", () => {
      const goal = "I want to build a voice cloning system";
      const result = parseGoal(goal);
      expect(result.keywords).not.toContain("the");
      expect(result.keywords).not.toContain("a");
      expect(result.keywords).not.toContain("to");
    });
  });

  describe("matchDomains", () => {
    it("matches voice cloning domain", () => {
      const goal = "Build a voice cloning system with speaker embedding";
      const domains = matchDomains(goal);
      expect(domains[0].slug).toBe("voice-clone");
      expect(domains[0].matchScore).toBeGreaterThan(0);
    });

    it("matches prosody domain", () => {
      const goal = "Improve emotion transfer and prosody in speech synthesis";
      const domains = matchDomains(goal);
      expect(domains[0].slug).toBe("prosody");
    });

    it("matches TTS domain", () => {
      const goal = "Develop a neural TTS system with a vocoder";
      const domains = matchDomains(goal);
      expect(domains[0].slug).toBe("tts");
    });

    it("matches NLP domain", () => {
      const goal = "Fine-tune a language model for text generation using transformers";
      const domains = matchDomains(goal);
      expect(domains[0].slug).toBe("nlp");
    });

    it("matches vision domain", () => {
      const goal = "Build an image classification system using CNN";
      const domains = matchDomains(goal);
      expect(domains[0].slug).toBe("vision");
    });

    it("returns multiple domain matches sorted by score", () => {
      const goal = "Build a multimodal voice cloning system";
      const domains = matchDomains(goal);
      expect(domains.length).toBeGreaterThan(1);
      // Should be sorted by matchScore descending
      for (let i = 1; i < domains.length; i++) {
        expect(domains[i - 1].matchScore).toBeGreaterThanOrEqual(domains[i].matchScore);
      }
    });

    it("returns default domains for unrecognized goals", () => {
      const goal = "Do something interesting";
      const domains = matchDomains(goal);
      expect(domains.length).toBeGreaterThan(0);
    });
  });

  describe("suggestPapers", () => {
    it("suggests foundational papers", () => {
      const parseResult = {
        objective: "Build voice cloning",
        domain: "voice-clone",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: ["voice", "cloning"],
      };
      const domains = [{ slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 }];

      const suggestions = suggestPapers(parseResult, domains);
      expect(suggestions.some(s => s.category === "foundational")).toBe(true);
    });

    it("suggests technique-specific papers", () => {
      const parseResult = {
        objective: "Build TTS",
        domain: "tts",
        techniques: ["FastSpeech", "VITS"],
        outcomes: [],
        constraints: [],
        keywords: ["tts"],
      };
      const domains = [{ slug: "tts", name: "TTS", description: "", keywords: [], matchScore: 0.8 }];

      const suggestions = suggestPapers(parseResult, domains);
      expect(suggestions.some(s => s.category === "technique")).toBe(true);
    });

    it("suggests recent papers", () => {
      const parseResult = {
        objective: "Research NLP",
        domain: "nlp",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: ["nlp"],
      };
      const domains = [{ slug: "nlp", name: "NLP", description: "", keywords: [], matchScore: 0.8 }];

      const suggestions = suggestPapers(parseResult, domains);
      expect(suggestions.some(s => s.category === "recent")).toBe(true);
    });

    it("limits suggestions to 10", () => {
      const parseResult = {
        objective: "Build voice cloning",
        domain: "voice-clone",
        techniques: ["A", "B", "C", "D", "E"],
        outcomes: ["X", "Y", "Z"],
        constraints: [],
        keywords: [],
      };
      const domains = [{ slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 }];

      const suggestions = suggestPapers(parseResult, domains);
      expect(suggestions.length).toBeLessThanOrEqual(10);
    });
  });

  describe("generatePlan", () => {
    it("generates a complete research plan", () => {
      const parseResult = {
        objective: "Build voice cloning system",
        domain: "voice-clone",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 };

      const plan = generatePlan(parseResult, domain, []);

      expect(plan.title).toBeTruthy();
      expect(plan.summary).toBeTruthy();
      expect(plan.steps.length).toBeGreaterThan(0);
      expect(plan.totalHours).toBeGreaterThan(0);
      expect(plan.timelineWeeks).toBeGreaterThan(0);
      expect(plan.milestones.length).toBeGreaterThan(0);
    });

    it("includes all required step types", () => {
      const parseResult = {
        objective: "Research project",
        domain: "nlp",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "nlp", name: "NLP", description: "", keywords: [], matchScore: 0.8 };

      const plan = generatePlan(parseResult, domain, []);

      const stepTypes = new Set(plan.steps.map(s => s.type));
      expect(stepTypes.has("research")).toBe(true);
      expect(stepTypes.has("implementation")).toBe(true);
      expect(stepTypes.has("evaluation")).toBe(true);
    });

    it("sets up step dependencies correctly", () => {
      const parseResult = {
        objective: "Research project",
        domain: "nlp",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "nlp", name: "NLP", description: "", keywords: [], matchScore: 0.8 };

      const plan = generatePlan(parseResult, domain, []);

      // First step should have no dependencies
      expect(plan.steps[0].dependencies).toHaveLength(0);

      // Later steps should depend on earlier ones
      for (let i = 1; i < plan.steps.length; i++) {
        expect(plan.steps[i].dependencies.some(d => d < plan.steps[i].order)).toBe(true);
      }
    });

    it("includes technique names in implementation step", () => {
      const parseResult = {
        objective: "Build model",
        domain: "tts",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "tts", name: "TTS", description: "", keywords: [], matchScore: 0.8 };
      const techniques = [
        { name: "FastSpeech", description: "Fast TTS", isMainContribution: false },
        { name: "Transformer", description: "Attention", isMainContribution: false },
      ];

      const plan = generatePlan(parseResult, domain, techniques);

      const implStep = plan.steps.find(s => s.title === "Proposed Method Implementation");
      expect(implStep?.description).toContain("FastSpeech");
    });
  });

  describe("estimateResources", () => {
    it("estimates GPU resources for compute-intensive domains", () => {
      const parseResult = {
        objective: "Build voice cloning",
        domain: "voice-clone",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 };
      const plan = { totalHours: 50, timelineWeeks: 3, steps: [], milestones: [], title: "", summary: "" };

      const resources = estimateResources(parseResult, domain, plan);

      const gpuResource = resources.find(r => r.name === "GPU");
      expect(gpuResource).toBeDefined();
      expect(gpuResource?.isCritical).toBe(true);
    });

    it("estimates time based on plan", () => {
      const parseResult = {
        objective: "Research project",
        domain: "nlp",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "nlp", name: "NLP", description: "", keywords: [], matchScore: 0.8 };
      const plan = { totalHours: 96, timelineWeeks: 5, steps: [], milestones: [], title: "", summary: "" };

      const resources = estimateResources(parseResult, domain, plan);

      const timeResource = resources.find(r => r.type === "time");
      expect(timeResource).toBeDefined();
      expect(timeResource?.estimate).toContain("96 hours");
      expect(timeResource?.estimate).toContain("5 weeks");
    });

    it("includes expertise requirements", () => {
      const parseResult = {
        objective: "Build model",
        domain: "vision",
        techniques: ["CNN", "ResNet"],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "vision", name: "Computer Vision", description: "", keywords: [], matchScore: 0.8 };
      const plan = { totalHours: 50, timelineWeeks: 3, steps: [], milestones: [], title: "", summary: "" };

      const resources = estimateResources(parseResult, domain, plan);

      const expertiseResource = resources.find(r => r.type === "expertise");
      expect(expertiseResource).toBeDefined();
      expect(expertiseResource?.estimate).toContain("Computer Vision");
    });
  });

  describe("assessGoalComplexity", () => {
    it("returns simple for basic goals", () => {
      const parseResult = {
        objective: "Build a model",
        domain: "tts",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "tts", name: "TTS", description: "", keywords: [], matchScore: 0.8 };
      const plan = { totalHours: 40, timelineWeeks: 2, steps: [], milestones: [], title: "", summary: "" };

      const { complexity } = assessGoalComplexity(parseResult, domain, plan);
      expect(complexity).toBe("simple");
    });

    it("returns higher complexity for many techniques", () => {
      const parseResult = {
        objective: "Build complex model",
        domain: "multimodal",
        techniques: ["A", "B", "C", "D", "E"],
        outcomes: ["X", "Y", "Z"],
        constraints: ["time limit"],
        keywords: [],
      };
      const domain = { slug: "multimodal", name: "Multimodal", description: "", keywords: [], matchScore: 0.8 };
      const plan = { totalHours: 100, timelineWeeks: 5, steps: [], milestones: [], title: "", summary: "" };

      const { complexity } = assessGoalComplexity(parseResult, domain, plan);
      expect(["complex", "research"]).toContain(complexity);
    });

    it("provides complexity reason", () => {
      const parseResult = {
        objective: "Build model",
        domain: "voice-clone",
        techniques: ["A", "B", "C", "D"],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 };
      const plan = { totalHours: 50, timelineWeeks: 3, steps: [], milestones: [], title: "", summary: "" };

      const { reason } = assessGoalComplexity(parseResult, domain, plan);
      expect(reason).toBeTruthy();
    });
  });

  describe("extractGoalTechniques", () => {
    it("extracts techniques from parsed goal", () => {
      const parseResult = {
        objective: "Build TTS",
        domain: "tts",
        techniques: ["VITS", "FastSpeech"],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "tts", name: "TTS", description: "", keywords: [], matchScore: 0.8 };

      const techniques = extractGoalTechniques(parseResult, domain);

      expect(techniques.some(t => t.name === "VITS")).toBe(true);
      expect(techniques.some(t => t.name === "FastSpeech")).toBe(true);
    });

    it("adds domain-specific default techniques", () => {
      const parseResult = {
        objective: "Build voice cloning",
        domain: "voice-clone",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 };

      const techniques = extractGoalTechniques(parseResult, domain);

      expect(techniques.some(t => t.name === "Speaker Embedding")).toBe(true);
    });

    it("avoids duplicate techniques", () => {
      const parseResult = {
        objective: "Build model",
        domain: "voice-clone",
        techniques: ["Speaker Embedding"],
        outcomes: [],
        constraints: [],
        keywords: [],
      };
      const domain = { slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 };

      const techniques = extractGoalTechniques(parseResult, domain);

      const speakerEmbeddingCount = techniques.filter(
        t => t.name.toLowerCase() === "speaker embedding"
      ).length;
      expect(speakerEmbeddingCount).toBe(1);
    });
  });

  describe("calculateConfidence", () => {
    it("returns high confidence for well-defined goals", () => {
      const parseResult = {
        objective: "Build a voice cloning system with emotion preservation",
        domain: "voice-clone",
        techniques: ["VITS", "Emotion Embedding"],
        outcomes: ["natural speech", "emotion accuracy"],
        constraints: [],
        keywords: ["voice", "cloning", "emotion", "speech", "model", "system"],
      };
      const domains = [{ slug: "voice-clone", name: "Voice Cloning", description: "", keywords: [], matchScore: 0.8 }];

      const confidence = calculateConfidence(parseResult, domains);
      expect(confidence).toBeGreaterThanOrEqual(0.7);
    });

    it("returns lower confidence for vague goals", () => {
      const parseResult = {
        objective: "Do ML",
        domain: "",
        techniques: [],
        outcomes: [],
        constraints: [],
        keywords: ["ml"],
      };
      const domains = [{ slug: "nlp", name: "NLP", description: "", keywords: [], matchScore: 0.1 }];

      const confidence = calculateConfidence(parseResult, domains);
      expect(confidence).toBeLessThan(0.5);
    });
  });

  describe("analyzeGoal", () => {
    it("performs complete goal analysis", () => {
      const goal = "I want to build a voice cloning system that can transfer emotion from one speaker to another using neural TTS";

      const analysis = analyzeGoal(goal);

      expect(analysis.originalGoal).toBe(goal);
      expect(analysis.concepts.length).toBeGreaterThan(0);
      expect(analysis.recommendedDomain).toBeDefined();
      expect(analysis.paperSuggestions.length).toBeGreaterThan(0);
      expect(analysis.plan.steps.length).toBeGreaterThan(0);
      expect(analysis.resources.length).toBeGreaterThan(0);
      expect(["simple", "moderate", "complex", "research"]).toContain(analysis.complexity);
      expect(analysis.confidence).toBeGreaterThan(0);
      expect(analysis.analyzedAt).toBeDefined();
    });

    it("identifies correct domain for voice cloning goal", () => {
      const goal = "Create a zero-shot voice cloning model with speaker embedding";

      const analysis = analyzeGoal(goal);

      expect(analysis.recommendedDomain.slug).toBe("voice-clone");
    });

    it("identifies correct domain for prosody goal", () => {
      const goal = "Improve emotion transfer and prosody control in expressive TTS";

      const analysis = analyzeGoal(goal);

      expect(["prosody", "tts"]).toContain(analysis.recommendedDomain.slug);
    });

    it("provides alternative domains", () => {
      const goal = "Build a neural TTS system with voice cloning capabilities";

      const analysis = analyzeGoal(goal);

      expect(analysis.alternativeDomains.length).toBeGreaterThan(0);
    });

    it("generates actionable plan", () => {
      const goal = "Implement a transformer-based language model for code generation";

      const analysis = analyzeGoal(goal);

      expect(analysis.plan.steps.length).toBeGreaterThan(5);
      expect(analysis.plan.totalHours).toBeGreaterThan(0);
      expect(analysis.plan.milestones.length).toBeGreaterThan(0);
    });
  });

  describe("Constants", () => {
    it("has all research domains defined", () => {
      expect(RESEARCH_DOMAINS.length).toBeGreaterThan(5);
      for (const domain of RESEARCH_DOMAINS) {
        expect(domain.slug).toBeTruthy();
        expect(domain.name).toBeTruthy();
        expect(domain.keywords.length).toBeGreaterThan(0);
      }
    });

    it("has step type labels", () => {
      expect(STEP_TYPE_LABELS.research).toBe("Research");
      expect(STEP_TYPE_LABELS.implementation).toBe("Implementation");
      expect(STEP_TYPE_LABELS.evaluation).toBe("Evaluation");
    });

    it("has resource type labels", () => {
      expect(RESOURCE_TYPE_LABELS.compute).toBe("Compute");
      expect(RESOURCE_TYPE_LABELS.data).toBe("Data");
      expect(RESOURCE_TYPE_LABELS.time).toBe("Time");
    });

    it("has paper category labels", () => {
      expect(PAPER_CATEGORY_LABELS.foundational).toBe("Foundational");
      expect(PAPER_CATEGORY_LABELS.recent).toBe("Recent");
      expect(PAPER_CATEGORY_LABELS.technique).toBe("Technique");
    });
  });
});
