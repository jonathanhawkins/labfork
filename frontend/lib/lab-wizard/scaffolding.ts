/**
 * Lab Scaffolding Service
 *
 * Creates lab directory structure and configuration files.
 */

import type {
  LabConfig,
  HardwareConfig,
  ResearchConfig,
  InitialTask,
} from "./types";
import type { DomainConfig } from "@/lib/domain/types";

/**
 * Generated domain.yaml content
 */
export interface DomainYamlContent {
  /** Raw YAML string */
  yaml: string;
  /** Parsed config object */
  config: DomainConfig;
}

/**
 * Lab scaffolding result
 */
export interface ScaffoldingResult {
  success: boolean;
  /** Created lab ID */
  labId?: string;
  /** Domain slug */
  domainSlug?: string;
  /** Created files */
  files?: string[];
  /** Created tasks IDs */
  taskIds?: string[];
  /** Error message */
  error?: string;
}

/**
 * Generate domain.yaml from domain config
 */
export function generateDomainYaml(domain: Partial<DomainConfig>): string {
  if (!domain?.slug) {
    throw new Error("Domain slug is required");
  }

  const domainConfig: DomainConfig = {
    name: domain.name || "Research Lab",
    slug: domain.slug,
    description: domain.description || "AI Research Lab",
    difficulty: domain.difficulty || "intermediate",
    branding: {
      primaryColor: domain.branding?.primaryColor || "#3b82f6",
      accentColor: domain.branding?.accentColor || "#22c55e",
      backgroundStyle: domain.branding?.backgroundStyle || "sky",
    },
    scene: domain.scene || {
      props: [],
      decorations: {
        plants: true,
        floatingCubes: true,
        particles: true,
      },
    },
    research: domain.research || {
      arxivCategories: ["cs.LG"],
      keywords: ["machine learning"],
    },
    hardware: domain.hardware,
    version: "1.0",
  };

  return generateYamlString(domainConfig);
}

/**
 * Generate domain.yaml from wizard config (full config)
 */
export function generateDomainYamlFromConfig(config: LabConfig): DomainYamlContent {
  const domain = config.domain;
  if (!domain?.slug) {
    throw new Error("Domain slug is required");
  }

  const domainConfig: DomainConfig = {
    name: domain.name || "Research Lab",
    slug: domain.slug,
    description: domain.description || "AI Research Lab",
    difficulty: domain.difficulty || "intermediate",
    branding: {
      primaryColor: domain.branding?.primaryColor || "#3b82f6",
      accentColor: domain.branding?.accentColor || "#22c55e",
      backgroundStyle: domain.branding?.backgroundStyle || "sky",
    },
    scene: domain.scene || {
      props: [],
      decorations: {
        plants: true,
        floatingCubes: true,
        particles: true,
      },
    },
    research: domain.research || {
      arxivCategories: ["cs.LG"],
      keywords: ["machine learning"],
    },
    hardware: domain.hardware,
    version: "1.0",
  };

  // Generate YAML (simplified - in production use js-yaml)
  const yaml = generateYamlString(domainConfig);

  return {
    yaml,
    config: domainConfig,
  };
}

/**
 * Generate YAML string from config object
 */
export function generateYamlString(config: Record<string, any>): string {
  // Simple YAML serializer (for testing - production would use js-yaml)
  const lines: string[] = [];

  function serialize(obj: any, indent: number = 0): void {
    const prefix = "  ".repeat(indent);

    if (Array.isArray(obj)) {
      if (obj.length === 0) {
        lines.push("[]");
      } else {
        for (const item of obj) {
          if (typeof item === "object" && item !== null) {
            lines.push(`${prefix}-`);
            serialize(item, indent + 1);
          } else {
            lines.push(`${prefix}- ${item}`);
          }
        }
      }
    } else if (typeof obj === "object" && obj !== null) {
      for (const [key, value] of Object.entries(obj)) {
        if (value === undefined || value === null) continue;

        if (Array.isArray(value)) {
          if (value.length === 0) {
            lines.push(`${prefix}${key}: []`);
          } else {
            lines.push(`${prefix}${key}:`);
            serialize(value, indent + 1);
          }
        } else if (typeof value === "object") {
          lines.push(`${prefix}${key}:`);
          serialize(value, indent + 1);
        } else {
          lines.push(`${prefix}${key}: ${value}`);
        }
      }
    } else {
      lines.push(`${prefix}${obj}`);
    }
  }

  serialize(config);
  return lines.join("\n");
}

/**
 * Generate YAML string from DomainConfig object
 */
function generateDomainYamlString(config: DomainConfig): string {
  const lines: string[] = [];

  lines.push(`name: "${config.name}"`);
  lines.push(`slug: "${config.slug}"`);
  lines.push(`description: "${config.description}"`);
  lines.push(`difficulty: "${config.difficulty || "intermediate"}"`);
  lines.push("");

  lines.push("branding:");
  lines.push(`  primaryColor: "${config.branding.primaryColor}"`);
  lines.push(`  accentColor: "${config.branding.accentColor}"`);
  lines.push(`  backgroundStyle: "${config.branding.backgroundStyle}"`);
  lines.push("");

  lines.push("research:");
  lines.push("  arxivCategories:");
  for (const cat of config.research.arxivCategories) {
    lines.push(`    - "${cat}"`);
  }
  lines.push("  keywords:");
  for (const kw of config.research.keywords) {
    lines.push(`    - "${kw}"`);
  }
  lines.push("");

  lines.push("scene:");
  lines.push("  props: []");
  lines.push("  decorations:");
  lines.push("    plants: true");
  lines.push("    floatingCubes: true");
  lines.push("    particles: true");
  lines.push("");

  if (config.hardware) {
    lines.push("hardware:");
    if (config.hardware.minGpuVram) {
      lines.push(`  minGpuVram: ${config.hardware.minGpuVram}`);
    }
    if (config.hardware.recommendedGpuVram) {
      lines.push(`  recommendedGpuVram: ${config.hardware.recommendedGpuVram}`);
    }
    lines.push("");
  }

  lines.push(`version: "1.0"`);

  return lines.join("\n");
}

/**
 * Generate hardware configuration section
 */
export function generateHardwareSection(hardware: HardwareConfig): string {
  const lines: string[] = [];

  lines.push("# Hardware Configuration");
  lines.push(`type: ${hardware.type}`);

  if (hardware.type === "local" && hardware.local) {
    lines.push("local:");
    if (hardware.local.gpu) {
      lines.push(`  gpu: "${hardware.local.gpu.name}"`);
      lines.push(`  vram: ${hardware.local.gpu.vram}`);
    }
    if (hardware.local.ollamaInstalled) {
      lines.push(`  ollama: true`);
    }
  }

  if (hardware.type === "remote-ssh" && hardware.ssh) {
    lines.push("remote:");
    lines.push(`  host: "${hardware.ssh.host}"`);
    lines.push(`  user: "${hardware.ssh.user}"`);
    if (hardware.ssh.port && hardware.ssh.port !== 22) {
      lines.push(`  port: ${hardware.ssh.port}`);
    }
  }

  if (hardware.type === "cloud" && hardware.cloud) {
    lines.push("cloud:");
    lines.push(`  provider: "${hardware.cloud.provider}"`);
    if (hardware.cloud.region) {
      lines.push(`  region: "${hardware.cloud.region}"`);
    }
    if (hardware.cloud.instanceType) {
      lines.push(`  instanceType: "${hardware.cloud.instanceType}"`);
    }
  }

  return lines.join("\n");
}

/**
 * Generate initial tasks from research config
 */
export function generateInitialTasksFromConfig(
  config: LabConfig
): InitialTask[] {
  // Return goal tasks if they exist
  if (config.research.goal?.initialTasks && config.research.goal.initialTasks.length > 0) {
    return config.research.goal.initialTasks;
  }

  // Return empty array if no tasks from goal
  return [];
}

/**
 * Create lab via API
 */
export async function createLab(config: LabConfig): Promise<ScaffoldingResult> {
  try {
    const response = await fetch("/api/lab/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });

    const data = await response.json();

    if (data.success) {
      return {
        success: true,
        labId: data.labId,
        domainSlug: data.domainSlug,
        files: data.files,
        taskIds: data.taskIds,
      };
    }

    return {
      success: false,
      error: data.error || "Failed to create lab",
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to create lab",
    };
  }
}

/**
 * Validate lab config before creation
 */
export function validateLabConfig(config: LabConfig): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // Domain validation
  if (config.createNewDomain) {
    if (!config.domain?.slug) {
      errors.push("Domain slug is required");
    }
    if (!config.domain?.name) {
      errors.push("Domain name is required");
    }
  } else if (!config.existingDomainSlug) {
    errors.push("Must select an existing domain or create new one");
  }

  // Hardware validation
  if (config.hardware.type === "remote-ssh") {
    if (!config.hardware.ssh?.host) {
      errors.push("SSH host is required");
    }
    if (!config.hardware.ssh?.user) {
      errors.push("SSH username is required");
    }
    if (config.hardware.ssh?.testStatus !== "success") {
      errors.push("SSH connection must be tested successfully");
    }
  }

  if (config.hardware.type === "cloud") {
    if (!config.hardware.cloud?.provider) {
      errors.push("Cloud provider is required");
    }
    if (!config.hardware.cloud?.apiKey) {
      errors.push("Cloud API key is required");
    }
  }

  // Research validation
  if (config.research.path === "papers") {
    if (!config.research.papers || config.research.papers.length === 0) {
      errors.push("At least one paper is required");
    }
  } else if (config.research.path === "goal") {
    if (!config.research.goal?.goalText) {
      errors.push("Research goal is required");
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Get lab directory structure
 */
export function getLabDirectoryStructure(slug: string): {
  root: string;
  files: string[];
} {
  return {
    root: `.domains/${slug}`,
    files: [
      "domain.yaml",
      "prompts/research.md",
      "prompts/implementation.md",
      "prompts/evaluation.md",
    ],
  };
}

/**
 * Default prompt templates
 */
export const DEFAULT_PROMPTS = {
  research: `# Research Agent Prompt

You are a research assistant for {domain_name}.

## Your Role
- Find and analyze relevant papers
- Summarize key findings
- Identify promising techniques

## Guidelines
- Focus on practical, implementable approaches
- Consider hardware constraints
- Prioritize recent publications
`,

  implementation: `# Implementation Agent Prompt

You are an implementation assistant for {domain_name}.

## Your Role
- Write clean, well-documented code
- Follow project conventions
- Test implementations thoroughly

## Guidelines
- Start with minimal viable implementations
- Document assumptions and limitations
- Create reusable components
`,

  evaluation: `# Evaluation Agent Prompt

You are an evaluation assistant for {domain_name}.

## Your Role
- Design fair evaluation protocols
- Run benchmarks and tests
- Analyze results objectively

## Guidelines
- Use consistent metrics
- Compare against baselines
- Document methodology
`,
};

/**
 * Generate prompt from template
 */
export function generatePrompt(
  type: "research" | "implementation" | "evaluation",
  slug: string,
  domain: Partial<DomainConfig>
): string {
  const template = DEFAULT_PROMPTS[type] || DEFAULT_PROMPTS.research;
  return template
    .replace(/{domain_name}/g, domain.name || slug)
    .replace(/{domain_slug}/g, slug);
}

/**
 * Estimate lab creation time
 */
export function estimateCreationTime(config: LabConfig): {
  seconds: number;
  description: string;
} {
  let seconds = 2; // Base time

  if (config.createNewDomain) {
    seconds += 3; // Create domain files
  }

  if (config.research.goal?.initialTasks?.length) {
    seconds += config.research.goal.initialTasks.length * 0.5;
  }

  if (config.research.papers?.length) {
    seconds += config.research.papers.length * 0.5;
  }

  return {
    seconds,
    description:
      seconds < 5
        ? "Just a moment..."
        : `About ${Math.ceil(seconds)} seconds...`,
  };
}

/**
 * Mock scaffolding result for testing
 */
export const MOCK_SCAFFOLDING_RESULT: ScaffoldingResult = {
  success: true,
  labId: "lab_1234567890",
  domainSlug: "voice-clone",
  files: [
    ".domains/voice-clone/domain.yaml",
    ".domains/voice-clone/prompts/research.md",
    ".domains/voice-clone/prompts/implementation.md",
    ".domains/voice-clone/prompts/evaluation.md",
  ],
  taskIds: ["task_1", "task_2", "task_3"],
};
