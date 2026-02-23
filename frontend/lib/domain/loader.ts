/**
 * Domain Configuration Loader
 *
 * Provides functions to load and validate domain configurations from YAML files.
 */

import * as yaml from 'yaml';
import * as fs from 'fs';
import * as path from 'path';
import {
  DomainConfig,
  isDomainConfig,
  mergeWithDefaults,
  DEFAULT_DOMAIN_CONFIG,
} from './types';

/**
 * Error thrown when domain configuration is invalid or missing
 */
export class DomainConfigError extends Error {
  constructor(
    message: string,
    public slug: string,
    public cause?: Error
  ) {
    super(message);
    this.name = 'DomainConfigError';
  }
}

/**
 * Get the path to the domains directory
 */
export function getDomainsPath(): string {
  // Try to find .domains relative to various locations
  const possiblePaths = [
    // From frontend directory (development)
    path.join(process.cwd(), '..', '.domains'),
    // From project root
    path.join(process.cwd(), '.domains'),
    // Absolute path fallback
    '/Users/light/dev/web-apps/labfork/.domains',
  ];

  for (const p of possiblePaths) {
    if (fs.existsSync(p)) {
      return p;
    }
  }

  // Default to relative path even if it doesn't exist yet
  return path.join(process.cwd(), '..', '.domains');
}

/**
 * Get the path to a specific domain's directory
 */
export function getDomainPath(slug: string): string {
  return path.join(getDomainsPath(), slug);
}

/**
 * Get the path to a domain's config file
 */
export function getDomainConfigPath(slug: string): string {
  return path.join(getDomainPath(slug), 'domain.yaml');
}

/**
 * Check if a domain exists
 */
export function domainExists(slug: string): boolean {
  const configPath = getDomainConfigPath(slug);
  return fs.existsSync(configPath);
}

/**
 * List all available domains
 */
export function listDomains(): string[] {
  const domainsPath = getDomainsPath();

  if (!fs.existsSync(domainsPath)) {
    return [];
  }

  return fs
    .readdirSync(domainsPath)
    .filter((name) => {
      // Skip hidden files and _template
      if (name.startsWith('.') || name.startsWith('_')) {
        return false;
      }

      // Check if it has a domain.yaml
      const configPath = path.join(domainsPath, name, 'domain.yaml');
      return fs.existsSync(configPath);
    });
}

/**
 * Load raw YAML content from domain config file
 */
export function loadDomainYaml(slug: string): Record<string, unknown> {
  const configPath = getDomainConfigPath(slug);

  if (!fs.existsSync(configPath)) {
    throw new DomainConfigError(
      `Domain config not found: ${configPath}`,
      slug
    );
  }

  try {
    const content = fs.readFileSync(configPath, 'utf-8');
    const parsed = yaml.parse(content);

    if (!parsed || typeof parsed !== 'object') {
      throw new DomainConfigError(
        `Invalid YAML in domain config: expected object, got ${typeof parsed}`,
        slug
      );
    }

    return parsed as Record<string, unknown>;
  } catch (error) {
    if (error instanceof DomainConfigError) {
      throw error;
    }
    throw new DomainConfigError(
      `Failed to parse domain config: ${error instanceof Error ? error.message : 'Unknown error'}`,
      slug,
      error instanceof Error ? error : undefined
    );
  }
}

/**
 * Validate a domain configuration object
 */
export function validateDomainConfig(config: unknown): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!config || typeof config !== 'object') {
    return { valid: false, errors: ['Config must be an object'] };
  }

  const c = config as Record<string, unknown>;

  // Required string fields
  if (typeof c.name !== 'string' || !c.name.trim()) {
    errors.push('name is required and must be a non-empty string');
  }
  if (typeof c.slug !== 'string' || !c.slug.trim()) {
    errors.push('slug is required and must be a non-empty string');
  } else if (!/^[a-z0-9-]+$/.test(c.slug as string)) {
    errors.push('slug must contain only lowercase letters, numbers, and hyphens');
  }
  if (typeof c.description !== 'string' || !c.description.trim()) {
    errors.push('description is required and must be a non-empty string');
  }

  // Required branding object
  if (!c.branding || typeof c.branding !== 'object') {
    errors.push('branding is required and must be an object');
  } else {
    const branding = c.branding as Record<string, unknown>;
    if (typeof branding.primaryColor !== 'string') {
      errors.push('branding.primaryColor is required');
    }
    if (typeof branding.accentColor !== 'string') {
      errors.push('branding.accentColor is required');
    }
  }

  // Required scene object
  if (!c.scene || typeof c.scene !== 'object') {
    errors.push('scene is required and must be an object');
  } else {
    const scene = c.scene as Record<string, unknown>;
    if (!Array.isArray(scene.props)) {
      errors.push('scene.props is required and must be an array');
    }
  }

  // Required research object
  if (!c.research || typeof c.research !== 'object') {
    errors.push('research is required and must be an object');
  } else {
    const research = c.research as Record<string, unknown>;
    if (!Array.isArray(research.arxivCategories)) {
      errors.push('research.arxivCategories is required and must be an array');
    }
    if (!Array.isArray(research.keywords)) {
      errors.push('research.keywords is required and must be an array');
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Load and validate a domain configuration
 *
 * @param slug - The domain slug (e.g., "voice-clone")
 * @returns The validated and merged domain configuration
 * @throws DomainConfigError if the config is missing or invalid
 */
export function loadDomainConfig(slug: string): DomainConfig {
  // Load raw YAML
  const rawConfig = loadDomainYaml(slug);

  // Validate
  const validation = validateDomainConfig(rawConfig);
  if (!validation.valid) {
    throw new DomainConfigError(
      `Invalid domain configuration:\n  - ${validation.errors.join('\n  - ')}`,
      slug
    );
  }

  // Merge with defaults
  const config = mergeWithDefaults(rawConfig as Partial<DomainConfig>);

  // Final type check
  if (!isDomainConfig(config)) {
    throw new DomainConfigError(
      'Domain configuration failed final type check',
      slug
    );
  }

  return config;
}

/**
 * Load a domain configuration, returning null if not found
 */
export function loadDomainConfigSafe(slug: string): DomainConfig | null {
  try {
    return loadDomainConfig(slug);
  } catch {
    return null;
  }
}

/**
 * Load a prompt template from a domain
 */
export function loadPromptTemplate(
  slug: string,
  templateName: 'research' | 'implementation' | 'evaluation'
): string | null {
  const domainPath = getDomainPath(slug);
  const templatePath = path.join(domainPath, 'prompts', `${templateName}.md`);

  if (!fs.existsSync(templatePath)) {
    return null;
  }

  try {
    return fs.readFileSync(templatePath, 'utf-8');
  } catch {
    return null;
  }
}

/**
 * Get default domain configuration (for when no domain is specified)
 */
export function getDefaultDomainConfig(): DomainConfig {
  return mergeWithDefaults({
    name: 'Default Lab',
    slug: 'default',
    description: 'A general-purpose research lab',
  });
}

/**
 * Async version of loadDomainConfig for use in React components
 */
export async function loadDomainConfigAsync(slug: string): Promise<DomainConfig> {
  // In a real async scenario, this might fetch from an API
  // For now, we just wrap the sync version
  return new Promise((resolve, reject) => {
    try {
      const config = loadDomainConfig(slug);
      resolve(config);
    } catch (error) {
      reject(error);
    }
  });
}
