/**
 * Knowledge Graph Module
 *
 * Exports all knowledge graph functionality for the meta-research
 * intelligence system.
 */

// Types
export * from "./types";

// Graph data structure
import { KnowledgeGraph as KnowledgeGraphClass, getGlobalGraph, resetGlobalGraph } from "./graph";
export { KnowledgeGraphClass as KnowledgeGraph, getGlobalGraph, resetGlobalGraph };

// Factory function for creating new knowledge graphs
export function createKnowledgeGraph(): KnowledgeGraphClass {
  return new KnowledgeGraphClass();
}

// Query engine
export {
  QueryEngine,
  NodeQueryBuilder,
  EdgeQueryBuilder,
  createQueryEngine,
} from "./query";
export type { QueryResult, AggregationResult } from "./query";

// Builders
export {
  TechniqueBuilder,
  createTechniqueBuilder,
  extractTechniquesFromPapers,
  createManualTechnique,
} from "./builders/technique-builder";
export type { TechniqueExtraction, TechniquePattern } from "./builders/technique-builder";
