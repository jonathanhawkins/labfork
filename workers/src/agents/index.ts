/**
 * Agents Module
 *
 * Exports all agent-related types, personas, and utilities.
 */

export {
  // Types
  type AgentPersona,
  type Specialization,
  // Constants
  AGENT_PERSONAS,
  ALL_AGENTS,
  // Functions
  getAgentPersona,
  getAgentSystemPrompt,
  findAgentsForTopic,
  createCollaborationPrompt,
  // Default export
  default,
} from './personas';
