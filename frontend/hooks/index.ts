/**
 * Custom React Hooks
 *
 * Centralized exports for all custom hooks used in the application.
 */

// Agent Status Hooks
export {
  useAgentStatus,
  useAgentWorkLog,
  useSingleAgentStatus,
  workLogToActivityLog,
  getAgentStatusColor,
  getWorkingAgentsCount,
  type AgentStatus,
  type AgentDetailedStatus,
  type WorkLogEntry,
  type TaskSummary,
  type UseAgentStatusOptions,
  type UseAgentStatusResult,
  type UseAgentWorkLogOptions,
  type UseAgentWorkLogResult,
  type UseSingleAgentStatusOptions,
  type UseSingleAgentStatusResult,
} from './useAgentStatus';
