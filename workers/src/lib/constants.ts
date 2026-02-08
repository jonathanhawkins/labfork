/**
 * Shared constants for task and device status values.
 * Use these instead of hardcoded strings to prevent typos.
 */

export const TaskStatus = {
  PENDING: 'pending',
  ASSIGNED: 'assigned',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  FAILED: 'failed',
  BLOCKED: 'blocked',
  TIMEOUT: 'timeout',
} as const;

export type TaskStatusType = (typeof TaskStatus)[keyof typeof TaskStatus];

export const DeviceStatus = {
  ONLINE: 'online',
  BUSY: 'busy',
  OFFLINE: 'offline',
  PAUSED: 'paused',
} as const;

export type DeviceStatusType = (typeof DeviceStatus)[keyof typeof DeviceStatus];

export const AgentStatus = {
  IDLE: 'idle',
  WORKING: 'working',
  ERROR: 'error',
} as const;

export type AgentStatusType = (typeof AgentStatus)[keyof typeof AgentStatus];
