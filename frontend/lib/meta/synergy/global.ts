/**
 * Global Synergy Discovery Instance
 *
 * Provides a singleton synergy discovery engine for the application.
 */

import { getGlobalGraph } from "../knowledge-graph";
import { SynergyDiscovery } from "./discovery";
import { SynergyProposal, ProposalStatus, ExploredCombination, SynergyDiscoveryConfig } from "./types";

let globalSynergyDiscovery: SynergyDiscovery | null = null;
let proposalStore: Map<string, SynergyProposal> = new Map();

/**
 * Get the global synergy discovery instance
 */
export function getGlobalSynergyDiscovery(): SynergyDiscovery {
  if (!globalSynergyDiscovery) {
    const graph = getGlobalGraph();
    globalSynergyDiscovery = new SynergyDiscovery(graph);
  }
  return globalSynergyDiscovery;
}

/**
 * Reset the global synergy discovery instance
 */
export function resetGlobalSynergyDiscovery(): void {
  globalSynergyDiscovery = null;
  proposalStore.clear();
}

/**
 * Update synergy discovery configuration
 */
export function updateSynergyConfig(
  config: Partial<SynergyDiscoveryConfig>
): void {
  const discovery = getGlobalSynergyDiscovery();
  discovery.updateConfig(config);
}

/**
 * Store a proposal
 */
export function storeProposal(proposal: SynergyProposal): void {
  proposalStore.set(proposal.id, proposal);
}

/**
 * Store multiple proposals
 */
export function storeProposals(proposals: SynergyProposal[]): void {
  for (const proposal of proposals) {
    proposalStore.set(proposal.id, proposal);
  }
}

/**
 * Get a proposal by ID
 */
export function getProposal(id: string): SynergyProposal | undefined {
  return proposalStore.get(id);
}

/**
 * Get all proposals
 */
export function getAllProposals(): SynergyProposal[] {
  return Array.from(proposalStore.values());
}

/**
 * Update a proposal's status
 */
export function updateProposalStatus(
  id: string,
  status: ProposalStatus,
  notes?: string
): SynergyProposal | undefined {
  const proposal = proposalStore.get(id);
  if (!proposal) return undefined;

  const updated: SynergyProposal = {
    ...proposal,
    status,
    updatedAt: new Date(),
    notes: notes || proposal.notes,
  };

  proposalStore.set(id, updated);
  return updated;
}

/**
 * Get proposals by status
 */
export function getProposalsByStatus(status: ProposalStatus): SynergyProposal[] {
  return getAllProposals().filter((p) => p.status === status);
}

/**
 * Get explored combinations
 */
export function getExploredCombinations(): ExploredCombination[] {
  const discovery = getGlobalSynergyDiscovery();
  return discovery.getExploredCombinations();
}
