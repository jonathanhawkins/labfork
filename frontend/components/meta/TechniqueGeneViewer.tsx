"use client";

/**
 * TechniqueGeneViewer Component
 *
 * Visualizes the genetic representation of a technique, showing its genes
 * and allowing comparison between chromosomes.
 */

import React, { useState, useMemo } from "react";
import {
  Chromosome,
  Gene,
  GeneType,
  MutationRecord,
  getGeneTypeLabel,
  getGeneTypeColor,
} from "@/lib/meta/evolution";

interface TechniqueGeneViewerProps {
  chromosome: Chromosome;
  compareWith?: Chromosome;
  showMutations?: boolean;
  onGeneClick?: (geneId: string) => void;
  className?: string;
}

export function TechniqueGeneViewer({
  chromosome,
  compareWith,
  showMutations = true,
  onGeneClick,
  className = "",
}: TechniqueGeneViewerProps) {
  const [expandedGenes, setExpandedGenes] = useState<Set<string>>(new Set());
  const [filterType, setFilterType] = useState<GeneType | "all">("all");

  // Group genes by type
  const groupedGenes = useMemo(() => {
    const groups: Record<GeneType, Gene[]> = {
      architecture: [],
      training: [],
      conditioning: [],
      loss: [],
      hyperparameter: [],
      data: [],
      performance: [],
    };

    chromosome.genes.forEach((gene) => {
      groups[gene.type].push(gene);
    });

    return groups;
  }, [chromosome.genes]);

  // Get comparison data
  const getCompareGene = (geneId: string): Gene | undefined => {
    return compareWith?.genes.find((g) => g.name === chromosome.genes.find(g2 => g2.id === geneId)?.name);
  };

  const toggleGene = (geneId: string) => {
    const newExpanded = new Set(expandedGenes);
    if (newExpanded.has(geneId)) {
      newExpanded.delete(geneId);
    } else {
      newExpanded.add(geneId);
    }
    setExpandedGenes(newExpanded);
  };

  const filteredGenes = filterType === "all"
    ? chromosome.genes
    : chromosome.genes.filter((g) => g.type === filterType);

  return (
    <div className={`bg-white border rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-lg">{chromosome.name}</h3>
            <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
              <span>Generation {chromosome.generation}</span>
              <span>Fitness: {(chromosome.fitness * 100).toFixed(1)}%</span>
              <span>{chromosome.genes.length} genes</span>
            </div>
          </div>
          {chromosome.isElite && (
            <span className="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm font-medium rounded-full">
              Elite
            </span>
          )}
        </div>
      </div>

      {/* Filter */}
      <div className="px-4 py-3 bg-gray-50 border-b">
        <div className="flex items-center gap-2 overflow-x-auto">
          <span className="text-sm text-gray-500 flex-shrink-0">Filter:</span>
          <button
            onClick={() => setFilterType("all")}
            className={`px-3 py-1 text-sm rounded-full ${
              filterType === "all"
                ? "bg-gray-800 text-white"
                : "bg-gray-200 text-gray-700 hover:bg-gray-300"
            }`}
          >
            All
          </button>
          {(Object.keys(groupedGenes) as GeneType[]).map((type) => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-3 py-1 text-sm rounded-full flex items-center gap-1 ${
                filterType === type
                  ? "text-white"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
              style={{
                backgroundColor: filterType === type ? getGeneTypeColor(type) : undefined,
              }}
            >
              {getGeneTypeLabel(type)}
              <span className="text-xs opacity-70">({groupedGenes[type].length})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Genes Grid */}
      <div className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {filteredGenes.map((gene) => (
            <GeneCard
              key={gene.id}
              gene={gene}
              compareGene={compareWith ? getCompareGene(gene.id) : undefined}
              mutations={showMutations ? chromosome.mutations.filter((m) => m.geneId === gene.id) : []}
              expanded={expandedGenes.has(gene.id)}
              onToggle={() => toggleGene(gene.id)}
              onClick={() => onGeneClick?.(gene.id)}
            />
          ))}
        </div>
      </div>

      {/* Fitness Components */}
      <div className="px-4 py-3 border-t bg-gray-50">
        <div className="text-sm font-medium mb-2">Fitness Breakdown</div>
        <div className="grid grid-cols-5 gap-2">
          {Object.entries(chromosome.fitnessComponents).map(([key, value]) => (
            <div key={key} className="text-center">
              <div className="text-lg font-semibold text-gray-700">
                {Math.round(value * 100)}%
              </div>
              <div className="text-xs text-gray-500 capitalize">{key}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * Individual Gene Card
 */
interface GeneCardProps {
  gene: Gene;
  compareGene?: Gene;
  mutations: MutationRecord[];
  expanded: boolean;
  onToggle: () => void;
  onClick?: () => void;
}

function GeneCard({
  gene,
  compareGene,
  mutations,
  expanded,
  onToggle,
  onClick,
}: GeneCardProps) {
  const formatValue = (value: number | string | boolean): string => {
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (typeof value === "number") return value.toFixed(4);
    return String(value);
  };

  const hasChanged = compareGene && gene.value !== compareGene.value;

  return (
    <div
      className={`border rounded-lg overflow-hidden ${
        hasChanged ? "border-yellow-400" : ""
      }`}
    >
      <div
        className="p-3 flex items-center gap-3 cursor-pointer hover:bg-gray-50"
        onClick={onToggle}
      >
        <div
          className="w-3 h-3 rounded-full flex-shrink-0"
          style={{ backgroundColor: getGeneTypeColor(gene.type) }}
        />
        <div className="flex-1 min-w-0">
          <div className="font-medium text-sm truncate">{gene.name}</div>
          <div className="text-xs text-gray-500">{getGeneTypeLabel(gene.type)}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-sm">{formatValue(gene.value)}</div>
          {hasChanged && compareGene && (
            <div className="text-xs text-yellow-600">
              was: {formatValue(compareGene.value)}
            </div>
          )}
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </div>

      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t bg-gray-50 space-y-2">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Mutation Rate:</span>
              <span className="ml-1 font-medium">{(gene.mutationRate * 100).toFixed(0)}%</span>
            </div>
            <div>
              <span className="text-gray-500">Weight:</span>
              <span className="ml-1 font-medium">{gene.weight.toFixed(2)}</span>
            </div>
            {gene.minValue !== undefined && (
              <div>
                <span className="text-gray-500">Range:</span>
                <span className="ml-1 font-medium">
                  {gene.minValue} - {gene.maxValue}
                </span>
              </div>
            )}
            {gene.allowedValues && (
              <div className="col-span-2">
                <span className="text-gray-500">Options:</span>
                <span className="ml-1 font-medium">
                  {gene.allowedValues.map((v) => formatValue(v)).join(", ")}
                </span>
              </div>
            )}
            <div>
              <span className="text-gray-500">Dominant:</span>
              <span className="ml-1 font-medium">{gene.dominant ? "Yes" : "No"}</span>
            </div>
            {gene.sourceId && (
              <div className="col-span-2 truncate">
                <span className="text-gray-500">Source:</span>
                <span className="ml-1 font-medium font-mono text-xs">
                  {gene.sourceId.slice(0, 20)}...
                </span>
              </div>
            )}
          </div>

          {mutations.length > 0 && (
            <div className="pt-2 border-t">
              <div className="text-xs font-medium text-gray-700 mb-1">
                Mutation History ({mutations.length})
              </div>
              <div className="space-y-1">
                {mutations.slice(0, 3).map((mutation, idx) => (
                  <div
                    key={idx}
                    className="text-xs bg-white p-1.5 rounded border"
                  >
                    <span className="text-gray-500">{mutation.type}:</span>{" "}
                    <span className="font-mono">
                      {formatValue(mutation.originalValue)} →{" "}
                      {formatValue(mutation.newValue)}
                    </span>
                    {mutation.fitnessImpact !== undefined && (
                      <span
                        className={`ml-2 ${
                          mutation.fitnessImpact > 0
                            ? "text-green-600"
                            : mutation.fitnessImpact < 0
                            ? "text-red-600"
                            : "text-gray-500"
                        }`}
                      >
                        ({mutation.fitnessImpact > 0 ? "+" : ""}
                        {(mutation.fitnessImpact * 100).toFixed(1)}%)
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {onClick && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClick();
              }}
              className="w-full mt-2 px-3 py-1.5 text-xs text-blue-600 bg-blue-50 rounded hover:bg-blue-100"
            >
              View Details
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Compact chromosome comparison view
 */
interface ChromosomeCompareProps {
  chromosomes: Chromosome[];
  onSelect?: (chromosomeId: string) => void;
  selectedId?: string;
  className?: string;
}

export function ChromosomeCompare({
  chromosomes,
  onSelect,
  selectedId,
  className = "",
}: ChromosomeCompareProps) {
  const sortedChromosomes = [...chromosomes].sort((a, b) => b.fitness - a.fitness);

  return (
    <div className={`space-y-2 ${className}`}>
      {sortedChromosomes.map((chr, idx) => (
        <div
          key={chr.id}
          className={`p-3 border rounded-lg cursor-pointer transition-colors ${
            selectedId === chr.id
              ? "border-blue-500 bg-blue-50"
              : "hover:bg-gray-50"
          }`}
          onClick={() => onSelect?.(chr.id)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-lg font-bold text-gray-400">#{idx + 1}</span>
              <div>
                <div className="font-medium">{chr.name}</div>
                <div className="text-xs text-gray-500">
                  Gen {chr.generation} | {chr.genes.length} genes
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xl font-bold text-blue-600">
                {(chr.fitness * 100).toFixed(1)}%
              </div>
              {chr.isElite && (
                <span className="text-xs text-yellow-600">Elite</span>
              )}
            </div>
          </div>

          {/* Mini fitness bar */}
          <div className="mt-2 flex gap-1">
            {Object.entries(chr.fitnessComponents).map(([key, value]) => (
              <div
                key={key}
                className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden"
                title={`${key}: ${Math.round(value * 100)}%`}
              >
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${value * 100}%` }}
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/**
 * Gene distribution chart
 */
interface GeneDistributionProps {
  chromosome: Chromosome;
  className?: string;
}

export function GeneDistribution({ chromosome, className = "" }: GeneDistributionProps) {
  const distribution = useMemo(() => {
    const counts: Record<GeneType, number> = {
      architecture: 0,
      training: 0,
      conditioning: 0,
      loss: 0,
      hyperparameter: 0,
      data: 0,
      performance: 0,
    };

    chromosome.genes.forEach((gene) => {
      counts[gene.type]++;
    });

    return Object.entries(counts)
      .filter(([_, count]) => count > 0)
      .map(([type, count]) => ({
        type: type as GeneType,
        count,
        percentage: (count / chromosome.genes.length) * 100,
      }));
  }, [chromosome.genes]);

  return (
    <div className={`bg-white border rounded-lg p-4 ${className}`}>
      <h4 className="font-medium mb-3">Gene Distribution</h4>
      <div className="space-y-2">
        {distribution.map(({ type, count, percentage }) => (
          <div key={type} className="flex items-center gap-3">
            <div
              className="w-3 h-3 rounded-full flex-shrink-0"
              style={{ backgroundColor: getGeneTypeColor(type) }}
            />
            <div className="flex-1">
              <div className="flex justify-between text-sm">
                <span>{getGeneTypeLabel(type)}</span>
                <span className="text-gray-500">{count}</span>
              </div>
              <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden mt-1">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: getGeneTypeColor(type),
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default TechniqueGeneViewer;
