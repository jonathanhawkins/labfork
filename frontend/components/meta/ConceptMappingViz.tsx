"use client";

/**
 * ConceptMappingViz Component
 *
 * Visualizes domain concept mappings showing relationships
 * between source and target domain concepts.
 */

import React, { useState, useMemo } from "react";

interface ConceptMapping {
  source: {
    id: string;
    name: string;
    abstractionLevel?: number;
  };
  target: {
    id: string;
    name: string;
    abstractionLevel?: number;
  };
  mappingType: "equivalent" | "analogous" | "partial" | "generalization" | "specialization" | "composition";
  similarity: number;
  justification?: string;
  transformation?: string;
  confidence: number;
}

interface DomainInfo {
  id: string;
  name: string;
  concepts: Array<{
    id: string;
    name: string;
    abstractionLevel?: number;
  }>;
}

interface ConceptMappingVizProps {
  sourceDomain: DomainInfo;
  targetDomain: DomainInfo;
  mappings: ConceptMapping[];
  mappingStrength?: number;
  quality?: "excellent" | "good" | "moderate" | "poor" | "none";
  onMappingClick?: (mapping: ConceptMapping) => void;
  className?: string;
}

export function ConceptMappingViz({
  sourceDomain,
  targetDomain,
  mappings,
  mappingStrength = 0,
  quality = "moderate",
  onMappingClick,
  className = "",
}: ConceptMappingVizProps) {
  const [selectedMapping, setSelectedMapping] = useState<ConceptMapping | null>(null);
  const [hoveredMapping, setHoveredMapping] = useState<ConceptMapping | null>(null);

  const handleMappingClick = (mapping: ConceptMapping) => {
    setSelectedMapping(mapping === selectedMapping ? null : mapping);
    onMappingClick?.(mapping);
  };

  // Layout calculations
  const layout = useMemo(() => {
    const width = 600;
    const height = Math.max(
      sourceDomain.concepts.length,
      targetDomain.concepts.length
    ) * 50 + 100;
    const leftX = 100;
    const rightX = width - 100;
    const topPadding = 60;
    const conceptSpacing = 50;

    const sourcePositions = new Map<string, { x: number; y: number }>();
    const targetPositions = new Map<string, { x: number; y: number }>();

    sourceDomain.concepts.forEach((concept, idx) => {
      sourcePositions.set(concept.id, {
        x: leftX,
        y: topPadding + idx * conceptSpacing,
      });
    });

    targetDomain.concepts.forEach((concept, idx) => {
      targetPositions.set(concept.id, {
        x: rightX,
        y: topPadding + idx * conceptSpacing,
      });
    });

    return { width, height, leftX, rightX, sourcePositions, targetPositions };
  }, [sourceDomain, targetDomain]);

  return (
    <div className={`bg-white border rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Concept Mapping</h3>
            <p className="text-sm text-gray-500">
              {sourceDomain.name} → {targetDomain.name}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              Strength: {(mappingStrength * 100).toFixed(0)}%
            </span>
            <QualityBadge quality={quality} />
          </div>
        </div>
      </div>

      {/* Visualization */}
      <div className="p-4 overflow-auto">
        <svg width={layout.width} height={layout.height}>
          {/* Domain labels */}
          <text
            x={layout.leftX}
            y={30}
            textAnchor="middle"
            className="text-sm font-medium fill-gray-700"
          >
            {sourceDomain.name}
          </text>
          <text
            x={layout.rightX}
            y={30}
            textAnchor="middle"
            className="text-sm font-medium fill-gray-700"
          >
            {targetDomain.name}
          </text>

          {/* Mapping lines */}
          {mappings.map((mapping, idx) => {
            const sourcePos = layout.sourcePositions.get(mapping.source.id);
            const targetPos = layout.targetPositions.get(mapping.target.id);

            if (!sourcePos || !targetPos) return null;

            const isSelected = selectedMapping?.source.id === mapping.source.id &&
              selectedMapping?.target.id === mapping.target.id;
            const isHovered = hoveredMapping?.source.id === mapping.source.id &&
              hoveredMapping?.target.id === mapping.target.id;

            return (
              <g
                key={idx}
                onClick={() => handleMappingClick(mapping)}
                onMouseEnter={() => setHoveredMapping(mapping)}
                onMouseLeave={() => setHoveredMapping(null)}
                className="cursor-pointer"
              >
                <path
                  d={`M ${sourcePos.x + 60} ${sourcePos.y}
                      C ${(sourcePos.x + targetPos.x) / 2} ${sourcePos.y},
                        ${(sourcePos.x + targetPos.x) / 2} ${targetPos.y},
                        ${targetPos.x - 60} ${targetPos.y}`}
                  fill="none"
                  stroke={getMappingColor(mapping.mappingType, isSelected || isHovered)}
                  strokeWidth={isSelected || isHovered ? 3 : 2}
                  strokeOpacity={mapping.similarity}
                  strokeDasharray={mapping.mappingType === "partial" ? "5,5" : undefined}
                />
                {/* Similarity indicator */}
                <circle
                  cx={(sourcePos.x + targetPos.x) / 2}
                  cy={(sourcePos.y + targetPos.y) / 2}
                  r={10}
                  fill="white"
                  stroke={getMappingColor(mapping.mappingType, false)}
                  strokeWidth={1}
                />
                <text
                  x={(sourcePos.x + targetPos.x) / 2}
                  y={(sourcePos.y + targetPos.y) / 2 + 4}
                  textAnchor="middle"
                  className="text-xs fill-gray-600"
                >
                  {(mapping.similarity * 100).toFixed(0)}
                </text>
              </g>
            );
          })}

          {/* Source concepts */}
          {sourceDomain.concepts.map((concept) => {
            const pos = layout.sourcePositions.get(concept.id);
            if (!pos) return null;

            const hasMappings = mappings.some((m) => m.source.id === concept.id);

            return (
              <g key={concept.id} transform={`translate(${pos.x}, ${pos.y})`}>
                <rect
                  x={-55}
                  y={-15}
                  width={110}
                  height={30}
                  rx={4}
                  fill={hasMappings ? "#dbeafe" : "#f3f4f6"}
                  stroke={hasMappings ? "#3b82f6" : "#d1d5db"}
                />
                <text
                  textAnchor="middle"
                  y={5}
                  className="text-sm fill-gray-700"
                >
                  {concept.name.length > 12
                    ? concept.name.slice(0, 12) + "..."
                    : concept.name}
                </text>
              </g>
            );
          })}

          {/* Target concepts */}
          {targetDomain.concepts.map((concept) => {
            const pos = layout.targetPositions.get(concept.id);
            if (!pos) return null;

            const hasMappings = mappings.some((m) => m.target.id === concept.id);

            return (
              <g key={concept.id} transform={`translate(${pos.x}, ${pos.y})`}>
                <rect
                  x={-55}
                  y={-15}
                  width={110}
                  height={30}
                  rx={4}
                  fill={hasMappings ? "#dcfce7" : "#f3f4f6"}
                  stroke={hasMappings ? "#22c55e" : "#d1d5db"}
                />
                <text
                  textAnchor="middle"
                  y={5}
                  className="text-sm fill-gray-700"
                >
                  {concept.name.length > 12
                    ? concept.name.slice(0, 12) + "..."
                    : concept.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Selected Mapping Details */}
      {selectedMapping && (
        <div className="p-4 border-t bg-gray-50">
          <MappingDetails mapping={selectedMapping} />
        </div>
      )}

      {/* Legend */}
      <div className="p-4 border-t flex flex-wrap items-center gap-4 text-xs text-gray-600">
        <LegendItem color="#3b82f6" label="Equivalent" />
        <LegendItem color="#22c55e" label="Analogous" />
        <LegendItem color="#f59e0b" label="Partial" dashed />
        <LegendItem color="#8b5cf6" label="Generalization" />
        <LegendItem color="#ec4899" label="Specialization" />
      </div>
    </div>
  );
}

// Helper components

function QualityBadge({ quality }: { quality: string }) {
  const colors: Record<string, string> = {
    excellent: "bg-green-100 text-green-700",
    good: "bg-lime-100 text-lime-700",
    moderate: "bg-yellow-100 text-yellow-700",
    poor: "bg-orange-100 text-orange-700",
    none: "bg-gray-100 text-gray-700",
  };

  return (
    <span className={`px-2 py-1 text-xs rounded-full ${colors[quality]}`}>
      {quality.charAt(0).toUpperCase() + quality.slice(1)}
    </span>
  );
}

function LegendItem({
  color,
  label,
  dashed,
}: {
  color: string;
  label: string;
  dashed?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <div
        className="w-6 h-0.5"
        style={{
          backgroundColor: color,
          backgroundImage: dashed
            ? `repeating-linear-gradient(90deg, ${color} 0, ${color} 3px, transparent 3px, transparent 6px)`
            : undefined,
        }}
      />
      <span>{label}</span>
    </div>
  );
}

function getMappingColor(
  type: ConceptMapping["mappingType"],
  highlighted: boolean
): string {
  const colors: Record<string, string> = {
    equivalent: "#3b82f6",
    analogous: "#22c55e",
    partial: "#f59e0b",
    generalization: "#8b5cf6",
    specialization: "#ec4899",
    composition: "#06b6d4",
  };

  const color = colors[type] || "#9ca3af";
  return highlighted ? color : color + "aa";
}

function MappingDetails({ mapping }: { mapping: ConceptMapping }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium">
          {mapping.source.name} → {mapping.target.name}
        </span>
        <span
          className="px-2 py-0.5 text-xs rounded-full"
          style={{
            backgroundColor: getMappingColor(mapping.mappingType, false) + "33",
            color: getMappingColor(mapping.mappingType, true),
          }}
        >
          {mapping.mappingType}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-500">Similarity:</span>{" "}
          <span className="font-mono">{(mapping.similarity * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span className="text-gray-500">Confidence:</span>{" "}
          <span className="font-mono">{(mapping.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>

      {mapping.justification && (
        <p className="text-sm text-gray-600">{mapping.justification}</p>
      )}

      {mapping.transformation && (
        <div className="text-sm">
          <span className="text-gray-500">Transformation:</span>{" "}
          <span className="text-gray-700">{mapping.transformation}</span>
        </div>
      )}
    </div>
  );
}

/**
 * DomainAnalogyCard Component
 *
 * Displays a domain analogy with examples.
 */

interface DomainAnalogy {
  id: string;
  sourcePattern: string;
  targetPattern: string;
  description: string;
  strength: number;
  examples: Array<{
    source: string;
    target: string;
    explanation: string;
  }>;
}

interface DomainAnalogyCardProps {
  analogy: DomainAnalogy;
  className?: string;
}

export function DomainAnalogyCard({
  analogy,
  className = "",
}: DomainAnalogyCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`border rounded-lg ${className}`}>
      <div
        className="p-4 cursor-pointer hover:bg-gray-50"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-medium">{analogy.sourcePattern}</span>
            <span className="text-gray-400">→</span>
            <span className="font-medium">{analogy.targetPattern}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">
              Strength: {(analogy.strength * 100).toFixed(0)}%
            </span>
            <span className="text-gray-400">{expanded ? "▲" : "▼"}</span>
          </div>
        </div>
        <p className="text-sm text-gray-600 mt-1">{analogy.description}</p>
      </div>

      {expanded && analogy.examples.length > 0 && (
        <div className="p-4 border-t bg-gray-50">
          <h5 className="text-sm font-medium text-gray-700 mb-2">Examples</h5>
          <div className="space-y-3">
            {analogy.examples.map((example, idx) => (
              <div key={idx} className="text-sm">
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
                    {example.source}
                  </span>
                  <span className="text-gray-400">→</span>
                  <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded">
                    {example.target}
                  </span>
                </div>
                <p className="text-gray-600 text-xs">{example.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * MappingMatrix Component
 *
 * Matrix view of all concept mappings between domains.
 */

interface MappingMatrixProps {
  sourceDomain: DomainInfo;
  targetDomain: DomainInfo;
  mappings: ConceptMapping[];
  className?: string;
}

export function MappingMatrix({
  sourceDomain,
  targetDomain,
  mappings,
  className = "",
}: MappingMatrixProps) {
  // Create mapping lookup
  const mappingLookup = new Map<string, ConceptMapping>();
  for (const mapping of mappings) {
    mappingLookup.set(`${mapping.source.id}-${mapping.target.id}`, mapping);
  }

  return (
    <div className={`bg-white border rounded-lg overflow-auto ${className}`}>
      <table className="min-w-full text-sm">
        <thead>
          <tr>
            <th className="p-2 border-b border-r bg-gray-50" />
            {targetDomain.concepts.map((concept) => (
              <th
                key={concept.id}
                className="p-2 border-b bg-gray-50 text-center font-medium text-gray-700"
              >
                <span className="block max-w-[80px] truncate" title={concept.name}>
                  {concept.name}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sourceDomain.concepts.map((sourceConcept) => (
            <tr key={sourceConcept.id}>
              <td className="p-2 border-b border-r bg-gray-50 font-medium text-gray-700">
                <span
                  className="block max-w-[100px] truncate"
                  title={sourceConcept.name}
                >
                  {sourceConcept.name}
                </span>
              </td>
              {targetDomain.concepts.map((targetConcept) => {
                const mapping = mappingLookup.get(
                  `${sourceConcept.id}-${targetConcept.id}`
                );

                return (
                  <td
                    key={targetConcept.id}
                    className="p-2 border-b text-center"
                    title={
                      mapping
                        ? `${mapping.mappingType}: ${(mapping.similarity * 100).toFixed(0)}%`
                        : "No mapping"
                    }
                  >
                    {mapping ? (
                      <div
                        className="w-8 h-8 mx-auto rounded flex items-center justify-center text-xs text-white font-medium"
                        style={{
                          backgroundColor: getMappingColor(
                            mapping.mappingType,
                            false
                          ),
                          opacity: 0.5 + mapping.similarity * 0.5,
                        }}
                      >
                        {(mapping.similarity * 100).toFixed(0)}
                      </div>
                    ) : (
                      <div className="w-8 h-8 mx-auto rounded bg-gray-100" />
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ConceptMappingViz;
