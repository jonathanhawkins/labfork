"use client";

/**
 * LineageTree Component
 *
 * Interactive tree visualization of technique ancestry showing
 * parent-child relationships, genetic contributions, and mutation points.
 */

import React, { useState, useMemo, useCallback } from "react";

interface LineageNode {
  id: string;
  name: string;
  generation: number;
  fitness: number;
  children: LineageNode[];
  isAlive?: boolean;
  mutations?: Array<{
    geneId: string;
    type: string;
    fitnessImpact?: number;
  }>;
  parentContribution?: number;
}

interface LineageTreeProps {
  root: LineageNode;
  currentGeneration?: number;
  onNodeClick?: (node: LineageNode) => void;
  onNodeHover?: (node: LineageNode | null) => void;
  highlightAlive?: boolean;
  showMutations?: boolean;
  className?: string;
}

export function LineageTree({
  root,
  currentGeneration,
  onNodeClick,
  onNodeHover,
  highlightAlive = true,
  showMutations = true,
  className = "",
}: LineageTreeProps) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Calculate tree layout
  const layout = useMemo(() => calculateTreeLayout(root), [root]);

  const handleNodeClick = useCallback(
    (node: LineageNode) => {
      setSelectedNodeId(node.id === selectedNodeId ? null : node.id);
      onNodeClick?.(node);
    },
    [selectedNodeId, onNodeClick]
  );

  const handleNodeHover = useCallback(
    (node: LineageNode | null) => {
      setHoveredNodeId(node?.id || null);
      onNodeHover?.(node);
    },
    [onNodeHover]
  );

  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.25, 2));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.5));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className={`bg-white border rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between">
        <div>
          <h3 className="font-semibold">Lineage Tree</h3>
          <p className="text-sm text-gray-500">
            Root: {root.name} | Depth: {layout.maxDepth} | Nodes: {layout.totalNodes}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleZoomOut}
            className="p-1 rounded hover:bg-gray-100"
            title="Zoom Out"
          >
            -
          </button>
          <span className="text-sm text-gray-500">{Math.round(zoom * 100)}%</span>
          <button
            onClick={handleZoomIn}
            className="p-1 rounded hover:bg-gray-100"
            title="Zoom In"
          >
            +
          </button>
          <button
            onClick={handleReset}
            className="px-2 py-1 text-xs bg-gray-100 rounded hover:bg-gray-200"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Tree View */}
      <div
        className="overflow-auto"
        style={{ height: 400 }}
      >
        <svg
          width={layout.width * zoom + 100}
          height={layout.height * zoom + 100}
          className="cursor-move"
        >
          <g transform={`translate(${50 + pan.x}, ${50 + pan.y}) scale(${zoom})`}>
            {/* Edges */}
            {layout.edges.map((edge, idx) => (
              <path
                key={idx}
                d={`M ${edge.x1} ${edge.y1} C ${edge.x1} ${(edge.y1 + edge.y2) / 2}, ${edge.x2} ${(edge.y1 + edge.y2) / 2}, ${edge.x2} ${edge.y2}`}
                fill="none"
                stroke={
                  edge.nodeId === selectedNodeId || edge.parentId === selectedNodeId
                    ? "#3b82f6"
                    : "#e5e7eb"
                }
                strokeWidth={
                  edge.nodeId === selectedNodeId || edge.parentId === selectedNodeId
                    ? 2
                    : 1
                }
              />
            ))}

            {/* Nodes */}
            {layout.nodes.map((nodeLayout) => (
              <g
                key={nodeLayout.node.id}
                transform={`translate(${nodeLayout.x}, ${nodeLayout.y})`}
                onClick={() => handleNodeClick(nodeLayout.node)}
                onMouseEnter={() => handleNodeHover(nodeLayout.node)}
                onMouseLeave={() => handleNodeHover(null)}
                className="cursor-pointer"
              >
                {/* Node circle */}
                <circle
                  r={20}
                  fill={getNodeColor(nodeLayout.node, highlightAlive, selectedNodeId, hoveredNodeId)}
                  stroke={
                    nodeLayout.node.id === selectedNodeId
                      ? "#3b82f6"
                      : nodeLayout.node.id === hoveredNodeId
                      ? "#60a5fa"
                      : "transparent"
                  }
                  strokeWidth={2}
                />

                {/* Fitness indicator */}
                <circle
                  r={15}
                  fill="none"
                  stroke="#fff"
                  strokeWidth={2}
                  strokeDasharray={`${nodeLayout.node.fitness * 94} 94`}
                  strokeDashoffset={0}
                  transform="rotate(-90)"
                  opacity={0.5}
                />

                {/* Generation label */}
                <text
                  y={5}
                  textAnchor="middle"
                  className="text-xs fill-white font-medium pointer-events-none"
                >
                  G{nodeLayout.node.generation}
                </text>

                {/* Mutation indicator */}
                {showMutations &&
                  nodeLayout.node.mutations &&
                  nodeLayout.node.mutations.length > 0 && (
                    <circle
                      cx={15}
                      cy={-15}
                      r={6}
                      fill="#f59e0b"
                      stroke="#fff"
                      strokeWidth={1}
                    />
                  )}

                {/* Elite indicator */}
                {nodeLayout.node.isAlive && (
                  <circle
                    cx={-15}
                    cy={-15}
                    r={6}
                    fill="#22c55e"
                    stroke="#fff"
                    strokeWidth={1}
                  />
                )}
              </g>
            ))}
          </g>
        </svg>
      </div>

      {/* Selected Node Details */}
      {selectedNodeId && (
        <div className="p-4 border-t bg-gray-50">
          <NodeDetails
            node={layout.nodes.find((n) => n.node.id === selectedNodeId)?.node}
          />
        </div>
      )}

      {/* Legend */}
      <div className="p-4 border-t flex items-center gap-6 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Alive</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-gray-400" />
          <span>Extinct</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span>Has Mutations</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-1 bg-gray-300" />
          <span>Parent-Child Link</span>
        </div>
      </div>
    </div>
  );
}

// Layout calculation
interface NodeLayout {
  node: LineageNode;
  x: number;
  y: number;
  depth: number;
}

interface EdgeLayout {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  nodeId: string;
  parentId: string;
}

interface TreeLayout {
  nodes: NodeLayout[];
  edges: EdgeLayout[];
  width: number;
  height: number;
  maxDepth: number;
  totalNodes: number;
}

function calculateTreeLayout(root: LineageNode): TreeLayout {
  const nodes: NodeLayout[] = [];
  const edges: EdgeLayout[] = [];
  const nodeSpacingX = 60;
  const nodeSpacingY = 80;

  let maxX = 0;
  let maxY = 0;
  let totalNodes = 0;

  // First pass: count nodes at each depth
  const nodesPerDepth = new Map<number, number>();

  function countNodes(node: LineageNode, depth: number): void {
    nodesPerDepth.set(depth, (nodesPerDepth.get(depth) || 0) + 1);
    totalNodes++;
    for (const child of node.children) {
      countNodes(child, depth + 1);
    }
  }

  countNodes(root, 0);
  const maxDepth = Math.max(...Array.from(nodesPerDepth.keys()));

  // Second pass: position nodes
  const xOffsets = new Map<number, number>();
  for (let i = 0; i <= maxDepth; i++) {
    xOffsets.set(i, 0);
  }

  function positionNode(
    node: LineageNode,
    depth: number,
    parentX?: number,
    parentY?: number,
    parentId?: string
  ): void {
    const xOffset = xOffsets.get(depth) || 0;
    const x = xOffset * nodeSpacingX;
    const y = depth * nodeSpacingY;

    xOffsets.set(depth, xOffset + 1);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);

    nodes.push({ node, x, y, depth });

    if (parentX !== undefined && parentY !== undefined && parentId) {
      edges.push({
        x1: parentX,
        y1: parentY + 20,
        x2: x,
        y2: y - 20,
        nodeId: node.id,
        parentId,
      });
    }

    for (const child of node.children) {
      positionNode(child, depth + 1, x, y, node.id);
    }
  }

  positionNode(root, 0);

  return {
    nodes,
    edges,
    width: maxX + nodeSpacingX,
    height: maxY + nodeSpacingY,
    maxDepth,
    totalNodes,
  };
}

function getNodeColor(
  node: LineageNode,
  highlightAlive: boolean,
  selectedId: string | null,
  hoveredId: string | null
): string {
  if (node.id === selectedId) return "#3b82f6";
  if (node.id === hoveredId) return "#60a5fa";
  if (highlightAlive && node.isAlive) return "#22c55e";
  if (node.fitness >= 0.8) return "#059669";
  if (node.fitness >= 0.6) return "#0891b2";
  if (node.fitness >= 0.4) return "#6366f1";
  return "#9ca3af";
}

function NodeDetails({ node }: { node?: LineageNode }) {
  if (!node) return null;

  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <h4 className="font-medium text-sm">{node.name}</h4>
        <p className="text-xs text-gray-500">ID: {node.id}</p>
      </div>
      <div className="text-right">
        <div className="text-sm">
          Fitness: <span className="font-mono">{(node.fitness * 100).toFixed(1)}%</span>
        </div>
        <div className="text-xs text-gray-500">
          Generation {node.generation}
        </div>
      </div>
      {node.mutations && node.mutations.length > 0 && (
        <div className="col-span-2">
          <p className="text-xs text-gray-500">
            Mutations: {node.mutations.length}
          </p>
          <div className="flex flex-wrap gap-1 mt-1">
            {node.mutations.slice(0, 3).map((m, i) => (
              <span
                key={i}
                className="px-1.5 py-0.5 text-xs bg-yellow-100 text-yellow-700 rounded"
              >
                {m.type}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * LineageTimeline Component
 *
 * Horizontal timeline view of lineage with fitness trajectory.
 */

interface LineageTimelineProps {
  fitnessTrajectory: Array<{ generation: number; fitness: number }>;
  keyMutations?: Array<{
    generation: number;
    fitnessImprovement: number;
    description?: string;
  }>;
  className?: string;
}

export function LineageTimeline({
  fitnessTrajectory,
  keyMutations = [],
  className = "",
}: LineageTimelineProps) {
  if (fitnessTrajectory.length === 0) {
    return (
      <div className={`p-4 text-center text-gray-500 ${className}`}>
        No trajectory data
      </div>
    );
  }

  const width = 500;
  const height = 100;
  const padding = { left: 40, right: 20, top: 20, bottom: 30 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const minGen = Math.min(...fitnessTrajectory.map((d) => d.generation));
  const maxGen = Math.max(...fitnessTrajectory.map((d) => d.generation));
  const genRange = maxGen - minGen || 1;

  const xScale = chartWidth / genRange;
  const yScale = chartHeight;

  return (
    <div className={`bg-white border rounded-lg p-4 ${className}`}>
      <h4 className="text-sm font-medium text-gray-700 mb-2">Fitness Trajectory</h4>
      <svg width={width} height={height}>
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {/* Fitness line */}
          <path
            d={fitnessTrajectory
              .map((d, i) => {
                const x = (d.generation - minGen) * xScale;
                const y = chartHeight - d.fitness * yScale;
                return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
              })
              .join(" ")}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={2}
          />

          {/* Data points */}
          {fitnessTrajectory.map((d) => (
            <circle
              key={d.generation}
              cx={(d.generation - minGen) * xScale}
              cy={chartHeight - d.fitness * yScale}
              r={3}
              fill="#3b82f6"
            />
          ))}

          {/* Key mutations */}
          {keyMutations.map((m, i) => (
            <g
              key={i}
              transform={`translate(${(m.generation - minGen) * xScale}, ${chartHeight + 10})`}
            >
              <line y1={-chartHeight - 10} y2={0} stroke="#f59e0b" strokeDasharray="2,2" />
              <polygon points="0,-5 -4,0 4,0" fill="#f59e0b" />
            </g>
          ))}

          {/* X-axis */}
          <line x1={0} y1={chartHeight} x2={chartWidth} y2={chartHeight} stroke="#e5e7eb" />
          <text x={chartWidth / 2} y={chartHeight + 20} textAnchor="middle" className="text-xs fill-gray-500">
            Generation
          </text>

          {/* Y-axis */}
          <line x1={0} y1={0} x2={0} y2={chartHeight} stroke="#e5e7eb" />
          <text x={-30} y={chartHeight / 2} textAnchor="middle" className="text-xs fill-gray-500" transform={`rotate(-90, -30, ${chartHeight / 2})`}>
            Fitness
          </text>
        </g>
      </svg>
    </div>
  );
}

export default LineageTree;
