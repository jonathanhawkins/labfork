"use client";

/**
 * GapLandscape Component
 *
 * Visualizes the research landscape showing techniques, gaps, and opportunities
 * as an interactive node-edge graph.
 */

import React, { useState, useMemo, useCallback } from "react";
import {
  ResearchLandscape,
  LandscapeNode,
  LandscapeEdge,
  LandscapeCluster,
} from "@/lib/meta/gaps";

interface GapLandscapeProps {
  landscape: ResearchLandscape;
  width?: number;
  height?: number;
  onNodeClick?: (nodeId: string, nodeType: string) => void;
  onGapClick?: (gapId: string) => void;
  showClusters?: boolean;
  showLabels?: boolean;
  highlightGaps?: boolean;
  className?: string;
}

export function GapLandscape({
  landscape,
  width = 800,
  height = 600,
  onNodeClick,
  onGapClick,
  showClusters = true,
  showLabels = true,
  highlightGaps = true,
  className = "",
}: GapLandscapeProps) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // Scale positions to canvas size with padding
  const padding = 50;
  const scaleX = useCallback(
    (x: number) => padding + x * (width - 2 * padding),
    [width]
  );
  const scaleY = useCallback(
    (y: number) => padding + y * (height - 2 * padding),
    [height]
  );

  // Get connected edges for a node
  const getConnectedEdges = useCallback(
    (nodeId: string): LandscapeEdge[] => {
      return landscape.edges.filter(
        (e) => e.source === nodeId || e.target === nodeId
      );
    },
    [landscape.edges]
  );

  // Get connected nodes for hover highlighting
  const connectedNodes = useMemo(() => {
    if (!hoveredNode) return new Set<string>();
    const connected = new Set<string>();
    getConnectedEdges(hoveredNode).forEach((edge) => {
      connected.add(edge.source);
      connected.add(edge.target);
    });
    return connected;
  }, [hoveredNode, getConnectedEdges]);

  const handleNodeClick = (node: LandscapeNode) => {
    setSelectedNode(node.id);
    if (node.type === "gap") {
      onGapClick?.(node.id);
    } else {
      onNodeClick?.(node.id, node.type);
    }
  };

  const handleZoom = (delta: number) => {
    setZoom((prev) => Math.max(0.5, Math.min(2, prev + delta)));
  };

  return (
    <div className={`relative bg-gray-50 rounded-lg overflow-hidden ${className}`}>
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2">
        <button
          onClick={() => handleZoom(0.1)}
          className="w-8 h-8 bg-white border rounded shadow flex items-center justify-center hover:bg-gray-50"
          title="Zoom in"
        >
          +
        </button>
        <button
          onClick={() => handleZoom(-0.1)}
          className="w-8 h-8 bg-white border rounded shadow flex items-center justify-center hover:bg-gray-50"
          title="Zoom out"
        >
          -
        </button>
        <button
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
          }}
          className="w-8 h-8 bg-white border rounded shadow flex items-center justify-center hover:bg-gray-50 text-xs"
          title="Reset view"
        >
          R
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-white border rounded shadow p-3">
        <div className="text-xs font-medium mb-2">Legend</div>
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-xs">Technique</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-xs">Gap</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-xs">Opportunity</span>
          </div>
          <div className="flex items-center gap-2 mt-1 pt-1 border-t">
            <div className="w-6 h-0.5 bg-gray-400" />
            <span className="text-xs">Connection</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-0.5 border-t-2 border-dashed border-red-400" />
            <span className="text-xs">Gap Link</span>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="absolute top-4 left-4 z-10 bg-white border rounded shadow p-3">
        <div className="text-xs font-medium mb-2">Landscape Stats</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <div className="text-gray-500">Nodes:</div>
          <div className="font-medium">{landscape.nodes.length}</div>
          <div className="text-gray-500">Edges:</div>
          <div className="font-medium">{landscape.edges.length}</div>
          <div className="text-gray-500">Gaps:</div>
          <div className="font-medium text-red-600">{landscape.gaps.length}</div>
          <div className="text-gray-500">Coverage:</div>
          <div className="font-medium">
            {Math.round(landscape.coverageScore * 100)}%
          </div>
        </div>
      </div>

      {/* SVG Canvas */}
      <svg
        width={width}
        height={height}
        viewBox={`${-pan.x} ${-pan.y} ${width / zoom} ${height / zoom}`}
        className="cursor-move"
      >
        {/* Cluster backgrounds */}
        {showClusters &&
          landscape.clusters.map((cluster) => (
            <ClusterBackground
              key={cluster.id}
              cluster={cluster}
              nodes={landscape.nodes}
              scaleX={scaleX}
              scaleY={scaleY}
            />
          ))}

        {/* Edges */}
        {landscape.edges.map((edge) => {
          const sourceNode = landscape.nodes.find((n) => n.id === edge.source);
          const targetNode = landscape.nodes.find((n) => n.id === edge.target);

          if (!sourceNode || !targetNode) return null;

          const isHighlighted =
            hoveredNode &&
            (edge.source === hoveredNode || edge.target === hoveredNode);

          return (
            <line
              key={edge.id}
              x1={scaleX(sourceNode.x)}
              y1={scaleY(sourceNode.y)}
              x2={scaleX(targetNode.x)}
              y2={scaleY(targetNode.y)}
              stroke={isHighlighted ? edge.color : `${edge.color}60`}
              strokeWidth={isHighlighted ? 2 : 1}
              strokeDasharray={edge.dashed ? "5,5" : undefined}
              className="transition-all duration-200"
            />
          );
        })}

        {/* Nodes */}
        {landscape.nodes.map((node) => {
          const isHovered = hoveredNode === node.id;
          const isConnected = connectedNodes.has(node.id);
          const isSelected = selectedNode === node.id;
          const opacity =
            hoveredNode && !isHovered && !isConnected ? 0.3 : 1;

          return (
            <g key={node.id} style={{ opacity }}>
              {/* Node circle */}
              <circle
                cx={scaleX(node.x)}
                cy={scaleY(node.y)}
                r={node.size * 8 * (isHovered ? 1.3 : 1)}
                fill={node.color}
                stroke={isSelected ? "#000" : isHovered ? "#333" : "white"}
                strokeWidth={isSelected ? 3 : isHovered ? 2 : 1}
                className="cursor-pointer transition-all duration-200"
                onMouseEnter={() => setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onClick={() => handleNodeClick(node)}
              />

              {/* Gap indicator pulse */}
              {highlightGaps && node.type === "gap" && (
                <circle
                  cx={scaleX(node.x)}
                  cy={scaleY(node.y)}
                  r={node.size * 12}
                  fill="none"
                  stroke={node.color}
                  strokeWidth={1}
                  className="animate-ping opacity-30"
                />
              )}

              {/* Label */}
              {showLabels && (isHovered || node.type === "gap") && (
                <text
                  x={scaleX(node.x)}
                  y={scaleY(node.y) + node.size * 8 + 12}
                  textAnchor="middle"
                  className="text-xs fill-gray-700 pointer-events-none"
                  style={{ fontSize: "10px" }}
                >
                  {node.label.length > 20
                    ? `${node.label.slice(0, 20)}...`
                    : node.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Tooltip */}
      {hoveredNode && (
        <NodeTooltip
          node={landscape.nodes.find((n) => n.id === hoveredNode)!}
          connectedEdges={getConnectedEdges(hoveredNode)}
        />
      )}
    </div>
  );
}

/**
 * Cluster background visualization
 */
function ClusterBackground({
  cluster,
  nodes,
  scaleX,
  scaleY,
}: {
  cluster: LandscapeCluster;
  nodes: LandscapeNode[];
  scaleX: (x: number) => number;
  scaleY: (y: number) => number;
}) {
  const clusterNodes = nodes.filter((n) => cluster.nodeIds.includes(n.id));
  if (clusterNodes.length < 2) return null;

  // Calculate bounding box
  const padding = 30;
  const xs = clusterNodes.map((n) => scaleX(n.x));
  const ys = clusterNodes.map((n) => scaleY(n.y));
  const minX = Math.min(...xs) - padding;
  const maxX = Math.max(...xs) + padding;
  const minY = Math.min(...ys) - padding;
  const maxY = Math.max(...ys) + padding;

  return (
    <g>
      <rect
        x={minX}
        y={minY}
        width={maxX - minX}
        height={maxY - minY}
        fill="#f3f4f6"
        stroke="#e5e7eb"
        strokeWidth={1}
        rx={8}
        opacity={0.5}
      />
      <text
        x={minX + 5}
        y={minY + 15}
        className="text-xs fill-gray-400"
        style={{ fontSize: "10px" }}
      >
        {cluster.label}
      </text>
    </g>
  );
}

/**
 * Node tooltip
 */
function NodeTooltip({
  node,
  connectedEdges,
}: {
  node: LandscapeNode;
  connectedEdges: LandscapeEdge[];
}) {
  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 bg-white border rounded-lg shadow-lg p-3 pointer-events-none">
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: node.color }}
        />
        <span className="font-medium">{node.label}</span>
      </div>
      <div className="text-xs text-gray-500 space-y-1">
        <div>Type: {node.type}</div>
        <div>Connections: {connectedEdges.length}</div>
        {node.metadata.category && (
          <div>Category: {String(node.metadata.category)}</div>
        )}
        {node.metadata.severity && (
          <div>Severity: {String(node.metadata.severity)}</div>
        )}
      </div>
    </div>
  );
}

/**
 * Compact landscape summary
 */
interface LandscapeSummaryProps {
  landscape: ResearchLandscape;
  onViewFull?: () => void;
  className?: string;
}

export function LandscapeSummary({
  landscape,
  onViewFull,
  className = "",
}: LandscapeSummaryProps) {
  const gapsByType = useMemo(() => {
    const counts: Record<string, number> = {};
    landscape.gaps.forEach((gap) => {
      counts[gap.type] = (counts[gap.type] || 0) + 1;
    });
    return counts;
  }, [landscape.gaps]);

  return (
    <div className={`bg-white border rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">Research Landscape: {landscape.domain}</h3>
        {onViewFull && (
          <button
            onClick={onViewFull}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            View Full Map
          </button>
        )}
      </div>

      <div className="grid grid-cols-4 gap-4 mb-4">
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-blue-600">
            {landscape.nodes.filter((n) => n.type === "technique").length}
          </div>
          <div className="text-xs text-gray-500">Techniques</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-red-600">
            {landscape.gaps.length}
          </div>
          <div className="text-xs text-gray-500">Gaps</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-green-600">
            {landscape.opportunities.length}
          </div>
          <div className="text-xs text-gray-500">Opportunities</div>
        </div>
        <div className="text-center p-3 bg-gray-50 rounded">
          <div className="text-2xl font-bold text-purple-600">
            {Math.round(landscape.coverageScore * 100)}%
          </div>
          <div className="text-xs text-gray-500">Coverage</div>
        </div>
      </div>

      {/* Gap breakdown */}
      {Object.keys(gapsByType).length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Gap Types</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(gapsByType).map(([type, count]) => (
              <span
                key={type}
                className="text-xs px-2 py-1 bg-red-50 text-red-700 rounded"
              >
                {type.replace("_", " ")}: {count}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default GapLandscape;
