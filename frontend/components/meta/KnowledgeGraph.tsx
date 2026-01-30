"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Text, Line } from "@react-three/drei";
import * as THREE from "three";
import {
  GraphNode,
  GraphEdge,
  NodeType,
  getNodeTypeColor,
  getEdgeTypeColor,
  getNodeTypeLabel,
} from "@/lib/meta/knowledge-graph/types";

// ============================================================================
// Types
// ============================================================================

interface KnowledgeGraphProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick?: (node: GraphNode) => void;
  onNodeHover?: (node: GraphNode | null) => void;
  selectedNodeId?: string;
  highlightedNodeIds?: string[];
  filterNodeTypes?: NodeType[];
  showLabels?: boolean;
  className?: string;
}

interface NodePosition {
  id: string;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
}

// ============================================================================
// Force-Directed Layout
// ============================================================================

function useForceLayout(
  nodes: GraphNode[],
  edges: GraphEdge[],
  iterations: number = 100
): Map<string, NodePosition> {
  const positions = useMemo(() => {
    const nodePositions = new Map<string, NodePosition>();

    // Initialize random positions
    nodes.forEach((node, i) => {
      const angle = (i / nodes.length) * Math.PI * 2;
      const radius = 5 + Math.random() * 3;
      nodePositions.set(node.id, {
        id: node.id,
        x: Math.cos(angle) * radius + (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5) * 4,
        z: Math.sin(angle) * radius + (Math.random() - 0.5) * 2,
        vx: 0,
        vy: 0,
        vz: 0,
      });
    });

    // Build edge map for quick lookup
    const edgeMap = new Map<string, Set<string>>();
    edges.forEach((edge) => {
      if (!edgeMap.has(edge.sourceId)) {
        edgeMap.set(edge.sourceId, new Set());
      }
      if (!edgeMap.has(edge.targetId)) {
        edgeMap.set(edge.targetId, new Set());
      }
      edgeMap.get(edge.sourceId)!.add(edge.targetId);
      edgeMap.get(edge.targetId)!.add(edge.sourceId);
    });

    // Force-directed layout iterations
    const repulsion = 50;
    const attraction = 0.1;
    const damping = 0.8;

    for (let iter = 0; iter < iterations; iter++) {
      const alpha = 1 - iter / iterations;

      // Apply repulsion between all nodes
      nodes.forEach((node1) => {
        const pos1 = nodePositions.get(node1.id)!;
        nodes.forEach((node2) => {
          if (node1.id === node2.id) return;
          const pos2 = nodePositions.get(node2.id)!;

          const dx = pos1.x - pos2.x;
          const dy = pos1.y - pos2.y;
          const dz = pos1.z - pos2.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;

          const force = (repulsion * alpha) / (dist * dist);
          pos1.vx += (dx / dist) * force;
          pos1.vy += (dy / dist) * force;
          pos1.vz += (dz / dist) * force;
        });
      });

      // Apply attraction along edges
      edges.forEach((edge) => {
        const pos1 = nodePositions.get(edge.sourceId);
        const pos2 = nodePositions.get(edge.targetId);
        if (!pos1 || !pos2) return;

        const dx = pos2.x - pos1.x;
        const dy = pos2.y - pos1.y;
        const dz = pos2.z - pos1.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.1;

        const force = dist * attraction * alpha;
        pos1.vx += (dx / dist) * force;
        pos1.vy += (dy / dist) * force;
        pos1.vz += (dz / dist) * force;
        pos2.vx -= (dx / dist) * force;
        pos2.vy -= (dy / dist) * force;
        pos2.vz -= (dz / dist) * force;
      });

      // Apply velocities and damping
      nodePositions.forEach((pos) => {
        pos.x += pos.vx;
        pos.y += pos.vy;
        pos.z += pos.vz;
        pos.vx *= damping;
        pos.vy *= damping;
        pos.vz *= damping;
      });
    }

    return nodePositions;
  }, [nodes, edges, iterations]);

  return positions;
}

// ============================================================================
// 3D Node Component
// ============================================================================

interface Node3DProps {
  node: GraphNode;
  position: [number, number, number];
  isSelected: boolean;
  isHighlighted: boolean;
  showLabel: boolean;
  onClick: () => void;
  onHover: (hovering: boolean) => void;
}

function Node3D({
  node,
  position,
  isSelected,
  isHighlighted,
  showLabel,
  onClick,
  onHover,
}: Node3DProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const color = getNodeTypeColor(node.type);
  const scale = isSelected ? 1.5 : isHighlighted ? 1.3 : hovered ? 1.2 : 1;
  const size = node.type === "paper" ? 0.3 : node.type === "technique" ? 0.4 : 0.35;

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.scale.lerp(
        new THREE.Vector3(scale, scale, scale),
        0.1
      );
    }
  });

  const handlePointerOver = useCallback(() => {
    setHovered(true);
    onHover(true);
  }, [onHover]);

  const handlePointerOut = useCallback(() => {
    setHovered(false);
    onHover(false);
  }, [onHover]);

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onClick={(e) => {
          e.stopPropagation();
          onClick();
        }}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        {node.type === "technique" ? (
          <octahedronGeometry args={[size]} />
        ) : node.type === "paper" ? (
          <boxGeometry args={[size * 1.5, size * 0.8, size * 0.2]} />
        ) : node.type === "lab" ? (
          <cylinderGeometry args={[size, size, size * 1.2, 6]} />
        ) : (
          <sphereGeometry args={[size, 16, 16]} />
        )}
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 0.5 : isHighlighted ? 0.3 : hovered ? 0.2 : 0.1}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>
      {(showLabel || hovered || isSelected) && (
        <Text
          position={[0, size + 0.3, 0]}
          fontSize={0.25}
          color="white"
          anchorX="center"
          anchorY="bottom"
          outlineWidth={0.02}
          outlineColor="black"
        >
          {node.name.length > 20 ? node.name.substring(0, 20) + "..." : node.name}
        </Text>
      )}
      {isSelected && (
        <mesh>
          <ringGeometry args={[size + 0.1, size + 0.15, 32]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.5} />
        </mesh>
      )}
    </group>
  );
}

// ============================================================================
// 3D Edge Component
// ============================================================================

interface Edge3DProps {
  edge: GraphEdge;
  startPos: [number, number, number];
  endPos: [number, number, number];
  isHighlighted: boolean;
}

function Edge3D({ edge, startPos, endPos, isHighlighted }: Edge3DProps) {
  const color = getEdgeTypeColor(edge.type);
  const opacity = isHighlighted ? 0.8 : 0.3;
  const lineWidth = isHighlighted ? 2 : 1;

  // Calculate midpoint for curved edge
  const midX = (startPos[0] + endPos[0]) / 2;
  const midY = (startPos[1] + endPos[1]) / 2 + 0.5;
  const midZ = (startPos[2] + endPos[2]) / 2;

  const points: [number, number, number][] = [startPos, [midX, midY, midZ], endPos];

  return (
    <Line
      points={points}
      color={color}
      lineWidth={lineWidth}
      transparent
      opacity={opacity}
    />
  );
}

// ============================================================================
// Scene Component
// ============================================================================

interface SceneProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  positions: Map<string, NodePosition>;
  selectedNodeId?: string;
  highlightedNodeIds?: string[];
  showLabels: boolean;
  onNodeClick: (node: GraphNode) => void;
  onNodeHover: (node: GraphNode | null) => void;
}

function Scene({
  nodes,
  edges,
  positions,
  selectedNodeId,
  highlightedNodeIds = [],
  showLabels,
  onNodeClick,
  onNodeHover,
}: SceneProps) {
  const highlightSet = useMemo(
    () => new Set(highlightedNodeIds),
    [highlightedNodeIds]
  );

  // Get connected node IDs for selected node
  const connectedNodes = useMemo(() => {
    if (!selectedNodeId) return new Set<string>();
    const connected = new Set<string>();
    edges.forEach((edge) => {
      if (edge.sourceId === selectedNodeId) {
        connected.add(edge.targetId);
      }
      if (edge.targetId === selectedNodeId) {
        connected.add(edge.sourceId);
      }
    });
    return connected;
  }, [selectedNodeId, edges]);

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />

      {/* Edges */}
      {edges.map((edge) => {
        const startPos = positions.get(edge.sourceId);
        const endPos = positions.get(edge.targetId);
        if (!startPos || !endPos) return null;

        const isHighlighted =
          edge.sourceId === selectedNodeId ||
          edge.targetId === selectedNodeId;

        return (
          <Edge3D
            key={edge.id}
            edge={edge}
            startPos={[startPos.x, startPos.y, startPos.z]}
            endPos={[endPos.x, endPos.y, endPos.z]}
            isHighlighted={isHighlighted}
          />
        );
      })}

      {/* Nodes */}
      {nodes.map((node) => {
        const pos = positions.get(node.id);
        if (!pos) return null;

        const isSelected = node.id === selectedNodeId;
        const isHighlighted =
          highlightSet.has(node.id) || connectedNodes.has(node.id);

        return (
          <Node3D
            key={node.id}
            node={node}
            position={[pos.x, pos.y, pos.z]}
            isSelected={isSelected}
            isHighlighted={isHighlighted}
            showLabel={showLabels}
            onClick={() => onNodeClick(node)}
            onHover={(hovering) => onNodeHover(hovering ? node : null)}
          />
        );
      })}

      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={3}
        maxDistance={30}
      />
    </>
  );
}

// ============================================================================
// Legend Component
// ============================================================================

function Legend({ nodeTypes }: { nodeTypes: NodeType[] }) {
  return (
    <div className="absolute bottom-4 left-4 bg-black/70 rounded-lg p-3 text-sm">
      <div className="text-white font-medium mb-2">Node Types</div>
      <div className="space-y-1">
        {nodeTypes.map((type) => (
          <div key={type} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: getNodeTypeColor(type) }}
            />
            <span className="text-gray-300">{getNodeTypeLabel(type)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Node Details Panel
// ============================================================================

interface NodeDetailsPanelProps {
  node: GraphNode | null;
  onClose: () => void;
}

function NodeDetailsPanel({ node, onClose }: NodeDetailsPanelProps) {
  if (!node) return null;

  return (
    <div className="absolute top-4 right-4 w-80 bg-gray-900/95 rounded-lg p-4 text-white shadow-xl">
      <div className="flex justify-between items-start mb-3">
        <div>
          <div
            className="text-xs font-medium px-2 py-0.5 rounded inline-block mb-1"
            style={{ backgroundColor: getNodeTypeColor(node.type) + "40" }}
          >
            {getNodeTypeLabel(node.type)}
          </div>
          <h3 className="font-semibold text-lg">{node.name}</h3>
        </div>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white text-xl"
        >
          ×
        </button>
      </div>

      {node.description && (
        <p className="text-gray-300 text-sm mb-3">{node.description}</p>
      )}

      {node.tags.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {node.tags.map((tag) => (
            <span
              key={tag}
              className="text-xs px-2 py-0.5 bg-gray-700 rounded-full text-gray-300"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      <div className="text-xs text-gray-500">
        Created: {new Date(node.createdAt).toLocaleDateString()}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function KnowledgeGraph({
  nodes,
  edges,
  onNodeClick,
  onNodeHover,
  selectedNodeId,
  highlightedNodeIds,
  filterNodeTypes,
  showLabels = false,
  className = "",
}: KnowledgeGraphProps) {
  const [localSelectedId, setLocalSelectedId] = useState<string | undefined>(
    selectedNodeId
  );
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);

  // Filter nodes by type if specified
  const filteredNodes = useMemo(() => {
    if (!filterNodeTypes || filterNodeTypes.length === 0) return nodes;
    return nodes.filter((n) => filterNodeTypes.includes(n.type));
  }, [nodes, filterNodeTypes]);

  // Filter edges to only include those between filtered nodes
  const filteredEdges = useMemo(() => {
    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    return edges.filter(
      (e) => nodeIds.has(e.sourceId) && nodeIds.has(e.targetId)
    );
  }, [edges, filteredNodes]);

  // Calculate positions
  const positions = useForceLayout(filteredNodes, filteredEdges);

  // Get unique node types for legend
  const nodeTypes = useMemo(() => {
    const types = new Set(filteredNodes.map((n) => n.type));
    return Array.from(types) as NodeType[];
  }, [filteredNodes]);

  const handleNodeClick = useCallback(
    (node: GraphNode) => {
      setLocalSelectedId(node.id);
      onNodeClick?.(node);
    },
    [onNodeClick]
  );

  const handleNodeHover = useCallback(
    (node: GraphNode | null) => {
      setHoveredNode(node);
      onNodeHover?.(node);
    },
    [onNodeHover]
  );

  const selectedNode = useMemo(
    () => filteredNodes.find((n) => n.id === localSelectedId) || null,
    [filteredNodes, localSelectedId]
  );

  if (filteredNodes.length === 0) {
    return (
      <div
        className={`flex items-center justify-center bg-gray-900 text-gray-400 ${className}`}
      >
        No nodes to display
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <Canvas
        camera={{ position: [0, 5, 15], fov: 60 }}
        style={{ background: "linear-gradient(to bottom, #0f172a, #1e293b)" }}
      >
        <Scene
          nodes={filteredNodes}
          edges={filteredEdges}
          positions={positions}
          selectedNodeId={localSelectedId}
          highlightedNodeIds={highlightedNodeIds}
          showLabels={showLabels}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
        />
      </Canvas>

      <Legend nodeTypes={nodeTypes} />

      {hoveredNode && !selectedNode && (
        <div className="absolute top-4 left-4 bg-black/70 rounded-lg p-2 text-sm text-white">
          <div className="font-medium">{hoveredNode.name}</div>
          <div className="text-gray-400 text-xs">
            {getNodeTypeLabel(hoveredNode.type)}
          </div>
        </div>
      )}

      <NodeDetailsPanel
        node={selectedNode}
        onClose={() => setLocalSelectedId(undefined)}
      />

      {/* Controls hint */}
      <div className="absolute bottom-4 right-4 text-xs text-gray-500">
        Drag to rotate | Scroll to zoom | Click node for details
      </div>
    </div>
  );
}

export default KnowledgeGraph;
