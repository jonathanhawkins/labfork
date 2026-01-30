/**
 * FireflyNetworkDemo
 *
 * Interactive Three.js visualization of the Firefly mesh network.
 * Users can click to add nodes and watch them auto-connect.
 */

"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";

interface Node {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  energy: number;
  brightness: number;
  connections: number[];
}

interface Connection {
  from: number;
  to: number;
  strength: number;
}

interface FireflyNetworkDemoProps {
  className?: string;
  initialNodes?: number;
  maxNodes?: number;
  connectionRange?: number;
}

export function FireflyNetworkDemo({
  className = "",
  initialNodes = 5,
  maxNodes = 30,
  connectionRange = 120,
}: FireflyNetworkDemoProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [stats, setStats] = useState({
    nodeCount: 0,
    connectionCount: 0,
    coverageArea: 0,
    networkHealth: 100,
  });
  const [isInitialized, setIsInitialized] = useState(false);
  const animationRef = useRef<number>(0);
  const nodesRef = useRef<Node[]>([]);

  // Initialize nodes
  useEffect(() => {
    if (!canvasRef.current || isInitialized) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const initialNodeList: Node[] = [];

    for (let i = 0; i < initialNodes; i++) {
      initialNodeList.push({
        id: i,
        x: Math.random() * (rect.width - 60) + 30,
        y: Math.random() * (rect.height - 60) + 30,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        energy: 0.7 + Math.random() * 0.3,
        brightness: 0.5 + Math.random() * 0.5,
        connections: [],
      });
    }

    setNodes(initialNodeList);
    nodesRef.current = initialNodeList;
    setIsInitialized(true);
  }, [initialNodes, isInitialized]);

  // Calculate connections
  const calculateConnections = useCallback(
    (nodeList: Node[]): Connection[] => {
      const connections: Connection[] = [];

      for (let i = 0; i < nodeList.length; i++) {
        nodeList[i].connections = [];
        for (let j = i + 1; j < nodeList.length; j++) {
          const dx = nodeList[i].x - nodeList[j].x;
          const dy = nodeList[i].y - nodeList[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < connectionRange) {
            const strength = 1 - dist / connectionRange;
            connections.push({ from: i, to: j, strength });
            nodeList[i].connections.push(j);
            nodeList[j].connections.push(i);
          }
        }
      }

      return connections;
    },
    [connectionRange]
  );

  // Animation loop
  useEffect(() => {
    if (!canvasRef.current || nodes.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    let time = 0;

    const animate = () => {
      time += 0.016;
      ctx.clearRect(0, 0, rect.width, rect.height);

      // Update node positions (gentle floating)
      const nodeList = nodesRef.current;
      nodeList.forEach((node) => {
        node.x += node.vx;
        node.y += node.vy;

        // Bounce off edges
        if (node.x < 20 || node.x > rect.width - 20) node.vx *= -1;
        if (node.y < 20 || node.y > rect.height - 20) node.vy *= -1;

        // Keep in bounds
        node.x = Math.max(20, Math.min(rect.width - 20, node.x));
        node.y = Math.max(20, Math.min(rect.height - 20, node.y));

        // Pulsing brightness
        node.brightness = 0.6 + 0.4 * Math.sin(time * 2 + node.id);
      });

      // Calculate and draw connections
      const connections = calculateConnections(nodeList);

      // Draw connection lines with gradient
      connections.forEach((conn) => {
        const fromNode = nodeList[conn.from];
        const toNode = nodeList[conn.to];

        const gradient = ctx.createLinearGradient(
          fromNode.x,
          fromNode.y,
          toNode.x,
          toNode.y
        );
        const alpha = Math.floor(conn.strength * 60)
          .toString(16)
          .padStart(2, "0");
        gradient.addColorStop(0, `#FFB84D${alpha}`);
        gradient.addColorStop(0.5, `#FFD700${alpha}`);
        gradient.addColorStop(1, `#FFB84D${alpha}`);

        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 1 + conn.strength * 2;
        ctx.stroke();

        // Animated data packets
        const packetProgress = (time * 0.5 + conn.from * 0.1) % 1;
        const packetX =
          fromNode.x + (toNode.x - fromNode.x) * packetProgress;
        const packetY =
          fromNode.y + (toNode.y - fromNode.y) * packetProgress;

        ctx.beginPath();
        ctx.arc(packetX, packetY, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 215, 0, ${conn.strength * 0.8})`;
        ctx.fill();
      });

      // Draw nodes (fireflies)
      nodeList.forEach((node) => {
        // Outer glow
        const glowGradient = ctx.createRadialGradient(
          node.x,
          node.y,
          0,
          node.x,
          node.y,
          30 * node.brightness
        );
        glowGradient.addColorStop(0, `rgba(255, 184, 77, ${0.4 * node.brightness})`);
        glowGradient.addColorStop(0.5, `rgba(255, 215, 0, ${0.2 * node.brightness})`);
        glowGradient.addColorStop(1, "rgba(255, 184, 77, 0)");

        ctx.beginPath();
        ctx.arc(node.x, node.y, 30 * node.brightness, 0, Math.PI * 2);
        ctx.fillStyle = glowGradient;
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(node.x, node.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${255}, ${200 + Math.floor(55 * node.energy)}, ${100})`;
        ctx.fill();

        // Energy ring
        ctx.beginPath();
        ctx.arc(node.x, node.y, 10, 0, Math.PI * 2 * node.energy);
        ctx.strokeStyle = `rgba(34, 197, 94, ${0.8 * node.energy})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Update stats
      const totalConnections = connections.length;
      const avgConnections =
        nodeList.length > 0 ? totalConnections / nodeList.length : 0;
      const coverage = Math.min(
        100,
        Math.floor((nodeList.length * connectionRange * connectionRange * Math.PI) / (rect.width * rect.height) * 100)
      );

      setStats({
        nodeCount: nodeList.length,
        connectionCount: totalConnections,
        coverageArea: coverage,
        networkHealth: Math.min(100, Math.floor(avgConnections * 50)),
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, calculateConnections]);

  // Handle canvas click to add node
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (nodes.length >= maxNodes) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const newNode: Node = {
      id: Date.now(),
      x,
      y,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      energy: 0.8 + Math.random() * 0.2,
      brightness: 0.7,
      connections: [],
    };

    const updatedNodes = [...nodesRef.current, newNode];
    nodesRef.current = updatedNodes;
    setNodes(updatedNodes);
  };

  return (
    <div className={`relative ${className}`}>
      {/* Canvas */}
      <div className="relative rounded-2xl overflow-hidden bg-gradient-to-b from-slate-900 to-slate-950 border border-amber-500/20">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className="w-full h-[400px] cursor-crosshair"
          style={{ touchAction: "none" }}
        />

        {/* Overlay instructions */}
        <div className="absolute top-4 left-4 text-amber-400/60 text-sm pointer-events-none">
          Click anywhere to add a firefly node
        </div>

        {/* Stats overlay */}
        <div className="absolute bottom-4 left-4 right-4 flex gap-4 text-xs">
          <div className="px-3 py-2 rounded-lg bg-slate-900/80 backdrop-blur border border-amber-500/20">
            <div className="text-amber-400 font-semibold">{stats.nodeCount}</div>
            <div className="text-gray-500">Nodes</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-slate-900/80 backdrop-blur border border-amber-500/20">
            <div className="text-amber-400 font-semibold">{stats.connectionCount}</div>
            <div className="text-gray-500">Links</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-slate-900/80 backdrop-blur border border-amber-500/20">
            <div className="text-amber-400 font-semibold">{stats.coverageArea}%</div>
            <div className="text-gray-500">Coverage</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-slate-900/80 backdrop-blur border border-amber-500/20">
            <div className="text-green-400 font-semibold">{stats.networkHealth}%</div>
            <div className="text-gray-500">Health</div>
          </div>
        </div>
      </div>

      {/* Caption */}
      <p className="mt-4 text-center text-gray-400 text-sm">
        This is what we&apos;re building - a self-organizing mesh network of solar lights.
        <br />
        <span className="text-amber-400">
          {nodes.length >= maxNodes
            ? "Maximum nodes reached!"
            : `Add up to ${maxNodes - nodes.length} more nodes`}
        </span>
      </p>
    </div>
  );
}

export default FireflyNetworkDemo;
