import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { fetchAPI } from '../api';

const NODE_COLORS = {
  individual: '#3b82f6',
  business: '#22c55e',
  shell_company: '#ef4444',
};

function formatINR(value) {
  if (value == null) return '—';
  const n = Number(value);
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  if (n >= 1e3) return `₹${(n / 1e3).toFixed(1)}K`;
  return `₹${n.toLocaleString('en-IN')}`;
}

function normalize(data) {
  const rawLinks = data.links || data.edges || [];
  // Backend returns camelCase (isFraud, fraudCount, txCount, avgAmount, mlScore)
  // — normalize so layout code can rely on snake_case across the board.
  const links = rawLinks.map((l) => ({
    ...l,
    amount: l.amount ?? l.weight ?? 0,
    is_fraud: l.is_fraud ?? l.isFraud ?? false,
    fraud_count: l.fraud_count ?? l.fraudCount ?? 0,
    transaction_count: l.transaction_count ?? l.txCount ?? 0,
    avg_amount: l.avg_amount ?? l.avgAmount ?? 0,
    ml_score: l.ml_score ?? l.mlScore ?? null,
  }));
  const nodes = (data.nodes || []).map((n) => {
    const degree = n.degree ?? 1;
    return {
      ...n,
      risk_score: n.risk_score ?? n.riskScore ?? 0,
      is_fraud: n.is_fraud ?? n.isFraud ?? false,
      val: Math.min(6, 1.5 + Math.sqrt(degree) * 0.6),
    };
  });
  return {
    nodes, links,
    totalNodes: data.total_nodes ?? nodes.length,
    totalEdges: data.total_edges ?? links.length,
    showingNodes: data.showing_nodes ?? nodes.length,
    showingEdges: data.showing_edges ?? links.length,
    appliedLimit: data.applied_limit ?? 0,
  };
}

// Map an amount to a hue/lightness for the heatmap palette.
function amountColor(amount, maxAmount, isFraud) {
  if (amount === 0 || amount == null) return 'transparent';
  const t = Math.log10(amount + 1) / Math.log10(maxAmount + 1);
  if (isFraud) {
    // Red ramp
    const l = 90 - t * 50;
    return `hsl(0, 75%, ${l}%)`;
  }
  // Blue ramp
  const l = 90 - t * 50;
  return `hsl(220, 70%, ${l}%)`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix layout
// ─────────────────────────────────────────────────────────────────────────────

function MatrixLayout({ data, selectedNodeId, onSelectNode, onHoverEdge }) {
  const [sortBy, setSortBy] = useState('type');
  const containerRef = useRef(null);
  const [containerWidth, setContainerWidth] = useState(800);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([entry]) => {
      setContainerWidth(Math.floor(entry.contentRect.width));
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const { sortedNodes, cellSize, labelWidth, edgeIndex, maxAmount } = useMemo(() => {
    const nodes = [...data.nodes];
    const cmpFns = {
      type: (a, b) => (a.type || '').localeCompare(b.type || ''),
      branch: (a, b) => (a.branch || '').localeCompare(b.branch || ''),
      risk: (a, b) => (b.risk_score || 0) - (a.risk_score || 0),
      degree: (a, b) => (b.degree || 0) - (a.degree || 0),
      name: (a, b) => (a.name || a.id).localeCompare(b.name || b.id),
    };
    nodes.sort(cmpFns[sortBy] || cmpFns.type);
    // Compute cell + label sizes that fit the container
    const labelW = 110;
    const available = containerWidth - labelW - 20;
    const cs = Math.max(6, Math.min(16, Math.floor(available / nodes.length)));

    // Edge lookup: source → target → edge
    const idx = new Map();
    let max = 1;
    for (const e of data.links) {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      const amt = e.amount || e.weight || 0;
      if (amt > max) max = amt;
      if (!idx.has(s)) idx.set(s, new Map());
      idx.get(s).set(t, { ...e, source: s, target: t });
    }
    return { sortedNodes: nodes, cellSize: cs, labelWidth: labelW, edgeIndex: idx, maxAmount: max };
  }, [data, sortBy, containerWidth]);

  const N = sortedNodes.length;
  const gridSize = N * cellSize;
  const totalW = labelWidth + gridSize + 30;     // row labels + grid + padding
  const totalH = labelWidth + gridSize + 30;     // col labels + grid + padding
  const idIndex = useMemo(() => {
    const m = new Map();
    sortedNodes.forEach((n, i) => m.set(n.id, i));
    return m;
  }, [sortedNodes]);
  const selIdx = selectedNodeId != null ? idIndex.get(selectedNodeId) : null;

  const [hover, setHover] = useState(null); // {row, col, edge}

  return (
    <div ref={containerRef} className="w-full h-full overflow-auto bg-white">
      <div className="p-3 flex items-center gap-3 border-b border-gray-200 sticky top-0 bg-white z-10">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Sort rows/cols by</span>
        {['type', 'branch', 'risk', 'degree', 'name'].map(s => (
          <button
            key={s}
            onClick={() => setSortBy(s)}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors capitalize ${
              sortBy === s ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {s}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-3 text-[11px] text-gray-600">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 inline-block" style={{ backgroundColor: 'hsl(220, 70%, 50%)' }} />
            Normal flow
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 inline-block" style={{ backgroundColor: 'hsl(0, 75%, 50%)' }} />
            Fraud flow
          </span>
          <span>· cell = sender→receiver</span>
        </div>
      </div>

      <div className="relative" style={{ width: totalW, height: totalH }}>
        <svg width={totalW} height={totalH} className="font-sans">
          {/* Column labels — rotated -45° */}
          {sortedNodes.map((n, i) => (
            <text
              key={`col-${n.id}`}
              x={labelWidth + i * cellSize + cellSize / 2}
              y={labelWidth - 4}
              fontSize={Math.max(8, cellSize - 2)}
              fill={selIdx === i ? '#1e1b4b' : '#374151'}
              fontWeight={selIdx === i ? 700 : 400}
              textAnchor="start"
              transform={`rotate(-60, ${labelWidth + i * cellSize + cellSize / 2}, ${labelWidth - 4})`}
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() => onSelectNode(n.id)}
            >
              {(n.name || n.id).slice(0, 14)}
            </text>
          ))}

          {/* Row labels */}
          {sortedNodes.map((n, i) => (
            <text
              key={`row-${n.id}`}
              x={labelWidth - 4}
              y={labelWidth + i * cellSize + cellSize / 2 + 3}
              fontSize={Math.max(8, cellSize - 2)}
              fill={selIdx === i ? '#1e1b4b' : '#374151'}
              fontWeight={selIdx === i ? 700 : 400}
              textAnchor="end"
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() => onSelectNode(n.id)}
            >
              {(n.name || n.id).slice(0, 14)}
            </text>
          ))}

          {/* Type color bar — column header strip */}
          {sortedNodes.map((n, i) => (
            <rect
              key={`coltype-${n.id}`}
              x={labelWidth + i * cellSize}
              y={labelWidth - 2}
              width={cellSize}
              height={2}
              fill={NODE_COLORS[n.type] || '#9ca3af'}
              opacity={0.7}
            />
          ))}
          {/* Type color bar — row header strip */}
          {sortedNodes.map((n, i) => (
            <rect
              key={`rowtype-${n.id}`}
              x={labelWidth - 2}
              y={labelWidth + i * cellSize}
              width={2}
              height={cellSize}
              fill={NODE_COLORS[n.type] || '#9ca3af'}
              opacity={0.7}
            />
          ))}

          {/* Grid background */}
          <rect
            x={labelWidth}
            y={labelWidth}
            width={gridSize}
            height={gridSize}
            fill="#fafafa"
            stroke="#e5e7eb"
          />

          {/* Cells with edges */}
          {sortedNodes.map((rowNode, i) => {
            const rowEdges = edgeIndex.get(rowNode.id);
            if (!rowEdges) return null;
            return Array.from(rowEdges.entries()).map(([targetId, edge]) => {
              const j = idIndex.get(targetId);
              if (j == null) return null;
              const isFraud = edge.is_fraud || edge.fraud_count > 0 || edge.fraud;
              const color = amountColor(edge.amount || edge.weight || 0, maxAmount, isFraud);
              const isHover = hover && hover.row === i && hover.col === j;
              return (
                <rect
                  key={`cell-${rowNode.id}-${targetId}`}
                  x={labelWidth + j * cellSize}
                  y={labelWidth + i * cellSize}
                  width={cellSize}
                  height={cellSize}
                  fill={color}
                  stroke={isHover ? '#0f172a' : isFraud ? '#dc2626' : 'transparent'}
                  strokeWidth={isHover ? 2 : isFraud ? 0.8 : 0}
                  onMouseEnter={() => {
                    setHover({ row: i, col: j, edge, rowNode, colNode: sortedNodes[j] });
                    onHoverEdge?.(edge);
                  }}
                  onMouseLeave={() => {
                    setHover(null);
                    onHoverEdge?.(null);
                  }}
                  onClick={() => onSelectNode(rowNode.id)}
                  style={{ cursor: 'pointer' }}
                />
              );
            });
          })}

          {/* Selected row + column highlight */}
          {selIdx != null && (
            <>
              <rect
                x={labelWidth + selIdx * cellSize}
                y={labelWidth}
                width={cellSize}
                height={gridSize}
                fill="rgba(99, 102, 241, 0.08)"
                stroke="rgba(99, 102, 241, 0.4)"
              />
              <rect
                x={labelWidth}
                y={labelWidth + selIdx * cellSize}
                width={gridSize}
                height={cellSize}
                fill="rgba(99, 102, 241, 0.08)"
                stroke="rgba(99, 102, 241, 0.4)"
              />
            </>
          )}
        </svg>

        {/* Tooltip */}
        {hover && (
          <div
            className="absolute pointer-events-none bg-gray-900 text-white text-xs rounded-md px-2.5 py-1.5 shadow-lg z-20"
            style={{
              left: Math.min(labelWidth + hover.col * cellSize + cellSize + 8, totalW - 220),
              top: labelWidth + hover.row * cellSize + cellSize,
            }}
          >
            <div className="font-medium">{hover.rowNode.name || hover.rowNode.id}</div>
            <div className="text-gray-400">→ {hover.colNode.name || hover.colNode.id}</div>
            <div className="mt-1 font-mono">{formatINR(hover.edge.amount || hover.edge.weight)}</div>
            {hover.edge.transaction_count != null && (
              <div className="text-gray-400 text-[10px]">{hover.edge.transaction_count} txns</div>
            )}
            {(hover.edge.is_fraud || hover.edge.fraud_count > 0) && (
              <div className="text-red-400 text-[10px] font-semibold">FRAUD EDGE</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Arc layout — entities along x-axis, edges as arcs above
// ─────────────────────────────────────────────────────────────────────────────

function ArcLayout({ data, selectedNodeId, onSelectNode }) {
  const [sortBy, setSortBy] = useState('risk');
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 800, height: 500 });
  const [hover, setHover] = useState(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.floor(entry.contentRect.width),
        height: Math.max(400, Math.floor(entry.contentRect.height)),
      });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const { sortedNodes, edges, idIndex, maxAmount } = useMemo(() => {
    const nodes = [...data.nodes];
    const cmpFns = {
      risk: (a, b) => (b.risk_score || 0) - (a.risk_score || 0),
      type: (a, b) => (a.type || '').localeCompare(b.type || ''),
      branch: (a, b) => (a.branch || '').localeCompare(b.branch || ''),
      degree: (a, b) => (b.degree || 0) - (a.degree || 0),
      name: (a, b) => (a.name || a.id).localeCompare(b.name || b.id),
    };
    nodes.sort(cmpFns[sortBy] || cmpFns.risk);
    const idx = new Map();
    nodes.forEach((n, i) => idx.set(n.id, i));
    let max = 1;
    const es = (data.links || []).map(e => {
      const s = typeof e.source === 'object' ? e.source.id : e.source;
      const t = typeof e.target === 'object' ? e.target.id : e.target;
      const amt = e.amount || e.weight || 0;
      if (amt > max) max = amt;
      return { ...e, source: s, target: t, amount: amt };
    });
    return { sortedNodes: nodes, edges: es, idIndex: idx, maxAmount: max };
  }, [data, sortBy]);

  const padLeft = 16;
  const padRight = 16;
  const padBottom = 110;
  const padTop = 20;
  const xStep = Math.max(8, (size.width - padLeft - padRight) / Math.max(sortedNodes.length - 1, 1));
  const baselineY = size.height - padBottom;
  const arcMaxY = padTop;

  function xFor(idx) { return padLeft + idx * xStep; }

  // Sort edges so big amounts draw on top
  const sortedEdges = [...edges].sort((a, b) => a.amount - b.amount);

  return (
    <div ref={containerRef} className="w-full h-full overflow-auto bg-white">
      <div className="p-3 flex items-center gap-3 border-b border-gray-200 sticky top-0 bg-white z-10">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Sort by</span>
        {['risk', 'type', 'branch', 'degree', 'name'].map(s => (
          <button
            key={s}
            onClick={() => setSortBy(s)}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors capitalize ${
              sortBy === s ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {s}
          </button>
        ))}
        <span className="ml-auto text-[11px] text-gray-500">
          Arc height = distance between entities · color = amount/fraud
        </span>
      </div>
      <div className="relative">
        <svg width={size.width} height={size.height}>
          {/* Arcs */}
          {sortedEdges.map((e, k) => {
            const i = idIndex.get(e.source);
            const j = idIndex.get(e.target);
            if (i == null || j == null) return null;
            const x1 = xFor(i);
            const x2 = xFor(j);
            const mid = (x1 + x2) / 2;
            const dist = Math.abs(x2 - x1);
            const arcHeight = Math.min(baselineY - arcMaxY, dist * 0.5 + 20);
            const ctrlY = baselineY - arcHeight;
            const isFraud = e.is_fraud || e.fraud_count > 0 || e.fraud;
            const isSelHit = selectedNodeId && (e.source === selectedNodeId || e.target === selectedNodeId);
            const color = amountColor(e.amount, maxAmount, isFraud);
            const opacity = selectedNodeId
              ? (isSelHit ? 0.85 : 0.05)
              : (isFraud ? 0.55 : 0.25);
            const width = Math.max(0.5, Math.log10(e.amount + 1) * 0.5);
            const isHover = hover && hover.k === k;
            return (
              <path
                key={k}
                d={`M ${x1} ${baselineY} Q ${mid} ${ctrlY} ${x2} ${baselineY}`}
                fill="none"
                stroke={color}
                strokeWidth={isHover ? width + 2 : width}
                opacity={isHover ? 1 : opacity}
                onMouseEnter={() => setHover({ k, edge: e })}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: 'pointer' }}
              />
            );
          })}

          {/* Baseline */}
          <line x1={padLeft} x2={size.width - padRight} y1={baselineY} y2={baselineY} stroke="#e5e7eb" />

          {/* Nodes */}
          {sortedNodes.map((n, i) => {
            const x = xFor(i);
            const isSel = selectedNodeId === n.id;
            const r = isSel ? 5 : 3.5;
            return (
              <g key={n.id} onClick={() => onSelectNode(n.id)} style={{ cursor: 'pointer' }}>
                <circle
                  cx={x}
                  cy={baselineY}
                  r={r}
                  fill={NODE_COLORS[n.type] || '#9ca3af'}
                  stroke={isSel ? '#0f172a' : '#ffffff'}
                  strokeWidth={isSel ? 2 : 0.8}
                />
                <text
                  x={x}
                  y={baselineY + 14}
                  fontSize={Math.max(7, Math.min(10, xStep - 2))}
                  fill={isSel ? '#0f172a' : '#4b5563'}
                  textAnchor="end"
                  transform={`rotate(-60, ${x}, ${baselineY + 14})`}
                  fontWeight={isSel ? 700 : 400}
                >
                  {(n.name || n.id).slice(0, 16)}
                </text>
              </g>
            );
          })}
        </svg>

        {hover && (
          <div
            className="absolute pointer-events-none bg-gray-900 text-white text-xs rounded-md px-2.5 py-1.5 shadow-lg z-20"
            style={{
              left: Math.min(xFor(idIndex.get(hover.edge.source)) + 8, size.width - 200),
              top: Math.max(8, baselineY - 80),
            }}
          >
            <div className="font-medium">
              {(data.nodes.find(n => n.id === hover.edge.source)?.name || hover.edge.source)}
              <span className="text-gray-400"> → </span>
              {(data.nodes.find(n => n.id === hover.edge.target)?.name || hover.edge.target)}
            </div>
            <div className="mt-1 font-mono">{formatINR(hover.edge.amount)}</div>
            {(hover.edge.is_fraud || hover.edge.fraud_count > 0) && (
              <div className="text-red-400 text-[10px] font-semibold">FRAUD EDGE</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Force layout — improved with hover-highlight + curved edges + opacity
// ─────────────────────────────────────────────────────────────────────────────

function ForceLayout({ data, selectedNodeId, hoveredNodeId, onSelectNode, onHoverNode, size }) {
  const ref = useRef();

  // Build neighbour set: ids connected to hovered node
  const focusedId = hoveredNodeId || selectedNodeId;
  const neighbours = useMemo(() => {
    if (!focusedId) return null;
    const s = new Set([focusedId]);
    for (const l of data.links) {
      const src = typeof l.source === 'object' ? l.source.id : l.source;
      const tgt = typeof l.target === 'object' ? l.target.id : l.target;
      if (src === focusedId) s.add(tgt);
      if (tgt === focusedId) s.add(src);
    }
    return s;
  }, [data.links, focusedId]);

  const maxAmount = useMemo(() => {
    let m = 1;
    for (const l of data.links) {
      const a = l.amount || l.weight || 0;
      if (a > m) m = a;
    }
    return m;
  }, [data.links]);

  useEffect(() => {
    if (!ref.current || data.nodes.length === 0) return;
    const t = setTimeout(() => {
      try { ref.current.zoomToFit(400, 60); } catch { /* not mounted */ }
    }, 200);
    return () => clearTimeout(t);
  }, [data, size.width, size.height]);

  const paintNode = useCallback((node, ctx, scale) => {
    const focused = focusedId === node.id;
    const dimmed = focusedId && !neighbours?.has(node.id);
    const r = focused ? 7 : Math.max(2.5, node.val || 3);
    ctx.globalAlpha = dimmed ? 0.2 : 1;
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = NODE_COLORS[node.type] || '#6b7280';
    ctx.fill();
    if (focused) {
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2.5 / scale;
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 0.6 / scale;
      ctx.stroke();
    }
    if (focused || scale >= 1.4) {
      const label = (node.name || node.id).slice(0, 20);
      const font = Math.max(9 / scale, 2);
      ctx.font = `${focused ? '600 ' : ''}${font}px ui-sans-serif`;
      ctx.fillStyle = '#1f2937';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(label, node.x, node.y + r + 1.5);
    }
    ctx.globalAlpha = 1;
  }, [focusedId, neighbours]);

  const paintLink = useCallback((link, ctx, scale) => {
    const sId = typeof link.source === 'object' ? link.source.id : link.source;
    const tId = typeof link.target === 'object' ? link.target.id : link.target;
    const focused = focusedId && (sId === focusedId || tId === focusedId);
    const dimmed = focusedId && !focused;
    const amt = link.amount || link.weight || 1;
    const isFraud = link.is_fraud || link.fraud || link.fraud_count > 0;
    const t = Math.log10(amt + 1) / Math.log10(maxAmount + 1);
    const width = Math.max(0.4, Math.min(5, t * 3 + 0.4));
    const baseOpacity = focused ? 0.95 : dimmed ? 0.04 : (isFraud ? 0.7 : 0.18 + t * 0.4);

    const sx = link.source.x ?? 0;
    const sy = link.source.y ?? 0;
    const tx = link.target.x ?? 0;
    const ty = link.target.y ?? 0;
    // Curved edge — bezier control point offset perpendicular to line direction
    const dx = tx - sx;
    const dy = ty - sy;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const offset = Math.min(30, dist * 0.15);
    const mx = (sx + tx) / 2 - (dy / dist) * offset;
    const my = (sy + ty) / 2 + (dx / dist) * offset;

    ctx.globalAlpha = baseOpacity;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.quadraticCurveTo(mx, my, tx, ty);
    ctx.strokeStyle = isFraud ? '#dc2626' : t > 0.6 ? '#4338ca' : '#94a3b8';
    ctx.lineWidth = width / scale;
    ctx.stroke();

    // Arrowhead at target
    const ang = Math.atan2(ty - my, tx - mx);
    const headLen = Math.max(4, width * 1.2) / scale;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(
      tx - headLen * Math.cos(ang - Math.PI / 7),
      ty - headLen * Math.sin(ang - Math.PI / 7),
    );
    ctx.moveTo(tx, ty);
    ctx.lineTo(
      tx - headLen * Math.cos(ang + Math.PI / 7),
      ty - headLen * Math.sin(ang + Math.PI / 7),
    );
    ctx.stroke();
    ctx.globalAlpha = 1;
  }, [focusedId, maxAmount]);

  if (data.nodes.length === 0) return null;

  return (
    <ForceGraph2D
      ref={ref}
      graphData={data}
      width={size.width}
      height={size.height}
      nodeVal="val"
      nodeRelSize={3}
      nodeCanvasObject={paintNode}
      nodeCanvasObjectMode={() => 'replace'}
      linkCanvasObject={paintLink}
      linkCanvasObjectMode={() => 'replace'}
      linkDirectionalArrowLength={0}
      onNodeClick={(n) => onSelectNode(n.id)}
      onNodeHover={(n) => onHoverNode(n ? n.id : null)}
      backgroundColor="#fafafa"
      cooldownTicks={120}
      d3VelocityDecay={0.5}
      enableNodeDrag={true}
      enableZoomInteraction={true}
      enablePanInteraction={true}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Graph page
// ─────────────────────────────────────────────────────────────────────────────

export default function Graph() {
  // Main graph
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [fraudOnly, setFraudOnly] = useState(false);
  const [highRiskOnly, setHighRiskOnly] = useState(false);
  const [minAmount, setMinAmount] = useState(0);
  const [topEdgePct, setTopEdgePct] = useState(100);   // % of strongest edges to keep
  // Server-side cap — IBM AML has 100k+ entities, so the first paint must be
  // bounded. limit=0 lets the backend return everything.
  const [limit, setLimit] = useState(200);

  // Scope + layout
  const [view, setView] = useState('full');           // 'full' | 'subgraph'
  const [layout, setLayout] = useState('matrix');     // 'force' | 'matrix' | 'arc'

  // Subgraph
  const [entitySearch, setEntitySearch] = useState('');
  const [selectedEntityId, setSelectedEntityId] = useState('');
  const [subhops, setSubhops] = useState(2);
  const [subgraphData, setSubgraphData] = useState({ nodes: [], links: [] });
  const [subgraphLoading, setSubgraphLoading] = useState(false);
  const [subgraphError, setSubgraphError] = useState(null);

  // Selection / hover
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [hoveredNodeId, setHoveredNodeId] = useState(null);

  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 800, height: 500 });

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.max(300, Math.floor(width)), height: Math.max(300, Math.floor(height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Fetch main graph
  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams();
    if (fraudOnly) params.set('fraud_only', 'true');
    if (highRiskOnly) params.set('high_risk_only', 'true');
    if (minAmount > 0) params.set('min_amount', String(minAmount));
    params.set('limit', String(limit));

    fetchAPI(`/api/graph?${params.toString()}`)
      .then((d) => setGraphData(normalize(d)))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [fraudOnly, highRiskOnly, minAmount, limit]);

  // Reset selection when view changes
  useEffect(() => { setSelectedNodeId(null); setHoveredNodeId(null); }, [view, layout]);

  // Searchable entity list
  const entityList = useMemo(() => {
    const q = entitySearch.trim().toLowerCase();
    const nodes = graphData.nodes;
    if (!q) return nodes.slice(0, 50);
    return nodes
      .filter((n) =>
        (n.name || '').toLowerCase().includes(q) ||
        (n.id || '').toLowerCase().includes(q) ||
        (n.type || '').toLowerCase().includes(q)
      )
      .slice(0, 50);
  }, [graphData.nodes, entitySearch]);

  const loadSubgraph = useCallback(async (entityId, hops) => {
    setSubgraphLoading(true);
    setSubgraphError(null);
    setSubgraphData({ nodes: [], links: [] });
    try {
      const data = await fetchAPI(`/api/graph/${entityId}?hops=${hops}`);
      setSubgraphData(normalize(data));
      setView('subgraph');
    } catch (e) {
      setSubgraphError(e.message);
    } finally {
      setSubgraphLoading(false);
    }
  }, []);

  // Apply top-N% edge filter to the active dataset
  const baseData = view === 'full' ? graphData : subgraphData;
  const filteredData = useMemo(() => {
    if (topEdgePct >= 100) return baseData;
    const sorted = [...baseData.links].sort(
      (a, b) => (b.amount || b.weight || 0) - (a.amount || a.weight || 0)
    );
    const keep = Math.max(1, Math.floor(sorted.length * (topEdgePct / 100)));
    const links = sorted.slice(0, keep);
    const keepIds = new Set();
    for (const l of links) {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      keepIds.add(s); keepIds.add(t);
    }
    const nodes = baseData.nodes.filter((n) => keepIds.has(n.id));
    return { nodes, links };
  }, [baseData, topEdgePct]);

  const stats = useMemo(() => {
    if (!filteredData.nodes.length) return null;
    const fraud = filteredData.links.filter((l) => l.is_fraud || l.fraud || l.fraud_count > 0).length;
    const totalAmt = filteredData.links.reduce((s, l) => s + (l.amount || l.weight || 0), 0);
    const highRisk = filteredData.nodes.filter((n) => (n.risk_score || 0) >= 0.5).length;
    const shellCount = filteredData.nodes.filter((n) => n.type === 'shell_company').length;
    return {
      nodes: filteredData.nodes.length, links: filteredData.links.length,
      fraud, highRisk, shellCount, totalAmt,
    };
  }, [filteredData]);

  const selectedNode = useMemo(
    () => filteredData.nodes.find((n) => n.id === selectedNodeId) || null,
    [filteredData, selectedNodeId]
  );

  const selectedEdges = useMemo(() => {
    if (!selectedNode) return [];
    return filteredData.links.filter((l) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return s === selectedNode.id || t === selectedNode.id;
    });
  }, [filteredData, selectedNode]);

  const activeLoading = view === 'full' ? loading : subgraphLoading;
  const activeError = view === 'full' ? error : subgraphError;

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* Header */}
      <div className="px-6 pt-6 pb-2 flex items-baseline justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Network Graph</h1>
          <p className="text-sm text-gray-500 mt-1">
            Three layouts for {graphData.nodes.length}-node bank graph. Matrix shows every edge in a fixed grid (best for the full graph),
            Arc reveals clusters along a line, Force is best for focused subgraphs.
          </p>
        </div>
        {!loading && !error && graphData.totalNodes != null && (
          <p className="text-xs text-gray-500 shrink-0">
            Showing <span className="font-semibold text-gray-800">{graphData.showingNodes}</span> of
            {' '}<span className="font-semibold text-gray-800">{graphData.totalNodes.toLocaleString()}</span> entities,
            {' '}<span className="font-semibold text-gray-800">{graphData.showingEdges}</span> of
            {' '}<span className="font-semibold text-gray-800">{graphData.totalEdges.toLocaleString()}</span> edges
          </p>
        )}
      </div>

      {/* Tabs + filters */}
      <div className="px-6 py-3 bg-white border-y border-gray-200 flex flex-wrap items-center gap-x-4 gap-y-2">
        {/* Scope */}
        <div className="inline-flex rounded-lg border border-gray-300 p-0.5">
          <button
            onClick={() => setView('full')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              view === 'full' ? 'bg-indigo-600 text-white' : 'text-gray-700 hover:bg-gray-50'
            }`}
          >
            Full Graph
          </button>
          <button
            onClick={() => subgraphData.nodes.length && setView('subgraph')}
            disabled={!subgraphData.nodes.length}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              view === 'subgraph' ? 'bg-indigo-600 text-white' : 'text-gray-700 hover:bg-gray-50'
            }`}
          >
            Subgraph {subgraphData.nodes.length ? `(${subgraphData.nodes.length})` : ''}
          </button>
        </div>

        {/* Layout */}
        <div className="inline-flex rounded-lg border border-gray-300 p-0.5">
          {[
            { k: 'matrix', label: 'Matrix' },
            { k: 'arc',    label: 'Arc' },
            { k: 'force',  label: 'Force' },
          ].map(L => (
            <button
              key={L.k}
              onClick={() => setLayout(L.k)}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                layout === L.k ? 'bg-emerald-600 text-white' : 'text-gray-700 hover:bg-gray-50'
              }`}
            >
              {L.label}
            </button>
          ))}
        </div>

        {view === 'full' && (
          <>
            <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
              <input
                type="checkbox"
                checked={fraudOnly}
                onChange={(e) => setFraudOnly(e.target.checked)}
                className="rounded border-gray-300 text-red-600 focus:ring-red-500"
              />
              Fraud only
            </label>

            <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
              <input
                type="checkbox"
                checked={highRiskOnly}
                onChange={(e) => setHighRiskOnly(e.target.checked)}
                className="rounded border-gray-300 text-amber-600 focus:ring-amber-500"
              />
              High risk only
            </label>

            <label className="flex items-center gap-2 text-sm text-gray-700">
              Min ₹
              <input
                type="number"
                value={minAmount}
                onChange={(e) => setMinAmount(Math.max(0, Number(e.target.value)))}
                min={0}
                step={100000}
                placeholder="0"
                className="w-28 rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </label>
          </>
        )}

        {/* Top-edges slider */}
        <label className="flex items-center gap-2 text-sm text-gray-700">
          Top {topEdgePct}% edges
          <input
            type="range"
            min={5}
            max={100}
            step={5}
            value={topEdgePct}
            onChange={(e) => setTopEdgePct(Number(e.target.value))}
            className="w-32"
          />
        </label>

        {/* Server-side cap — IBM AML has 100k+ nodes so the first paint must be bounded. */}
        <label className="flex items-center gap-2 text-sm text-gray-700">
          Show
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          >
            <option value={50}>Top 50 by risk</option>
            <option value={100}>Top 100</option>
            <option value={200}>Top 200</option>
            <option value={500}>Top 500</option>
            <option value={1000}>Top 1000</option>
            <option value={0}>All (slow)</option>
          </select>
        </label>

        {/* Legend */}
        <div className="ml-auto flex items-center gap-3 text-xs text-gray-600">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Individual
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-green-500" /> Business
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500" /> Shell Co.
          </span>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <aside className="w-72 shrink-0 border-r border-gray-200 bg-white overflow-y-auto">
          {/* Entity picker */}
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Subgraph around entity
            </h3>
            <input
              type="text"
              value={entitySearch}
              onChange={(e) => setEntitySearch(e.target.value)}
              placeholder="Search entities..."
              className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <div className="mt-2 max-h-56 overflow-y-auto border border-gray-100 rounded-md divide-y divide-gray-100">
              {entityList.length === 0 ? (
                <p className="text-xs text-gray-400 p-3 text-center">No matches</p>
              ) : entityList.map((e) => {
                const isSel = selectedEntityId === e.id;
                return (
                  <button
                    key={e.id}
                    onClick={() => setSelectedEntityId(e.id)}
                    className={`w-full text-left px-3 py-1.5 text-xs transition-colors ${
                      isSel ? 'bg-indigo-50 text-indigo-700 font-medium' : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <div className="font-medium truncate">{e.name || e.id}</div>
                    <div className="text-[10px] text-gray-400 capitalize">{(e.type || '').replace('_', ' ')}</div>
                  </button>
                );
              })}
            </div>
            <label className="mt-3 flex items-center gap-2 text-xs text-gray-700">
              Hops
              <input
                type="number"
                min={1} max={4}
                value={subhops}
                onChange={(e) => setSubhops(Math.max(1, Math.min(4, Number(e.target.value))))}
                className="w-14 px-2 py-1 border border-gray-300 rounded text-xs"
              />
            </label>
            <button
              onClick={() => selectedEntityId && loadSubgraph(selectedEntityId, subhops)}
              disabled={!selectedEntityId || subgraphLoading}
              className="mt-2 w-full px-3 py-1.5 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {subgraphLoading ? 'Loading...' : 'Load Subgraph'}
            </button>
          </div>

          {/* Stats */}
          {stats && (
            <div className="p-4 border-b border-gray-200">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                {view === 'full' ? 'Full' : 'Subgraph'} · {layout}
              </h3>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-sm">
                <div>
                  <dt className="text-[10px] uppercase text-gray-500">Nodes</dt>
                  <dd className="font-semibold text-gray-900">{stats.nodes}</dd>
                </div>
                <div>
                  <dt className="text-[10px] uppercase text-gray-500">Edges</dt>
                  <dd className="font-semibold text-gray-900">{stats.links}</dd>
                </div>
                <div>
                  <dt className="text-[10px] uppercase text-gray-500">Fraud Edges</dt>
                  <dd className="font-semibold text-rose-700">{stats.fraud}</dd>
                </div>
                <div>
                  <dt className="text-[10px] uppercase text-gray-500">High Risk</dt>
                  <dd className="font-semibold text-amber-700">{stats.highRisk}</dd>
                </div>
                <div>
                  <dt className="text-[10px] uppercase text-gray-500">Shell Cos.</dt>
                  <dd className="font-semibold text-red-700">{stats.shellCount}</dd>
                </div>
                <div>
                  <dt className="text-[10px] uppercase text-gray-500">Total Flow</dt>
                  <dd className="font-semibold text-gray-900">{formatINR(stats.totalAmt)}</dd>
                </div>
              </dl>
            </div>
          )}

          {/* Selected detail */}
          {selectedNode && (
            <div className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Selected</h3>
                <button
                  onClick={() => setSelectedNodeId(null)}
                  className="text-gray-400 hover:text-gray-600 text-lg leading-none"
                >×</button>
              </div>
              <p className="text-sm font-semibold text-gray-900 truncate">{selectedNode.name || selectedNode.id}</p>
              <p className="text-[10px] font-mono text-gray-400 truncate">{selectedNode.id}</p>
              <div className="mt-2 flex flex-wrap items-center gap-1.5">
                <span className={`text-[10px] px-2 py-0.5 rounded font-medium capitalize ${
                  selectedNode.type === 'shell_company' ? 'bg-red-100 text-red-700'
                  : selectedNode.type === 'business' ? 'bg-green-100 text-green-700'
                  : 'bg-blue-100 text-blue-700'
                }`}>
                  {(selectedNode.type || 'unknown').replace('_', ' ')}
                </span>
                {selectedNode.is_fraud && (
                  <span className="text-[10px] px-2 py-0.5 rounded font-medium bg-red-100 text-red-700">FRAUD</span>
                )}
              </div>
              <dl className="mt-3 space-y-2 text-xs">
                {selectedNode.branch && (
                  <div>
                    <dt className="text-gray-500">Branch</dt>
                    <dd className="font-medium text-gray-900">{selectedNode.branch}</dd>
                  </div>
                )}
                <div>
                  <dt className="text-gray-500">Risk score</dt>
                  <dd className="font-medium text-gray-900">{Number(selectedNode.risk_score || 0).toFixed(3)}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Degree</dt>
                  <dd className="font-medium text-gray-900">{selectedNode.degree || '—'}</dd>
                </div>
                {selectedNode.turnover != null && (
                  <div>
                    <dt className="text-gray-500">Turnover</dt>
                    <dd className="font-medium text-gray-900">{formatINR(selectedNode.turnover)}</dd>
                  </div>
                )}
                {selectedNode.net_flow != null && (
                  <div>
                    <dt className="text-gray-500">Net flow</dt>
                    <dd className={`font-medium ${selectedNode.net_flow >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                      {formatINR(selectedNode.net_flow)}
                    </dd>
                  </div>
                )}
              </dl>

              {selectedEdges.length > 0 && (
                <div className="mt-4">
                  <h4 className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                    Connections ({selectedEdges.length})
                  </h4>
                  <ul className="space-y-1.5 max-h-48 overflow-y-auto">
                    {selectedEdges.slice(0, 30).map((l, i) => {
                      const sId = typeof l.source === 'object' ? l.source.id : l.source;
                      const tId = typeof l.target === 'object' ? l.target.id : l.target;
                      const otherId = sId === selectedNode.id ? tId : sId;
                      const other = filteredData.nodes.find((n) => n.id === otherId);
                      const out = sId === selectedNode.id;
                      const isFraud = l.is_fraud || l.fraud || l.fraud_count > 0;
                      return (
                        <li key={i} className="text-xs border-l-2 pl-2"
                            style={{ borderColor: isFraud ? '#dc2626' : '#cbd5e1' }}>
                          <p className="font-medium text-gray-800 truncate">
                            {out ? '→' : '←'} {other?.name || otherId}
                          </p>
                          <p className="text-gray-500">
                            {formatINR(l.amount || l.weight)}
                            {l.transaction_count ? ` · ${l.transaction_count} txns` : ''}
                          </p>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}

              <button
                onClick={() => loadSubgraph(selectedNode.id, subhops)}
                disabled={subgraphLoading}
                className="mt-3 w-full px-3 py-1.5 text-xs bg-indigo-50 text-indigo-700 border border-indigo-200 rounded hover:bg-indigo-100 disabled:opacity-50"
              >
                Focus subgraph on this entity
              </button>
            </div>
          )}
        </aside>

        {/* Canvas */}
        <div ref={containerRef} className="flex-1 relative bg-gray-50 min-w-0 overflow-hidden">
          {activeLoading && (
            <div className="absolute inset-0 z-20 flex items-center justify-center text-gray-400">
              <div className="animate-pulse text-lg">Loading...</div>
            </div>
          )}
          {activeError && (
            <div className="absolute inset-0 z-20 flex items-center justify-center text-red-600 px-6 text-center">
              Error: {activeError}
            </div>
          )}
          {!activeLoading && !activeError && filteredData.nodes.length === 0 && (
            <div className="absolute inset-0 z-20 flex items-center justify-center text-gray-400 px-6 text-center">
              {view === 'subgraph' ? 'Pick an entity in the sidebar and click "Load Subgraph"' : 'No nodes match the current filters'}
            </div>
          )}
          {filteredData.nodes.length > 0 && layout === 'force' && (
            <ForceLayout
              data={filteredData}
              selectedNodeId={selectedNodeId}
              hoveredNodeId={hoveredNodeId}
              onSelectNode={setSelectedNodeId}
              onHoverNode={setHoveredNodeId}
              size={size}
            />
          )}
          {filteredData.nodes.length > 0 && layout === 'matrix' && (
            <MatrixLayout
              data={filteredData}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
            />
          )}
          {filteredData.nodes.length > 0 && layout === 'arc' && (
            <ArcLayout
              data={filteredData}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
            />
          )}
        </div>
      </div>
    </div>
  );
}
