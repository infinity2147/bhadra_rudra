/*
 * Minimal SVG Sankey for fund-flow journeys.
 * Scrollable viewport + wheel/button zoom (matches Force graph UX on Journey page).
 */

import { useMemo, useState, useRef, useCallback, useEffect, useId } from 'react';

function formatINR(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

const NODE_FILL = {
  individual: '#3b82f6',
  business: '#22c55e',
  shell_company: '#ef4444',
};

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

export default function Sankey({
  nodes = [],
  links = [],
  width: containerWidth = 900,
  onNodeClick,
  onLinkClick,
}) {
  const uid = useId().replace(/:/g, '');
  const scrollRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const panDrag = useRef(null);

  const layout = useMemo(() => {
    if (!nodes.length || !links.length) return null;

    const sideOrder = {
      upstream: 0, source: 0,
      alert: 1, focus: 1, loop: 1,
      neighbor: 2, downstream: 2,
    };
    const groups = {};
    for (const n of nodes) {
      const col = sideOrder[n.side] ?? 1;
      (groups[col] = groups[col] || []).push(n);
    }
    const colKeys = Object.keys(groups).map(Number).sort((a, b) => a - b);
    if (colKeys.length < 2) return null;

    const maxColLen = Math.max(...colKeys.map((c) => groups[c].length));
    const nodeWidth = 18;
    // Adapt node height when a column is dense, so we don't end up with a
    // single 2000px-tall strip the user has to scroll through (the failure
    // mode visible in the original bug screenshot).
    const targetHeight = 460;
    const nodeGap = maxColLen > 12 ? 4 : 12;
    const nodeHeight = Math.max(
      10,
      Math.min(28, Math.floor((targetHeight - 64) / maxColLen) - nodeGap),
    );
    const padX = 32;
    const padY = 24;
    const labelPad = 140;

    // Take the full available container width so the columns actually spread
    // out instead of clustering on the left.
    const contentWidth = Math.max(720, containerWidth);
    const contentHeight = Math.max(
      320,
      Math.min(targetHeight, maxColLen * (nodeHeight + nodeGap) + padY * 2),
    );
    const plotWidth = contentWidth - 2 * padX - labelPad;
    const colWidth = plotWidth / Math.max(colKeys.length - 1, 1);

    const throughput = {};
    for (const l of links) {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      throughput[s] = (throughput[s] || 0) + l.amount;
      throughput[t] = (throughput[t] || 0) + l.amount;
    }

    const positioned = {};
    colKeys.forEach((col, colIndex) => {
      const arr = groups[col];
      const sorted = [...arr].sort((a, b) => (throughput[b.id] || 0) - (throughput[a.id] || 0));
      const x = padX + colIndex * colWidth - nodeWidth / 2;
      const totalH = sorted.length * nodeHeight + (sorted.length - 1) * nodeGap;
      const startY = padY + Math.max(0, (contentHeight - padY * 2 - totalH) / 2);
      sorted.forEach((n, i) => {
        positioned[n.id] = {
          ...n,
          x,
          y: startY + i * (nodeHeight + nodeGap),
          w: nodeWidth,
          h: nodeHeight,
        };
      });
    });

    const safeLinks = links.filter((l) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return positioned[s] && positioned[t];
    });

    const maxAmount = Math.max(...safeLinks.map((l) => l.amount || 1));
    function edgeWidth(amount) {
      const minW = 2;
      const maxW = 22;
      const ratio = Math.log10((amount || 1) + 1) / Math.log10(maxAmount + 1);
      return Math.max(minW, minW + ratio * (maxW - minW));
    }

    return { positioned, safeLinks, contentWidth, contentHeight, edgeWidth };
  }, [nodes, links, containerWidth]);

  // Reset zoom when graph data changes
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
      scrollRef.current.scrollLeft = 0;
    }
  }, [nodes, links]);

  const applyZoom = useCallback((factor) => {
    setZoom((z) => clamp(z * factor, 0.35, 4));
  }, []);

  const resetView = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
      scrollRef.current.scrollLeft = 0;
    }
  }, []);

  // Wheel zoom on the graph panel (don't bubble to page scroll)
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;
      applyZoom(factor);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [applyZoom, layout]);

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    panDrag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    if (!panDrag.current) return;
    setPan({
      x: e.clientX - panDrag.current.x,
      y: e.clientY - panDrag.current.y,
    });
  };
  const onPointerUp = (e) => {
    panDrag.current = null;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
  };

  if (!layout) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm px-4 text-center">
        Not enough columns to render a Sankey — the alert is a single-column cluster (likely a closed cycle).
      </div>
    );
  }

  const { positioned, safeLinks, contentWidth, contentHeight, edgeWidth } = layout;
  const scaledW = contentWidth * zoom;
  const scaledH = contentHeight * zoom;

  return (
    <div className="relative w-full h-full min-h-0 flex flex-col">
      <div className="absolute top-2 right-2 z-20 flex items-center gap-1">
        <button
          type="button"
          onClick={() => applyZoom(1.2)}
          className="w-7 h-7 rounded border border-gray-300 bg-white text-gray-700 text-sm shadow-sm hover:bg-gray-50"
          title="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={() => applyZoom(1 / 1.2)}
          className="w-7 h-7 rounded border border-gray-300 bg-white text-gray-700 text-sm shadow-sm hover:bg-gray-50"
          title="Zoom out"
        >
          −
        </button>
        <button
          type="button"
          onClick={resetView}
          className="px-2 h-7 rounded border border-gray-300 bg-white text-gray-600 text-xs shadow-sm hover:bg-gray-50"
          title="Reset zoom"
        >
          Reset
        </button>
        <span className="text-[10px] text-gray-500 tabular-nums ml-1">{Math.round(zoom * 100)}%</span>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 min-h-0 w-full overflow-auto overscroll-contain bg-gray-50/50 cursor-grab active:cursor-grabbing"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
      >
        <div
          style={{
            width: scaledW,
            height: scaledH,
            minWidth: scaledW,
            minHeight: scaledH,
            transform: `translate(${pan.x}px, ${pan.y}px)`,
            transformOrigin: '0 0',
          }}
          className="bg-white"
        >
          <svg
            width={contentWidth}
            height={contentHeight}
            viewBox={`0 0 ${contentWidth} ${contentHeight}`}
            className="block"
            style={{
              width: scaledW,
              height: scaledH,
            }}
          >
            <defs>
              <marker id={`arrow-${uid}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
              </marker>
              <marker id={`arrow-red-${uid}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill="#dc2626" />
              </marker>
            </defs>

            {safeLinks.map((l, i) => {
              const s = typeof l.source === 'object' ? l.source.id : l.source;
              const t = typeof l.target === 'object' ? l.target.id : l.target;
              const a = positioned[s];
              const b = positioned[t];
              if (!a || !b) return null;
              const x1 = a.x + a.w;
              const y1 = a.y + a.h / 2;
              const x2 = b.x;
              const y2 = b.y + b.h / 2;
              const cx = (x1 + x2) / 2;
              const w = edgeWidth(l.amount || 1);
              const isFraud = (l.flags || []).includes('contains_fraud_txn') || (l.fraud_count || 0) > 0;
              const mlHigh = (l.ml_score ?? 0) >= 0.6;
              const stroke = isFraud ? '#dc2626' : mlHigh ? '#f59e0b' : '#94a3b8';
              const arrow = isFraud ? `url(#arrow-red-${uid})` : `url(#arrow-${uid})`;
              return (
                <g key={i} onClick={() => onLinkClick?.(l)} style={{ cursor: 'pointer' }}>
                  <path
                    d={`M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`}
                    fill="none"
                    stroke={stroke}
                    strokeOpacity={isFraud ? 0.65 : 0.45}
                    strokeWidth={w}
                    markerEnd={arrow}
                  />
                  {l.amount >= 500_000 && (
                    <text x={cx} y={(y1 + y2) / 2 - w / 2 - 4} textAnchor="middle" fontSize="11" fill="#475569">
                      {formatINR(l.amount)}
                    </text>
                  )}
                </g>
              );
            })}

            {Object.values(positioned).map((n) => {
              const fill = NODE_FILL[n.type] || '#6b7280';
              const isFocus = n.side === 'focus' || n.side === 'alert';
              const flagged = (n.flags || []).length > 0;
              // Scale label font with row height — keeps text legible when
              // nodes are tall and stops it overlapping when they're tight.
              const labelFont = Math.max(9, Math.min(12, n.h - 4));
              return (
                <g key={n.id} onClick={() => onNodeClick?.(n)} style={{ cursor: 'pointer' }}>
                  <rect
                    x={n.x}
                    y={n.y}
                    width={n.w}
                    height={n.h}
                    fill={fill}
                    stroke={isFocus ? '#1e1b4b' : flagged ? '#dc2626' : 'rgba(0,0,0,0.2)'}
                    strokeWidth={isFocus ? 2 : 1}
                    rx={3}
                  />
                  <text
                    x={n.x + n.w + 6}
                    y={n.y + n.h / 2 + labelFont / 3}
                    fontSize={labelFont}
                    fill="#111827"
                    fontWeight={isFocus ? 600 : 400}
                  >
                    {(n.name || n.id).slice(0, 28)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <p className="shrink-0 px-2 py-1 text-[10px] text-gray-400 border-t border-gray-100 bg-white">
        Scroll to explore · Wheel or +/- to zoom · Drag to pan
      </p>
    </div>
  );
}
