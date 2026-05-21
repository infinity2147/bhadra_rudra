/*
 * Minimal SVG Sankey for fund-flow journeys.
 *
 * Why custom: recharts has no Sankey, react-d3-tree only does trees, and the
 * heavy d3-sankey package would be overkill. Sankey layout is just:
 *   - group nodes into vertical columns by depth/side
 *   - sort within each column to minimise edge crossings
 *   - draw smooth cubic curves between columns with thickness ∝ amount
 *
 * We render two columns when "alert mode" doesn't have upstream/downstream:
 * fall back to a layered force graph instead (handled by the parent).
 */

import { useMemo } from 'react';

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

export default function Sankey({ nodes = [], links = [], height = 480, onNodeClick, onLinkClick }) {
  const layout = useMemo(() => {
    if (!nodes.length || !links.length) return null;

    // Column = depth + side bucket
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
    const numCols = Math.max(...Object.keys(groups).map(Number)) + 1;
    if (numCols < 2) return null;

    // Width allocation
    const width = 900;
    const padX = 100;
    const colWidth = (width - 2 * padX) / Math.max(numCols - 1, 1);
    const nodeWidth = 18;
    const nodeGap = 8;

    // Compute per-node throughput (sum of incoming + outgoing edge amounts)
    const throughput = {};
    for (const l of links) {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      throughput[s] = (throughput[s] || 0) + l.amount;
      throughput[t] = (throughput[t] || 0) + l.amount;
    }

    // Assign x/y positions to nodes (sorted by throughput descending within col)
    const positioned = {};
    Object.entries(groups).forEach(([col, arr]) => {
      const sorted = [...arr].sort((a, b) => (throughput[b.id] || 0) - (throughput[a.id] || 0));
      const x = padX + Number(col) * colWidth - nodeWidth / 2;
      const totalH = sorted.length * 28 + (sorted.length - 1) * nodeGap;
      const startY = Math.max(20, (height - totalH) / 2);
      sorted.forEach((n, i) => {
        positioned[n.id] = {
          ...n,
          x,
          y: startY + i * (28 + nodeGap),
          w: nodeWidth,
          h: 28,
        };
      });
    });

    // Filter links to those with both endpoints positioned
    const safeLinks = links.filter((l) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return positioned[s] && positioned[t];
    });

    // Edge thickness scale (log)
    const maxAmount = Math.max(...safeLinks.map(l => l.amount || 1));
    function edgeWidth(amount) {
      const minW = 1, maxW = 18;
      const ratio = Math.log10((amount || 1) + 1) / Math.log10(maxAmount + 1);
      return Math.max(minW, minW + ratio * (maxW - minW));
    }

    return { positioned, safeLinks, width, edgeWidth };
  }, [nodes, links, height]);

  if (!layout) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Not enough columns to render a Sankey — the alert is a single-column cluster (likely a closed cycle).
      </div>
    );
  }

  const { positioned, safeLinks, width, edgeWidth } = layout;

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
        </marker>
        <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#dc2626" />
        </marker>
      </defs>

      {/* Links */}
      {safeLinks.map((l, i) => {
        const s = typeof l.source === 'object' ? l.source.id : l.source;
        const t = typeof l.target === 'object' ? l.target.id : l.target;
        const a = positioned[s], b = positioned[t];
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
        const arrow = isFraud ? 'url(#arrow-red)' : 'url(#arrow)';
        return (
          <g key={i} onClick={() => onLinkClick?.(l)} style={{ cursor: 'pointer' }}>
            <path
              d={`M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`}
              fill="none"
              stroke={stroke}
              strokeOpacity={isFraud ? 0.6 : 0.4}
              strokeWidth={w}
              markerEnd={arrow}
            />
            {/* Label only for high-value edges */}
            {l.amount >= 500_000 && (
              <text x={cx} y={(y1 + y2) / 2 - w / 2 - 3} textAnchor="middle" fontSize="10" fill="#475569">
                {formatINR(l.amount)}
              </text>
            )}
          </g>
        );
      })}

      {/* Nodes */}
      {Object.values(positioned).map((n) => {
        const fill = NODE_FILL[n.type] || '#6b7280';
        const isFocus = n.side === 'focus' || n.side === 'alert';
        const flagged = (n.flags || []).length > 0;
        return (
          <g key={n.id} onClick={() => onNodeClick?.(n)} style={{ cursor: 'pointer' }}>
            <rect
              x={n.x} y={n.y} width={n.w} height={n.h}
              fill={fill}
              stroke={isFocus ? '#1e1b4b' : flagged ? '#dc2626' : 'rgba(0,0,0,0.2)'}
              strokeWidth={isFocus ? 2 : 1}
              rx={3}
            />
            <text
              x={n.x + n.w + 4}
              y={n.y + n.h / 2 + 3}
              fontSize="11"
              fill="#111827"
              fontWeight={isFocus ? 600 : 400}
            >
              {(n.name || n.id).slice(0, 24)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
