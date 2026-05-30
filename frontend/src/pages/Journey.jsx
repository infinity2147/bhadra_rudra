import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { fetchAPI, downloadFromAPI } from '../api';

const NODE_FILL = {
  individual: '#3b82f6',
  business: '#22c55e',
  shell_company: '#ef4444',
  focus: '#1e1b4b',
};

const SIDE_LABEL = {
  upstream: 'Sources',
  focus: 'Focal Entity',
  loop: 'In Cycle',
  downstream: 'Destinations',
  alert: 'Alert Entity',
  neighbor: 'Counterparty',
};

function formatINR(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

function FlagPill({ text, tone = 'amber' }) {
  const tones = {
    red: 'bg-red-50 text-red-700 ring-red-200',
    amber: 'bg-amber-50 text-amber-800 ring-amber-200',
    indigo: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
    gray: 'bg-gray-50 text-gray-700 ring-gray-200',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium ring-1 ${tones[tone]}`}>
      {text}
    </span>
  );
}

function GraphLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-600">
      <span className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-blue-500 shrink-0" /> Individual
      </span>
      <span className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-green-500 shrink-0" /> Business
      </span>
      <span className="flex items-center gap-1.5">
        <span className="w-2.5 h-2.5 rounded-full bg-red-500 shrink-0" /> Shell Co.
      </span>
      <span className="flex items-center gap-1.5">
        <span className="w-5 h-0.5 bg-red-600 shrink-0" /> Fraud edge
      </span>
      <span className="flex items-center gap-1.5">
        <span className="w-5 h-0.5 bg-amber-500 shrink-0" /> High ML score
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Flow Story view — replaces Sankey
//
// What it shows: sources → focus → destinations as readable cards with
// curved arrows. Hard-caps at top-N edges by amount so it never becomes a
// hairball, no matter how many entities the trace returns.
// ─────────────────────────────────────────────────────────────────────────────

function FlowStoryView({ data, focusId, panelSize, selectedNodeId, onSelectNode }) {
  const TOP_EDGES = 18;       // never show more — keeps the diagram readable
  const MAX_PER_COL = 9;      // max nodes per column

  const { columns, edges } = useMemo(() => {
    if (!data?.nodes?.length) return { columns: { upstream: [], focus: [], downstream: [] }, edges: [] };

    // Normalise links — react-force-graph used to mutate source/target into
    // node refs; defensive normalisation keeps the comparisons working.
    const allLinks = data.links.map(l => ({
      ...l,
      source: typeof l.source === 'object' ? l.source.id : l.source,
      target: typeof l.target === 'object' ? l.target.id : l.target,
    }));

    const nodeMap = new Map(data.nodes.map(n => [n.id, n]));
    const focus = focusId
      || data.nodes.find(n => n.side === 'focus' || n.side === 'alert')?.id
      || data.nodes[0]?.id;
    const alertSet = new Set(
      data.nodes.filter(n => n.side === 'focus' || n.side === 'alert').map(n => n.id),
    );
    alertSet.add(focus);

    // 1. Rank top-N edges by raw amount.  No "boost focus": for funnel and
    //    smurfing patterns the genuinely highest-amount edges already touch
    //    the alert entities, and the boost was burying them.
    const ranked = [...allLinks].sort((a, b) => b.amount - a.amount).slice(0, TOP_EDGES);

    // 2. Build per-node net-flow stats from those edges + always include
    //    the alert entities in the visible set, even if no top-N edge
    //    touches them (so the focus column is never empty).
    const outFlow = new Map();
    const inFlow  = new Map();
    const visible = new Set(alertSet);
    for (const e of ranked) {
      visible.add(e.source); visible.add(e.target);
      outFlow.set(e.source, (outFlow.get(e.source) || 0) + e.amount);
      inFlow.set(e.target,  (inFlow.get(e.target)  || 0) + e.amount);
    }

    // 3. Classify each visible node into a column by net-flow direction.
    //    Alert entities default to focus; balanced nodes (in≈out) also focus.
    const upstream = [];
    const focusCol = [];
    const downstream = [];
    for (const id of visible) {
      const out = outFlow.get(id) || 0;
      const inn = inFlow.get(id) || 0;
      const isAlert = alertSet.has(id);
      if (isAlert && out + inn === 0) {
        focusCol.push(id);
      } else if (out > inn * 1.5 && !isAlert) {
        upstream.push(id);
      } else if (inn > out * 1.5 && !isAlert) {
        downstream.push(id);
      } else {
        focusCol.push(id);  // balanced, mid-chain, or alert
      }
    }

    // 4. Sort each column by total activity (most active first), cap at MAX_PER_COL.
    const activity = id => (outFlow.get(id) || 0) + (inFlow.get(id) || 0);
    const sortAndCap = arr => arr.sort((a, b) => activity(b) - activity(a)).slice(0, MAX_PER_COL);
    const finalUp    = sortAndCap(upstream);
    const finalFocus = sortAndCap(focusCol);
    const finalDown  = sortAndCap(downstream);
    const finalVisible = new Set([...finalUp, ...finalFocus, ...finalDown]);

    return {
      columns: {
        upstream:   finalUp.map(id => nodeMap.get(id)).filter(Boolean),
        focus:      finalFocus.map(id => nodeMap.get(id)).filter(Boolean),
        downstream: finalDown.map(id => nodeMap.get(id)).filter(Boolean),
      },
      edges: ranked.filter(e => finalVisible.has(e.source) && finalVisible.has(e.target)),
    };
  }, [data, focusId]);

  const width = panelSize.width;
  const height = Math.max(panelSize.height, 60 + 70 * Math.max(
    columns.upstream.length, columns.focus.length, columns.downstream.length, 1,
  ));
  const colWidth = width / 3;
  const cardW = Math.min(190, colWidth - 24);
  const cardH = 56;
  const headerY = 22;

  function positionFor(colIdx, rowIdx, total) {
    const cx = colIdx * colWidth + colWidth / 2;
    const usable = height - headerY - 50;
    const step = usable / Math.max(total, 1);
    const cy = headerY + 30 + step * (rowIdx + 0.5);
    return { cx, cy };
  }

  const positions = useMemo(() => {
    const p = {};
    columns.upstream.forEach((n, i) => { p[n.id] = positionFor(0, i, columns.upstream.length); });
    columns.focus.forEach((n, i) =>    { p[n.id] = positionFor(1, i, columns.focus.length); });
    columns.downstream.forEach((n, i) => { p[n.id] = positionFor(2, i, columns.downstream.length); });
    return p;
  }, [columns, width, height]);   // eslint-disable-line react-hooks/exhaustive-deps

  const visibleEdges = edges.filter(e => positions[e.source] && positions[e.target]);
  const maxAmount = Math.max(...visibleEdges.map(e => e.amount), 1);

  const [hoverEdge, setHoverEdge] = useState(null);

  function edgeColor(e) {
    if (e.fraud_count > 0 || e.is_fraud) return '#dc2626';
    if ((e.ml_score ?? 0) >= 0.6) return '#f59e0b';
    return '#94a3b8';
  }

  const empty = !columns.upstream.length && !columns.focus.length && !columns.downstream.length;
  if (empty) {
    return (
      <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
        No flow data to display for this trace.
      </div>
    );
  }

  return (
    <div className="absolute inset-0 overflow-auto">
      <svg width={width} height={height} className="bg-gradient-to-b from-white to-gray-50">
        {/* Column headers */}
        {[
          { x: colWidth * 0.5, label: 'SOURCES', count: columns.upstream.length },
          { x: colWidth * 1.5, label: 'FOCUS',   count: columns.focus.length },
          { x: colWidth * 2.5, label: 'DESTINATIONS', count: columns.downstream.length },
        ].map(h => (
          <g key={h.label}>
            <text x={h.x} y={headerY} textAnchor="middle"
                  className="text-[11px] font-semibold uppercase tracking-wider"
                  fill="#6b7280">
              {h.label}
              {h.count > 0 && <tspan fill="#9ca3af" fontWeight={400}>  ({h.count})</tspan>}
            </text>
            <line x1={h.x - 30} x2={h.x + 30} y1={headerY + 4} y2={headerY + 4}
                  stroke="#e5e7eb" />
          </g>
        ))}

        {/* Curved edges first so cards sit on top */}
        {visibleEdges.map((e, i) => {
          const s = positions[e.source];
          const t = positions[e.target];
          const isHover = hoverEdge === i;
          const color = edgeColor(e);
          const sx = s.cx + cardW / 2;
          const sy = s.cy;
          const tx = t.cx - cardW / 2;
          const ty = t.cy;
          const midX = (sx + tx) / 2;
          const w = Math.max(1.5, Math.min(7, 1 + Math.log10(e.amount + 1) * 0.9));
          return (
            <g key={`e-${i}`}
               onMouseEnter={() => setHoverEdge(i)}
               onMouseLeave={() => setHoverEdge(null)}
               style={{ cursor: 'pointer' }}>
              <path
                d={`M ${sx} ${sy} C ${midX} ${sy}, ${midX} ${ty}, ${tx} ${ty}`}
                fill="none"
                stroke={color}
                strokeWidth={isHover ? w + 2 : w}
                opacity={isHover ? 1 : 0.55}
              />
              {/* Amount label, centered on the arc */}
              <g transform={`translate(${midX}, ${(sy + ty) / 2 - 4})`}>
                <rect x={-32} y={-9} width={64} height={16} rx={8}
                      fill="white" stroke={color} strokeWidth={0.75}
                      opacity={isHover ? 1 : 0.9} />
                <text x={0} y={3} textAnchor="middle"
                      className="text-[10px] font-semibold tabular-nums"
                      fill={color}>
                  {formatINR(e.amount)}
                </text>
              </g>
              {/* Arrow head */}
              <polygon
                points={`${tx},${ty} ${tx - 6},${ty - 4} ${tx - 6},${ty + 4}`}
                fill={color}
                opacity={isHover ? 1 : 0.7}
              />
            </g>
          );
        })}

        {/* Node cards */}
        {Object.entries(positions).map(([id, pos]) => {
          const n = data.nodes.find(nn => nn.id === id);
          if (!n) return null;
          const isSel = selectedNodeId === id;
          const isFocus = n.side === 'focus' || n.side === 'alert' || id === focusId;
          const fill = NODE_FILL[n.type] || '#9ca3af';
          const flags = (n.flags || []).slice(0, 2);
          return (
            <g key={`n-${id}`}
               transform={`translate(${pos.cx - cardW / 2}, ${pos.cy - cardH / 2})`}
               style={{ cursor: 'pointer' }}
               onClick={() => onSelectNode(id)}>
              <rect width={cardW} height={cardH} rx={6}
                    fill="white"
                    stroke={isSel ? '#1e1b4b' : isFocus ? '#4338ca' : '#cbd5e1'}
                    strokeWidth={isSel ? 2.5 : isFocus ? 2 : 1} />
              {/* Type color strip */}
              <rect x={0} y={0} width={4} height={cardH} fill={fill} rx={2} />
              <text x={12} y={18} className="text-[11px] font-semibold" fill="#1f2937">
                {(n.name || id).slice(0, 22)}
              </text>
              <text x={12} y={32} className="text-[10px]" fill="#6b7280">
                {(n.type || '').replace('_', ' ')}
                {n.branch ? ` · ${n.branch.slice(0, 10)}` : ''}
              </text>
              {flags.length > 0 && (
                <g transform={`translate(12, 38)`}>
                  {flags.map((f, i) => (
                    <g key={f} transform={`translate(${i * 64}, 0)`}>
                      <rect width={60} height={12} rx={3} fill="#fef3c7" stroke="#fcd34d" />
                      <text x={30} y={9} textAnchor="middle"
                            className="text-[8px] font-medium" fill="#92400e">
                        {f.replace(/_/g, ' ').slice(0, 11)}
                      </text>
                    </g>
                  ))}
                </g>
              )}
              {/* Risk score badge */}
              {n.risk_score >= 0.4 && (
                <g transform={`translate(${cardW - 30}, 4)`}>
                  <rect width={26} height={14} rx={3}
                        fill={n.risk_score >= 0.7 ? '#fee2e2' : '#fef3c7'}
                        stroke={n.risk_score >= 0.7 ? '#dc2626' : '#f59e0b'}
                        strokeWidth={0.5} />
                  <text x={13} y={11} textAnchor="middle"
                        className="text-[9px] font-bold tabular-nums"
                        fill={n.risk_score >= 0.7 ? '#991b1b' : '#92400e'}>
                    {(n.risk_score * 100).toFixed(0)}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Hover tooltip */}
        {hoverEdge != null && visibleEdges[hoverEdge] && (() => {
          const e = visibleEdges[hoverEdge];
          const s = positions[e.source];
          const t = positions[e.target];
          const tx = (s.cx + t.cx) / 2 + cardW / 4;
          const ty = (s.cy + t.cy) / 2 + 30;
          return (
            <g transform={`translate(${tx}, ${ty})`}>
              <rect x={-90} y={-2} width={180} height={56} rx={6}
                    fill="#111827" opacity={0.92} />
              <text x={-82} y={14} className="text-[10px]" fill="white">
                <tspan fontWeight={600}>{formatINR(e.amount)}</tspan>
                <tspan fill="#9ca3af"> · {e.txn_count} txn{e.txn_count > 1 ? 's' : ''}</tspan>
              </text>
              <text x={-82} y={28} className="text-[10px]" fill="#9ca3af">
                {e.first_seen?.slice(0, 16)} – {e.last_seen?.slice(0, 16)}
              </text>
              <text x={-82} y={42} className="text-[10px]" fill="#9ca3af">
                velocity {e.txn_velocity?.toFixed(2) || '—'}/h
                {e.ml_score != null && ` · ML ${(e.ml_score * 100).toFixed(0)}`}
                {e.fraud_count > 0 && <tspan fill="#fca5a5" fontWeight={600}> · FRAUD</tspan>}
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline view — each entity gets a row, transactions plotted in time
//
// Reveals: burst patterns (clusters of dots), layering chains (diagonal
// stripes from upper rows down), cycles (criss-crossing in both directions).
// ─────────────────────────────────────────────────────────────────────────────

function TimelineView({ data, panelSize, selectedNodeId, onSelectNode }) {
  const MAX_ENTITIES = 14;

  const { entities, txns, tMin, tMax, entityIdx } = useMemo(() => {
    const allTxns = (data.timeline || []).filter(t => t.timestamp);
    if (!allTxns.length) return { entities: [], txns: [], tMin: 0, tMax: 1, entityIdx: new Map() };

    // Count entity involvement (sender or receiver)
    const counts = new Map();
    for (const t of allTxns) {
      counts.set(t.sender_id, (counts.get(t.sender_id) || 0) + 1);
      counts.set(t.receiver_id, (counts.get(t.receiver_id) || 0) + 1);
    }
    // Top entities by involvement (focus + alert entities ranked first)
    const focusSet = new Set(
      data.nodes.filter(n => n.side === 'focus' || n.side === 'alert').map(n => n.id),
    );
    const sortedIds = [...counts.keys()].sort((a, b) => {
      const af = focusSet.has(a) ? 1 : 0;
      const bf = focusSet.has(b) ? 1 : 0;
      if (af !== bf) return bf - af;
      return (counts.get(b) || 0) - (counts.get(a) || 0);
    }).slice(0, MAX_ENTITIES);

    const idx = new Map(sortedIds.map((id, i) => [id, i]));
    const nodeMap = new Map(data.nodes.map(n => [n.id, n]));
    const ents = sortedIds.map(id => ({
      id,
      name: nodeMap.get(id)?.name || id,
      type: nodeMap.get(id)?.type,
      risk: nodeMap.get(id)?.risk_score || 0,
      txns: counts.get(id) || 0,
    }));

    const times = allTxns.map(t => new Date(t.timestamp).getTime()).filter(t => !isNaN(t));
    return {
      entities: ents,
      txns: allTxns.filter(t => idx.has(t.sender_id) && idx.has(t.receiver_id)),
      tMin: Math.min(...times),
      tMax: Math.max(...times),
      entityIdx: idx,
    };
  }, [data]);

  if (!entities.length) {
    return (
      <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
        No timeline data to display.
      </div>
    );
  }

  const labelW = 170;
  const padTop = 30;
  const padBottom = 26;
  const rowH = Math.max(28, Math.floor((panelSize.height - padTop - padBottom) / entities.length));
  const innerW = panelSize.width - labelW - 30;
  const totalH = padTop + rowH * entities.length + padBottom;
  const tRange = tMax - tMin || 1;

  function xFor(ts) {
    const t = new Date(ts).getTime();
    if (isNaN(t)) return labelW;
    return labelW + ((t - tMin) / tRange) * innerW;
  }

  // Time-axis tick labels
  const tickCount = 5;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => {
    const t = tMin + (tRange * i) / tickCount;
    return { t, x: xFor(new Date(t).toISOString()) };
  });

  const [hoverTxn, setHoverTxn] = useState(null);

  return (
    <div className="absolute inset-0 overflow-auto">
      <svg width={Math.max(panelSize.width, 800)} height={Math.max(totalH, panelSize.height)}>
        {/* Entity rows */}
        {entities.map((e, i) => {
          const y = padTop + i * rowH + rowH / 2;
          const isSel = selectedNodeId === e.id;
          const fill = NODE_FILL[e.type] || '#9ca3af';
          return (
            <g key={e.id}>
              <rect x={0} y={padTop + i * rowH} width={panelSize.width} height={rowH}
                    fill={i % 2 === 0 ? '#fafafa' : 'white'} opacity={isSel ? 0.5 : 1} />
              <rect x={2} y={padTop + i * rowH + 4} width={3} height={rowH - 8}
                    fill={fill} rx={1.5} />
              <text x={labelW - 10} y={y - 3} textAnchor="end"
                    className="text-[11px] font-medium"
                    fill={isSel ? '#1e1b4b' : '#1f2937'}
                    fontWeight={isSel ? 700 : 500}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onSelectNode(e.id)}>
                {e.name.slice(0, 22)}
              </text>
              <text x={labelW - 10} y={y + 9} textAnchor="end"
                    className="text-[9px]" fill="#9ca3af">
                {e.txns} txn{e.txns > 1 ? 's' : ''}
                {e.risk >= 0.4 && <tspan fill="#dc2626"> · risk {(e.risk * 100).toFixed(0)}</tspan>}
              </text>
              <line x1={labelW} x2={labelW + innerW} y1={y} y2={y}
                    stroke="#e5e7eb" strokeDasharray="2,3" />
            </g>
          );
        })}

        {/* Time-axis ticks */}
        {ticks.map((tk, i) => (
          <g key={i}>
            <line x1={tk.x} x2={tk.x} y1={padTop - 4} y2={padTop + entities.length * rowH}
                  stroke="#f3f4f6" />
            <text x={tk.x} y={padTop + entities.length * rowH + 16}
                  textAnchor="middle" className="text-[9px]" fill="#9ca3af">
              {new Date(tk.t).toISOString().slice(5, 16).replace('T', ' ')}
            </text>
          </g>
        ))}

        {/* Transactions: vertical arrows from sender row to receiver row */}
        {txns.map((t, i) => {
          const si = entityIdx.get(t.sender_id);
          const ri = entityIdx.get(t.receiver_id);
          if (si == null || ri == null) return null;
          const x = xFor(t.timestamp);
          const y1 = padTop + si * rowH + rowH / 2;
          const y2 = padTop + ri * rowH + rowH / 2;
          const isFraud = !!t.is_fraud;
          const color = isFraud ? '#dc2626' : '#3b82f6';
          const isHover = hoverTxn === i;
          return (
            <g key={i}
               onMouseEnter={() => setHoverTxn(i)}
               onMouseLeave={() => setHoverTxn(null)}
               style={{ cursor: 'pointer' }}>
              <line x1={x} x2={x} y1={y1} y2={y2}
                    stroke={color} strokeWidth={isHover ? 2.5 : 1.2}
                    opacity={isHover ? 1 : 0.7} />
              <circle cx={x} cy={y1} r={isHover ? 4 : 2.5} fill={color}
                      opacity={isHover ? 1 : 0.85} />
              <polygon
                points={`${x},${y2} ${x - 3.5},${y2 + (y2 > y1 ? -5 : 5)} ${x + 3.5},${y2 + (y2 > y1 ? -5 : 5)}`}
                fill={color} opacity={isHover ? 1 : 0.85}
              />
            </g>
          );
        })}

        {/* Tooltip */}
        {hoverTxn != null && txns[hoverTxn] && (() => {
          const t = txns[hoverTxn];
          const x = Math.min(xFor(t.timestamp) + 8, panelSize.width - 220);
          const y = padTop + 4;
          return (
            <g transform={`translate(${x}, ${y})`}>
              <rect x={0} y={0} width={210} height={66} rx={6}
                    fill="#111827" opacity={0.93} />
              <text x={8} y={16} className="text-[10px]" fill="white" fontWeight={600}>
                {formatINR(t.amount)}
              </text>
              <text x={8} y={30} className="text-[10px]" fill="#9ca3af">
                {(t.sender_name || t.sender_id || '').slice(0, 24)} →
              </text>
              <text x={8} y={42} className="text-[10px]" fill="#9ca3af">
                {(t.receiver_name || t.receiver_id || '').slice(0, 24)}
              </text>
              <text x={8} y={56} className="text-[10px]" fill="#9ca3af">
                {(t.timestamp || '').slice(0, 16)} · {t.transaction_type || ''}
                {t.is_fraud && <tspan fill="#fca5a5" fontWeight={600}> · FRAUD</tspan>}
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}

export default function Journey() {
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const alertParam = params.get('alert');
  const entityParam = params.get('entity');

  const [mode, setMode] = useState(alertParam ? 'alert' : 'entity'); // 'alert' | 'entity'
  const [alertId, setAlertId] = useState(alertParam || '');
  const [entityId, setEntityId] = useState(entityParam || '');
  const [direction, setDirection] = useState('both');
  const [hops, setHops] = useState(2);
  const [minAmount, setMinAmount] = useState(0);
  const [includeNeighbors, setIncludeNeighbors] = useState(false);

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [alertOptions, setAlertOptions] = useState([]);
  const [entityOptions, setEntityOptions] = useState([]);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [viewMode, setViewMode] = useState('flow'); // 'flow' | 'timeline'
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const graphPanelRef = useRef(null);
  const graphContainerRef = useRef(null);   // the whole lower section, what goes fullscreen
  const [panelSize, setPanelSize] = useState({ width: 800, height: 600 });
  const [panelMode, setPanelMode] = useState('large');  // 'compact' | 'large' | 'fullscreen'
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Sync the React state with the browser's actual fullscreen status
  useEffect(() => {
    const onFsChange = () => {
      const fs = !!document.fullscreenElement;
      setIsFullscreen(fs);
      if (!fs && panelMode === 'fullscreen') setPanelMode('large');
    };
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, [panelMode]);

  function toggleFullscreen() {
    const el = graphContainerRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
    } else {
      el.requestFullscreen?.().then(() => setPanelMode('fullscreen')).catch(() => {});
    }
  }

  useEffect(() => {
    const el = graphPanelRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setPanelSize({
        width: Math.max(280, Math.floor(width)),
        height: Math.max(200, Math.floor(height)),
      });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Load alert and entity options
  useEffect(() => {
    fetchAPI('/api/alerts')
      .then(d => setAlertOptions((d.alerts || []).slice(0, 200)))
      .catch(() => {});
    fetchAPI('/api/entities')
      .then(d => setEntityOptions(d.entities || []))
      .catch(() => {});
  }, []);

  const fetchTrace = useCallback(async () => {
    setLoading(true); setError(null); setSelectedNodeId(null);
    try {
      let url;
      if (mode === 'alert' && alertId) {
        const qs = includeNeighbors ? '?include_neighbors=true' : '';
        url = `/api/journey/alert/${alertId}${qs}`;
      } else if (mode === 'entity' && entityId) {
        const p = new URLSearchParams({ direction, hops: String(hops) });
        if (minAmount > 0) p.set('min_amount', String(minAmount));
        url = `/api/journey/${entityId}?${p.toString()}`;
      } else {
        setData(null); setLoading(false); return;
      }
      const result = await fetchAPI(url);
      setData(result);
      // Sync URL
      const next = new URLSearchParams();
      if (mode === 'alert') next.set('alert', alertId);
      else next.set('entity', entityId);
      setParams(next, { replace: true });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [mode, alertId, entityId, direction, hops, minAmount, includeNeighbors, setParams]);

  // Auto-fetch when key inputs change
  useEffect(() => {
    if ((mode === 'alert' && alertId) || (mode === 'entity' && entityId)) {
      fetchTrace();
    }
  }, [mode, alertId, entityId, direction, hops, minAmount, includeNeighbors, fetchTrace]);

  // SHAP explain when an alert is loaded
  useEffect(() => {
    if (mode !== 'alert' || !alertId) { setExplanation(null); return; }
    setExplainLoading(true);
    fetchAPI(`/api/alerts/${alertId}/explain`)
      .then((d) => setExplanation(d && !d.error ? d : null))
      .catch(() => setExplanation(null))
      .finally(() => setExplainLoading(false));
  }, [mode, alertId]);

  // Decide effective view mode: force graph for cycles or dense traces,
  // Selected node + edges
  const selectedNode = data?.nodes?.find(n => n.id === selectedNodeId);
  const selectedEdges = data?.links?.filter(
    l => l.source === selectedNodeId || l.target === selectedNodeId
  ) || [];

  return (
    <div className="w-full max-w-full overflow-x-hidden">
      {/* Header */}
      <div className="px-6 pt-6 pb-3">
        <h1 className="text-2xl font-bold text-gray-900">Fund Journey Tracer</h1>
        <p className="text-sm text-gray-500 mt-1">
          Map the end-to-end movement of funds — pick an alert to retrace its chain, or a focal entity to walk forward and backward through the graph
        </p>
      </div>

      {/* Controls */}
      <div className="px-6 py-3 border-b border-gray-200 bg-white flex items-center gap-4 flex-wrap">
        <div className="inline-flex rounded-lg border border-gray-300 p-0.5">
          <button
            onClick={() => setMode('alert')}
            className={`px-3 py-1 text-sm rounded-md transition-colors ${
              mode === 'alert' ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            From Alert
          </button>
          <button
            onClick={() => setMode('entity')}
            className={`px-3 py-1 text-sm rounded-md transition-colors ${
              mode === 'entity' ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            From Entity
          </button>
        </div>

        {mode === 'alert' ? (
          <>
            <select
              value={alertId}
              onChange={e => setAlertId(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-sm w-72 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">Select alert…</option>
              {alertOptions.map(a => (
                <option key={a.alert_id} value={a.alert_id}>
                  {a.alert_id} — {a.pattern_type} ({a.severity})
                </option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={includeNeighbors}
                onChange={e => setIncludeNeighbors(e.target.checked)}
                className="rounded border-gray-300"
              />
              Show 1-hop neighbours
            </label>
          </>
        ) : (
          <>
            <select
              value={entityId}
              onChange={e => setEntityId(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-sm w-64 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              <option value="">Select entity…</option>
              {entityOptions.map(e => (
                <option key={e.entity_id} value={e.entity_id}>
                  {e.name} ({e.type})
                </option>
              ))}
            </select>
            <select
              value={direction}
              onChange={e => setDirection(e.target.value)}
              className="px-3 py-1.5 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              <option value="both">Both directions</option>
              <option value="forward">Outgoing only</option>
              <option value="backward">Incoming only</option>
            </select>
            <label className="text-sm text-gray-700">
              Hops
              <input
                type="number"
                min={1}
                max={4}
                value={hops}
                onChange={e => setHops(Number(e.target.value))}
                className="ml-1.5 w-14 px-2 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </label>
            <label className="text-sm text-gray-700">
              Min amount
              <input
                type="number"
                step={100000}
                value={minAmount}
                onChange={e => setMinAmount(Number(e.target.value))}
                className="ml-1.5 w-28 px-2 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
              />
            </label>
          </>
        )}

        {/* View mode toggle */}
        <div className="inline-flex rounded-lg border border-gray-300 p-0.5 ml-auto">
          {[
            { k: 'flow',     label: 'Flow Story' },
            { k: 'timeline', label: 'Timeline' },
          ].map((m) => (
            <button
              key={m.k}
              onClick={() => setViewMode(m.k)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                viewMode === m.k ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>

        {mode === 'alert' && alertId && (
          <button
            onClick={() => downloadFromAPI(`/api/fiu/package/${alertId}`, `FIU_${alertId}.zip`)}
            className="px-3 py-1.5 text-sm bg-emerald-600 text-white rounded-md hover:bg-emerald-700"
          >
            Download FIU Package
          </button>
        )}
      </div>

      {/* Summary cards */}
      {data && data.summary && (
        <div className="px-6 py-3 bg-gray-50 border-b border-gray-200 grid grid-cols-2 md:grid-cols-5 gap-3">
          <div>
            <p className="text-[11px] uppercase tracking-wide text-gray-500">Entities in Journey</p>
            <p className="text-lg font-bold text-gray-900">{data.summary.n_nodes}</p>
          </div>
          <div>
            <p className="text-[11px] uppercase tracking-wide text-gray-500">Flow Links</p>
            <p className="text-lg font-bold text-gray-900">{data.summary.n_links}</p>
          </div>
          {data.summary.total_inflow != null && (
            <div>
              <p className="text-[11px] uppercase tracking-wide text-gray-500">Inflow to Focus</p>
              <p className="text-lg font-bold text-emerald-700">{formatINR(data.summary.total_inflow)}</p>
            </div>
          )}
          {data.summary.total_outflow != null && (
            <div>
              <p className="text-[11px] uppercase tracking-wide text-gray-500">Outflow from Focus</p>
              <p className="text-lg font-bold text-rose-700">{formatINR(data.summary.total_outflow)}</p>
            </div>
          )}
          <div>
            <p className="text-[11px] uppercase tracking-wide text-gray-500">Flagged Txns in Journey</p>
            <p className="text-lg font-bold text-red-700">{data.summary.n_fraud_txns ?? 0}</p>
          </div>
        </div>
      )}

      {/* Red flags */}
      {data && data.summary?.red_flags?.length > 0 && (
        <div className="px-6 py-2.5 bg-red-50 border-b border-red-200 flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-red-800 uppercase tracking-wide">Red Flags</span>
          {data.summary.red_flags.map((f, i) => (
            <FlagPill key={i} text={f} tone="red" />
          ))}
        </div>
      )}

      {/* Dominant paths — top fund corridors */}
      {data && data.dominant_paths?.length > 0 && (
        <div className="px-6 py-3 bg-white border-b border-gray-200">
          <h3 className="text-sm font-semibold text-gray-800 mb-2">
            Dominant Flow Corridors
            <span className="text-xs font-normal text-gray-500 ml-2">
              Top {data.dominant_paths.length} highest-throughput paths (risk-weighted Dijkstra)
            </span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {data.dominant_paths.map((dp, i) => (
              <div key={i} className="rounded-lg border border-gray-200 bg-gray-50 p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono text-gray-500">#{i + 1}</span>
                  <span className="text-xs font-semibold text-rose-700">
                    risk {(dp.path_risk_score ?? 0).toFixed(2)}
                  </span>
                </div>
                <p className="text-xs text-gray-700 font-medium">
                  {(dp.path || []).map((n, j) => (
                    <span key={j}>
                      {j > 0 && <span className="text-gray-400 mx-1">→</span>}
                      <span
                        className="text-indigo-700 cursor-pointer hover:underline"
                        onClick={() => setSelectedNodeId(n)}
                      >
                        {(data.nodes?.find((node) => node.id === n)?.name || n).slice(0, 18)}
                      </span>
                    </span>
                  ))}
                </p>
                <div className="mt-1.5 flex items-center justify-between text-[11px] text-gray-600">
                  <span>{dp.hops} hops</span>
                  <span className="font-medium">{formatINR(dp.bottleneck_amount)}</span>
                </div>
                {dp.path_flags?.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {dp.path_flags.slice(0, 4).map((f, j) => (
                      <FlagPill key={j} text={f.replace(/_/g, ' ')} tone="amber" />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Flow distribution + terminal classification side-by-side */}
      {data && (data.flow_distribution?.length > 0 || data.terminal_classification) && (
        <div className="px-6 py-3 bg-white border-b border-gray-200 grid grid-cols-1 lg:grid-cols-2 gap-4">
          {data.flow_distribution?.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">
                Parallel-Path Flow Distribution
                <span className="text-xs font-normal text-gray-500 ml-2">
                  Max-flow analysis — reveals layering across multiple chains
                </span>
              </h3>
              <div className="space-y-2">
                {data.flow_distribution.map((fd, i) => {
                  const sinkNode = data.nodes?.find((n) => n.id === fd.sink);
                  return (
                    <div key={i} className="rounded-md border border-gray-200 p-2.5 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-gray-800">
                          to{' '}
                          <span
                            className="text-indigo-700 cursor-pointer hover:underline"
                            onClick={() => setSelectedNodeId(fd.sink)}
                          >
                            {(sinkNode?.name || fd.sink).slice(0, 24)}
                          </span>
                        </span>
                        <span className="font-semibold text-gray-900">{formatINR(fd.max_flow_amount)}</span>
                      </div>
                      <p className="text-[11px] text-gray-500 mt-0.5">
                        {fd.n_parallel_paths_estimate} parallel paths • {fd.top_edges?.length || 0} edges contribute
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {data.terminal_classification && Object.keys(data.terminal_classification).length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">
                Where the Funds Ended Up
                <span className="text-xs font-normal text-gray-500 ml-2">
                  Terminal-destination classification
                </span>
              </h3>
              <div className="space-y-1.5">
                {Object.entries(data.terminal_classification).map(([category, ids]) => {
                  const palette = {
                    cash_out: 'bg-rose-50 border-rose-200 text-rose-800',
                    cross_border: 'bg-amber-50 border-amber-200 text-amber-800',
                    conversion: 'bg-violet-50 border-violet-200 text-violet-800',
                    layered: 'bg-gray-50 border-gray-200 text-gray-700',
                  };
                  return (
                    <div key={category} className={`rounded-md border p-2.5 text-xs ${palette[category] || palette.layered}`}>
                      <div className="flex items-center justify-between">
                        <span className="font-semibold capitalize">{category.replace('_', ' ')}</span>
                        <span className="font-mono">{ids.length}</span>
                      </div>
                      <p className="mt-1 opacity-80">
                        {ids.slice(0, 3).map((id) => {
                          const n = data.nodes?.find((node) => node.id === id);
                          return n?.name || id;
                        }).join(', ')}
                        {ids.length > 3 && <span> +{ids.length - 3} more</span>}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SHAP explanation */}
      {mode === 'alert' && (explanation || explainLoading) && (
        <div className="px-6 py-3 bg-violet-50 border-b border-violet-200">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-semibold text-violet-900">ML Score Explanation (SHAP)</h3>
            {explanation && (
              <span className="text-xs text-violet-700">
                Predicted fraud probability: <strong>{(explanation.predicted_proba * 100).toFixed(1)}%</strong>
                {' '}for edge <span className="font-mono">{explanation.edge}</span>
              </span>
            )}
          </div>
          {explainLoading && (
            <p className="text-xs text-violet-700">Computing SHAP values...</p>
          )}
          {explanation && (
            <div className="space-y-1.5">
              {(explanation.top_features || []).slice(0, 6).map((f) => {
                const pct = Math.abs(f.shap) / Math.max(...(explanation.top_features || []).map(x => Math.abs(x.shap))) * 100;
                const positive = f.shap > 0;
                return (
                  <div key={f.feature} className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs min-w-0">
                    <span className="min-w-0 flex-1 basis-40 text-violet-900 truncate">{f.feature}</span>
                    <span className="font-mono text-violet-700 shrink-0">val: {Number(f.value).toFixed(2)}</span>
                    <div className="w-full sm:flex-1 sm:min-w-[120px] relative h-2 bg-violet-200 rounded">
                      <div
                        className={`absolute top-0 h-2 rounded ${positive ? 'left-1/2 bg-red-500' : 'right-1/2 bg-emerald-500'}`}
                        style={{ width: `${pct / 2}%` }}
                      />
                      <div className="absolute top-0 left-1/2 w-px h-2 bg-violet-400" />
                    </div>
                    <span className={`font-mono w-16 text-right ${positive ? 'text-red-700' : 'text-emerald-700'}`}>
                      {f.shap > 0 ? '+' : ''}{f.shap.toFixed(3)}
                    </span>
                  </div>
                );
              })}
              <p className="text-[10px] text-violet-600 mt-2 italic">
                Positive (red) = pushed score up. Negative (green) = pulled it down. Top features only.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Timeline */}
      {data && data.timeline && data.timeline.length > 0 && (
        <div className="border-t border-gray-200 bg-white w-full max-w-full overflow-hidden">
          <div className="px-6 py-2 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-800">
              Transaction Timeline ({data.timeline.length})
            </h3>
          </div>
          <div className="overflow-x-hidden w-full">
            <table className="w-full table-fixed text-xs">
              <thead className="bg-gray-50">
                <tr className="text-left text-gray-500">
                  <th className="w-[18%] px-4 py-1.5 font-medium">Timestamp</th>
                  <th className="w-[18%] px-2 py-1.5 font-medium">Sender</th>
                  <th className="w-[18%] px-2 py-1.5 font-medium">Receiver</th>
                  <th className="w-[12%] px-2 py-1.5 font-medium text-right">Amount</th>
                  <th className="w-[12%] px-2 py-1.5 font-medium">Channel</th>
                  <th className="w-[10%] px-2 py-1.5 font-medium">Rail</th>
                  <th className="w-[12%] px-2 py-1.5 font-medium">Flag</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {data.timeline.slice().reverse().map((t, i) => (
                  <tr key={i} className={t.is_fraud ? 'bg-red-50/40' : ''}>
                    <td className="px-4 py-1.5 font-mono text-gray-600 truncate">{(t.timestamp || '').slice(0, 19)}</td>
                    <td className="px-2 py-1.5 truncate" title={t.sender_name}>{t.sender_name}</td>
                    <td className="px-2 py-1.5 truncate" title={t.receiver_name}>{t.receiver_name}</td>
                    <td className="px-2 py-1.5 text-right font-medium truncate">{formatINR(t.amount)}</td>
                    <td className="px-2 py-1.5 text-gray-600 truncate">{t.channel}</td>
                    <td className="px-2 py-1.5 text-gray-600 truncate">{t.transaction_type}</td>
                    <td className="px-2 py-1.5 truncate">
                      {t.is_fraud
                        ? <span className="text-red-700 font-medium">{t.fraud_pattern?.replace(/_/g, ' ')}</span>
                        : <span className="text-gray-400">normal</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Selected entity detail (upper panel) */}
      {selectedNode && data && (
        <div className="px-6 py-4 border-t border-gray-200 bg-white w-full max-w-full">
          <div className="flex items-start justify-between gap-3 mb-3 min-w-0">
            <div className="min-w-0">
              <p className="text-[10px] font-mono text-gray-400 truncate">{selectedNode.id}</p>
              <h3 className="font-semibold text-gray-900 truncate">{selectedNode.name}</h3>
              <p className="text-xs text-gray-500 capitalize mt-0.5 truncate">
                {selectedNode.type} • {selectedNode.branch}
              </p>
            </div>
            <button onClick={() => setSelectedNodeId(null)} className="text-gray-400 text-lg leading-none shrink-0">&times;</button>
          </div>
          <div className="space-y-3 text-sm min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              {(selectedNode.flags || []).map((f, i) => {
                const redFlags = ['shell_company', 'part_of_cycle', 'transit_node', 'outflow_zscore_anomaly'];
                const tone = redFlags.includes(f) ? 'red'
                           : f === 'velocity_burst' ? 'amber'
                           : 'amber';
                return <FlagPill key={i} text={f.replace(/_/g, ' ')} tone={tone} />;
              })}
              {SIDE_LABEL[selectedNode.side] && <FlagPill text={SIDE_LABEL[selectedNode.side]} tone="indigo" />}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div>
                <p className="text-[10px] uppercase tracking-wide text-gray-500">Risk Score</p>
                <p className="font-semibold">{(selectedNode.risk_score || 0).toFixed(3)}</p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-wide text-gray-500">Product</p>
                <p className="font-semibold truncate">{selectedNode.product || 'N/A'}</p>
              </div>
            </div>
            <button
              onClick={() => navigate('/entities')}
              className="text-xs px-3 py-1.5 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Open in Entity Explorer
            </button>
            {selectedEdges.length > 0 && (
              <div className="pt-2 border-t border-gray-200">
                <h4 className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                  Incident Edges ({selectedEdges.length})
                </h4>
                <ul className="space-y-1.5">
                  {selectedEdges.slice(0, 12).map((l, i) => {
                    const otherId = l.source === selectedNode.id ? l.target : l.source;
                    const other = data.nodes.find(n => n.id === otherId);
                    const isFraud = (l.flags || []).includes('contains_fraud_txn');
                    return (
                      <li key={i} className="text-xs border-l-2 pl-2 min-w-0"
                          style={{ borderColor: isFraud ? '#dc2626' : '#cbd5e1' }}>
                        <p className="font-medium text-gray-800 truncate">
                          {l.source === selectedNode.id ? '→' : '←'} {other?.name || otherId}
                        </p>
                        <p className="text-gray-500 truncate">
                          {formatINR(l.amount)} over {l.txn_count} txn{l.txn_count > 1 ? 's' : ''}
                          {l.ml_score != null && <> • ML {(l.ml_score * 100).toFixed(0)}</>}
                          {l.txn_velocity != null && l.txn_velocity > 1 && (
                            <> • <span className="text-amber-700 font-medium">{l.txn_velocity.toFixed(1)}/h</span></>
                          )}
                        </p>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Lower panel — graph + legend.  Three sizing modes:
            compact  ~360 px  (small but visible)
            large    ~720 px  (default — fits most laptop screens)
            fullscreen        (browser-native, takes the whole viewport)
       */}
      <div
        ref={graphContainerRef}
        className={`border-t border-gray-200 bg-gray-50 flex flex-col w-full max-w-full overflow-hidden ${
          panelMode === 'fullscreen'
            ? 'fixed inset-0 z-50 h-screen'
            : panelMode === 'large'
            ? 'h-[min(75vh,720px)] min-h-[420px]'
            : 'h-[min(42vh,400px)] min-h-[280px]'
        }`}
      >
        <div className="shrink-0 px-4 py-2 border-b border-gray-200 bg-white flex flex-wrap items-center gap-x-4 gap-y-2 min-w-0">
          <h3 className="text-sm font-semibold text-gray-800 shrink-0">Fund Flow</h3>
          <GraphLegend />
          {data && (
            <span className="text-xs text-gray-500 ml-2 shrink-0">
              View: {viewMode === 'flow' ? 'Flow Story' : 'Timeline'}
            </span>
          )}
          {/* Size controls — pushed to the right */}
          <div className="ml-auto flex items-center gap-1.5 shrink-0">
            <div className="inline-flex rounded-md border border-gray-300 overflow-hidden text-xs">
              <button
                onClick={() => setPanelMode('compact')}
                disabled={panelMode === 'fullscreen'}
                className={`px-2 py-1 transition-colors ${
                  panelMode === 'compact' ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50 disabled:opacity-50'
                }`}
                title="Compact view"
              >
                Compact
              </button>
              <button
                onClick={() => setPanelMode('large')}
                disabled={panelMode === 'fullscreen'}
                className={`px-2 py-1 border-l border-gray-300 transition-colors ${
                  panelMode === 'large' ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50 disabled:opacity-50'
                }`}
                title="Large view"
              >
                Large
              </button>
            </div>
            <button
              onClick={toggleFullscreen}
              className="px-2 py-1 text-xs border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 inline-flex items-center gap-1"
              title={isFullscreen ? 'Exit fullscreen (Esc)' : 'Fullscreen'}
            >
              {isFullscreen ? (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 9V5h4M5 9V5h4m6 10v4h-4m4-4h4m-4 4v-4m-10 0H5v4h4" />
                  </svg>
                  Exit
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4 8V4h4m8 0h4v4m0 8v4h-4M8 20H4v-4" />
                  </svg>
                  Fullscreen
                </>
              )}
            </button>
          </div>
        </div>
        <div
          ref={graphPanelRef}
          className="flex-1 min-h-0 min-w-0 w-full relative bg-gradient-to-b from-white to-gray-50"
          style={{
            overflowX: 'hidden',
            overflowY: 'hidden',
          }}
        >
          {loading && (
            <div className="absolute inset-0 z-20 flex items-center justify-center text-gray-400">
              Tracing fund journey...
            </div>
          )}
          {error && (
            <div className="absolute inset-0 z-20 flex items-center justify-center text-red-600 px-4 text-center">
              {error}
            </div>
          )}
          {!loading && !data && (
            <div className="absolute inset-0 z-20 flex items-center justify-center text-gray-400 text-sm px-4 text-center">
              Pick an alert or entity above to trace the fund flow.
            </div>
          )}
          {data && viewMode === 'flow' && (
            <FlowStoryView
              data={data}
              focusId={data.entity?.id}
              panelSize={panelSize}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
            />
          )}
          {data && viewMode === 'timeline' && (
            <TimelineView
              data={data}
              panelSize={panelSize}
              selectedNodeId={selectedNodeId}
              onSelectNode={setSelectedNodeId}
            />
          )}
        </div>
      </div>
    </div>
  );
}
