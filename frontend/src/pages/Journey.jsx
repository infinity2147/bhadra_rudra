import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import ForceGraph2D from 'react-force-graph-2d';
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

  const fgRef = useRef();

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

  // Build graph data with positions:
  //   - entity-mode trace: upstream/focus/downstream get layered columns
  //   - alert-mode trace: let the force layout settle (cycles look better as ring)
  //   - if there are neighbors, expand to 3 columns (alert + neighbor)
  const graphData = useMemo(() => {
    if (!data) return { nodes: [], links: [] };
    const isAlertMode = data.direction === 'alert_scope';
    const hasMultipleSides = new Set(data.nodes.map(n => n.side)).size > 1;

    let nodes;
    if (isAlertMode && !hasMultipleSides) {
      // Pure alert chain — let force layout shape it
      nodes = data.nodes.map(n => ({ ...n }));
    } else {
      const sideOrder = { upstream: 0, source: 0, focus: 1, alert: 1, loop: 1, neighbor: 2, downstream: 2 };
      const groups = {};
      for (const n of data.nodes) {
        const col = sideOrder[n.side] ?? 1;
        groups[col] = groups[col] || [];
        groups[col].push(n);
      }
      nodes = [];
      Object.keys(groups).forEach(col => {
        const arr = groups[col];
        const colX = (Number(col) - 1) * 350;
        arr.forEach((n, i) => {
          const colY = (i - arr.length / 2) * 60;
          nodes.push({ ...n, fx: colX, fy: colY });
        });
      });
    }
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = (data.links || []).filter(l => nodeIds.has(l.source) && nodeIds.has(l.target));
    return { nodes, links };
  }, [data]);

  // Node paint
  const paintNode = useCallback((node, ctx, globalScale) => {
    const isFocus = node.side === 'focus' || node.side === 'alert';
    const isShell = node.type === 'shell_company';
    const hasFraud = (node.flags || []).includes('contains_fraud_txn');
    const r = isFocus ? 12 : 7;

    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = isShell ? '#ef4444' : NODE_FILL[node.type] || '#6b7280';
    ctx.fill();

    // outline
    if (selectedNodeId === node.id) {
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 3 / globalScale;
      ctx.stroke();
    } else if (isFocus) {
      ctx.strokeStyle = '#1e1b4b';
      ctx.lineWidth = 2 / globalScale;
      ctx.stroke();
    } else if (isShell) {
      ctx.strokeStyle = '#7f1d1d';
      ctx.lineWidth = 1.5 / globalScale;
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.lineWidth = 0.5 / globalScale;
      ctx.stroke();
    }

    const label = (node.name || node.id).slice(0, 22);
    const fontSize = Math.max(11 / globalScale, 2);
    ctx.font = `${fontSize}px ui-sans-serif`;
    ctx.fillStyle = '#111827';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(label, node.x, node.y + r + 2);
  }, [selectedNodeId]);

  // Link paint
  const paintLink = useCallback((link, ctx, globalScale) => {
    const isFraud = (link.flags || []).includes('contains_fraud_txn') || (link.fraud_count || 0) > 0;
    const mlHigh = (link.ml_score ?? 0) >= 0.6;
    const amount = link.amount || 1;
    const width = Math.max(0.6, Math.min(6, Math.log10(amount + 1) * 0.6));

    ctx.beginPath();
    const sx = link.source.x ?? 0, sy = link.source.y ?? 0;
    const tx = link.target.x ?? 0, ty = link.target.y ?? 0;
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.strokeStyle = isFraud ? '#dc2626' : mlHigh ? '#f59e0b' : '#cbd5e1';
    ctx.lineWidth = width / globalScale;
    ctx.stroke();

    // arrow head
    const angle = Math.atan2(ty - sy, tx - sx);
    const headLen = 4 / globalScale;
    const baseX = tx - Math.cos(angle) * 10;
    const baseY = ty - Math.sin(angle) * 10;
    ctx.beginPath();
    ctx.moveTo(baseX, baseY);
    ctx.lineTo(baseX - headLen * Math.cos(angle - Math.PI / 6),
                baseY - headLen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(baseX, baseY);
    ctx.lineTo(baseX - headLen * Math.cos(angle + Math.PI / 6),
                baseY - headLen * Math.sin(angle + Math.PI / 6));
    ctx.strokeStyle = isFraud ? '#dc2626' : mlHigh ? '#f59e0b' : '#94a3b8';
    ctx.stroke();
  }, []);

  // Selected node + edges
  const selectedNode = data?.nodes?.find(n => n.id === selectedNodeId);
  const selectedEdges = data?.links?.filter(
    l => l.source === selectedNodeId || l.target === selectedNodeId
  ) || [];

  return (
    <div className="h-full flex flex-col">
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

        {mode === 'alert' && alertId && (
          <button
            onClick={() => downloadFromAPI(`/api/fiu/package/${alertId}`, `FIU_${alertId}.zip`)}
            className="ml-auto px-3 py-1.5 text-sm bg-emerald-600 text-white rounded-md hover:bg-emerald-700"
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

      {/* Main: graph + detail + timeline */}
      <div className="flex-1 flex min-h-0">
        {/* Graph canvas */}
        <div className="flex-1 relative min-h-0 bg-gradient-to-b from-white to-gray-50">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400">
              Tracing fund journey...
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center text-red-600">
              {error}
            </div>
          )}
          {!loading && !data && (
            <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
              Pick an alert or entity to begin tracing.
            </div>
          )}
          {data && (
            <>
              <ForceGraph2D
                ref={fgRef}
                graphData={graphData}
                nodeCanvasObject={paintNode}
                nodeCanvasObjectMode={() => 'replace'}
                linkCanvasObject={paintLink}
                linkCanvasObjectMode={() => 'replace'}
                onNodeClick={n => setSelectedNodeId(n.id)}
                onLinkClick={l => {
                  const sId = typeof l.source === 'object' ? l.source.id : l.source;
                  setSelectedNodeId(sId);
                }}
                cooldownTicks={50}
                d3VelocityDecay={0.5}
                enableNodeDrag={false}
                backgroundColor="rgba(0,0,0,0)"
              />
              {/* Legend */}
              <div className="absolute bottom-3 left-3 bg-white/95 backdrop-blur rounded-lg border border-gray-200 px-3 py-2 text-xs space-y-1 shadow-sm">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-blue-500" /> Individual
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-green-500" /> Business
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-red-500" /> Shell Company
                </div>
                <hr className="border-gray-200" />
                <div className="flex items-center gap-2">
                  <span className="w-6 h-0.5 bg-red-600" /> Fraud edge
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-6 h-0.5 bg-amber-500" /> High ML score
                </div>
              </div>
            </>
          )}
        </div>

        {/* Right detail rail */}
        {selectedNode && (
          <div className="w-80 border-l border-gray-200 bg-white overflow-y-auto p-4 shrink-0">
            <div className="flex items-start justify-between mb-3">
              <div>
                <p className="text-[10px] font-mono text-gray-400">{selectedNode.id}</p>
                <h3 className="font-semibold text-gray-900">{selectedNode.name}</h3>
                <p className="text-xs text-gray-500 capitalize mt-0.5">
                  {selectedNode.type} • {selectedNode.branch}
                </p>
              </div>
              <button onClick={() => setSelectedNodeId(null)} className="text-gray-400 text-lg leading-none">&times;</button>
            </div>
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-2 flex-wrap">
                {(selectedNode.flags || []).map((f, i) => (
                  <FlagPill
                    key={i}
                    text={f.replace(/_/g, ' ')}
                    tone={f === 'shell_company' || f === 'part_of_cycle' ? 'red' : 'amber'}
                  />
                ))}
                {SIDE_LABEL[selectedNode.side] && <FlagPill text={SIDE_LABEL[selectedNode.side]} tone="indigo" />}
              </div>
              <div className="grid grid-cols-2 gap-2 pt-2">
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-gray-500">Risk Score</p>
                  <p className="font-semibold">{(selectedNode.risk_score || 0).toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-gray-500">Product</p>
                  <p className="font-semibold">{selectedNode.product || 'N/A'}</p>
                </div>
              </div>
              <button
                onClick={() => navigate(`/entities`)}
                className="w-full text-xs px-2 py-1.5 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Open in Entity Explorer
              </button>
              {selectedEdges.length > 0 && (
                <div className="pt-2 border-t border-gray-200">
                  <h4 className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                    Incident Edges ({selectedEdges.length})
                  </h4>
                  <ul className="space-y-1.5 max-h-72 overflow-y-auto">
                    {selectedEdges.slice(0, 30).map((l, i) => {
                      const otherId = l.source === selectedNode.id ? l.target : l.source;
                      const other = data.nodes.find(n => n.id === otherId);
                      const isFraud = (l.flags || []).includes('contains_fraud_txn');
                      return (
                        <li key={i} className="text-xs border-l-2 pl-2"
                            style={{ borderColor: isFraud ? '#dc2626' : '#cbd5e1' }}>
                          <p className="font-medium text-gray-800">
                            {l.source === selectedNode.id ? '→' : '←'} {other?.name || otherId}
                          </p>
                          <p className="text-gray-500">
                            {formatINR(l.amount)} over {l.txn_count} txn{l.txn_count > 1 ? 's' : ''}
                            {l.ml_score != null && <> • ML {(l.ml_score * 100).toFixed(0)}</>}
                          </p>
                          {(l.flags || []).length > 0 && (
                            <p className="text-[10px] text-gray-500 mt-0.5">
                              {l.flags.map(f => f.replace(/_/g, ' ')).join(' • ')}
                            </p>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Timeline */}
      {data && data.timeline && data.timeline.length > 0 && (
        <div className="border-t border-gray-200 bg-white max-h-64 overflow-y-auto">
          <div className="px-6 py-2 border-b border-gray-100 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-800">
              Transaction Timeline ({data.timeline.length})
            </h3>
          </div>
          <table className="w-full text-xs">
            <thead className="bg-gray-50 sticky top-0">
              <tr className="text-left text-gray-500">
                <th className="px-6 py-1.5 font-medium">Timestamp</th>
                <th className="px-3 py-1.5 font-medium">Sender</th>
                <th className="px-3 py-1.5 font-medium">Receiver</th>
                <th className="px-3 py-1.5 font-medium text-right">Amount</th>
                <th className="px-3 py-1.5 font-medium">Channel</th>
                <th className="px-3 py-1.5 font-medium">Rail</th>
                <th className="px-3 py-1.5 font-medium">Flag</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {data.timeline.slice().reverse().map((t, i) => (
                <tr key={i} className={t.is_fraud ? 'bg-red-50/40' : ''}>
                  <td className="px-6 py-1.5 font-mono text-gray-600 whitespace-nowrap">{(t.timestamp || '').slice(0, 19)}</td>
                  <td className="px-3 py-1.5">{t.sender_name}</td>
                  <td className="px-3 py-1.5">{t.receiver_name}</td>
                  <td className="px-3 py-1.5 text-right font-medium">{formatINR(t.amount)}</td>
                  <td className="px-3 py-1.5 text-gray-600">{t.channel}</td>
                  <td className="px-3 py-1.5 text-gray-600">{t.transaction_type}</td>
                  <td className="px-3 py-1.5">
                    {t.is_fraud
                      ? <span className="text-red-700 font-medium">{t.fraud_pattern?.replace(/_/g, ' ')}</span>
                      : <span className="text-gray-400">normal</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
