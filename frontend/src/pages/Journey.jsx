import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import ForceGraph2D from 'react-force-graph-2d';
import { fetchAPI, downloadFromAPI } from '../api';
import Sankey from '../components/Sankey';

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
  const [viewMode, setViewMode] = useState('auto'); // 'auto' | 'sankey' | 'force'
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const fgRef = useRef();
  const graphPanelRef = useRef(null);
  const [panelSize, setPanelSize] = useState({ width: 800, height: 300 });

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
  // Sankey only for small linear/branching flows where columns are readable.
  const effectiveView = useMemo(() => {
    if (viewMode !== 'auto') return viewMode;
    if (!data) return 'force';
    const sides = new Set((data.nodes || []).map((n) => n.side));
    const nodeCount = (data.nodes || []).length;
    // Sankey only stays readable up to ~25 nodes. Beyond that, columns get
    // crushed into vertical strips — force-directed is the right call.
    if (nodeCount > 25) return 'force';
    const hasMulti = sides.size > 1;
    return hasMulti ? 'sankey' : 'force';
  }, [data, viewMode]);

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
      const colKeys = Object.keys(groups).map(Number);
      const minCol = Math.min(...colKeys);
      const maxCol = Math.max(...colKeys);
      const colSpan = maxCol - minCol || 1;
      const hSpread = panelSize.width * 0.82;
      Object.keys(groups).forEach(col => {
        const arr = groups[col];
        const colX = ((Number(col) - minCol) / colSpan - 0.5) * hSpread;
        arr.forEach((n, i) => {
          const colY = (i - arr.length / 2) * 56;
          nodes.push({ ...n, fx: colX, fy: colY });
        });
      });
    }
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = (data.links || []).filter(l => nodeIds.has(l.source) && nodeIds.has(l.target));
    return { nodes, links };
  }, [data, panelSize.width]);

  // Fit force graph inside the lower panel when data or size changes
  useEffect(() => {
    if (effectiveView !== 'force' || !fgRef.current || !graphData.nodes.length) return;
    const t = setTimeout(() => {
      try {
        fgRef.current.zoomToFit(400, 48);
      } catch {
        /* graph may not be mounted yet */
      }
    }, 150);
    return () => clearTimeout(t);
  }, [graphData, effectiveView, panelSize.width, panelSize.height]);

  // Node paint
  const paintNode = useCallback((node, ctx, globalScale) => {
    const isFocus = node.side === 'focus' || node.side === 'alert';
    const isShell = node.type === 'shell_company';
    const r = isFocus ? 7 : 4.5;

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

    const label = (node.name || node.id).slice(0, 18);
    const fontSize = Math.max(9 / globalScale, 2);
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
          {['auto', 'sankey', 'force'].map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className={`px-2.5 py-1 text-xs rounded-md transition-colors capitalize ${
                effectiveView === m && viewMode !== 'auto'
                  ? 'bg-indigo-600 text-white'
                  : viewMode === 'auto' && m === 'auto'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              {m}
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

      {/* Lower panel — graph + legend (part of page scroll) */}
      <div className="border-t border-gray-200 bg-gray-50 flex flex-col h-[min(42vh,400px)] min-h-[280px] w-full max-w-full overflow-hidden">
        <div className="shrink-0 px-4 py-2 border-b border-gray-200 bg-white flex flex-wrap items-center gap-x-4 gap-y-2 min-w-0">
          <h3 className="text-sm font-semibold text-gray-800 shrink-0">Fund Flow</h3>
          <GraphLegend />
          {data && (
            <span className="text-xs text-gray-500 ml-auto shrink-0 capitalize">
              View: {effectiveView}
            </span>
          )}
        </div>
        <div
          ref={graphPanelRef}
          className="flex-1 min-h-0 min-w-0 w-full relative overflow-hidden bg-gradient-to-b from-white to-gray-50"
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
          {data && effectiveView === 'sankey' && (
            <div className="absolute inset-0 min-h-0">
              <Sankey
                nodes={data.nodes}
                links={data.links}
                width={panelSize.width}
                onNodeClick={(n) => setSelectedNodeId(n.id)}
                onLinkClick={(l) => {
                  const sId = typeof l.source === 'object' ? l.source.id : l.source;
                  setSelectedNodeId(sId);
                }}
              />
            </div>
          )}
          {data && effectiveView === 'force' && (
            <div className="absolute inset-0">
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
                enableZoomInteraction={true}
                enablePanInteraction={true}
                backgroundColor="#fafafa"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
