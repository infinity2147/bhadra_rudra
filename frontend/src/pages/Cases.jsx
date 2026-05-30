import { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { fetchAPI, postAPI, downloadFromAPI, getRole } from '../api';
import SeverityBadge from '../components/SeverityBadge';

const STATUS_TABS = [
  { key: 'OPEN', label: 'Open' },
  { key: 'INVESTIGATING', label: 'Investigating' },
  { key: 'SAR_FILED', label: 'SAR Filed' },
  { key: 'ESCALATED', label: 'Escalated' },
  { key: 'DISMISSED', label: 'Dismissed' },
  { key: 'ALL', label: 'All' },
];

const STATUS_COLORS = {
  OPEN: 'bg-amber-100 text-amber-800 ring-1 ring-amber-200',
  INVESTIGATING: 'bg-indigo-100 text-indigo-800 ring-1 ring-indigo-200',
  SAR_FILED: 'bg-emerald-100 text-emerald-800 ring-1 ring-emerald-200',
  ESCALATED: 'bg-rose-100 text-rose-800 ring-1 ring-rose-200',
  DISMISSED: 'bg-gray-100 text-gray-700 ring-1 ring-gray-200',
};

const SEVERITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

function formatINR(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

function StatusPill({ status }) {
  const cls = STATUS_COLORS[status] || 'bg-gray-100 text-gray-700';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium ${cls}`}>
      {(status || 'OPEN').replace('_', ' ')}
    </span>
  );
}

function MLScoreBar({ score }) {
  if (score == null) return <span className="text-xs text-gray-400">—</span>;
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? 'bg-red-500' : pct >= 40 ? 'bg-amber-500' : 'bg-emerald-500';
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-gray-600">{pct}</span>
    </div>
  );
}

const FLAG_TONES = {
  shell_company:           'bg-red-100 text-red-800 ring-red-200',
  part_of_cycle:           'bg-red-100 text-red-800 ring-red-200',
  transit_node:            'bg-red-100 text-red-800 ring-red-200',
  outflow_zscore_anomaly:  'bg-red-100 text-red-800 ring-red-200',
  high_risk:               'bg-amber-100 text-amber-800 ring-amber-200',
  velocity_burst:          'bg-amber-100 text-amber-800 ring-amber-200',
  dormant_then_active:     'bg-amber-100 text-amber-800 ring-amber-200',
  multi_branch_activity:   'bg-indigo-100 text-indigo-800 ring-indigo-200',
};

function FlagPill({ flag }) {
  const tone = FLAG_TONES[flag] || 'bg-gray-100 text-gray-700 ring-gray-200';
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ring-1 ${tone}`}>
      {flag.replace(/_/g, ' ')}
    </span>
  );
}

export default function Cases() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const focusedId = params.get('focus');
  const role = getRole();

  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeStatus, setActiveStatus] = useState('OPEN');
  const [selectedId, setSelectedId] = useState(focusedId || null);
  const [caseDetail, setCaseDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [note, setNote] = useState('');
  const [search, setSearch] = useState('');
  const [explanation, setExplanation] = useState(null);
  const [chainStatus, setChainStatus] = useState(null);
  const [entityFlags, setEntityFlags] = useState({});      // {entity_id: [flags]}
  const [tracePreview, setTracePreview] = useState(null);  // {dominant_paths: [...]}
  const [mlBand, setMlBand] = useState('ALL');             // 'ALL' | 'HIGH' | 'MED' | 'LOW' | 'NONE'
  const [groupByIncident, setGroupByIncident] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchAPI('/api/alerts')
      .then(data => setAlerts(data.alerts || []))
      .catch(() => setAlerts([]))
      .finally(() => setLoading(false));
  }, []);

  const refreshAlerts = useCallback(async () => {
    const data = await fetchAPI('/api/alerts');
    setAlerts(data.alerts || []);
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setCaseDetail(null); setNote(''); setExplanation(null); setChainStatus(null);
      setEntityFlags({}); setTracePreview(null);
      return;
    }
    setDetailLoading(true);
    setEntityFlags({}); setTracePreview(null);
    Promise.all([
      fetchAPI(`/api/cases/${selectedId}`),
      fetchAPI(`/api/alerts/${selectedId}/explain`).catch(() => null),
    ])
      .then(([c, e]) => {
        setCaseDetail(c);
        setExplanation(e && !e.error ? e : null);

        // Fetch entity flags + a small alert-scoped trace preview in parallel.
        // The alert object lives in the `alerts` list, so look it up there.
        const alert = alerts.find(a => a.alert_id === selectedId);
        const entityIds = (alert?.entities || []).slice(0, 8);

        Promise.all(
          entityIds.map(eid =>
            fetchAPI(`/api/entities/${eid}`)
              .then(d => [eid, d.flags || []])
              .catch(() => [eid, []])
          )
        ).then(pairs => setEntityFlags(Object.fromEntries(pairs)));

        // Pull the journey for the alert's first entity to get dominant_paths.
        // trace_for_alert doesn't compute them — use the entity-mode endpoint.
        if (entityIds[0]) {
          fetchAPI(`/api/journey/${entityIds[0]}?hops=3&direction=forward`)
            .then(d => setTracePreview({
              dominant_paths: d.dominant_paths || [],
              terminal_classification: d.terminal_classification || {},
            }))
            .catch(() => setTracePreview(null));
        }
      })
      .catch(() => { setCaseDetail(null); setExplanation(null); })
      .finally(() => setDetailLoading(false));
  }, [selectedId, alerts]);

  const filtered = useMemo(() => {
    let xs = alerts;
    if (activeStatus !== 'ALL') {
      xs = xs.filter(a => (a.case_status || 'OPEN') === activeStatus);
    }
    if (mlBand !== 'ALL') {
      xs = xs.filter(a => {
        const s = a.ml_score;
        if (mlBand === 'NONE') return s == null;
        if (s == null) return false;
        if (mlBand === 'HIGH') return s >= 0.7;
        if (mlBand === 'MED') return s >= 0.4 && s < 0.7;
        if (mlBand === 'LOW') return s < 0.4;
        return true;
      });
    }
    if (search.trim()) {
      const q = search.toLowerCase();
      xs = xs.filter(a =>
        (a.pattern_type || '').toLowerCase().includes(q) ||
        (a.alert_id || '').toLowerCase().includes(q) ||
        (a.entity_names || []).some(n => (n || '').toLowerCase().includes(q))
      );
    }
    return [...xs].sort((a, b) => {
      const sevDiff = (SEVERITY_ORDER[a.severity] ?? 99) - (SEVERITY_ORDER[b.severity] ?? 99);
      if (sevDiff !== 0) return sevDiff;
      return (b.total_flow ?? 0) - (a.total_flow ?? 0);
    });
  }, [alerts, activeStatus, mlBand, search]);

  // Group filtered alerts by incident_id for the grouped view.
  const groupedFiltered = useMemo(() => {
    if (!groupByIncident) return null;
    const groups = new Map();
    for (const a of filtered) {
      const key = a.incident_id || '__no_incident__';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(a);
    }
    // Sort groups by worst severity within them
    return Array.from(groups.entries()).sort(([, a], [, b]) => {
      const aSev = Math.min(...a.map(x => SEVERITY_ORDER[x.severity] ?? 99));
      const bSev = Math.min(...b.map(x => SEVERITY_ORDER[x.severity] ?? 99));
      return aSev - bSev;
    });
  }, [filtered, groupByIncident]);

  const counts = useMemo(() => {
    const c = { OPEN: 0, INVESTIGATING: 0, SAR_FILED: 0, ESCALATED: 0, DISMISSED: 0, ALL: alerts.length };
    for (const a of alerts) {
      const s = a.case_status || 'OPEN';
      if (c[s] != null) c[s] += 1;
    }
    return c;
  }, [alerts]);

  const selectedAlert = useMemo(
    () => alerts.find(a => a.alert_id === selectedId),
    [alerts, selectedId]
  );

  async function disposeCase(status) {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      await postAPI(`/api/cases/${selectedId}/dispose`, { status, note });
      setNote('');
      await refreshAlerts();
      const c = await fetchAPI(`/api/cases/${selectedId}`);
      setCaseDetail(c);
    } catch (e) {
      alert(e.message);
    } finally {
      setActionLoading(false);
    }
  }

  async function addNote() {
    if (!selectedId || !note.trim()) return;
    setActionLoading(true);
    try {
      const c = await postAPI(`/api/cases/${selectedId}/note`, { note });
      setNote('');
      setCaseDetail(c);
    } finally {
      setActionLoading(false);
    }
  }

  async function downloadFiuPackage() {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      await downloadFromAPI(`/api/fiu/package/${selectedId}`, `FIU_${selectedId}.zip`);
    } finally {
      setActionLoading(false);
    }
  }

  async function verifyChain() {
    if (!selectedId) return;
    setActionLoading(true);
    try {
      const c = await fetchAPI(`/api/cases/${selectedId}/verify`);
      setChainStatus(c);
    } catch (e) {
      setChainStatus({ error: e.message });
    } finally {
      setActionLoading(false);
    }
  }

  // Compute which actions are allowed given current role
  const canSAR = role === 'SUPERVISOR' || role === 'ADMIN';
  const canDismiss = role === 'SUPERVISOR' || role === 'ADMIN';
  const canVerify = role === 'SUPERVISOR' || role === 'ADMIN';

  // Single-row renderer reused by both the flat list and the grouped list.
  function renderCaseRow(a) {
    const isSelected = a.alert_id === selectedId;
    return (
      <li
        key={a.alert_id}
        onClick={() => {
          setSelectedId(a.alert_id);
          setParams({ focus: a.alert_id });
        }}
        className={`px-6 py-4 cursor-pointer transition-colors ${
          isSelected ? 'bg-indigo-50' : 'hover:bg-gray-50'
        }`}
      >
        <div className="flex items-start gap-4">
          <div className="flex flex-col gap-1.5">
            <SeverityBadge severity={a.severity} />
            <StatusPill status={a.case_status} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-baseline gap-2 flex-wrap">
              <span className="text-sm font-semibold text-gray-900">{a.pattern_type}</span>
              <span className="text-xs text-gray-400 font-mono">{a.alert_id}</span>
              {a.incident_id && !groupByIncident && (
                <button
                  onClick={(e) => { e.stopPropagation(); navigate(`/incidents`); }}
                  className="text-xs text-indigo-600 hover:underline"
                >
                  {a.incident_id}
                </button>
              )}
            </div>
            <p className="text-sm text-gray-600 mt-1 line-clamp-2">{a.description}</p>
            <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
              <span>{a.entities?.length || 0} entities</span>
              <span>•</span>
              <span className="font-medium text-gray-700">{formatINR(a.total_flow)}</span>
              <span>•</span>
              <span>Conf. {a.confidence}%</span>
            </div>
          </div>
          <div className="text-right shrink-0">
            <p className="text-[11px] text-gray-400 uppercase tracking-wide mb-1">ML</p>
            <MLScoreBar score={a.ml_score} />
          </div>
        </div>
      </li>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-6 pb-3">
        <h1 className="text-2xl font-bold text-gray-900">Case Workbench</h1>
        <p className="text-sm text-gray-500 mt-1">
          Triage fraud alerts, record decisions, generate FIU evidence packages. SHAP shows why the ML model flagged each one.
        </p>
      </div>

      {/* Status tabs + search */}
      <div className="px-6 border-b border-gray-200 bg-white">
        <nav className="flex gap-1 -mb-px overflow-x-auto">
          {STATUS_TABS.map(t => (
            <button
              key={t.key}
              type="button"
              onClick={() => setActiveStatus(t.key)}
              className={`px-3 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                activeStatus === t.key
                  ? 'border-indigo-600 text-indigo-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {t.label}
              <span className="ml-2 inline-flex items-center justify-center min-w-[1.25rem] px-1.5 py-0.5 rounded-full text-[11px] font-semibold bg-gray-100 text-gray-700">
                {counts[t.key]}
              </span>
            </button>
          ))}
          <div className="ml-auto py-2 flex items-center gap-2">
            <input
              type="text"
              placeholder="Search alerts, entities..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-56 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            <label className="flex items-center gap-1.5 text-xs text-gray-700 whitespace-nowrap cursor-pointer">
              <input
                type="checkbox"
                checked={groupByIncident}
                onChange={(e) => setGroupByIncident(e.target.checked)}
                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              />
              Group by incident
            </label>
          </div>
        </nav>

        {/* ML score band chips */}
        <div className="flex items-center gap-1.5 py-2 pb-3 text-xs">
          <span className="text-gray-500 uppercase tracking-wide font-semibold mr-1">ML score:</span>
          {[
            { key: 'ALL', label: 'All' },
            { key: 'HIGH', label: '≥ 70', tone: 'bg-red-50 text-red-700 ring-red-200' },
            { key: 'MED', label: '40–70', tone: 'bg-amber-50 text-amber-700 ring-amber-200' },
            { key: 'LOW', label: '< 40', tone: 'bg-emerald-50 text-emerald-700 ring-emerald-200' },
            { key: 'NONE', label: 'No score', tone: 'bg-gray-50 text-gray-600 ring-gray-200' },
          ].map(b => {
            const active = mlBand === b.key;
            return (
              <button
                key={b.key}
                onClick={() => setMlBand(b.key)}
                className={`px-2.5 py-0.5 rounded-full ring-1 transition-colors ${
                  active
                    ? 'bg-indigo-600 text-white ring-indigo-600'
                    : b.tone || 'bg-white text-gray-600 ring-gray-200 hover:bg-gray-50'
                }`}
              >
                {b.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 flex min-h-0">
        {/* Cases list */}
        <div className={`${selectedId ? 'w-3/5' : 'w-full'} overflow-y-auto bg-gray-50 border-r border-gray-200 transition-all`}>
          {loading ? (
            <div className="p-12 text-center text-gray-400">Loading cases...</div>
          ) : filtered.length === 0 ? (
            <div className="p-12 text-center text-gray-400">No cases in this category.</div>
          ) : groupedFiltered ? (
            <div className="bg-white">
              {groupedFiltered.map(([incidentId, items]) => (
                <div key={incidentId} className="border-b-2 border-gray-300">
                  <div className="px-6 py-2 bg-indigo-50/40 border-b border-indigo-100 flex items-center justify-between">
                    <p className="text-xs font-semibold text-indigo-900 uppercase tracking-wide">
                      {incidentId === '__no_incident__' ? 'No incident' : `Incident ${incidentId}`}
                      <span className="ml-2 text-gray-500 font-normal normal-case">
                        ({items.length} alert{items.length === 1 ? '' : 's'})
                      </span>
                    </p>
                    {incidentId !== '__no_incident__' && (
                      <button
                        onClick={() => navigate(`/incidents`)}
                        className="text-xs text-indigo-600 hover:underline"
                      >View incident →</button>
                    )}
                  </div>
                  <ul className="divide-y divide-gray-200">
                    {items.map(a => renderCaseRow(a))}
                  </ul>
                </div>
              ))}
            </div>
          ) : (
            <ul className="divide-y divide-gray-200 bg-white">
              {filtered.map(a => renderCaseRow(a))}
            </ul>
          )}
        </div>

        {/* Detail panel */}
        {selectedId && (
          <div className="w-2/5 overflow-y-auto bg-white">
            {detailLoading || !selectedAlert ? (
              <div className="p-12 text-center text-gray-400">Loading case...</div>
            ) : (
              <div className="p-6 space-y-5">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-xs font-mono text-gray-400">{selectedAlert.alert_id}</p>
                    <h2 className="text-lg font-bold text-gray-900 mt-0.5">{selectedAlert.pattern_type}</h2>
                    <div className="flex items-center gap-2 mt-2">
                      <SeverityBadge severity={selectedAlert.severity} />
                      <StatusPill status={caseDetail?.status} />
                      {caseDetail?.incident_id && (
                        <button
                          onClick={() => navigate('/incidents')}
                          className="text-xs text-indigo-600 hover:underline"
                        >
                          {caseDetail.incident_id}
                        </button>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => { setSelectedId(null); setParams({}); }}
                    className="text-gray-400 hover:text-gray-600 text-xl leading-none"
                  >&times;</button>
                </div>

                <div>
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Description</h3>
                  <p className="text-sm text-gray-700 leading-relaxed">{selectedAlert.description}</p>
                </div>

                {selectedAlert.recommendation && (
                  <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2">
                    <p className="text-[11px] font-semibold text-amber-700 uppercase tracking-wide">Recommended Action</p>
                    <p className="text-xs text-amber-900 mt-1 leading-relaxed">{selectedAlert.recommendation}</p>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-gray-500">Total Flow</p>
                    <p className="font-semibold text-gray-900 mt-0.5">{formatINR(selectedAlert.total_flow)}</p>
                  </div>
                  <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-gray-500">Detector Confidence</p>
                    <p className="font-semibold text-gray-900 mt-0.5">{selectedAlert.confidence}%</p>
                  </div>
                  <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-gray-500">ML Score</p>
                    <div className="mt-1"><MLScoreBar score={selectedAlert.ml_score} /></div>
                  </div>
                  <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                    <p className="text-[11px] uppercase tracking-wide text-gray-500">Entities</p>
                    <p className="font-semibold text-gray-900 mt-0.5">{selectedAlert.entities?.length || 0}</p>
                  </div>
                </div>

                {/* SHAP — visual top-features bar chart */}
                {explanation && (
                  <div className="rounded-lg bg-violet-50 border border-violet-200 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-semibold text-violet-900 uppercase tracking-wide">Why this score? (SHAP)</p>
                      <p className="text-xs text-violet-700">
                        Predicted: <strong>{(explanation.predicted_proba * 100).toFixed(1)}%</strong>
                        {explanation.edge && (
                          <span className="ml-2 font-mono text-[10px] text-violet-500">{explanation.edge}</span>
                        )}
                      </p>
                    </div>
                    <div className="space-y-1.5">
                      {(() => {
                        const features = (explanation.top_features || []).slice(0, 6);
                        if (features.length === 0) return (
                          <p className="text-xs text-violet-700">No feature attributions available.</p>
                        );
                        const maxAbs = Math.max(...features.map(x => Math.abs(x.shap))) || 1;
                        return features.map((f) => {
                          const pct = (Math.abs(f.shap) / maxAbs) * 100;
                          const positive = f.shap > 0;
                          return (
                            <div key={f.feature} className="flex items-center gap-2 text-[11px]">
                              <span className="w-32 text-violet-900 truncate" title={f.feature}>{f.feature}</span>
                              <span className="font-mono text-violet-600 shrink-0 w-16 text-right">{Number(f.value).toFixed(2)}</span>
                              <div className="flex-1 relative h-2 bg-violet-200/60 rounded">
                                <div
                                  className={`absolute top-0 h-2 rounded ${positive ? 'left-1/2 bg-red-500' : 'right-1/2 bg-emerald-500'}`}
                                  style={{ width: `${pct / 2}%` }}
                                />
                                <div className="absolute top-0 left-1/2 w-px h-2 bg-violet-400/80" />
                              </div>
                              <span className={`font-mono w-14 text-right shrink-0 ${positive ? 'text-red-700' : 'text-emerald-700'}`}>
                                {f.shap > 0 ? '+' : ''}{f.shap.toFixed(3)}
                              </span>
                            </div>
                          );
                        });
                      })()}
                    </div>
                    <p className="text-[10px] text-violet-600 mt-2 italic">
                      Red = pushed score up. Green = pulled it down. Width = magnitude.
                    </p>
                  </div>
                )}

                {/* Dominant-path preview — top fund corridor downstream of the alert */}
                {tracePreview && tracePreview.dominant_paths?.length > 0 && (
                  <div className="rounded-lg bg-rose-50 border border-rose-200 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-xs font-semibold text-rose-900 uppercase tracking-wide">
                        Where did the funds go?
                      </p>
                      <button
                        onClick={() => navigate(`/journey?alert=${selectedId}`)}
                        className="text-[11px] text-rose-700 hover:text-rose-900 hover:underline"
                      >
                        Full trace →
                      </button>
                    </div>
                    {tracePreview.dominant_paths.slice(0, 2).map((dp, i) => (
                      <div key={i} className="text-xs mb-1.5 last:mb-0">
                        <div className="flex items-baseline gap-2 mb-0.5">
                          <span className="font-semibold text-rose-700">
                            risk {(dp.path_risk_score ?? 0).toFixed(2)}
                          </span>
                          <span className="text-rose-900 font-medium">{formatINR(dp.bottleneck_amount)}</span>
                          <span className="text-rose-600">• {dp.hops} hops</span>
                        </div>
                        <p className="text-rose-900 leading-snug">
                          {(dp.path || []).join(' → ')}
                        </p>
                        {dp.path_flags?.length > 0 && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {dp.path_flags.slice(0, 5).map((f) => <FlagPill key={f} flag={f} />)}
                          </div>
                        )}
                      </div>
                    ))}
                    {Object.keys(tracePreview.terminal_classification || {}).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-rose-200">
                        <p className="text-[10px] uppercase text-rose-700 tracking-wide font-semibold mb-1">
                          Terminal destinations
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {Object.entries(tracePreview.terminal_classification).map(([cat, ids]) => (
                            <span key={cat} className="text-[11px] text-rose-900">
                              <span className="font-semibold capitalize">{cat.replace('_', ' ')}</span>
                              <span className="text-rose-600 ml-1">×{ids.length}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {selectedAlert.entities && selectedAlert.entities.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">Entity Chain</h3>
                    <div className="space-y-1.5">
                      {selectedAlert.entities.map((eid, i) => {
                        const name = selectedAlert.entity_names?.[i] || eid;
                        const flags = entityFlags[eid] || [];
                        return (
                          <div key={eid} className="flex items-start gap-2 px-2 py-1.5 rounded-md bg-indigo-50 border border-indigo-100">
                            <span className="text-xs font-medium text-indigo-900 shrink-0">
                              {i + 1}. {name}
                            </span>
                            {flags.length > 0 && (
                              <div className="flex flex-wrap gap-1 min-w-0">
                                {flags.map((f) => <FlagPill key={f} flag={f} />)}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="space-y-2 pt-2 border-t border-gray-200">
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => navigate(`/journey?alert=${selectedId}`)}
                      className="px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      Trace Journey
                    </button>
                    <button
                      onClick={downloadFiuPackage}
                      disabled={actionLoading}
                      className="px-3 py-2 text-sm bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50"
                    >
                      FIU Package
                    </button>
                  </div>

                  <textarea
                    value={note}
                    onChange={e => setNote(e.target.value)}
                    placeholder="Add an investigator note (optional)..."
                    rows={2}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />

                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => disposeCase('INVESTIGATING')}
                      disabled={actionLoading}
                      className="px-3 py-2 text-sm bg-indigo-50 text-indigo-700 ring-1 ring-indigo-200 rounded-lg hover:bg-indigo-100 transition-colors disabled:opacity-50"
                    >
                      Investigating
                    </button>
                    <button
                      onClick={() => disposeCase('SAR_FILED')}
                      disabled={actionLoading || !canSAR}
                      title={!canSAR ? 'Requires SUPERVISOR role' : undefined}
                      className="px-3 py-2 text-sm bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 rounded-lg hover:bg-emerald-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      File SAR {!canSAR && '🔒'}
                    </button>
                    <button
                      onClick={() => disposeCase('ESCALATED')}
                      disabled={actionLoading}
                      className="px-3 py-2 text-sm bg-rose-50 text-rose-700 ring-1 ring-rose-200 rounded-lg hover:bg-rose-100 transition-colors disabled:opacity-50"
                    >
                      Escalate
                    </button>
                    <button
                      onClick={() => disposeCase('DISMISSED')}
                      disabled={actionLoading || !canDismiss}
                      title={!canDismiss ? 'Requires SUPERVISOR role' : undefined}
                      className="px-3 py-2 text-sm bg-gray-50 text-gray-700 ring-1 ring-gray-200 rounded-lg hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Dismiss {!canDismiss && '🔒'}
                    </button>
                  </div>
                  {note && (
                    <button
                      onClick={addNote}
                      disabled={actionLoading}
                      className="w-full px-3 py-2 text-sm text-indigo-600 hover:text-indigo-800 border border-dashed border-indigo-300 rounded-lg hover:bg-indigo-50 transition-colors disabled:opacity-50"
                    >
                      Add Note Only (no status change)
                    </button>
                  )}
                </div>

                {/* Hash chain verify */}
                <div className="pt-3 border-t border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Audit Log (tamper-evident)</h3>
                    <button
                      onClick={verifyChain}
                      disabled={!canVerify || actionLoading}
                      title={!canVerify ? 'Requires SUPERVISOR role' : undefined}
                      className="text-xs px-2 py-1 border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Verify chain {!canVerify && '🔒'}
                    </button>
                  </div>
                  {chainStatus && (
                    <div className={`mb-2 rounded-md px-3 py-2 text-xs ${
                      chainStatus.verified ? 'bg-emerald-50 text-emerald-900 ring-1 ring-emerald-200'
                                            : 'bg-red-50 text-red-900 ring-1 ring-red-200'
                    }`}>
                      {chainStatus.verified
                        ? `✓ Chain intact — ${chainStatus.entries} entries. Head: ${chainStatus.head_hash?.slice(0, 12)}…`
                        : `✗ Tampering detected at entry ${chainStatus.tampered_at}: ${chainStatus.message}`}
                    </div>
                  )}
                  {caseDetail?.audit_log && caseDetail.audit_log.length > 0 && (
                    <ol className="space-y-2">
                      {[...caseDetail.audit_log].reverse().map((entry, i) => (
                        <li key={i} className="text-xs border-l-2 border-indigo-200 pl-3">
                          <div className="flex items-center gap-2 text-gray-500">
                            <span className="font-mono">{entry.timestamp?.slice(0, 19).replace('T', ' ')}</span>
                            <span>•</span>
                            <span className="font-medium text-gray-700">{entry.author}</span>
                            <span>•</span>
                            <span className="text-indigo-600 font-medium">{entry.action}</span>
                          </div>
                          {entry.note && <p className="text-gray-700 mt-1">{entry.note}</p>}
                          {entry.this_hash && (
                            <p className="text-[10px] font-mono text-gray-400 mt-0.5">
                              hash: {entry.this_hash.slice(0, 16)}…
                            </p>
                          )}
                        </li>
                      ))}
                    </ol>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
