import { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchAPI, postAPI, downloadFromAPI } from '../api';
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

export default function Cases() {
  const navigate = useNavigate();
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeStatus, setActiveStatus] = useState('OPEN');
  const [selectedId, setSelectedId] = useState(null);
  const [caseDetail, setCaseDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [note, setNote] = useState('');
  const [search, setSearch] = useState('');

  // Load alerts on mount
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

  // Load case detail when selected
  useEffect(() => {
    if (!selectedId) {
      setCaseDetail(null);
      setNote('');
      return;
    }
    setDetailLoading(true);
    fetchAPI(`/api/cases/${selectedId}`)
      .then(setCaseDetail)
      .catch(() => setCaseDetail(null))
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  const filtered = useMemo(() => {
    let xs = alerts;
    if (activeStatus !== 'ALL') {
      xs = xs.filter(a => (a.case_status || 'OPEN') === activeStatus);
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
  }, [alerts, activeStatus, search]);

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

  return (
    <div className="h-full flex flex-col">
      <div className="px-6 pt-6 pb-3">
        <h1 className="text-2xl font-bold text-gray-900">Case Workbench</h1>
        <p className="text-sm text-gray-500 mt-1">
          Triage fraud alerts, record decisions, and generate FIU evidence packages
        </p>
      </div>

      {/* Status tabs + counts */}
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
          <div className="ml-auto py-2">
            <input
              type="text"
              placeholder="Search alerts, entities..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-64 px-3 py-1.5 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        </nav>
      </div>

      {/* Body — split: list + detail */}
      <div className="flex-1 flex min-h-0">
        {/* Cases list */}
        <div className={`${selectedId ? 'w-3/5' : 'w-full'} overflow-y-auto bg-gray-50 border-r border-gray-200 transition-all`}>
          {loading ? (
            <div className="p-12 text-center text-gray-400">Loading cases...</div>
          ) : filtered.length === 0 ? (
            <div className="p-12 text-center text-gray-400">No cases in this category.</div>
          ) : (
            <ul className="divide-y divide-gray-200 bg-white">
              {filtered.map(a => {
                const isSelected = a.alert_id === selectedId;
                const age = a.case_status ? '' : ''; // age would need timestamp from case
                return (
                  <li
                    key={a.alert_id}
                    onClick={() => setSelectedId(a.alert_id)}
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
                        </div>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">{a.description}</p>
                        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                          <span>{a.entities?.length || 0} entities</span>
                          <span>•</span>
                          <span className="font-medium text-gray-700">{formatINR(a.total_flow)}</span>
                          <span>•</span>
                          <span>Confidence {a.confidence}%</span>
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <p className="text-[11px] text-gray-400 uppercase tracking-wide mb-1">ML</p>
                        <MLScoreBar score={a.ml_score} />
                      </div>
                    </div>
                  </li>
                );
              })}
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
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-xs font-mono text-gray-400">{selectedAlert.alert_id}</p>
                    <h2 className="text-lg font-bold text-gray-900 mt-0.5">{selectedAlert.pattern_type}</h2>
                    <div className="flex items-center gap-2 mt-2">
                      <SeverityBadge severity={selectedAlert.severity} />
                      <StatusPill status={caseDetail?.status} />
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedId(null)}
                    className="text-gray-400 hover:text-gray-600 text-xl leading-none"
                  >
                    &times;
                  </button>
                </div>

                {/* Description */}
                <div>
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                    Description
                  </h3>
                  <p className="text-sm text-gray-700 leading-relaxed">{selectedAlert.description}</p>
                </div>

                {/* Recommendation */}
                {selectedAlert.recommendation && (
                  <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2">
                    <p className="text-[11px] font-semibold text-amber-700 uppercase tracking-wide">
                      Recommended Action
                    </p>
                    <p className="text-xs text-amber-900 mt-1 leading-relaxed">
                      {selectedAlert.recommendation}
                    </p>
                  </div>
                )}

                {/* Stats */}
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

                {/* Entity chain */}
                {selectedAlert.entity_names && selectedAlert.entity_names.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
                      Entity Chain
                    </h3>
                    <div className="flex flex-wrap gap-1.5">
                      {selectedAlert.entity_names.map((n, i) => (
                        <span key={i} className="inline-flex items-center px-2 py-1 rounded-md bg-indigo-50 text-indigo-700 text-xs font-medium">
                          {n}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Action buttons */}
                <div className="space-y-2 pt-2 border-t border-gray-200">
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => navigate(`/journey?alert=${selectedId}`)}
                      className="px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center justify-center gap-1.5"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                      Trace Journey
                    </button>
                    <button
                      onClick={downloadFiuPackage}
                      disabled={actionLoading}
                      className="px-3 py-2 text-sm bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors flex items-center justify-center gap-1.5 disabled:opacity-50"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      FIU Package
                    </button>
                  </div>

                  {/* Note input */}
                  <textarea
                    value={note}
                    onChange={e => setNote(e.target.value)}
                    placeholder="Add an investigator note (optional)..."
                    rows={2}
                    className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />

                  {/* Disposition buttons */}
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
                      disabled={actionLoading}
                      className="px-3 py-2 text-sm bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200 rounded-lg hover:bg-emerald-100 transition-colors disabled:opacity-50"
                    >
                      File SAR
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
                      disabled={actionLoading}
                      className="px-3 py-2 text-sm bg-gray-50 text-gray-700 ring-1 ring-gray-200 rounded-lg hover:bg-gray-100 transition-colors disabled:opacity-50"
                    >
                      Dismiss
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

                {/* Audit log */}
                {caseDetail?.audit_log && caseDetail.audit_log.length > 0 && (
                  <div className="pt-3 border-t border-gray-200">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      Audit Log
                    </h3>
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
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
