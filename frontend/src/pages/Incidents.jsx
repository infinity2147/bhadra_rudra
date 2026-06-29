import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { fetchAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';
import RegBadges from '../components/RegBadges';

function formatINR(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

const SEVERITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
// Cap rendered incident cards so 4k+ incidents don't choke the DOM.
const INC_CAP = 200;

// Pick the most severe legal_basis among an incident's alerts (prefer STR).
function pickLegalBasis(alerts) {
  const bases = (alerts || []).map((a) => a.legal_basis).filter(Boolean);
  if (bases.length === 0) return null;
  return bases.find((b) => b.includes('STR')) || bases[0];
}

export default function Incidents() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(params.get('id'));
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const d = await fetchAPI('/api/incidents');
        if (!cancelled) setIncidents((d.incidents || []).sort(
          (a, b) => (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9),
        ));
      } catch {
        if (!cancelled) setIncidents([]);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!selected) { setDetail(null); return; }
      try {
        const d = await fetchAPI(`/api/incidents/${selected}`);
        if (!cancelled) setDetail(d);
      } catch {
        if (!cancelled) setDetail(null);
      }
    })();
    return () => { cancelled = true; };
  }, [selected]);

  return (
    <div className="h-full flex flex-col">
      <div className="p-6 pb-2">
        <h1 className="text-2xl font-bold text-gray-900">Incidents</h1>
        <p className="text-sm text-gray-500 mt-1">
          {incidents.length} clustered incidents — alerts grouped by entity overlap. Triage these instead of the raw alert list.
        </p>
      </div>

      <div className="flex-1 flex min-h-0">
        <div className={`${selected ? 'w-1/2' : 'w-full'} overflow-y-auto p-6 transition-all`}>
          {loading ? (
            <div className="text-center text-gray-400 py-12">Loading incidents...</div>
          ) : incidents.length === 0 ? (
            <div className="text-center text-gray-400 py-12">No incidents yet.</div>
          ) : (
            <div className="space-y-3">
              {/* Cap rendered cards — painting 4k+ incidents is what made this
                  page slow (the data is already loaded). Sorted by severity. */}
              {incidents.slice(0, INC_CAP).map((inc) => {
                const isSel = inc.incident_id === selected;
                return (
                  <button
                    key={inc.incident_id}
                    onClick={() => { setSelected(inc.incident_id); setParams({ id: inc.incident_id }); }}
                    className={`w-full text-left rounded-xl border p-4 transition-colors ${
                      isSel
                        ? 'border-indigo-500 bg-indigo-50'
                        : 'border-gray-200 bg-white hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono text-gray-400">{inc.incident_id}</span>
                          <SeverityBadge severity={inc.severity} />
                        </div>
                        <h3 className="text-base font-semibold text-gray-900 mt-1">
                          {inc.primary_pattern}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                          {inc.primary_alert_description}
                        </p>
                        <div className="flex flex-wrap items-center gap-2 mt-2 text-xs text-gray-500">
                          <span><strong>{inc.alert_count}</strong> alerts</span>
                          <span>•</span>
                          <span><strong>{inc.n_entities}</strong> entities</span>
                          <span>•</span>
                          <span className="font-medium text-gray-700">{formatINR(inc.total_flow)}</span>
                          {inc.patterns.length > 1 && (
                            <>
                              <span>•</span>
                              <span>{inc.patterns.length} pattern types</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
              {incidents.length > INC_CAP && (
                <p className="text-center text-xs text-gray-500 py-2">
                  Showing top {INC_CAP} of {incidents.length.toLocaleString()} incidents (by severity).
                </p>
              )}
            </div>
          )}
        </div>

        {selected && (
          <div className="w-1/2 border-l border-gray-200 bg-white overflow-y-auto p-6">
            {!detail ? (
              <div className="text-center text-gray-400 py-12">Loading...</div>
            ) : (
              <>
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <p className="text-xs font-mono text-gray-400">{detail.incident_id}</p>
                    <h2 className="text-lg font-bold text-gray-900 mt-0.5">{detail.primary_pattern}</h2>
                    <div className="flex items-center gap-2 mt-2">
                      <SeverityBadge severity={detail.severity} />
                      <span className="text-xs text-gray-500">{detail.alert_count} alerts</span>
                      <span className="text-xs text-gray-500">{detail.n_entities} entities</span>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelected(null)}
                    className="text-gray-400 text-xl"
                  >&times;</button>
                </div>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                    <p className="text-[11px] uppercase text-gray-500">Total Flow</p>
                    <p className="text-base font-semibold mt-0.5">{formatINR(detail.total_flow)}</p>
                  </div>
                  <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
                    <p className="text-[11px] uppercase text-gray-500">Patterns</p>
                    <p className="text-base font-semibold mt-0.5">{detail.patterns?.length || 0}</p>
                  </div>
                </div>

                {/* Regulatory basis — most severe legal_basis across underlying alerts */}
                {pickLegalBasis(detail.alerts) && (
                  <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3">
                    <p className="text-xs text-red-800">
                      <span className="font-semibold">Regulatory basis:</span> {pickLegalBasis(detail.alerts)}
                    </p>
                  </div>
                )}

                {/* Patterns list */}
                {detail.patterns && detail.patterns.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      Pattern Types in Incident
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {detail.patterns.map((p) => (
                        <span key={p} className="inline-flex items-center px-2 py-1 rounded-md bg-rose-50 text-rose-700 text-xs font-medium">
                          {p}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Entities */}
                {detail.entity_names && detail.entity_names.length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      Involved Entities ({detail.n_entities}{detail.n_entities > 20 ? ' — top 20 shown' : ''})
                    </h4>
                    <div className="flex flex-wrap gap-1.5">
                      {detail.entity_names.map((n, i) => (
                        <span key={i} className="inline-flex items-center px-2 py-1 rounded-md bg-indigo-50 text-indigo-700 text-xs font-medium">
                          {n}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Underlying alerts */}
                <div className="mt-6">
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                    Underlying Alerts ({detail.alerts?.length || 0})
                  </h4>
                  <div className="space-y-2 max-h-[40vh] overflow-y-auto">
                    {(detail.alerts || []).slice(0, 50).map((a) => (
                      <button
                        key={a.alert_id}
                        onClick={() => navigate(`/cases?focus=${a.alert_id}`)}
                        className="w-full text-left rounded-lg border border-gray-200 p-3 hover:bg-gray-50"
                      >
                        <div className="flex items-center gap-2">
                          <SeverityBadge severity={a.severity} />
                          {a.tier && (
                            <span
                              title={a.tier === 1 ? 'Tier 1 — ML + rule agree' : a.tier === 2 ? 'Tier 2 — ML only' : 'Tier 3 — rule only'}
                              className={`px-1.5 py-0.5 rounded text-[10px] font-bold ring-1 ${
                                a.tier === 1 ? 'bg-emerald-100 text-emerald-800 ring-emerald-300'
                                : a.tier === 2 ? 'bg-indigo-100 text-indigo-800 ring-indigo-300'
                                : 'bg-gray-100 text-gray-700 ring-gray-300'}`}
                            >T{a.tier}</span>
                          )}
                          <span className="text-xs font-mono text-gray-400">{a.alert_id}</span>
                          <span className="ml-auto text-xs text-gray-500">{formatINR(a.total_flow)}</span>
                        </div>
                        <p className="text-xs text-gray-700 mt-1 line-clamp-2">{a.description}</p>
                        <div className="mt-1.5"><RegBadges alert={a} /></div>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="mt-4 flex gap-2">
                  <button
                    onClick={() => navigate(`/journey?alert=${detail.primary_alert_id}`)}
                    className="flex-1 px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                  >
                    Trace primary alert
                  </button>
                  <button
                    onClick={() => navigate(`/cases?focus=${detail.primary_alert_id}`)}
                    className="flex-1 px-3 py-2 text-sm bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
                  >
                    Open in Workbench
                  </button>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
