import { useState, useEffect } from 'react';
import { fetchAPI, postAPI, getRole } from '../api';

const FIELDS = [
  // Circular
  { key: 'circular_amount_tolerance', label: 'Circular: amount tolerance (cycle members)',
    detector: 'Circular Transactions', type: 'percent', min: 0, max: 0.5, step: 0.01 },
  { key: 'circular_min_total_flow', label: 'Circular: min total flow (INR)',
    detector: 'Circular Transactions', type: 'inr', min: 0, max: 10_000_000, step: 50_000 },
  { key: 'circular_max_cycle_length', label: 'Circular: max cycle length',
    detector: 'Circular Transactions', type: 'int', min: 3, max: 12 },
  { key: 'circular_max_alerts', label: 'Circular: cap on alerts emitted',
    detector: 'Circular Transactions', type: 'int', min: 10, max: 500 },
  // Layering
  { key: 'layering_min_chain_length', label: 'Layering: min chain length',
    detector: 'Rapid Layering', type: 'int', min: 3, max: 10 },
  { key: 'layering_max_chains', label: 'Layering: cap on chains',
    detector: 'Rapid Layering', type: 'int', min: 10, max: 1000 },
  { key: 'layering_decrease_ratio', label: 'Layering: minimum next/prev ratio',
    detector: 'Rapid Layering', type: 'percent', min: 0.5, max: 1.0, step: 0.05 },
  // Smurfing
  { key: 'smurfing_threshold', label: 'Smurfing: reporting threshold (INR)',
    detector: 'Smurfing / Structuring', type: 'inr', min: 100_000, max: 1_000_000, step: 10_000 },
  { key: 'smurfing_cluster_tolerance', label: 'Smurfing: cluster tolerance',
    detector: 'Smurfing / Structuring', type: 'percent', min: 0, max: 0.3, step: 0.01 },
  // Funnel
  { key: 'funnel_imbalance_threshold', label: 'Funnel: in/out imbalance threshold',
    detector: 'Shell Company Funnel', type: 'percent', min: 0.3, max: 1.0, step: 0.05 },
  { key: 'funnel_min_in_degree', label: 'Funnel: min sources',
    detector: 'Shell Company Funnel', type: 'int', min: 2, max: 10 },
  // Dormant
  { key: 'dormant_threshold_days', label: 'Dormant: gap in days',
    detector: 'Dormant Activation', type: 'int', min: 14, max: 180 },
  { key: 'dormant_z_score_threshold', label: 'Dormant: spike Z-score',
    detector: 'Dormant Activation', type: 'float', min: 1.5, max: 6, step: 0.1 },
  // ML
  { key: 'ml_alert_threshold', label: 'ML: alert when probability ≥',
    detector: 'ML Classifier', type: 'percent', min: 0.3, max: 0.95, step: 0.05 },
  // Fund Tracer annotation thresholds
  { key: 'tracer_structuring_threshold', label: 'Tracer: structuring threshold (INR)',
    detector: 'Fund Tracer', type: 'inr', min: 50_000, max: 1_000_000, step: 10_000 },
  { key: 'tracer_high_value_threshold', label: 'Tracer: high-value rail threshold (INR)',
    detector: 'Fund Tracer', type: 'inr', min: 100_000, max: 10_000_000, step: 100_000 },
  { key: 'tracer_baseline_z_threshold', label: 'Tracer: outflow Z-score anomaly threshold',
    detector: 'Fund Tracer', type: 'float', min: 1.5, max: 6, step: 0.1 },
  { key: 'tracer_burst_window_minutes', label: 'Tracer: burst window (minutes)',
    detector: 'Fund Tracer', type: 'int', min: 10, max: 240 },
  { key: 'tracer_burst_min_cluster', label: 'Tracer: burst min cluster size',
    detector: 'Fund Tracer', type: 'int', min: 2, max: 10 },
  { key: 'tracer_transit_window_hours', label: 'Tracer: transit-node window (hours)',
    detector: 'Fund Tracer', type: 'float', min: 0.5, max: 24, step: 0.5 },
  { key: 'tracer_transit_ratio_threshold', label: 'Tracer: transit-node ratio threshold',
    detector: 'Fund Tracer', type: 'percent', min: 0.3, max: 0.95, step: 0.05 },
];

function formatValue(val, type) {
  if (val == null) return '--';
  if (type === 'percent') return Number(val).toFixed(2);
  if (type === 'inr') return `₹${Number(val).toLocaleString('en-IN')}`;
  if (type === 'float') return Number(val).toFixed(2);
  return val;
}

export default function Settings() {
  const role = getRole();
  const isAdmin = role === 'ADMIN';
  const [current, setCurrent] = useState({});
  const [defaults, setDefaults] = useState({});
  const [draft, setDraft] = useState({});
  const [saving, setSaving] = useState(false);
  const [rerunning, setRerunning] = useState(false);
  const [rerunResult, setRerunResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAPI('/api/config/thresholds')
      .then((d) => {
        setCurrent(d.current || {});
        setDefaults(d.defaults || {});
        setDraft(d.current || {});
      })
      .catch((e) => setError(e.message));
  }, []);

  const dirty = JSON.stringify(draft) !== JSON.stringify(current);

  async function save() {
    setSaving(true); setError(null);
    try {
      const changed = {};
      for (const k of Object.keys(draft)) {
        if (JSON.stringify(draft[k]) !== JSON.stringify(current[k])) {
          changed[k] = draft[k];
        }
      }
      const r = await postAPI('/api/config/thresholds', changed);
      setCurrent(r.current);
      setDraft(r.current);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  async function reset() {
    setSaving(true);
    try {
      const r = await postAPI('/api/config/thresholds/reset', {});
      setCurrent(r.current);
      setDraft(r.current);
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  }

  async function rerun() {
    setRerunning(true); setRerunResult(null);
    try {
      const r = await postAPI('/api/config/rerun', {});
      setRerunResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setRerunning(false);
    }
  }

  // Group fields by detector
  const grouped = FIELDS.reduce((m, f) => {
    (m[f.detector] = m[f.detector] || []).push(f);
    return m;
  }, {});

  return (
    <div className="p-6 max-w-5xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Detector Settings</h1>
        <p className="text-sm text-gray-500 mt-1">
          Tune the per-detector knobs. Production banks would gate this behind a senior compliance officer.
        </p>
      </div>

      {!isAdmin && (
        <div className="rounded-xl bg-amber-50 border border-amber-200 px-4 py-3 text-sm">
          <p className="font-semibold text-amber-900">Read-only as {role}</p>
          <p className="text-amber-800 mt-1">
            Editing thresholds requires the <strong>ADMIN</strong> role. Switch role in the sidebar.
          </p>
        </div>
      )}

      {error && (
        <div className="rounded-xl bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        {Object.entries(grouped).map(([detector, fields]) => (
          <div key={detector} className="border-b border-gray-200 last:border-b-0">
            <div className="px-5 py-3 bg-gray-50 border-b border-gray-200">
              <h2 className="text-sm font-semibold text-gray-900">{detector}</h2>
            </div>
            <div className="p-5 space-y-4">
              {fields.map((f) => {
                const val = draft[f.key];
                const def = defaults[f.key];
                return (
                  <div key={f.key} className="grid grid-cols-12 gap-3 items-center">
                    <label className="col-span-5 text-sm text-gray-700">
                      {f.label}
                      <div className="text-[10px] text-gray-400">default {formatValue(def, f.type)}</div>
                    </label>
                    <input
                      type="range"
                      min={f.min}
                      max={f.max}
                      step={f.step || (f.type === 'int' ? 1 : 0.01)}
                      value={val ?? 0}
                      onChange={(e) => setDraft((d) => ({ ...d, [f.key]: Number(e.target.value) }))}
                      disabled={!isAdmin}
                      className="col-span-4 disabled:opacity-50"
                    />
                    <input
                      type="number"
                      step={f.step || (f.type === 'int' ? 1 : 0.01)}
                      value={val ?? 0}
                      onChange={(e) => setDraft((d) => ({ ...d, [f.key]: Number(e.target.value) }))}
                      disabled={!isAdmin}
                      className="col-span-2 px-2 py-1 border border-gray-300 rounded text-sm disabled:bg-gray-50"
                    />
                    <span className="col-span-1 text-xs text-gray-500">
                      {f.type === 'inr' ? '₹' : f.type === 'percent' ? '×' : ''}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3 sticky bottom-0 bg-white p-3 -mx-6 px-6 border-t border-gray-200">
        <button
          onClick={save}
          disabled={!isAdmin || !dirty || saving}
          className="px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          onClick={reset}
          disabled={!isAdmin || saving}
          className="px-4 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
        >
          Reset to defaults
        </button>
        <button
          onClick={rerun}
          disabled={!isAdmin || rerunning}
          className="ml-auto px-4 py-2 text-sm bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50"
        >
          {rerunning ? 'Re-running...' : 'Re-run all detectors'}
        </button>
      </div>

      {rerunResult && (
        <div className="rounded-xl bg-emerald-50 border border-emerald-200 p-4 text-sm">
          <p className="font-semibold text-emerald-900">Detection re-ran successfully.</p>
          <p className="text-emerald-800 mt-1">
            {rerunResult.alert_count} alerts → {rerunResult.incident_count} incidents.
          </p>
          <p className="text-xs text-emerald-700 mt-1">
            Critical: {rerunResult.summary?.critical_alerts ?? 0} • High: {rerunResult.summary?.high_alerts ?? 0} • Medium: {rerunResult.summary?.medium_alerts ?? 0}
          </p>
        </div>
      )}
    </div>
  );
}
