import { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';
import MetricCard from '../components/MetricCard';

const SEVERITY_OPTIONS = ['CRITICAL', 'HIGH', 'MEDIUM'];
const PATTERN_OPTIONS = [
  'Circular Transaction',
  'Rapid Layering',
  'Smurfing / Structuring',
  'Shell Company Funnel',
  'Dormant Activation',
  'Profile Mismatch',
];

const SEVERITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

function formatINR(amount) {
  return '₹' + Number(amount).toLocaleString('en-IN');
}

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedIds, setExpandedIds] = useState(new Set());

  // Filter state
  const [severityFilter, setSeverityFilter] = useState([]);
  const [patternFilter, setPatternFilter] = useState([]);

  // Dropdown open state
  const [severityOpen, setSeverityOpen] = useState(false);
  const [patternOpen, setPatternOpen] = useState(false);

  const buildQuery = useCallback(() => {
    const params = new URLSearchParams();
    severityFilter.forEach((s) => params.append('severity', s));
    patternFilter.forEach((p) => params.append('pattern_type', p));
    const qs = params.toString();
    return qs ? `?${qs}` : '';
  }, [severityFilter, patternFilter]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchAPI(`/api/alerts${buildQuery()}`)
      .then((data) => {
        if (!cancelled) {
          setAlerts(Array.isArray(data) ? data : data.alerts ?? []);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [buildQuery]);

  // Sort alerts by severity
  const sortedAlerts = useMemo(() => {
    return [...alerts].sort(
      (a, b) =>
        (SEVERITY_ORDER[a.severity] ?? 99) - (SEVERITY_ORDER[b.severity] ?? 99)
    );
  }, [alerts]);

  // Metrics
  const metrics = useMemo(() => {
    const critical = alerts.filter((a) => a.severity === 'CRITICAL').length;
    const high = alerts.filter((a) => a.severity === 'HIGH').length;
    const medium = alerts.filter((a) => a.severity === 'MEDIUM').length;
    const totalVolume = alerts.reduce((sum, a) => sum + (a.total_flow ?? 0), 0);
    return { critical, high, medium, totalVolume };
  }, [alerts]);

  const toggleExpanded = (id) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // --- Multiselect helpers ---
  const toggleValue = (arr, setArr, val) => {
    setArr((prev) =>
      prev.includes(val) ? prev.filter((v) => v !== val) : [...prev, val]
    );
  };

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClick = () => {
      setSeverityOpen(false);
      setPatternOpen(false);
    };
    if (severityOpen || patternOpen) {
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [severityOpen, patternOpen]);

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">
      {/* Header */}
      <h1 className="text-2xl font-bold text-gray-900">
        Fraud Alert Dashboard
      </h1>

      {/* Filter row */}
      <div className="flex flex-wrap gap-4 items-start">
        {/* Severity multiselect */}
        <div className="relative" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            onClick={() => setSeverityOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            Severity
            {severityFilter.length > 0 && (
              <span className="inline-flex items-center justify-center rounded-full bg-indigo-600 text-white text-xs w-5 h-5">
                {severityFilter.length}
              </span>
            )}
            <svg className="h-4 w-4 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
          {severityOpen && (
            <div className="absolute z-20 mt-1 w-48 rounded-lg border border-gray-200 bg-white shadow-lg">
              <div className="py-1">
                {SEVERITY_OPTIONS.map((opt) => (
                  <label
                    key={opt}
                    className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={severityFilter.includes(opt)}
                      onChange={() => toggleValue(severityFilter, setSeverityFilter, opt)}
                      className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    {opt}
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Pattern type multiselect */}
        <div className="relative" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            onClick={() => setPatternOpen((v) => !v)}
            className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            Pattern Type
            {patternFilter.length > 0 && (
              <span className="inline-flex items-center justify-center rounded-full bg-indigo-600 text-white text-xs w-5 h-5">
                {patternFilter.length}
              </span>
            )}
            <svg className="h-4 w-4 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
          {patternOpen && (
            <div className="absolute z-20 mt-1 w-64 rounded-lg border border-gray-200 bg-white shadow-lg">
              <div className="py-1">
                {PATTERN_OPTIONS.map((opt) => (
                  <label
                    key={opt}
                    className="flex items-center gap-2 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={patternFilter.includes(opt)}
                      onChange={() => toggleValue(patternFilter, setPatternFilter, opt)}
                      className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                    />
                    {opt}
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Clear filters */}
        {(severityFilter.length > 0 || patternFilter.length > 0) && (
          <button
            type="button"
            onClick={() => {
              setSeverityFilter([]);
              setPatternFilter([]);
            }}
            className="text-sm text-indigo-600 hover:text-indigo-800 font-medium self-center"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Critical Alerts" value={metrics.critical} color="red" />
        <MetricCard label="High Alerts" value={metrics.high} color="amber" />
        <MetricCard label="Medium Alerts" value={metrics.medium} color="blue" />
        <MetricCard
          label="Total Flagged Volume"
          value={formatINR(metrics.totalVolume)}
          color="purple"
        />
      </div>

      {/* Loading / Error */}
      {loading && (
        <div className="text-center py-12 text-gray-500">Loading alerts...</div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 p-4 text-red-700 text-sm">
          Failed to load alerts: {error}
        </div>
      )}

      {/* Alert list */}
      {!loading && !error && (
        <div className="space-y-3">
          {sortedAlerts.length === 0 && (
            <p className="text-center text-gray-400 py-12">
              No alerts match the current filters.
            </p>
          )}
          {sortedAlerts.map((alert) => {
            const isExpanded = expandedIds.has(alert.alert_id ?? alert.id);
            return (
              <div
                key={alert.alert_id ?? alert.id}
                className="rounded-xl border border-gray-200 bg-white shadow-sm"
              >
                {/* Collapsed header */}
                <button
                  type="button"
                  onClick={() => toggleExpanded(alert.alert_id ?? alert.id)}
                  className="w-full flex items-center gap-4 px-5 py-4 text-left hover:bg-gray-50 transition-colors"
                >
                  <SeverityBadge severity={alert.severity} />
                  <span className="text-sm font-semibold text-gray-800 flex-1">
                    {alert.pattern_type}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(alert.confidence != null
                      ? `${Number(alert.confidence).toFixed(1)}%`
                      : '')}
                  </span>
                  <span className="text-sm font-medium text-gray-700">
                    {alert.total_flow != null ? formatINR(alert.total_flow) : ''}
                  </span>
                  <svg
                    className={`h-5 w-5 text-gray-400 transition-transform ${
                      isExpanded ? 'rotate-180' : ''
                    }`}
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>

                {/* Expanded detail */}
                {isExpanded && (
                  <div className="border-t border-gray-100 px-5 py-4 space-y-4">
                    {/* Description */}
                    {alert.description && (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                          Description
                        </h4>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {alert.description}
                        </p>
                      </div>
                    )}

                    {/* Recommendation */}
                    {alert.recommendation && (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                          Recommendation
                        </h4>
                        <p className="text-sm text-gray-700 leading-relaxed">
                          {alert.recommendation}
                        </p>
                      </div>
                    )}

                    {/* Entity chain */}
                    {alert.entities && alert.entities.length > 0 && (
                      <div>
                        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                          Entity Chain
                        </h4>
                        <p className="text-sm text-gray-700">
                          {alert.entities.join(' → ')}
                        </p>
                      </div>
                    )}

                    {/* Metrics row */}
                    <div className="flex flex-wrap gap-6">
                      {alert.total_flow != null && (
                        <div>
                          <p className="text-xs text-gray-500">Total Flow</p>
                          <p className="text-sm font-semibold text-gray-800">
                            {formatINR(alert.total_flow)}
                          </p>
                        </div>
                      )}
                      {alert.confidence != null && (
                        <div>
                          <p className="text-xs text-gray-500">Confidence</p>
                          <p className="text-sm font-semibold text-gray-800">
                            {Number(alert.confidence).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {alert.entities && (
                        <div>
                          <p className="text-xs text-gray-500">Entity Count</p>
                          <p className="text-sm font-semibold text-gray-800">
                            {alert.entities.length}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
