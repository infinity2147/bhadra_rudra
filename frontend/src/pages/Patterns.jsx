import { useState, useEffect } from 'react';
import { fetchAPI } from '../api';
import MetricCard from '../components/MetricCard';

const TABS = [
  { key: 'circular', label: 'Circular' },
  { key: 'layering', label: 'Layering' },
  { key: 'smurfing', label: 'Smurfing' },
  { key: 'funnel', label: 'Shell Funnels' },
  { key: 'dormant', label: 'Dormant' },
  { key: 'profile', label: 'Profile Mismatch' },
];

const GRAPH_TABS = new Set(['circular', 'layering', 'smurfing', 'funnel']);

function formatINR(amount) {
  return '₹' + Number(amount).toLocaleString('en-IN');
}

function AlertCard({ alert, defaultExpanded = false }) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-gray-50 transition-colors"
      >
        <span className="text-sm font-semibold text-gray-800 flex-1">
          {alert.pattern_type || alert.alert_type || 'Alert'}
        </span>
        {alert.total_flow != null && (
          <span className="text-sm font-medium text-gray-700">
            {formatINR(alert.total_flow)}
          </span>
        )}
        <svg
          className={`h-5 w-5 text-gray-400 transition-transform ${
            expanded ? 'rotate-180' : ''
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

      {expanded && (
        <div className="border-t border-gray-100 px-5 py-4 space-y-3">
          {/* Description */}
          {alert.description && (
            <p className="text-sm text-gray-700 leading-relaxed">
              {alert.description}
            </p>
          )}

          {/* Entities */}
          {alert.entities && alert.entities.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Entities
              </p>
              <p className="text-sm text-gray-700">
                {alert.entities.join(' → ')}
              </p>
            </div>
          )}

          {/* Flow amounts */}
          {alert.flows && alert.flows.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Flow Amounts
              </p>
              <div className="flex flex-wrap gap-2">
                {alert.flows.map((f, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
                  >
                    {typeof f === 'number' ? formatINR(f) : String(f)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Z-score (dormant / profile alerts) */}
          {alert.z_score != null && (
            <div>
              <p className="text-xs text-gray-500">Z-Score</p>
              <p className="text-sm font-semibold text-gray-800">
                {Number(alert.z_score).toFixed(2)}
              </p>
            </div>
          )}

          {/* Mismatch description (profile alerts) */}
          {alert.mismatch && (
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Mismatch
              </p>
              <p className="text-sm text-gray-700">{alert.mismatch}</p>
            </div>
          )}

          {/* General metrics row */}
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
          </div>
        </div>
      )}
    </div>
  );
}

function TransactionTable({ transactions }) {
  if (!transactions || transactions.length === 0) {
    return (
      <p className="text-sm text-gray-400 py-4">
        No transaction data available for this pattern.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead>
          <tr className="bg-gray-50">
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Timestamp
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Sender
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Receiver
            </th>
            <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Amount
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Type
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Case ID
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {transactions.map((tx, i) => (
            <tr key={tx.case_id ?? tx.id ?? i} className="hover:bg-gray-50">
              <td className="px-4 py-3 text-gray-700 whitespace-nowrap">
                {tx.timestamp ?? tx.date ?? '-'}
              </td>
              <td className="px-4 py-3 text-gray-700 whitespace-nowrap">
                {tx.sender ?? '-'}
              </td>
              <td className="px-4 py-3 text-gray-700 whitespace-nowrap">
                {tx.receiver ?? '-'}
              </td>
              <td className="px-4 py-3 text-gray-700 text-right whitespace-nowrap font-medium">
                {tx.amount != null ? formatINR(tx.amount) : '-'}
              </td>
              <td className="px-4 py-3 text-gray-700 whitespace-nowrap">
                {tx.type ?? tx.transaction_type ?? '-'}
              </td>
              <td className="px-4 py-3 text-gray-700 whitespace-nowrap">
                {tx.case_id ?? tx.caseId ?? '-'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Patterns() {
  const [activeTab, setActiveTab] = useState('circular');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchAPI(`/api/patterns/${activeTab}`)
      .then((res) => {
        if (!cancelled) {
          setData(res);
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
  }, [activeTab]);

  // Normalise API response: support { alerts, transactions } or bare array
  const alerts = Array.isArray(data)
    ? data
    : data?.alerts ?? [];
  const transactions = Array.isArray(data?.transactions)
    ? data.transactions
    : [];

  const isGraphTab = GRAPH_TABS.has(activeTab);

  // Metrics for graph-type tabs
  const metrics = (() => {
    if (!isGraphTab) return null;
    const patternCount = alerts.length;
    const totalVolume = alerts.reduce(
      (sum, a) => sum + (a.total_flow ?? 0),
      0
    );
    return { patternCount, totalVolume };
  })();

  return (
    <div className="min-h-screen bg-gray-50 p-6 space-y-6">
      {/* Header */}
      <h1 className="text-2xl font-bold text-gray-900">
        Pattern Analysis
      </h1>

      {/* Tab bar */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex gap-6" aria-label="Tabs">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`whitespace-nowrap pb-3 pt-1 text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? 'border-b-2 border-indigo-600 text-indigo-600'
                  : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Loading / Error */}
      {loading && (
        <div className="text-center py-12 text-gray-500">
          Loading pattern data...
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 p-4 text-red-700 text-sm">
          Failed to load patterns: {error}
        </div>
      )}

      {!loading && !error && (
        <>
          {/* Metric cards for graph-type tabs */}
          {metrics && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <MetricCard
                label="Patterns Detected"
                value={metrics.patternCount}
                color="indigo"
              />
              <MetricCard
                label="Total Volume"
                value={formatINR(metrics.totalVolume)}
                color="purple"
              />
            </div>
          )}

          {/* Alert cards */}
          <div className="space-y-3">
            {alerts.length === 0 && (
              <p className="text-center text-gray-400 py-12">
                No alerts found for this pattern type.
              </p>
            )}
            {alerts.map((alert, i) => (
              <AlertCard
                key={alert.alert_id ?? alert.id ?? i}
                alert={alert}
              />
            ))}
          </div>

          {/* Transaction table for graph-type tabs */}
          {isGraphTab && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                Transactions
              </h3>
              <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
                <TransactionTable transactions={transactions} />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
