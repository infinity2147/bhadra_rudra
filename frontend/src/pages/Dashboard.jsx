import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Cell, ComposedChart, Line,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import MetricCard from '../components/MetricCard';
import { fetchAPI } from '../api';

const RISK_COLOR = {
  CRITICAL: '#dc2626', HIGH: '#f97316', MEDIUM: '#f59e0b', LOW: '#22c55e',
};

const CASE_COLOR = {
  OPEN: '#f59e0b', INVESTIGATING: '#6366f1', SAR_FILED: '#10b981',
  ESCALATED: '#e11d48', DISMISSED: '#94a3b8',
};

function formatCr(value) {
  const cr = value / 10_000_000;
  return `₹${cr.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} Cr`;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [bench, setBench] = useState(null);

  const load = () => {
    setLoading(true);
    fetchAPI('/api/dashboard')
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  async function runBenchmark() {
    setBench({ loading: true });
    try {
      const b = await fetchAPI('/api/benchmark/latency');
      setBench(b);
    } catch (e) {
      setBench({ error: e.message });
    }
  }

  const dailyData = useMemo(() => data?.daily_data || [], [data]);

  if (loading && !data) return <div className="flex items-center justify-center h-full text-gray-400">Loading dashboard...</div>;
  if (error) return <div className="flex items-center justify-center h-full text-red-600">Error: {error}</div>;
  if (!data) return null;

  const { kpis, risk_distribution, case_status_counts, amount_distribution } = data;
  const risk_data = Object.entries(risk_distribution || {}).map(([level, count]) => ({ level, count }));
  const case_data = Object.entries(case_status_counts || {}).map(([status, count]) => ({ status, count }));

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Fund Flow Intelligence — Overview</h1>
        <p className="text-sm text-gray-500 mt-1">
          {kpis.total_transactions.toLocaleString('en-IN')} transactions monitored across {Object.keys(risk_distribution || {}).length}
          {' '}risk tiers and {kpis.incidents} clustered incidents. Click any card to drill in.
        </p>
      </div>

      {/* Primary KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Total Transactions" value={kpis.total_transactions.toLocaleString('en-IN')} color="indigo" />
        <MetricCard label="Total Volume" value={formatCr(kpis.total_volume)} color="blue" />
        <MetricCard label="Fraud Volume" value={formatCr(kpis.fraud_volume)} delta={`${kpis.fraud_rate}% of total`} color="amber" />
        <MetricCard
          label="Incidents"
          value={(kpis.incidents || 0).toLocaleString('en-IN')}
          delta={`${kpis.total_alerts} raw alerts • ${kpis.critical_alerts} critical`}
          color="red"
          onClick={() => navigate('/incidents')}
        />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Open Cases"
          value={(case_status_counts?.OPEN || 0).toLocaleString('en-IN')}
          delta={`${case_status_counts?.INVESTIGATING || 0} investigating`}
          color="amber"
          onClick={() => navigate('/cases')}
        />
        <MetricCard
          label="ML F1 Score"
          value={kpis.model_f1 != null ? kpis.model_f1.toFixed(3) : '—'}
          delta={kpis.model_auc != null ? `AUC ${kpis.model_auc.toFixed(3)}` : ''}
          color="purple"
          onClick={() => navigate('/model')}
        />
        <MetricCard
          label="High-Risk Entities"
          value={(kpis.high_risk_entities || 0).toLocaleString('en-IN')}
          delta="risk score ≥ 0.5"
          color="amber"
        />
        <MetricCard
          label="SAR Filed"
          value={(case_status_counts?.SAR_FILED || 0).toLocaleString('en-IN')}
          delta={`${case_status_counts?.DISMISSED || 0} dismissed`}
          color="green"
          onClick={() => navigate('/cases')}
        />
      </div>

      {/* AML mule signals — new */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Velocity Bursts"
          value={(kpis.velocity_burst_entities || 0).toLocaleString('en-IN')}
          delta="≥3 txns within 60 min"
          color="amber"
          onClick={() => navigate('/entities')}
        />
        <MetricCard
          label="Transit-Node Mules"
          value={(kpis.transit_node_entities || 0).toLocaleString('en-IN')}
          delta="≥50% inflow exits in 1h"
          color="red"
          onClick={() => navigate('/entities')}
        />
        <MetricCard
          label="Fraud Rate"
          value={`${kpis.fraud_rate ?? 0}%`}
          delta={`${kpis.fraud_transactions?.toLocaleString('en-IN') || 0} of ${kpis.total_transactions?.toLocaleString('en-IN') || 0}`}
          color="red"
        />
        <MetricCard
          label="Total Alerts"
          value={(kpis.total_alerts || 0).toLocaleString('en-IN')}
          delta={`${kpis.critical_alerts || 0} critical`}
          color="indigo"
          onClick={() => navigate('/incidents')}
        />
      </div>

      {/* Latency benchmark */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 flex items-start justify-between flex-wrap gap-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">Detection latency vs T+1 batch</h3>
          <p className="text-xs text-gray-500 mt-0.5">
            The RBI's 2023 FRM framework mandates real-time detection. T+1 batch processing wastes 24 hours.
          </p>
          {bench?.vs_t_plus_1 && (
            <div className="mt-2 flex items-baseline gap-3 flex-wrap">
              <span className="text-2xl font-bold text-emerald-700">{bench.vs_t_plus_1.speedup_factor.toLocaleString('en-IN')}×</span>
              <span className="text-sm text-gray-600">faster than T+1</span>
              <span className="text-xs text-gray-500">
                ({bench.pipeline_ms.total.toFixed(0)} ms for full pipeline,
                 ~{bench.per_txn_ms.mean.toFixed(2)} ms per txn — p95 {bench.per_txn_ms.p95.toFixed(2)} ms)
              </span>
            </div>
          )}
        </div>
        <button
          onClick={runBenchmark}
          disabled={bench?.loading}
          className="px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
        >
          {bench?.loading ? 'Benchmarking...' : bench?.vs_t_plus_1 ? 'Re-run benchmark' : 'Run benchmark'}
        </button>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Daily Transaction Trends</h3>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={dailyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} stroke="#ef4444" />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} name="Transactions" />
              <Line yAxisId="right" type="monotone" dataKey="fraud_count" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} name="Fraud" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Amount Distribution (Normal vs Fraud)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={amount_distribution || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="bucket" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <Tooltip />
              <Legend />
              <Bar dataKey="normal_count" fill="#3b82f6" radius={[3, 3, 0, 0]} name="Normal" />
              <Bar dataKey="fraud_count" fill="#ef4444" radius={[3, 3, 0, 0]} name="Fraud" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Risk Distribution &amp; Case Status</h3>
          <div className="grid grid-cols-2 gap-3">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={risk_data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="level" tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <YAxis tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <Tooltip />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {risk_data.map((entry, i) => (
                    <Cell key={i} fill={RISK_COLOR[entry.level] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={case_data} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 10 }} stroke="#9ca3af" />
                <YAxis dataKey="status" type="category" tick={{ fontSize: 10 }} width={90} stroke="#9ca3af" />
                <Tooltip />
                <Bar dataKey="count" radius={[0, 3, 3, 0]}>
                  {case_data.map((entry, i) => (
                    <Cell key={i} fill={CASE_COLOR[entry.status] || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
