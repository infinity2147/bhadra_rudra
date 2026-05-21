import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, ComposedChart, Line,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import MetricCard from '../components/MetricCard';
import { fetchAPI } from '../api';

const PIE_COLORS = ['#6366f1', '#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6'];

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

function isoFromInput(s) {
  if (!s) return '';
  return s.replace('T', ' ') + ':00';
}

function inputFromIso(s) {
  if (!s) return '';
  return s.slice(0, 16); // 'YYYY-MM-DDTHH:MM'
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [until, setUntil] = useState('');         // '' = no time travel
  const [bench, setBench] = useState(null);

  const load = (untilParam) => {
    setLoading(true);
    const qs = untilParam ? `?until=${encodeURIComponent(untilParam)}` : '';
    fetchAPI(`/api/dashboard${qs}`)
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  // Once we know the window, set the slider to the end as default
  useEffect(() => {
    if (data?.time_window?.end && !until) {
      // leave 'until' empty so default fetch shows everything
    }
  }, [data, until]);

  function onTimeTravel(value) {
    setUntil(value);
    load(value ? isoFromInput(value) : null);
  }

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

  const { kpis, pattern_breakdown, risk_distribution, case_status_counts, amount_distribution, time_window } = data;
  const risk_data = Object.entries(risk_distribution || {}).map(([level, count]) => ({ level, count }));
  const case_data = Object.entries(case_status_counts || {}).map(([status, count]) => ({ status, count }));
  const pattern_data = (pattern_breakdown || []).map(p => ({
    ...p,
    pattern: (p.fraud_pattern || '').replace(/_/g, ' '),
  }));

  // Time-travel slider derives min/max from the full window
  const windowStart = time_window?.start || '';
  const windowEnd = time_window?.end || '';

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Fund Flow Intelligence — Overview</h1>
        <p className="text-sm text-gray-500 mt-1">
          {kpis.total_transactions.toLocaleString('en-IN')} transactions monitored across {Object.keys(risk_distribution || {}).length}
          {' '}risk tiers and {kpis.incidents} clustered incidents. Click any card to drill in.
        </p>
      </div>

      {/* Time-travel slider */}
      <div className="bg-white rounded-xl border border-gray-200 p-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <p className="text-sm font-semibold text-gray-800">Time travel</p>
            <p className="text-xs text-gray-500">
              Replay the dataset up to any point — KPIs and charts recompute on the sliced data.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="datetime-local"
              value={until || (windowEnd ? inputFromIso(windowEnd) : '')}
              min={windowStart ? inputFromIso(windowStart) : undefined}
              max={windowEnd ? inputFromIso(windowEnd) : undefined}
              onChange={(e) => onTimeTravel(e.target.value)}
              className="px-2 py-1.5 border border-gray-300 rounded text-sm"
            />
            {until && (
              <button
                onClick={() => onTimeTravel('')}
                className="text-xs text-indigo-600 hover:text-indigo-800"
              >
                Reset
              </button>
            )}
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={100}
          value={until && windowStart && windowEnd ? (() => {
            const lo = new Date(windowStart).getTime();
            const hi = new Date(windowEnd).getTime();
            const cur = new Date(isoFromInput(until)).getTime();
            return Math.round(((cur - lo) / (hi - lo)) * 100);
          })() : 100}
          onChange={(e) => {
            if (!windowStart || !windowEnd) return;
            const lo = new Date(windowStart).getTime();
            const hi = new Date(windowEnd).getTime();
            const pct = Number(e.target.value) / 100;
            const ms = lo + pct * (hi - lo);
            const iso = new Date(ms).toISOString().slice(0, 16);
            onTimeTravel(iso);
          }}
          className="w-full mt-3"
        />
        <div className="mt-1 flex items-center justify-between text-[10px] text-gray-400 font-mono">
          <span>{windowStart?.slice(0, 10) || '—'}</span>
          <span>{until || windowEnd?.slice(0, 16) || '—'}</span>
          <span>{windowEnd?.slice(0, 10) || '—'}</span>
        </div>
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
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Fraud Pattern Breakdown</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pattern_data}
                dataKey="count"
                nameKey="pattern"
                cx="50%" cy="50%" outerRadius={90}
                label={({ pattern, percent }) => `${pattern} (${(percent * 100).toFixed(0)}%)`}
                labelLine={{ stroke: '#9ca3af' }}
              >
                {pattern_data.map((_, index) => (
                  <Cell key={index} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
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
