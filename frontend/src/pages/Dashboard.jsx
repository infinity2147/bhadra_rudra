import { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
  Line,
} from 'recharts';
import MetricCard from '../components/MetricCard';
import { fetchAPI } from '../api';

const PIE_COLORS = ['#6366f1', '#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6'];

function formatCr(value) {
  const cr = value / 10_000_000;
  return `₹${cr.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} Cr`;
}

function formatINR(value) {
  return `₹${Number(value).toLocaleString('en-IN')}`;
}

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAPI('/api/dashboard')
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-pulse text-gray-400 text-lg">Loading dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-600 text-lg">Error: {error}</div>
      </div>
    );
  }

  const { kpis, daily_data, amount_stats, pattern_breakdown, risk_distribution: riskDistObj } = data;
  const risk_distribution = Object.entries(riskDistObj || {}).map(([level, count]) => ({ level, count }));

  // Build amount distribution from stats
  const amount_distribution = (() => {
    const normal = amount_stats?.normal || {};
    const fraud = amount_stats?.fraud || {};
    if (!normal.mean) return [];
    const buckets = [
      { label: '<₹50K', min: 0, max: 50000 },
      { label: '₹50K-2L', min: 50000, max: 200000 },
      { label: '₹2L-10L', min: 200000, max: 1000000 },
      { label: '₹10L-50L', min: 1000000, max: 5000000 },
      { label: '>₹50L', min: 5000000, max: Infinity },
    ];
    return buckets.map(b => ({ bucket: b.label, normal_count: Math.round(normal['50%'] || 0), fraud_count: Math.round(fraud['50%'] || 0) }));
  })();

  // Fix pattern_breakdown key names
  const pattern_data = (pattern_breakdown || []).map(p => ({
    ...p,
    pattern: p.fraud_pattern || p.pattern,
  }));

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">RUDRA Fund Flow Intelligence</h1>
        <p className="text-sm text-gray-500 mt-1">
          Real-time monitoring of transactions, fraud patterns, and risk distribution
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-5 gap-4">
        <MetricCard
          label="Total Transactions"
          value={Number(kpis.total_transactions).toLocaleString('en-IN')}
          color="indigo"
        />
        <MetricCard
          label="Total Volume"
          value={formatCr(kpis.total_volume)}
          color="blue"
        />
        <MetricCard
          label="Fraud Transactions"
          value={Number(kpis.fraud_transactions).toLocaleString('en-IN')}
          delta={`${kpis.fraud_rate}% of total`}
          color="red"
        />
        <MetricCard
          label="Fraud Volume"
          value={formatCr(kpis.fraud_volume)}
          color="amber"
        />
        <MetricCard
          label="Active Alerts"
          value={Number(kpis.total_alerts).toLocaleString('en-IN')}
          delta={`${kpis.critical_alerts} critical`}
          color="purple"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Top Left: Daily Transaction Counts with Fraud Overlay */}
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Daily Transaction Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={daily_data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} stroke="#ef4444" />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb' }}
                formatter={(value, name) => {
                  if (name === 'fraud_count') return [value, 'Fraud'];
                  return [Number(value).toLocaleString('en-IN'), name === 'count' ? 'Transactions' : name];
                }}
              />
              <Legend />
              <Bar yAxisId="left" dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} name="Transactions" />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="fraud_count"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ r: 3 }}
                name="Fraud"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Top Right: Amount Distribution */}
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Amount Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={amount_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="bucket" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb' }}
                formatter={(value, name) => [
                  Number(value).toLocaleString('en-IN'),
                  name === 'normal_count' ? 'Normal' : 'Fraud',
                ]}
              />
              <Legend />
              <Bar dataKey="normal_count" fill="#3b82f6" radius={[3, 3, 0, 0]} name="Normal" />
              <Bar dataKey="fraud_count" fill="#ef4444" radius={[3, 3, 0, 0]} name="Fraud" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Bottom Left: Pattern Breakdown Pie */}
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Fraud Pattern Breakdown</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pattern_data}
                dataKey="count"
                nameKey="fraud_pattern"
                cx="50%"
                cy="50%"
                outerRadius={100}
                label={({ fraud_pattern, percent }) => `${fraud_pattern} (${(percent * 100).toFixed(0)}%)`}
                labelLine={{ stroke: '#9ca3af' }}
              >
                {pattern_data.map((_, index) => (
                  <Cell key={index} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb' }}
                formatter={(value, name) => [Number(value).toLocaleString('en-IN'), name]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Bottom Right: Risk Distribution */}
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Risk Level Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={risk_distribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="level" tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <YAxis tick={{ fontSize: 11 }} stroke="#9ca3af" />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb' }}
                formatter={(value) => [Number(value).toLocaleString('en-IN'), 'Count']}
              />
              <Legend />
              <Bar dataKey="count" radius={[3, 3, 0, 0]} name="Count">
                {risk_distribution.map((entry, index) => {
                  const colorMap = {
                    low: '#22c55e',
                    medium: '#f59e0b',
                    high: '#f97316',
                    critical: '#ef4444',
                  };
                  return <Cell key={index} fill={colorMap[entry.level?.toLowerCase()] || '#6366f1'} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
