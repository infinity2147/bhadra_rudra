import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell, ComposedChart, Line,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import MetricCard from '../components/MetricCard';
import { fetchAPI } from '../api';

const PIE_COLORS = ['#6366f1', '#ef4444', '#f59e0b', '#22c55e', '#3b82f6', '#8b5cf6'];

const RISK_COLOR = {
  CRITICAL: '#dc2626',
  HIGH: '#f97316',
  MEDIUM: '#f59e0b',
  LOW: '#22c55e',
};

const CASE_COLOR = {
  OPEN: '#f59e0b',
  INVESTIGATING: '#6366f1',
  SAR_FILED: '#10b981',
  ESCALATED: '#e11d48',
  DISMISSED: '#94a3b8',
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

  useEffect(() => {
    fetchAPI('/api/dashboard')
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex items-center justify-center h-full text-gray-400">Loading dashboard...</div>;
  if (error) return <div className="flex items-center justify-center h-full text-red-600">Error: {error}</div>;

  const { kpis, daily_data, pattern_breakdown, risk_distribution, case_status_counts, amount_distribution } = data;

  const risk_data = Object.entries(risk_distribution || {}).map(([level, count]) => ({ level, count }));
  const case_data = Object.entries(case_status_counts || {}).map(([status, count]) => ({ status, count }));
  const pattern_data = (pattern_breakdown || []).map(p => ({
    ...p,
    pattern: (p.fraud_pattern || '').replace(/_/g, ' '),
  }));

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Fund Flow Intelligence — Overview</h1>
        <p className="text-sm text-gray-500 mt-1">
          {kpis.total_transactions.toLocaleString('en-IN')} transactions monitored across {Object.keys(risk_distribution || {}).length}
          {' '}risk tiers. Click any card to drill in.
        </p>
      </div>

      {/* Primary KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Total Transactions" value={kpis.total_transactions.toLocaleString('en-IN')} color="indigo" />
        <MetricCard label="Total Volume" value={formatCr(kpis.total_volume)} color="blue" />
        <MetricCard label="Fraud Volume" value={formatCr(kpis.fraud_volume)} delta={`${kpis.fraud_rate}% of total`} color="amber" />
        <MetricCard
          label="Open Cases"
          value={(case_status_counts?.OPEN || 0).toLocaleString('en-IN')}
          delta={`${kpis.critical_alerts} critical • ${kpis.total_alerts} total`}
          color="red"
          onClick={() => navigate('/cases')}
        />
      </div>

      {/* Secondary KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
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
        <MetricCard
          label="Live Stream"
          value="Inactive"
          delta="Click to start streaming"
          color="blue"
          onClick={() => navigate('/live')}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Daily Transaction Trends</h3>
          <ResponsiveContainer width="100%" height={280}>
            <ComposedChart data={daily_data || []}>
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
                cx="50%"
                cy="50%"
                outerRadius={90}
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
          <p className="text-xs text-gray-400 mt-2">
            Left: entity risk tiers • Right: case workflow state. Click <button onClick={() => navigate('/cases')} className="text-indigo-600 hover:underline">open cases</button> to triage.
          </p>
        </div>
      </div>
    </div>
  );
}
