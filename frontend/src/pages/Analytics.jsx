import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, LineChart, Line, Legend,
} from 'recharts';
import { fetchAPI } from '../api';

function formatCr(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

export default function Analytics() {
  const [channels, setChannels] = useState(null);
  const [branches, setBranches] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetchAPI('/api/analytics/channels'),
      fetchAPI('/api/analytics/branches'),
    ])
      .then(([c, b]) => {
        setChannels(c);
        setBranches(b);
      })
      .catch((e) => setError(e?.message || 'Failed to load analytics'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="p-12 text-center text-gray-400">Loading analytics...</div>;
  }

  if (error) {
    return (
      <div className="p-12 max-w-xl">
        <h1 className="text-xl font-bold text-gray-900">Analytics unavailable</h1>
        <p className="text-sm text-red-600 mt-2">Could not load analytics data: {error}</p>
        <p className="text-sm text-gray-500 mt-1">
          Check that the backend is running and the pipeline has been run for this dataset.
        </p>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Channel, Branch &amp; Product Analytics</h1>
        <p className="text-sm text-gray-500 mt-1">
          Where is fraud entering the bank? Slice volume and fraud-rate by the dimensions the RBI FRM framework cares about.
        </p>
      </div>

      {/* Channels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h2 className="text-base font-semibold text-gray-900">Volume by Initiation Channel</h2>
          <p className="text-xs text-gray-500 mt-0.5">How customers initiated each transaction</p>
          <div className="mt-3" style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={channels?.by_channel || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={v => v >= 1e7 ? `${(v / 1e7).toFixed(0)}Cr` : v >= 1e5 ? `${(v / 1e5).toFixed(0)}L` : v} />
                <Tooltip formatter={(v, name) => name === 'fraud_volume' ? [formatCr(v), 'Fraud Volume'] : [formatCr(v), 'Volume']} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="volume" name="Total Volume" fill="#6366f1" radius={[4, 4, 0, 0]} />
                <Bar dataKey="fraud_volume" name="Fraud Volume" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h2 className="text-base font-semibold text-gray-900">Fraud Rate by Channel</h2>
          <p className="text-xs text-gray-500 mt-0.5">Where do anomalies most often originate?</p>
          <div className="mt-3" style={{ height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={channels?.by_channel || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="channel" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={v => [`${v}%`, 'Fraud rate']} />
                <Bar dataKey="fraud_rate" radius={[4, 4, 0, 0]}>
                  {(channels?.by_channel || []).map((c, i) => (
                    <Cell key={i} fill={c.fraud_rate >= 30 ? '#dc2626' : c.fraud_rate >= 15 ? '#f59e0b' : '#22c55e'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Hourly */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h2 className="text-base font-semibold text-gray-900">Transaction Activity by Hour of Day</h2>
        <p className="text-xs text-gray-500 mt-0.5">Fraud spikes between 10pm–6am are a structuring signal</p>
        <div className="mt-3" style={{ height: 260 }}>
          <ResponsiveContainer>
            <LineChart data={(channels?.by_hour || []).map(h => ({ ...h, label: `${h.hour}:00` }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} stroke="#ef4444" />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line yAxisId="left" type="monotone" dataKey="count" name="Transactions" stroke="#6366f1" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="fraud_count" name="Fraud txns" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Branches table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-gray-200">
          <h2 className="text-base font-semibold text-gray-900">Branch Risk Heatmap</h2>
          <p className="text-xs text-gray-500 mt-0.5">All branches with their fraud rate (inbound + outbound combined)</p>
        </div>
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
              <th className="px-5 py-2.5 font-medium">Branch</th>
              <th className="px-3 py-2.5 font-medium text-right">Total Volume</th>
              <th className="px-3 py-2.5 font-medium text-right">Fraud Volume</th>
              <th className="px-3 py-2.5 font-medium text-right">Fraud Rate</th>
              <th className="px-3 py-2.5 font-medium text-right">Fraud Count</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {(branches?.branches || []).map(b => (
              <tr key={b.branch} className="hover:bg-gray-50">
                <td className="px-5 py-2 font-medium text-gray-900">{b.branch || '(unknown)'}</td>
                <td className="px-3 py-2 text-right tabular-nums">{formatCr(b.total_volume)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-rose-700">{formatCr(b.total_fraud_volume || 0)}</td>
                <td className="px-3 py-2 text-right">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-semibold ${
                    b.fraud_rate >= 30 ? 'bg-red-100 text-red-800' :
                    b.fraud_rate >= 15 ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'
                  }`}>
                    {b.fraud_rate}%
                  </span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{Math.round(b.total_fraud_count || 0).toLocaleString('en-IN')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
