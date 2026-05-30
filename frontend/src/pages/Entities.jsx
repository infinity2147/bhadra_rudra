import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { fetchAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';

const FLAG_TONES = {
  shell_company:           'bg-red-100 text-red-800 ring-red-200',
  part_of_cycle:           'bg-red-100 text-red-800 ring-red-200',
  transit_node:            'bg-red-100 text-red-800 ring-red-200',
  outflow_zscore_anomaly:  'bg-red-100 text-red-800 ring-red-200',
  high_risk:               'bg-amber-100 text-amber-800 ring-amber-200',
  velocity_burst:          'bg-amber-100 text-amber-800 ring-amber-200',
  dormant_then_active:     'bg-amber-100 text-amber-800 ring-amber-200',
  multi_branch_activity:   'bg-indigo-100 text-indigo-800 ring-indigo-200',
};

function FlagPill({ flag }) {
  const tone = FLAG_TONES[flag] || 'bg-gray-100 text-gray-700 ring-gray-200';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium ring-1 ${tone}`}>
      {flag.replace(/_/g, ' ')}
    </span>
  );
}

function formatCurrency(value) {
  if (value == null) return '--';
  return '₹' + Number(value).toLocaleString('en-IN');
}

function RiskScoreBar({ score }) {
  // score from API is 0-1; normalise to a 0-100 width.
  const value = Number(score ?? 0);
  const pct = value <= 1 ? Math.round(value * 100) : Math.round(value);
  const color = pct >= 70 ? 'bg-red-500' : pct >= 50 ? 'bg-orange-500' : pct >= 30 ? 'bg-yellow-500' : 'bg-green-500';
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-medium text-gray-600 tabular-nums">{pct}</span>
    </div>
  );
}

export default function Entities() {
  const navigate = useNavigate();
  const [entities, setEntities] = useState([]);
  const [search, setSearch] = useState('');
  const [riskLevel, setRiskLevel] = useState('ALL');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const debounceRef = useRef(null);

  useEffect(() => {
    setLoading(true);
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (riskLevel !== 'ALL') params.set('risk_level', riskLevel);

    fetchAPI(`/api/entities?${params.toString()}`)
      .then(data => {
        setEntities(Array.isArray(data) ? data : data.entities ?? []);
      })
      .catch(() => setEntities([]))
      .finally(() => setLoading(false));
  }, [search, riskLevel]);

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    setDetailLoading(true);
    fetchAPI(`/api/entities/${selected}`)
      .then(data => setDetail(data))
      .catch(() => setDetail(null))
      .finally(() => setDetailLoading(false));
  }, [selected]);

  function handleSearchChange(e) {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setSearch(e.target.value), 300);
  }

  const transactions = detail?.transactionHistory ?? detail?.transactions ?? [];
  const chartData = detail
    ? [
        { name: 'Sent', value: detail.sentVolume ?? detail.total_sent ?? 0, fill: '#ef4444' },
        { name: 'Received', value: detail.receivedVolume ?? detail.total_received ?? 0, fill: '#22c55e' },
        { name: 'Net Flow', value: detail.netFlow ?? detail.net_flow ?? 0, fill: '#6366f1' },
      ]
    : [];

  return (
    <div className="h-full flex flex-col">
      <div className="p-6 pb-0">
        <h1 className="text-2xl font-bold text-gray-900">Entity Risk Explorer</h1>
        <p className="text-sm text-gray-500 mt-1">Search and analyze entity risk profiles and transaction histories</p>
      </div>

      <div className="px-6 pt-4 flex gap-3 flex-wrap">
        <input
          type="text"
          placeholder="Search entities by name..."
          onChange={handleSearchChange}
          className="flex-1 min-w-[200px] px-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
        />
        <select
          value={riskLevel}
          onChange={e => setRiskLevel(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          <option value="ALL">All Risk Levels</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>
        <select
          value={typeFilter}
          onChange={e => setTypeFilter(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          <option value="ALL">All Types</option>
          <option value="individual">Individual</option>
          <option value="business">Business</option>
          <option value="shell_company">Shell Company</option>
        </select>
      </div>

      <div className="flex-1 flex gap-4 p-6 min-h-0">
        {/* Entity Table */}
        <div className={`${selected ? 'w-1/2' : 'w-full'} flex flex-col min-h-0 transition-all duration-200`}>
          <div className="flex-1 overflow-auto border border-gray-200 rounded-xl bg-white">
            {loading ? (
              <div className="flex items-center justify-center h-40 text-gray-400 text-sm">Loading entities...</div>
            ) : entities.length === 0 ? (
              <div className="flex items-center justify-center h-40 text-gray-400 text-sm">No entities found</div>
            ) : (
              <table className="w-full text-sm">
                <thead className="bg-gray-50 sticky top-0">
                  <tr>
                    <th className="text-left px-4 py-3 font-medium text-gray-600">Name</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-600">Type</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-600">Risk Score</th>
                    <th className="text-left px-4 py-3 font-medium text-gray-600">Risk Level</th>
                  </tr>
                </thead>
                <tbody>
                  {entities
                    .filter((entity) => typeFilter === 'ALL' || entity.type === typeFilter)
                    .map(entity => (
                    <tr
                      key={entity.entity_id ?? entity.id}
                      onClick={() => setSelected(entity.entity_id ?? entity.id)}
                      className={`border-t border-gray-100 cursor-pointer transition-colors ${
                        selected === (entity.entity_id ?? entity.id)
                          ? 'bg-indigo-50'
                          : 'hover:bg-gray-50'
                      }`}
                    >
                      <td className="px-4 py-3 font-medium">{entity.name}</td>
                      <td className="px-4 py-3 text-gray-600 capitalize">{(entity.type || '').replace('_', ' ')}</td>
                      <td className="px-4 py-3"><RiskScoreBar score={entity.risk_score ?? 0} /></td>
                      <td className="px-4 py-3"><SeverityBadge severity={entity.risk_level} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Detail Panel */}
        {selected && (
          <div className="w-1/2 flex flex-col min-h-0 border border-gray-200 rounded-xl bg-white overflow-hidden">
            {detailLoading ? (
              <div className="flex items-center justify-center h-40 text-gray-400 text-sm">Loading entity details...</div>
            ) : !detail ? (
              <div className="flex items-center justify-center h-40 text-gray-400 text-sm">Entity not found</div>
            ) : (
              <div className="flex-1 overflow-y-auto p-5 space-y-5">
                {/* Header */}
                <div className="flex items-start justify-between">
                  <div className="min-w-0">
                    <p className="text-[10px] font-mono text-gray-400">{detail.id}</p>
                    <h2 className="text-lg font-bold text-gray-900 truncate">{detail.name}</h2>
                    <p className="text-sm text-gray-500 capitalize">{(detail.type || '').replace('_', ' ')}{detail.branch ? ` — ${detail.branch}` : ''}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <SeverityBadge severity={detail.riskLevel || detail.risk_level} />
                    <button
                      onClick={() => setSelected(null)}
                      className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600"
                    >
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                    </button>
                  </div>
                </div>

                {/* Flags */}
                {detail.flags && detail.flags.length > 0 && (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {detail.flags.map((f) => <FlagPill key={f} flag={f} />)}
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() => navigate(`/journey?entity=${detail.id}`)}
                    className="flex-1 px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                  >
                    Trace fund journey
                  </button>
                  <button
                    onClick={() => navigate(`/graph`)}
                    className="px-3 py-2 text-sm bg-white border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                  >
                    View in graph
                  </button>
                </div>

                {/* Metric Cards */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl p-4 bg-indigo-50 text-indigo-700">
                    <p className="text-xs font-medium uppercase tracking-wide opacity-70">Risk Score</p>
                    <p className="text-2xl font-bold mt-1">{detail.riskScore != null ? Number(detail.riskScore).toFixed(3) : '--'}</p>
                  </div>
                  <div className="rounded-xl p-4 bg-blue-50 text-blue-700">
                    <p className="text-xs font-medium uppercase tracking-wide opacity-70">Total Transactions</p>
                    <p className="text-2xl font-bold mt-1">{detail.totalTransactions ?? '--'}</p>
                  </div>
                  <div className="rounded-xl p-4 bg-red-50 text-red-700">
                    <p className="text-xs font-medium uppercase tracking-wide opacity-70">Fraud Transactions</p>
                    <p className="text-2xl font-bold mt-1">{detail.fraudTransactions ?? '--'}</p>
                  </div>
                  <div className="rounded-xl p-4 bg-green-50 text-green-700">
                    <p className="text-xs font-medium uppercase tracking-wide opacity-70">Net Flow</p>
                    <p className="text-2xl font-bold mt-1">{formatCurrency(detail.netFlow ?? detail.net_flow)}</p>
                  </div>
                </div>

                {/* Flow Chart */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Flow Overview</h3>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 11 }} tickFormatter={v => v >= 100000 ? `${(v / 100000).toFixed(0)}L` : v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v} />
                        <Tooltip formatter={val => formatCurrency(val)} />
                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                          {chartData.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Transaction History */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Transaction History</h3>
                  <div className="overflow-auto border border-gray-200 rounded-lg max-h-72">
                    <table className="w-full text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          <th className="text-left px-3 py-2 font-medium text-gray-600">Timestamp</th>
                          <th className="text-left px-3 py-2 font-medium text-gray-600">Sender</th>
                          <th className="text-left px-3 py-2 font-medium text-gray-600">Receiver</th>
                          <th className="text-right px-3 py-2 font-medium text-gray-600">Amount</th>
                          <th className="text-left px-3 py-2 font-medium text-gray-600">Type</th>
                          <th className="text-center px-3 py-2 font-medium text-gray-600">Fraud</th>
                        </tr>
                      </thead>
                      <tbody>
                        {transactions.length === 0 ? (
                          <tr><td colSpan={6} className="text-center py-6 text-gray-400">No transactions</td></tr>
                        ) : transactions.map((tx, i) => (
                          <tr key={i} className={`border-t border-gray-100 ${tx.is_fraud ? 'bg-red-50/50' : ''}`}>
                            <td className="px-3 py-2 text-gray-500 whitespace-nowrap">{tx.timestamp ? new Date(tx.timestamp).toLocaleString('en-IN') : '--'}</td>
                            <td className="px-3 py-2">{tx.sender_name ?? tx.sender ?? '--'}</td>
                            <td className="px-3 py-2">{tx.receiver_name ?? tx.receiver ?? '--'}</td>
                            <td className="px-3 py-2 text-right font-medium">{formatCurrency(tx.amount)}</td>
                            <td className="px-3 py-2 text-gray-600 capitalize">{tx.transaction_type ?? tx.type ?? '--'}</td>
                            <td className="px-3 py-2 text-center">
                              {tx.is_fraud
                                ? <span className="text-red-600 font-semibold">Yes</span>
                                : <span className="text-gray-400">No</span>
                              }
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
