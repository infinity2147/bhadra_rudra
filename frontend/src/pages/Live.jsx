import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';

function formatINR(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

const PATTERN_TONE = {
  circular_transaction: 'bg-rose-100 text-rose-700',
  rapid_layering: 'bg-amber-100 text-amber-800',
  shell_funnel: 'bg-red-100 text-red-700',
  smurfing: 'bg-purple-100 text-purple-700',
  dormant_activation: 'bg-indigo-100 text-indigo-700',
  none: 'bg-gray-100 text-gray-500',
};

export default function Live() {
  const [running, setRunning] = useState(false);
  const [feed, setFeed] = useState([]);
  const [stats, setStats] = useState({ total: 0, fraud: 0, volume: 0, fraud_volume: 0 });
  const [tps, setTps] = useState(2); // transactions per second
  const intervalRef = useRef(null);
  const seqRef = useRef(0);

  const pull = useCallback(async () => {
    try {
      const data = await fetchAPI(`/api/live/inject?count=${tps}`);
      const stamped = (data.transactions || []).map(t => ({
        ...t,
        seq: ++seqRef.current,
      }));
      setFeed(prev => {
        const next = [...stamped.reverse(), ...prev].slice(0, 200);
        return next;
      });
      setStats(prev => ({
        total: prev.total + stamped.length,
        fraud: prev.fraud + stamped.filter(t => t.isFraud).length,
        volume: prev.volume + stamped.reduce((s, t) => s + (t.amount || 0), 0),
        fraud_volume: prev.fraud_volume + stamped.filter(t => t.isFraud).reduce((s, t) => s + (t.amount || 0), 0),
      }));
    } catch (e) {
      // swallow; user sees nothing happens
    }
  }, [tps]);

  useEffect(() => {
    if (!running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      return;
    }
    pull();
    intervalRef.current = setInterval(pull, 1000);
    return () => intervalRef.current && clearInterval(intervalRef.current);
  }, [running, pull]);

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Transaction Stream</h1>
          <p className="text-sm text-gray-500 mt-1">
            Simulated per-transaction ingestion with on-the-fly ML scoring and pattern classification.
            Each transaction passes through the same detectors used by the batch pipeline.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-700">
            TPS
            <input
              type="number"
              min={1}
              max={10}
              value={tps}
              onChange={e => setTps(Math.max(1, Math.min(10, Number(e.target.value))))}
              className="ml-2 w-16 px-2 py-1 border border-gray-300 rounded-md text-sm"
            />
          </label>
          <button
            onClick={() => setRunning(r => !r)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              running
                ? 'bg-red-600 text-white hover:bg-red-700'
                : 'bg-emerald-600 text-white hover:bg-emerald-700'
            }`}
          >
            {running ? '■ Stop Stream' : '▶ Start Stream'}
          </button>
          <button
            onClick={() => { setFeed([]); setStats({ total: 0, fraud: 0, volume: 0, fraud_volume: 0 }); seqRef.current = 0; }}
            className="px-3 py-2 text-sm border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
          >
            Clear
          </button>
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-4 gap-4">
        <div className="rounded-xl bg-indigo-50 text-indigo-700 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Streamed Txns</p>
          <p className="text-2xl font-bold mt-1">{stats.total.toLocaleString('en-IN')}</p>
        </div>
        <div className="rounded-xl bg-red-50 text-red-700 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Detected Fraud</p>
          <p className="text-2xl font-bold mt-1">{stats.fraud.toLocaleString('en-IN')}</p>
        </div>
        <div className="rounded-xl bg-blue-50 text-blue-700 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Total Volume</p>
          <p className="text-2xl font-bold mt-1">{formatINR(stats.volume)}</p>
        </div>
        <div className="rounded-xl bg-amber-50 text-amber-800 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Flagged Volume</p>
          <p className="text-2xl font-bold mt-1">{formatINR(stats.fraud_volume)}</p>
        </div>
      </div>

      {/* Feed */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${running ? 'bg-emerald-500 animate-pulse' : 'bg-gray-300'}`} />
            <span className="text-sm font-medium text-gray-700">
              {running ? 'Streaming live' : 'Stream paused'}
            </span>
          </div>
          <span className="text-xs text-gray-500 font-mono">last {feed.length} events</span>
        </div>
        <div className="max-h-[60vh] overflow-y-auto">
          {feed.length === 0 ? (
            <div className="p-12 text-center text-gray-400">
              No transactions yet. Press <span className="font-semibold">Start Stream</span> to begin.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="bg-gray-50 sticky top-0">
                <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
                  <th className="px-5 py-2 font-medium">Seq</th>
                  <th className="px-3 py-2 font-medium">Time</th>
                  <th className="px-3 py-2 font-medium">Sender</th>
                  <th className="px-3 py-2 font-medium">Receiver</th>
                  <th className="px-3 py-2 font-medium text-right">Amount</th>
                  <th className="px-3 py-2 font-medium">Rail</th>
                  <th className="px-3 py-2 font-medium">Channel</th>
                  <th className="px-3 py-2 font-medium">ML</th>
                  <th className="px-3 py-2 font-medium">Pattern</th>
                  <th className="px-3 py-2 font-medium">Severity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {feed.map(t => (
                  <tr key={t.seq} className={t.isFraud ? 'bg-red-50/40' : ''}>
                    <td className="px-5 py-1.5 text-xs font-mono text-gray-400">{t.seq}</td>
                    <td className="px-3 py-1.5 text-xs font-mono text-gray-500 whitespace-nowrap">{(t.timestamp || '').slice(11, 19)}</td>
                    <td className="px-3 py-1.5">{t.sender}</td>
                    <td className="px-3 py-1.5">{t.receiver}</td>
                    <td className="px-3 py-1.5 text-right font-medium tabular-nums">{formatINR(t.amount)}</td>
                    <td className="px-3 py-1.5 text-gray-600">{t.transaction_type}</td>
                    <td className="px-3 py-1.5 text-gray-600">{t.channel}</td>
                    <td className="px-3 py-1.5">
                      {t.mlScore != null ? (
                        <span className={`text-xs font-medium ${
                          t.mlScore >= 0.7 ? 'text-red-700' : t.mlScore >= 0.4 ? 'text-amber-700' : 'text-emerald-700'
                        }`}>
                          {(t.mlScore * 100).toFixed(0)}
                        </span>
                      ) : <span className="text-xs text-gray-300">—</span>}
                    </td>
                    <td className="px-3 py-1.5">
                      {t.pattern && t.pattern !== 'none' ? (
                        <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium ${PATTERN_TONE[t.pattern] || PATTERN_TONE.none}`}>
                          {t.pattern.replace(/_/g, ' ')}
                        </span>
                      ) : <span className="text-xs text-gray-400">normal</span>}
                    </td>
                    <td className="px-3 py-1.5">
                      {t.severity ? <SeverityBadge severity={t.severity} /> : <span className="text-xs text-gray-400">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
