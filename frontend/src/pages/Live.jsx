import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { fetchAPI, postAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';

function formatINR(n) {
  if (n == null) return '--';
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(2)} L`;
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

// HH:MM:SS for the monotonic ingest time. received_at is tz-aware UTC, so
// new Date() parses it correctly and renders the operator's local time.
function ingestTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return isNaN(d) ? '' : d.toLocaleTimeString('en-IN', { hour12: false });
}

// Honest signal flags emitted by the live feature extractor (src/live_scoring.py),
// surfaced verbatim by the ingestor. Each maps to a real computed feature.
// ML% text colour follows the τ-aligned severity (set by the backend), not a
// separate hardcoded cutoff — keeps the flag threshold uniform end-to-end.
const SEV_TEXT = { CRITICAL: 'text-red-700', HIGH: 'text-orange-700', MEDIUM: 'text-amber-700' };

const SIGNAL_TONE = {
  cycle: 'bg-rose-100 text-rose-700',
  'near-threshold': 'bg-amber-100 text-amber-800',
  'shell-sender': 'bg-red-100 text-red-700',
  'shell-receiver': 'bg-red-100 text-red-700',
  'high-value-rail': 'bg-indigo-100 text-indigo-700',
  'off-hours': 'bg-purple-100 text-purple-700',
};

// Reshape one /api/stream/recent event onto the flat row the table renders.
function mapEvent(e) {
  return {
    seq: e.seq,
    txnId: e.txn?.transaction_id,
    timestamp: e.txn?.timestamp,        // original transaction time (source data)
    receivedAt: e.received_at,          // when it was ingested/scored (monotonic)
    sender: e.txn?.sender_id,
    receiver: e.txn?.receiver_id,
    amount: e.txn?.amount,
    rail: e.txn?.transaction_type,
    channel: e.txn?.channel,
    mlScore: e.ml_score,
    severity: e.severity,
    signals: e.signals || [],
    latency: e.latency_ms?.total,
    isFraud: !!e.severity,           // real: severity is assigned by the ML scorer
    error: e.error,
  };
}

export default function Live() {
  const [running, setRunning] = useState(false);
  const [feed, setFeed] = useState([]);
  const [stats, setStats] = useState({ total: 0, fraud: 0, volume: 0, fraud_volume: 0, latency_samples: [] });
  const [tps, setTps] = useState(3);          // replay rate (events/sec)
  const [mode, setMode] = useState(null);     // "kafka" | "inproc"
  const [error, setError] = useState(null);

  const runningRef = useRef(false);
  const tpsRef = useRef(tps);
  const pollRef = useRef(null);
  const seenSeqRef = useRef(new Set());       // seqs already counted into cumulative stats
  const genRef = useRef(0);                   // bumped on reset; lets an in-flight poll drop stale data

  useEffect(() => { tpsRef.current = tps; }, [tps]);

  // Poll the ring buffer; mirror the newest window, accumulate cumulative stats
  // over genuinely-new seqs so totals keep climbing past the 500-event buffer.
  const poll = useCallback(async () => {
    const gen = genRef.current;
    try {
      const data = await fetchAPI('/api/stream/recent?limit=200');
      if (gen !== genRef.current) return;    // a reset happened mid-fetch — drop this stale batch
      const events = (data.events || []).map(mapEvent).filter(e => e.seq != null);
      setFeed(events.slice(0, 200));         // recent() already returns newest-first

      // Dedup over genuinely-new seqs HERE (poll body runs once per tick), not
      // inside the setStats updater — StrictMode invokes updaters twice, so a
      // ref mutation in the updater would double-add then zero out the counts.
      const fresh = [];
      for (const e of [...events].reverse()) {  // oldest-first for natural seq order
        if (seenSeqRef.current.has(e.seq)) continue;
        seenSeqRef.current.add(e.seq);
        fresh.push(e);
      }
      if (fresh.length === 0) return;
      setStats(prev => {                        // pure: reads only prev + fresh
        let { total, fraud, volume, fraud_volume } = prev;
        const lat = [...(prev.latency_samples || [])];
        for (const e of fresh) {
          total += 1;
          volume += e.amount || 0;
          if (e.isFraud) { fraud += 1; fraud_volume += e.amount || 0; }
          if (typeof e.latency === 'number') lat.push(e.latency);
        }
        return { total, fraud, volume, fraud_volume, latency_samples: lat.slice(-500) };
      });
    } catch (err) {
      setError(err.message);
    }
  }, []);

  // Drive traffic by replaying the loaded dataset onto the bus in short chunks.
  // Chunked (≈3s each) so toggling Stop is responsive. The backend consumer
  // scores each event through the same model the batch pipeline uses.
  const replayLoop = useCallback(async () => {
    while (runningRef.current) {
      try {
        await postAPI('/api/stream/replay', {
          rate: Math.max(1, tpsRef.current),
          total: Math.max(1, tpsRef.current) * 3,
          shuffle: true,
        });
      } catch (err) {
        setError(`Replay stopped: ${err.message}`);
        setRunning(false);
        break;
      }
    }
  }, []);

  useEffect(() => {
    runningRef.current = running;
    if (!running) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    let cancelled = false;
    (async () => {
      setError(null);
      try {
        const s = await postAPI('/api/stream/start', {});   // idempotent — ensures consumer is up
        setMode(s.mode);
      } catch (err) {
        setError(`Could not start stream: ${err.message}`);
        setRunning(false);
        return;
      }
      if (cancelled) return;
      poll();
      pollRef.current = setInterval(poll, 1000);
      replayLoop();   // fire-and-forget; self-terminates when runningRef flips false
    })();
    return () => {
      cancelled = true;
      runningRef.current = false;             // also stops replayLoop on unmount
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [running, poll, replayLoop]);

  // Burst detection — entities appearing ≥3 times in the current feed window
  const bursts = useMemo(() => {
    const counts = {};
    for (const t of feed) {
      counts[t.sender] = (counts[t.sender] || 0) + 1;
      counts[t.receiver] = (counts[t.receiver] || 0) + 1;
    }
    return Object.entries(counts)
      .filter(([, n]) => n >= 3)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);
  }, [feed]);

  // Hard reset: stop the stream, wipe the backend ring buffer + counters (so seq
  // restarts at 0), and clear the local view. The gen bump invalidates any poll
  // still in flight so it can't repopulate what we just cleared.
  async function resetStream() {
    genRef.current += 1;
    setRunning(false);
    setFeed([]);
    setStats({ total: 0, fraud: 0, volume: 0, fraud_volume: 0, latency_samples: [] });
    seenSeqRef.current = new Set();
    setMode(null);
    try {
      await postAPI('/api/stream/reset', {});
      setError(null);
    } catch (err) {
      setError(`Reset failed: ${err.message}`);
    }
  }

  return (
    <div className="p-6 space-y-6 max-w-[1600px] mx-auto">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Live Transaction Stream</h1>
          <p className="text-sm text-gray-500 mt-1">
            Real stream: transactions from the loaded dataset are replayed onto the ingest bus
            {mode ? <> (<span className="font-mono">{mode}</span> backend)</> : null} and scored
            per-transaction through the same ML model the batch pipeline uses. No synthetic data.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-sm text-gray-700">
            Rate (TPS)
            <input
              type="number"
              min={1}
              max={20}
              value={tps}
              onChange={e => setTps(Math.max(1, Math.min(20, Number(e.target.value))))}
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
            onClick={resetStream}
            className="px-3 py-2 text-sm border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            title="Stop the stream and wipe the buffer + counters (seq restarts at 0)"
          >
            ⟲ Reset
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {/* Burst alert */}
      {bursts.length > 0 && (
        <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <p className="text-sm font-semibold text-amber-900">
              Velocity burst — {bursts.length} {bursts.length === 1 ? 'entity' : 'entities'} active in the current window
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {bursts.map(([entityId, count]) => (
              <span key={entityId} className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-white border border-amber-200 text-xs">
                <span className="font-mono text-gray-600">{entityId}</span>
                <span className="font-bold text-amber-700">{count}×</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-5 gap-4">
        <div className="rounded-xl bg-indigo-50 text-indigo-700 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Streamed Txns</p>
          <p className="text-2xl font-bold mt-1">{stats.total.toLocaleString('en-IN')}</p>
        </div>
        <div className="rounded-xl bg-red-50 text-red-700 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Flagged (sev ≥ MEDIUM)</p>
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
        <div className="rounded-xl bg-emerald-50 text-emerald-700 p-4">
          <p className="text-xs font-medium opacity-70 uppercase tracking-wide">Per-Txn Latency</p>
          {(() => {
            const lats = stats.latency_samples || [];
            if (lats.length === 0) return <p className="text-xl font-bold mt-1">— ms</p>;
            const sorted = [...lats].sort((a, b) => a - b);
            const mean = lats.reduce((s, v) => s + v, 0) / lats.length;
            const p95 = sorted[Math.floor(0.95 * sorted.length)];
            return (
              <>
                <p className="text-2xl font-bold mt-1">{mean.toFixed(2)} ms</p>
                <p className="text-[11px] opacity-70 mt-0.5">p95 {p95?.toFixed(2)} ms • n={lats.length}</p>
              </>
            );
          })()}
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
                  <th className="px-3 py-2 font-medium">Ingest Time</th>
                  <th className="px-3 py-2 font-medium">Sender</th>
                  <th className="px-3 py-2 font-medium">Receiver</th>
                  <th className="px-3 py-2 font-medium text-right">Amount</th>
                  <th className="px-3 py-2 font-medium">Rail</th>
                  <th className="px-3 py-2 font-medium">Channel</th>
                  <th className="px-3 py-2 font-medium">ML</th>
                  <th className="px-3 py-2 font-medium">Signals</th>
                  <th className="px-3 py-2 font-medium">Severity</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {feed.map(t => (
                  <tr key={t.seq} className={t.isFraud ? 'bg-red-50/40' : ''}>
                    <td className="px-5 py-1.5 text-xs font-mono text-gray-400">{t.seq}</td>
                    <td className="px-3 py-1.5 font-mono whitespace-nowrap">
                      <div className="text-xs text-gray-600">{ingestTime(t.receivedAt)}</div>
                      <div className="text-[10px] text-gray-400">txn {(t.timestamp || '').slice(11, 19)}</div>
                    </td>
                    <td className="px-3 py-1.5 font-mono text-xs">{t.sender}</td>
                    <td className="px-3 py-1.5 font-mono text-xs">{t.receiver}</td>
                    <td className="px-3 py-1.5 text-right font-medium tabular-nums">{formatINR(t.amount)}</td>
                    <td className="px-3 py-1.5 text-gray-600">{t.rail}</td>
                    <td className="px-3 py-1.5 text-gray-600">{t.channel}</td>
                    <td className="px-3 py-1.5">
                      {t.mlScore != null ? (
                        <span className={`text-xs font-medium ${SEV_TEXT[t.severity] || 'text-emerald-700'}`}>
                          {(t.mlScore * 100).toFixed(0)}
                        </span>
                      ) : <span className="text-xs text-gray-300">—</span>}
                    </td>
                    <td className="px-3 py-1.5">
                      {t.signals.length > 0 ? (
                        <span className="flex flex-wrap gap-1">
                          {t.signals.map(sig => (
                            <span key={sig} className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-medium ${SIGNAL_TONE[sig] || 'bg-gray-100 text-gray-500'}`}>
                              {sig.replace(/-/g, ' ')}
                            </span>
                          ))}
                        </span>
                      ) : <span className="text-xs text-gray-400">—</span>}
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
