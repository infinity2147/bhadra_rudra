import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchAPI, postAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';
import EnsembleConsensus from '../components/EnsembleConsensus';

const CHANNELS = ['NetBanking', 'UPI', 'IMPS', 'ATM', 'MobileBanking'];
const RAILS = ['NEFT', 'RTGS', 'IMPS', 'UPI', 'Wire Transfer'];

// τ-aligned colour for the headline fraud probability. Null severity means the
// score fell below the watchlist threshold, so render it neutral.
const SEV_PROB_TONE = {
  CRITICAL: 'text-red-600',
  HIGH: 'text-orange-600',
  MEDIUM: 'text-yellow-600',
  LOW: 'text-green-600',
};

function inr(x) {
  if (x == null || x === '') return '0';
  return Number(x).toLocaleString('en-IN');
}

function Chip({ children }) {
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 text-[11px] font-medium">
      {children}
    </span>
  );
}

// One SHAP feature row: a bar centered at zero. Positive contribution pushes the
// fraud score up (right, red); negative pulls it down (left, blue). Width is
// scaled against the largest |shap| in the set so the strongest driver fills the half.
function ShapRow({ feature, shap, maxAbs }) {
  const v = Number(shap) || 0;
  const frac = maxAbs > 0 ? Math.min(1, Math.abs(v) / maxAbs) : 0;
  const widthPct = (frac * 50).toFixed(1);
  const positive = v >= 0;
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-40 truncate text-gray-700" title={feature}>{feature}</span>
      <div className="relative flex-1 h-3 bg-gray-50 rounded">
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-gray-300" />
        <div
          className={`absolute top-0 bottom-0 ${positive ? 'bg-red-500 left-1/2' : 'bg-blue-500 right-1/2'} rounded`}
          style={{ width: `${widthPct}%` }}
        />
      </div>
      <span className={`w-14 text-right tabular-nums font-medium ${positive ? 'text-red-600' : 'text-blue-600'}`}>
        {v >= 0 ? '+' : ''}{v.toFixed(3)}
      </span>
    </div>
  );
}

export default function SimulationStudio() {
  const [form, setForm] = useState({
    sender: 'SIM_SENDER',
    receiver: 'SIM_RECEIVER',
    amount: 1900000,
    channel: 'NetBanking',
    rail: 'NEFT',
  });
  const [result, setResult] = useState(null);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState(null);

  const [scenarios, setScenarios] = useState([]);
  const [injecting, setInjecting] = useState(null); // scenario name currently being injected
  const [toast, setToast] = useState(null);

  const [feed, setFeed] = useState([]);
  const [feedError, setFeedError] = useState(null);

  const toastTimer = useRef(null);

  function setField(key, value) {
    setForm(f => ({ ...f, [key]: value }));
  }

  function showToast(msg) {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }
  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current); }, []);

  // --- Live feed polling --------------------------------------------------
  // /api/stream/recent may return a bare array or {events:[...]} depending on
  // backend version; normalize to an array either way.
  const refreshFeed = useCallback(async () => {
    try {
      const data = await fetchAPI('/api/stream/recent?limit=25');
      const events = Array.isArray(data) ? data : (data?.events || []);
      setFeed(events.filter(e => e && e.seq != null));
      setFeedError(null);
    } catch (err) {
      setFeedError(err.message);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => { if (!cancelled) await refreshFeed(); })();
    const id = setInterval(refreshFeed, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, [refreshFeed]);

  // --- Scenarios ----------------------------------------------------------
  useEffect(() => {
    fetchAPI('/api/simulate/scenarios')
      .then(d => setScenarios(d?.scenarios || []))
      .catch(() => setScenarios([]));
  }, []);

  // --- Actions ------------------------------------------------------------
  async function score() {
    setScoring(true);
    setError(null);
    try {
      const res = await postAPI('/api/simulate/score', {
        sender: form.sender,
        receiver: form.receiver,
        amount: Number(form.amount),
        channel: form.channel,
        rail: form.rail,
      });
      setResult(res);
      // A scored sim txn is published to the stream — surface it in the feed right away.
      refreshFeed();
    } catch (err) {
      setError(err.message);
    } finally {
      setScoring(false);
    }
  }

  async function injectScenario(s) {
    setInjecting(s.name);
    setError(null);
    try {
      const res = await postAPI('/api/simulate/scenario/' + s.name);
      const n = res?.injected ?? res?.n_txns ?? s.n_txns ?? 0;
      showToast(`Injected ${s.label} (${n} txns)`);
      await refreshFeed();
    } catch (err) {
      setError(err.message);
    } finally {
      setInjecting(null);
    }
  }

  // --- Derived ------------------------------------------------------------
  const probTone = result
    ? (result.severity ? (SEV_PROB_TONE[result.severity] || 'text-gray-700') : 'text-gray-400')
    : 'text-gray-700';

  const shapTop = result?.shap
    ? [...result.shap]
        .filter(s => s && typeof s.shap === 'number')
        .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
        .slice(0, 8)
    : [];
  const shapMaxAbs = shapTop.reduce((m, s) => Math.max(m, Math.abs(s.shap)), 0);

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Simulation Studio</h1>
        <p className="text-sm text-gray-500">
          Build a transaction, score it through the live ML pipeline, and watch it stream — or inject a whole
          fraud scenario and see the detectors light up.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      {toast && (
        <div className="rounded-lg border border-emerald-300 bg-emerald-50 px-4 py-2 text-sm text-emerald-800">
          {toast}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: builder + scenarios */}
        <div className="space-y-6 lg:col-span-1">
          {/* Builder */}
          <div className="bg-white border border-gray-200 rounded-xl p-5">
            <h2 className="text-base font-semibold text-gray-900">Build a transaction</h2>
            <p className="text-xs text-gray-500 mt-0.5">Scored through the same model as the batch + stream paths</p>

            <div className="mt-4 space-y-3">
              <label className="block">
                <span className="text-xs font-medium text-gray-600">Sender</span>
                <input
                  type="text"
                  value={form.sender}
                  onChange={e => setField('sender', e.target.value)}
                  className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-gray-600">Receiver</span>
                <input
                  type="text"
                  value={form.receiver}
                  onChange={e => setField('receiver', e.target.value)}
                  className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md text-sm font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-gray-600">Amount (₹)</span>
                <input
                  type="number"
                  min={0}
                  value={form.amount}
                  onChange={e => setField('amount', e.target.value)}
                  className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md text-sm tabular-nums focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <span className="text-[11px] text-gray-400">₹{inr(form.amount)}</span>
              </label>
              <div className="grid grid-cols-2 gap-3">
                <label className="block">
                  <span className="text-xs font-medium text-gray-600">Channel</span>
                  <select
                    value={form.channel}
                    onChange={e => setField('channel', e.target.value)}
                    className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    {CHANNELS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="text-xs font-medium text-gray-600">Rail</span>
                  <select
                    value={form.rail}
                    onChange={e => setField('rail', e.target.value)}
                    className="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    {RAILS.map(r => <option key={r} value={r}>{r}</option>)}
                  </select>
                </label>
              </div>

              <button
                onClick={score}
                disabled={scoring}
                className="w-full mt-1 px-4 py-2.5 rounded-lg text-sm font-medium bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-60 transition-colors"
              >
                {scoring ? 'Scoring…' : 'Score & Stream'}
              </button>
            </div>
          </div>

          {/* Scenarios */}
          <div className="bg-white border border-gray-200 rounded-xl p-5">
            <h2 className="text-base font-semibold text-gray-900">Inject a scenario</h2>
            <p className="text-xs text-gray-500 mt-0.5">Replay a canned fraud pattern onto the stream</p>

            <div className="mt-4 space-y-2">
              {scenarios.length === 0 ? (
                <p className="text-xs text-gray-400 italic">No scenarios available</p>
              ) : (
                scenarios.map(s => (
                  <button
                    key={s.name}
                    onClick={() => injectScenario(s)}
                    disabled={injecting != null}
                    className="w-full text-left rounded-lg border border-gray-200 px-3 py-2.5 hover:border-indigo-300 hover:bg-indigo-50/40 disabled:opacity-60 transition-colors"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-bold text-gray-900">{s.label}</span>
                      <span className="shrink-0 inline-flex items-center px-2 py-0.5 rounded-full bg-gray-100 text-gray-600 text-[11px] font-medium">
                        {s.n_txns} txns
                      </span>
                    </div>
                    {s.description && <p className="text-xs text-gray-500 mt-0.5">{s.description}</p>}
                    {injecting === s.name && <p className="text-[11px] text-indigo-600 mt-1">Injecting…</p>}
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: result + feed */}
        <div className="space-y-6 lg:col-span-2">
          {/* Result */}
          {result ? (
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wide text-gray-500">Fraud probability</p>
                  <div className="flex items-baseline gap-3 mt-1">
                    <span className={`text-5xl font-bold tabular-nums ${probTone}`}>
                      {(Number(result.ml_score) * 100).toFixed(1)}%
                    </span>
                    {result.severity ? (
                      <SeverityBadge severity={result.severity} />
                    ) : (
                      <span className="text-sm text-gray-400">below watchlist</span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    scored in {result.latency_ms?.total ?? '—'} ms
                    {typeof result.threshold === 'number' && (
                      <> · τ = {result.threshold.toFixed(2)}</>
                    )}
                    {result.edge_exists === false && <> · new edge</>}
                    {result.published && <> · published to stream</>}
                  </p>
                </div>
              </div>

              {Array.isArray(result.signals) && result.signals.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {result.signals.map(sig => (
                    <Chip key={sig}>{String(sig).replace(/-/g, ' ')}</Chip>
                  ))}
                </div>
              )}

              {result.ensemble && (
                <div className="mt-5">
                  <EnsembleConsensus scores={result.ensemble} />
                </div>
              )}

              {shapTop.length > 0 && (
                <div className="mt-5">
                  <h3 className="text-base font-semibold text-gray-900">Why (SHAP)</h3>
                  <p className="text-xs text-gray-500 mt-0.5">
                    <span className="text-red-600 font-medium">Red</span> pushes fraud up,{' '}
                    <span className="text-blue-600 font-medium">blue</span> pulls it down
                  </p>
                  <div className="mt-3 space-y-2">
                    {shapTop.map(s => (
                      <ShapRow key={s.feature} feature={s.feature} shap={s.shap} maxAbs={shapMaxAbs} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white border border-dashed border-gray-300 rounded-xl p-12 text-center text-gray-400">
              Build a transaction and press <span className="font-semibold text-gray-500">Score &amp; Stream</span> to see the result.
            </div>
          )}

          {/* Mini live feed */}
          <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50">
              <span className="text-sm font-medium text-gray-700">Live stream (last 25)</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </div>
            {feedError ? (
              <div className="px-5 py-3 text-sm text-red-700 bg-red-50">{feedError}</div>
            ) : feed.length === 0 ? (
              <div className="p-8 text-center text-gray-400 text-sm">
                No transactions yet — score one or inject a scenario.
              </div>
            ) : (
              <ul className="divide-y divide-gray-100 max-h-[50vh] overflow-y-auto">
                {feed.map(e => {
                  const txn = e.txn || {};
                  const signals = e.signals || [];
                  return (
                    <li key={e.seq} className={`px-5 py-2.5 ${e.severity ? 'bg-red-50/40' : ''}`}>
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="font-mono text-xs text-gray-700 truncate">{txn.sender_id}</span>
                          <span className="text-gray-400">→</span>
                          <span className="font-mono text-xs text-gray-700 truncate">{txn.receiver_id}</span>
                        </div>
                        <div className="flex items-center gap-3 shrink-0">
                          <span className="text-xs font-medium tabular-nums text-gray-800">₹{inr(txn.amount)}</span>
                          {e.ml_score != null && (
                            <span className="text-xs tabular-nums text-gray-500">
                              {(Number(e.ml_score) * 100).toFixed(0)}%
                            </span>
                          )}
                          {e.severity && <SeverityBadge severity={e.severity} />}
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-3 mt-1">
                        <div className="flex flex-wrap gap-1">
                          {signals.map(sig => (
                            <span key={sig} className="inline-flex items-center px-1.5 py-0.5 rounded bg-gray-100 text-gray-500 text-[10px] font-medium">
                              {String(sig).replace(/-/g, ' ')}
                            </span>
                          ))}
                        </div>
                        {txn.transaction_type && (
                          <span className="text-[10px] text-gray-400 shrink-0">{txn.transaction_type}</span>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
