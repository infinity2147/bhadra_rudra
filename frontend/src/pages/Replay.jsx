import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { fetchAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';

// Pretty-print ₹ in Indian grouping. We keep the raw rupee figure (no Cr/L
// abbreviation) here because the playhead readout wants exact amounts.
function formatINR(n) {
  if (n == null || isNaN(Number(n))) return '--';
  return `₹${Number(n).toLocaleString('en-IN')}`;
}

// Compact ₹ for tight SVG labels.
function compactINR(n) {
  if (n == null || isNaN(Number(n))) return '--';
  const v = Number(n);
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2)}Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(2)}L`;
  if (v >= 1e3) return `₹${(v / 1e3).toFixed(1)}k`;
  return `₹${v.toLocaleString('en-IN')}`;
}

function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d)) return String(iso).slice(0, 19);
  return d.toLocaleString('en-IN', { hour12: false }).replace(',', '');
}

// Speed → tick interval (ms between revealed events).
const SPEEDS = [
  { label: '0.5x', tickMs: 800 },
  { label: '1x', tickMs: 400 },
  { label: '2x', tickMs: 200 },
  { label: '4x', tickMs: 100 },
];

const MAX_ENTITIES = 24;

// SVG horizontal layout — the left band holds entity labels + lane lines,
// the right band is the time axis the transfers are plotted along.
const VB_WIDTH = 1000;
const LANE_X0 = 220;   // left edge of the plotting area (after labels)
const LANE_X1 = 960;   // right edge of the plotting area
const TOP_PAD = 56;    // room for the playhead time label
const ROW_H = 34;

export default function Replay() {
  const [alerts, setAlerts] = useState([]);
  const [alertId, setAlertId] = useState('');
  const [data, setData] = useState(null);

  const [alertsLoading, setAlertsLoading] = useState(true);
  const [alertsError, setAlertsError] = useState(null);
  const [journeyLoading, setJourneyLoading] = useState(false);
  const [journeyError, setJourneyError] = useState(null);

  const [revealedCount, setRevealedCount] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedIdx, setSpeedIdx] = useState(1); // default 1x

  const intervalRef = useRef(null);

  // ── Load the alert list once ───────────────────────────────────────────────
  // eslint v10 forbids synchronous setState in an effect body, so the whole
  // fetch lives in an async IIFE — every setState then runs after an await,
  // guarded by `cancelled` so a quick unmount can't write into a dead component.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const d = await fetchAPI('/api/alerts');
        if (cancelled) return;
        const list = Array.isArray(d) ? d : (d.alerts || []);
        // Surface the cinematic patterns first — layering chains and circular
        // rings make the most compelling replays.
        const rank = (a) => {
          const p = (a.pattern_type || '').toLowerCase();
          if (p.includes('layering')) return 0;
          if (p.includes('circular') || p.includes('cycle')) return 1;
          return 2;
        };
        const sorted = [...list].sort((a, b) => rank(a) - rank(b));
        setAlerts(sorted);
        if (sorted.length) setAlertId(sorted[0].alert_id);
        setAlertsError(null);
      } catch (e) {
        if (!cancelled) setAlertsError(e.message);
      } finally {
        if (!cancelled) setAlertsLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ── Load the selected alert's fund journey ──────────────────────────────────
  // Same async-IIFE shape as above to keep every setState off the synchronous
  // effect path. `await null` yields a microtask before the reset writes so the
  // no-alert clear and the loading reset are never synchronous either.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      await null;
      if (cancelled) return;
      if (!alertId) { setData(null); return; }
      setJourneyLoading(true);
      setJourneyError(null);
      setPlaying(false);
      setRevealedCount(0);
      try {
        const d = await fetchAPI('/api/journey/alert/' + alertId);
        if (!cancelled) setData(d);
      } catch (e) {
        if (!cancelled) setJourneyError(e.message);
      } finally {
        if (!cancelled) setJourneyLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [alertId]);

  // ── Derive the ordered timeline + entity swimlanes ──────────────────────────
  const layout = useMemo(() => {
    const raw = (data?.timeline || []).filter(
      (t) => t && t.timestamp && t.sender_id && t.receiver_id
    );
    const events = [...raw].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
    if (!events.length) {
      return { events: [], entities: [], rowIdx: new Map(), tMin: 0, tMax: 1 };
    }

    // Count involvement so that, if we exceed the cap, we keep the busiest
    // entities (the ones that actually carry the ring).
    const counts = new Map();
    const names = new Map();
    for (const t of events) {
      counts.set(t.sender_id, (counts.get(t.sender_id) || 0) + 1);
      counts.set(t.receiver_id, (counts.get(t.receiver_id) || 0) + 1);
      if (t.sender_name && !names.has(t.sender_id)) names.set(t.sender_id, t.sender_name);
      if (t.receiver_name && !names.has(t.receiver_id)) names.set(t.receiver_id, t.receiver_name);
    }
    // Fall back to the journey nodes for any missing display name.
    for (const n of data?.nodes || []) {
      if (n?.id && n?.name && !names.has(n.id)) names.set(n.id, n.name);
    }

    const keptIds = [...counts.keys()]
      .sort((a, b) => (counts.get(b) || 0) - (counts.get(a) || 0))
      .slice(0, MAX_ENTITIES);
    const keptSet = new Set(keptIds);

    // Order the kept entities by first appearance in time → a sender tends to
    // sit above its receiver, so layering chains read as a downward cascade.
    const firstSeen = new Map();
    for (let i = 0; i < events.length; i++) {
      const t = events[i];
      if (keptSet.has(t.sender_id) && !firstSeen.has(t.sender_id)) firstSeen.set(t.sender_id, i);
      if (keptSet.has(t.receiver_id) && !firstSeen.has(t.receiver_id)) firstSeen.set(t.receiver_id, i);
    }
    const orderedIds = keptIds
      .filter((id) => firstSeen.has(id))
      .sort((a, b) => firstSeen.get(a) - firstSeen.get(b));

    const rowIdx = new Map(orderedIds.map((id, i) => [id, i]));
    const entities = orderedIds.map((id) => ({
      id,
      name: names.get(id) || id,
      txns: counts.get(id) || 0,
    }));

    // Only plot transfers whose BOTH endpoints survived the cap.
    const plotted = events.filter(
      (t) => rowIdx.has(t.sender_id) && rowIdx.has(t.receiver_id)
    );

    const times = plotted.map((t) => new Date(t.timestamp).getTime());
    const tMin = Math.min(...times);
    const tMax = Math.max(...times);

    return { events: plotted, entities, rowIdx, tMin, tMax };
  }, [data]);

  const { events, entities, rowIdx, tMin, tMax } = layout;
  const total = events.length;
  // Clamp at the point of use rather than via a state-writing effect — keeps
  // the value sane if the dataset shrinks under us without a cascading render.
  const safeCount = Math.min(revealedCount, total);

  // ── Playback loop ───────────────────────────────────────────────────────────
  const tickMs = SPEEDS[speedIdx].tickMs;

  // Stop helper — clears the running interval.
  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  // Toggle play/pause. Pressing play at the end rewinds to the start so the
  // ring re-assembles from scratch. (Kept out of the effect so the effect
  // never has to setState synchronously.)
  const togglePlay = useCallback(() => {
    if (playing) { setPlaying(false); return; }
    if (total === 0) return;
    if (revealedCount >= total) setRevealedCount(0);
    setPlaying(true);
  }, [playing, total, revealedCount]);

  useEffect(() => {
    if (!playing || total === 0) { stop(); return; }

    intervalRef.current = setInterval(() => {
      setRevealedCount((c) => {
        if (c >= total) {
          setPlaying(false);    // auto-pause at the end
          return total;
        }
        return c + 1;
      });
    }, tickMs);

    return stop;
  }, [playing, tickMs, total, stop]);

  // Clean up on unmount.
  useEffect(() => stop, [stop]);

  // ── Geometry helpers ────────────────────────────────────────────────────────
  const tRange = tMax - tMin || 1;
  const svgHeight = TOP_PAD + entities.length * ROW_H + 28;

  const xFor = useCallback(
    (iso) => {
      const t = new Date(iso).getTime();
      if (isNaN(t)) return LANE_X0;
      return LANE_X0 + ((t - tMin) / tRange) * (LANE_X1 - LANE_X0);
    },
    [tMin, tRange]
  );
  const yFor = useCallback((id) => TOP_PAD + (rowIdx.get(id) ?? 0) * ROW_H + ROW_H / 2, [rowIdx]);

  // Log-scaled stroke width keyed off the largest transfer.
  const maxAmount = useMemo(
    () => Math.max(1, ...events.map((t) => Number(t.amount) || 0)),
    [events]
  );
  const strokeFor = useCallback(
    (amt) => {
      const a = Number(amt) || 0;
      const w = 1.2 + (Math.log10(a + 1) / Math.log10(maxAmount + 1)) * 4.5;
      return Math.max(1.2, Math.min(6, w));
    },
    [maxAmount]
  );

  const visible = events.slice(0, safeCount);
  const lastEvent = safeCount > 0 ? events[safeCount - 1] : null;
  const revealedFraud = visible.reduce((acc, t) => acc + (t.is_fraud ? 1 : 0), 0);
  const playheadX = lastEvent ? xFor(lastEvent.timestamp) : LANE_X0;

  // ── Render ──────────────────────────────────────────────────────────────────
  const atEnd = total > 0 && safeCount >= total;
  const selectedAlert = alerts.find((a) => a.alert_id === alertId);

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Temporal Replay</h1>
        <p className="text-sm text-gray-500">Watch a fraud ring assemble transaction-by-transaction</p>
      </div>

      {/* Alert picker */}
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-gray-700">Alert</label>
        <select
          value={alertId}
          onChange={(e) => setAlertId(e.target.value)}
          disabled={alertsLoading || !!alertsError || alerts.length === 0}
          className="px-3 py-1.5 border border-gray-300 rounded-md text-sm w-96 max-w-full bg-white focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:bg-gray-100 disabled:text-gray-400"
        >
          {alertsLoading && <option>Loading alerts…</option>}
          {!alertsLoading && alerts.length === 0 && <option>No alerts available</option>}
          {alerts.map((a) => (
            <option key={a.alert_id} value={a.alert_id}>
              {a.alert_id} — {a.pattern_type} ({a.severity})
            </option>
          ))}
        </select>
        {selectedAlert && <SeverityBadge severity={selectedAlert.severity} />}
        {selectedAlert?.pattern_type && (
          <span className="text-sm text-gray-600">{selectedAlert.pattern_type}</span>
        )}
      </div>

      {alertsError && (
        <div className="rounded-md bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
          Could not load alerts: {alertsError}
        </div>
      )}

      {/* Control bar */}
      <div className="rounded-xl border border-gray-200 bg-white shadow-sm">
        <div className="flex flex-wrap items-center gap-4 px-4 py-3 border-b border-gray-100">
          <button
            onClick={togglePlay}
            disabled={total === 0 || journeyLoading}
            className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              playing
                ? 'bg-amber-500 text-white hover:bg-amber-600'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'
            }`}
          >
            {playing ? (
              <>
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
                Pause
              </>
            ) : (
              <>
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
                {atEnd ? 'Replay' : 'Play'}
              </>
            )}
          </button>

          <button
            onClick={() => { setPlaying(false); setRevealedCount(0); }}
            disabled={total === 0}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 4v6h6M20 20v-6h-6M20 9a8 8 0 00-14.9-2M4 15a8 8 0 0014.9 2" /></svg>
            Reset
          </button>

          {/* Speed selector */}
          <div className="inline-flex rounded-lg border border-gray-300 p-0.5">
            {SPEEDS.map((s, i) => (
              <button
                key={s.label}
                onClick={() => setSpeedIdx(i)}
                className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
                  speedIdx === i ? 'bg-indigo-600 text-white' : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>

          {/* Readout */}
          <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
            <span className="text-gray-700">
              Showing <span className="font-semibold text-gray-900 tabular-nums">{safeCount}</span>
              {' / '}
              <span className="font-semibold text-gray-900 tabular-nums">{total}</span> transactions
            </span>
            <span className="text-red-700 font-medium tabular-nums">{revealedFraud} flagged</span>
            <span className="font-mono text-xs text-gray-500">
              {lastEvent ? fmtTime(lastEvent.timestamp) : '—'}
            </span>
          </div>
        </div>

        {/* Seek slider */}
        <div className="px-4 py-3">
          <input
            type="range"
            min={0}
            max={total}
            value={revealedCount}
            onChange={(e) => { setPlaying(false); setRevealedCount(Number(e.target.value)); }}
            disabled={total === 0}
            className="w-full accent-indigo-600 cursor-pointer disabled:cursor-not-allowed"
            aria-label="Seek through transactions"
          />
          <div className="flex justify-between text-[11px] text-gray-400 font-mono mt-1">
            <span>{total ? fmtTime(events[0].timestamp).slice(0, 16) : '—'}</span>
            <span>{total ? fmtTime(events[total - 1].timestamp).slice(0, 16) : '—'}</span>
          </div>
        </div>
      </div>

      {/* Stage */}
      <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
        {journeyLoading && (
          <div className="flex items-center justify-center py-24 text-gray-400 text-sm">
            Loading fund journey…
          </div>
        )}

        {!journeyLoading && journeyError && (
          <div className="flex items-center justify-center py-24 text-red-600 text-sm px-6 text-center">
            Could not load journey: {journeyError}
          </div>
        )}

        {!journeyLoading && !journeyError && !alertId && (
          <div className="flex items-center justify-center py-24 text-gray-400 text-sm px-6 text-center">
            Select an alert above to replay its fund journey.
          </div>
        )}

        {!journeyLoading && !journeyError && alertId && total === 0 && (
          <div className="flex items-center justify-center py-24 text-gray-400 text-sm px-6 text-center">
            This alert has no time-stamped transactions to replay.
          </div>
        )}

        {!journeyLoading && !journeyError && total > 0 && (
          <div className="w-full overflow-x-auto">
            <svg
              viewBox={`0 0 ${VB_WIDTH} ${svgHeight}`}
              width="100%"
              style={{ minWidth: 640, height: svgHeight }}
              className="bg-gradient-to-b from-white to-gray-50"
            >
              <defs>
                <marker id="rp-arrow-red" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M0 0 L10 5 L0 10 z" fill="#dc2626" />
                </marker>
                <marker id="rp-arrow-slate" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M0 0 L10 5 L0 10 z" fill="#64748b" />
                </marker>
              </defs>

              {/* Entity lanes */}
              {entities.map((e, i) => {
                const y = TOP_PAD + i * ROW_H + ROW_H / 2;
                return (
                  <g key={e.id}>
                    <rect
                      x={0}
                      y={TOP_PAD + i * ROW_H}
                      width={VB_WIDTH}
                      height={ROW_H}
                      fill={i % 2 === 0 ? '#fafafa' : '#ffffff'}
                    />
                    <text x={LANE_X0 - 12} y={y - 2} textAnchor="end" fontSize="11" fontWeight="600" fill="#1f2937">
                      {String(e.name).slice(0, 26)}
                    </text>
                    <text x={LANE_X0 - 12} y={y + 9} textAnchor="end" fontSize="8.5" fill="#9ca3af">
                      {e.txns} txn{e.txns > 1 ? 's' : ''}
                    </text>
                    <line x1={LANE_X0} y1={y} x2={LANE_X1} y2={y} stroke="#e5e7eb" strokeDasharray="2,3" />
                  </g>
                );
              })}

              {/* Time-axis grid + labels */}
              {Array.from({ length: 6 }, (_, i) => {
                const t = tMin + (tRange * i) / 5;
                const x = LANE_X0 + ((LANE_X1 - LANE_X0) * i) / 5;
                return (
                  <g key={`tick-${i}`}>
                    <line x1={x} y1={TOP_PAD - 6} x2={x} y2={svgHeight - 22} stroke="#f3f4f6" />
                    <text x={x} y={svgHeight - 8} textAnchor="middle" fontSize="9" fill="#9ca3af">
                      {new Date(t).toISOString().slice(5, 16).replace('T', ' ')}
                    </text>
                  </g>
                );
              })}

              {/* Revealed transfers */}
              {visible.map((t, i) => {
                const x = xFor(t.timestamp);
                const y1 = yFor(t.sender_id);
                const y2 = yFor(t.receiver_id);
                const isFraud = !!t.is_fraud;
                const isLatest = i === revealedCount - 1;
                const color = isFraud ? '#dc2626' : '#64748b';
                const marker = isFraud ? 'url(#rp-arrow-red)' : 'url(#rp-arrow-slate)';
                const w = strokeFor(t.amount);
                // Gentle quadratic curve bowed in the direction of travel so
                // overlapping transfers between the same rows stay distinct.
                const dir = y2 >= y1 ? 1 : -1;
                const cx = x + Math.min(26, Math.abs(y2 - y1) * 0.35) * dir;
                const cy = (y1 + y2) / 2;
                return (
                  <g key={`${t.transaction_id || i}-${i}`} opacity={isLatest ? 1 : 0.82}>
                    <path
                      d={`M ${x} ${y1} Q ${cx} ${cy} ${x} ${y2}`}
                      fill="none"
                      stroke={color}
                      strokeWidth={isLatest ? w + 1.4 : w}
                      markerEnd={marker}
                      strokeLinecap="round"
                    >
                      {isLatest && (
                        <animate attributeName="opacity" values="0.35;1;0.7;1" dur="0.6s" repeatCount="1" />
                      )}
                    </path>
                    {/* Origin dot */}
                    <circle cx={x} cy={y1} r={isLatest ? 3.5 : 2.4} fill={color}>
                      {isLatest && <animate attributeName="r" values="6;3.5" dur="0.5s" repeatCount="1" />}
                    </circle>
                    {/* Amount label only for the newest transfer, to avoid clutter */}
                    {isLatest && (
                      <g transform={`translate(${cx + dir * 6}, ${cy})`}>
                        <rect x={2} y={-8} width={58} height={15} rx={7} fill="#ffffff" stroke={color} strokeWidth="0.75" />
                        <text x={31} y={3} textAnchor="middle" fontSize="9" fontWeight="700" fill={color}>
                          {compactINR(t.amount)}
                        </text>
                      </g>
                    )}
                  </g>
                );
              })}

              {/* Playhead */}
              {lastEvent && (
                <g>
                  <line x1={playheadX} y1={TOP_PAD - 10} x2={playheadX} y2={svgHeight - 22} stroke="#4338ca" strokeWidth="1.5" />
                  <circle cx={playheadX} cy={TOP_PAD - 10} r="3.5" fill="#4338ca" />
                  <g transform={`translate(${Math.max(LANE_X0, Math.min(playheadX, LANE_X1 - 150))}, 18)`}>
                    <rect x={0} y={-12} width={150} height={20} rx={5} fill="#4338ca" />
                    <text x={75} y={2} textAnchor="middle" fontSize="10" fontWeight="600" fill="#ffffff" fontFamily="ui-monospace, monospace">
                      {fmtTime(lastEvent.timestamp)}
                    </text>
                  </g>
                </g>
              )}
            </svg>
          </div>
        )}

        {/* Legend */}
        {!journeyLoading && !journeyError && total > 0 && (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 px-4 py-2.5 border-t border-gray-100 text-xs text-gray-600">
            <span className="flex items-center gap-1.5"><span className="w-5 h-0.5 bg-red-600 inline-block" /> Flagged transfer</span>
            <span className="flex items-center gap-1.5"><span className="w-5 h-0.5 bg-slate-500 inline-block" /> Normal transfer</span>
            <span className="flex items-center gap-1.5"><span className="w-0.5 h-3.5 bg-indigo-700 inline-block" /> Playhead (current time)</span>
            <span className="text-gray-400">Edge thickness ∝ log(amount). Rows ordered by first appearance.</span>
          </div>
        )}
      </div>

      {/* Last revealed transfer detail */}
      {!journeyLoading && !journeyError && lastEvent && (
        <div className="rounded-xl border border-gray-200 bg-white shadow-sm px-4 py-3">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <span className="text-[11px] uppercase tracking-wide text-gray-400">Latest transfer</span>
            <span className="font-medium text-gray-900">
              {(lastEvent.sender_name || lastEvent.sender_id)} <span className="text-gray-400 mx-1">→</span> {(lastEvent.receiver_name || lastEvent.receiver_id)}
            </span>
            <span className="font-semibold tabular-nums text-gray-900">{formatINR(lastEvent.amount)}</span>
            <span className="font-mono text-xs text-gray-500">{fmtTime(lastEvent.timestamp)}</span>
            {lastEvent.is_fraud
              ? <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-700">FLAGGED</span>
              : <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-gray-100 text-gray-600">normal</span>}
          </div>
        </div>
      )}
    </div>
  );
}
