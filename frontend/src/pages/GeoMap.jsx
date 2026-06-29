import { useEffect, useMemo, useState } from 'react';
import { fetchAPI } from '../api';

// --- geometry / scaling helpers -------------------------------------------

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

// India bounding box → SVG viewBox (0 0 800 900).
// Longitude 68→98 maps to x 40→760; latitude 37.5→6 maps to y 40→860.
// Latitude inverts: higher latitude (further north) = smaller y.
const LNG_MIN = 68;
const LNG_MAX = 98;
const LAT_MIN = 6;
const LAT_MAX = 37.5;
const X_MIN = 40;
const X_MAX = 760;
const Y_MIN = 40;
const Y_MAX = 860;

function project(lat, lng) {
  const x = X_MIN + ((lng - LNG_MIN) / (LNG_MAX - LNG_MIN)) * (X_MAX - X_MIN);
  // higher latitude => lower y, so map LAT_MAX to Y_MIN and LAT_MIN to Y_MAX
  const y = Y_MIN + ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * (Y_MAX - Y_MIN);
  return { x, y };
}

// Interpolate fraud-rate → colour: green (0) → amber (~0.15) → red (>=0.3).
function colorFor(rate) {
  const r = clamp(Number(rate) || 0, 0, 0.3) / 0.3; // 0..1
  // green #16a34a (22,163,74) → amber #f59e0b (245,158,11) → red #dc2626 (220,38,38)
  const green = [22, 163, 74];
  const amber = [245, 158, 11];
  const red = [220, 38, 38];
  let from;
  let to;
  let t;
  if (r < 0.5) {
    from = green;
    to = amber;
    t = r / 0.5;
  } else {
    from = amber;
    to = red;
    t = (r - 0.5) / 0.5;
  }
  const mix = (i) => Math.round(from[i] + (to[i] - from[i]) * t);
  return `rgb(${mix(0)}, ${mix(1)}, ${mix(2)})`;
}

function inr(x) {
  const n = Number(x || 0);
  if (n >= 1e7) return `${(n / 1e7).toFixed(2)} Cr`;
  if (n >= 1e5) return `${(n / 1e5).toFixed(2)} L`;
  return n.toLocaleString('en-IN');
}

function pct(x) {
  return `${((Number(x) || 0) * 100).toFixed(1)}%`;
}

// --- page ------------------------------------------------------------------

export default function GeoMap() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [hovered, setHovered] = useState(null);

  useEffect(() => {
    let active = true;
    fetchAPI('/api/geo/flows')
      .then((d) => {
        if (active) {
          setData(d);
          setError(null);
        }
      })
      .catch((e) => {
        if (active) setError(e.message || 'Failed to load map data');
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const cities = useMemo(() => (data?.cities ?? []), [data]);
  const flows = useMemo(() => (data?.flows ?? []), [data]);

  // Lookup city -> coordinates + metadata.
  const cityByName = useMemo(() => {
    const m = new Map();
    for (const c of cities) {
      if (c && c.city != null) m.set(c.city, c);
    }
    return m;
  }, [cities]);

  // Marker radius scaled by total volume (sqrt), clamped 6..26.
  const maxVolume = useMemo(() => {
    let max = 0;
    for (const c of cities) {
      const v = (Number(c.inflow) || 0) + (Number(c.outflow) || 0);
      if (v > max) max = v;
    }
    return max || 1;
  }, [cities]);

  function radiusFor(city) {
    const v = (Number(city.inflow) || 0) + (Number(city.outflow) || 0);
    const scaled = Math.sqrt(v / maxVolume); // 0..1
    return clamp(6 + scaled * 20, 6, 26);
  }

  // Top ~120 flows by amount, with both endpoints resolvable, as bezier arcs.
  const arcs = useMemo(() => {
    const maxAmount = flows.reduce(
      (mx, f) => Math.max(mx, Number(f.amount) || 0),
      0,
    ) || 1;
    return [...flows]
      .sort((a, b) => (Number(b.amount) || 0) - (Number(a.amount) || 0))
      .slice(0, 120)
      .map((f, i) => {
        const src = cityByName.get(f.source);
        const tgt = cityByName.get(f.target);
        if (!src || !tgt) return null;
        const p1 = project(src.lat, src.lng);
        const p2 = project(tgt.lat, tgt.lng);
        // Control point offset perpendicular to the midpoint for a gentle curve.
        const mx = (p1.x + p2.x) / 2;
        const my = (p1.y + p2.y) / 2;
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const len = Math.hypot(dx, dy) || 1;
        const offset = clamp(len * 0.18, 12, 90);
        const cx = mx + (-dy / len) * offset;
        const cy = my + (dx / len) * offset;
        const amount = Number(f.amount) || 0;
        const fraud = (Number(f.fraud_count) || 0) > 0;
        const width = clamp(1 + (3 * Math.log10(amount + 10)) / Math.log10(maxAmount + 10), 1, 6);
        return {
          key: `${f.source}->${f.target}-${i}`,
          d: `M ${p1.x.toFixed(1)} ${p1.y.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`,
          width,
          fraud,
          flow: f,
        };
      })
      .filter(Boolean);
  }, [flows, cityByName]);

  // Top hotspots by fraud volume.
  const hotspots = useMemo(
    () =>
      [...cities]
        .sort((a, b) => (Number(b.fraud_volume) || 0) - (Number(a.fraud_volume) || 0))
        .slice(0, 8),
    [cities],
  );

  if (loading) {
    return <div className="p-12 text-center text-gray-400">Loading fund-flow map...</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">India Fund-Flow Map</h1>
        <p className="text-sm text-gray-500">
          Inter-city fund flows and fraud hotspots across the branch network
        </p>
        <p className="text-xs text-amber-700 mt-1">
          Amounts and fraud are real. On anonymised datasets (IBM AML) the branch→city
          mapping is illustrative (deterministic), not actual geography.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {!error && cities.length === 0 && (
        <div className="rounded-xl border border-gray-200 bg-white px-4 py-12 text-center text-sm text-gray-400">
          No geographic fund-flow data available.
        </div>
      )}

      {!error && cities.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Map */}
          <div className="lg:col-span-2">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <svg
                viewBox="0 0 800 900"
                width="100%"
                className="w-full h-auto"
                role="img"
                aria-label="Map of India showing inter-city fund flows"
              >
                {/* Flow arcs first, so city dots sit on top */}
                <g>
                  {arcs.map((arc) => (
                    <path
                      key={arc.key}
                      d={arc.d}
                      fill="none"
                      stroke={arc.fraud ? '#dc2626' : '#94a3b8'}
                      strokeOpacity={arc.fraud ? 0.5 : 0.25}
                      strokeWidth={arc.width}
                      strokeLinecap="round"
                    />
                  ))}
                </g>

                {/* City markers */}
                <g>
                  {cities.map((c) => {
                    if (c.lat == null || c.lng == null) return null;
                    const { x, y } = project(c.lat, c.lng);
                    const r = radiusFor(c);
                    const isSelected = selected && selected.city === c.city;
                    const showLabel = r >= 12 || isSelected || (hovered && hovered === c.city);
                    return (
                      <g key={c.city}>
                        <circle
                          cx={x}
                          cy={y}
                          r={r}
                          fill={colorFor(c.fraud_rate)}
                          fillOpacity={0.85}
                          stroke={isSelected ? '#0f172a' : '#ffffff'}
                          strokeWidth={isSelected ? 2.5 : 1.5}
                          className="cursor-pointer"
                          onClick={() => setSelected(c)}
                          onMouseEnter={() => setHovered(c.city)}
                          onMouseLeave={() => setHovered(null)}
                        />
                        {showLabel && (
                          <text
                            x={x + r + 3}
                            y={y + 3}
                            className="text-[10px] fill-gray-700 pointer-events-none select-none"
                          >
                            {c.city}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>

            {/* Legend */}
            <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-gray-500">
              <div className="flex items-center gap-2">
                <svg width="58" height="20" aria-hidden="true">
                  <circle cx="8" cy="10" r="4" fill="#9ca3af" />
                  <circle cx="26" cy="10" r="7" fill="#9ca3af" />
                  <circle cx="48" cy="10" r="9" fill="#9ca3af" />
                </svg>
                <span>Marker size = total volume</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-flex h-3 w-20 rounded-full" style={{ background: 'linear-gradient(to right, #16a34a, #f59e0b, #dc2626)' }} />
                <span>Colour = fraud rate (0 → 30%+)</span>
              </div>
              <div className="flex items-center gap-2">
                <svg width="34" height="10" aria-hidden="true">
                  <line x1="2" y1="5" x2="32" y2="5" stroke="#dc2626" strokeOpacity="0.5" strokeWidth="3" />
                </svg>
                <span>Red arc = flow with fraud</span>
              </div>
            </div>

            <p className="mt-3 text-xs text-gray-400">
              Branches mapped to their city; unrecognised branches assigned deterministically for
              visualization.
            </p>
          </div>

          {/* Side panel */}
          <div className="space-y-6">
            <div className="rounded-xl border border-gray-200 bg-white p-5">
              <h2 className="text-base font-semibold text-gray-900">
                {selected ? selected.city : 'Select a city'}
              </h2>
              {selected ? (
                <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                  <div>
                    <dt className="text-xs text-gray-500">Inflow</dt>
                    <dd className="font-semibold text-gray-900">₹{inr(selected.inflow)}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Outflow</dt>
                    <dd className="font-semibold text-gray-900">₹{inr(selected.outflow)}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Transactions</dt>
                    <dd className="font-semibold text-gray-900">{Number(selected.txn_count || 0).toLocaleString('en-IN')}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Fraud volume</dt>
                    <dd className="font-semibold text-gray-900">₹{inr(selected.fraud_volume)}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-500">Fraud rate</dt>
                    <dd className="flex items-center gap-2 font-semibold text-gray-900">
                      <span
                        className="inline-block h-3 w-3 rounded-full"
                        style={{ background: colorFor(selected.fraud_rate) }}
                      />
                      {pct(selected.fraud_rate)}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p className="mt-2 text-sm text-gray-400">
                  Click a marker on the map or a hotspot below to see city-level fund flow and fraud
                  detail.
                </p>
              )}
            </div>

            <div className="rounded-xl border border-gray-200 bg-white p-5">
              <h2 className="text-base font-semibold text-gray-900">Top hotspots</h2>
              <p className="text-xs text-gray-500 mt-0.5">Cities ranked by fraud volume</p>
              <ul className="mt-3 divide-y divide-gray-100">
                {hotspots.map((c, i) => {
                  const isSelected = selected && selected.city === c.city;
                  return (
                    <li key={c.city}>
                      <button
                        type="button"
                        onClick={() => setSelected(c)}
                        className={`flex w-full items-center justify-between gap-3 py-2 text-left text-sm transition hover:bg-gray-50 ${
                          isSelected ? 'bg-gray-50' : ''
                        }`}
                      >
                        <span className="flex items-center gap-2 truncate">
                          <span className="w-4 text-xs text-gray-400">{i + 1}</span>
                          <span
                            className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ background: colorFor(c.fraud_rate) }}
                          />
                          <span className="truncate font-medium text-gray-900">{c.city}</span>
                        </span>
                        <span className="shrink-0 text-right">
                          <span className="block font-semibold text-gray-900">
                            ₹{inr(c.fraud_volume)}
                          </span>
                          <span className="block text-xs text-gray-400">{pct(c.fraud_rate)}</span>
                        </span>
                      </button>
                    </li>
                  );
                })}
                {hotspots.length === 0 && (
                  <li className="py-2 text-sm text-gray-400">No hotspots detected.</li>
                )}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
