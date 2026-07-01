import { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { fetchAPI, postAPI, getTgnPredictions } from '../api';

function MetricCard({ label, value, hint, tone = 'indigo' }) {
  const tones = {
    indigo: 'bg-indigo-50 text-indigo-900 border-indigo-200',
    emerald: 'bg-emerald-50 text-emerald-900 border-emerald-200',
    amber: 'bg-amber-50 text-amber-900 border-amber-200',
    rose: 'bg-rose-50 text-rose-900 border-rose-200',
  };
  return (
    <div className={`rounded-xl border p-5 ${tones[tone]}`}>
      <p className="text-[11px] font-semibold uppercase tracking-wide opacity-70">{label}</p>
      <p className="text-3xl font-bold mt-1 tabular-nums">{value}</p>
      {hint && <p className="text-xs mt-1 opacity-70">{hint}</p>}
    </div>
  );
}

export default function ModelMetrics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [error, setError] = useState(null);
  const [variants, setVariants] = useState([]);
  const [variant, setVariant] = useState('ibm_aml');
  const [tgnPredictions, setTgnPredictions] = useState(null);
  const [tgnLoading, setTgnLoading] = useState(false);

  const load = useCallback((v) => {
    setLoading(true);
    fetchAPI(`/api/ml/metrics?variant=${encodeURIComponent(v)}`)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchAPI('/api/ml/variants')
      .then((d) => { if (!cancelled) setVariants(d.variants || []); })
      .catch(() => { if (!cancelled) setVariants([]); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => { (async () => { await load(variant); })(); }, [load, variant]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (cancelled) return;
      setTgnLoading(true);
      setTgnPredictions(null);
      try {
        const d = await getTgnPredictions(variant);
        if (!cancelled) setTgnPredictions(d);
      } catch {
        if (!cancelled) setTgnPredictions(null);
      } finally {
        if (!cancelled) setTgnLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [variant]);

  async function retrain() {
    setRetraining(true);
    try {
      const out = await postAPI('/api/ml/retrain', {});
      setData({ trained: true, ...(out.metrics || {}) });
      // Refresh the variants list in case a new variant was created
      const vs = await fetchAPI('/api/ml/variants');
      setVariants(vs.variants || []);
      // Refresh TGN predictions after retrain
      const tgnData = await getTgnPredictions(variant);
      setTgnPredictions(tgnData);
    } catch (e) {
      setError(e.message);
    } finally {
      setRetraining(false);
    }
  }

  if (loading) return <div className="p-12 text-center text-gray-400">Loading model metrics...</div>;
  if (error) return <div className="p-12 text-red-600">{error}</div>;
  if (!data?.trained) {
    return (
      <div className="p-12 max-w-xl">
        <h1 className="text-2xl font-bold text-gray-900">ML Model</h1>
        <p className="text-gray-600 mt-2">{data?.message || 'Model not trained.'}</p>
        <button onClick={retrain} disabled={retraining} className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg">
          {retraining ? 'Training...' : 'Train Now'}
        </button>
      </div>
    );
  }

  const cm = data.confusion_matrix || [[0, 0], [0, 0]];
  const [tn, fp] = cm[0]; const [fn, tp] = cm[1];

  const variantLabel = data.variant === 'ibm_aml'
    ? 'IBM AML (public benchmark, 100k stratified sample)'
    : data.variant === 'paysim'
    ? 'PaySim mobile-money (public Kaggle dataset)'
    : data.variant;

  return (
    <div className="p-6 space-y-6 max-w-[1400px] mx-auto">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-gray-900">ML Model — Edge-level Fraud Classifier</h1>
          <p className="text-sm text-gray-500 mt-1">
            {data.model_kind === 'xgboost' ? 'XGBoost' : 'Gradient Boosting'} trained on {data.n_features} engineered features
            from the fund-flow graph. Evaluated on a stratified 20% held-out split.
          </p>
          <p className="text-xs text-indigo-700 mt-1 font-medium">
            Active dataset: {variantLabel}
          </p>
          {data.dataset_name && (
            <p className="text-xs text-gray-400 mt-0.5">Dataset: {data.dataset_name}</p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {variants.length > 0 && (
            <label className="text-xs text-gray-600 flex items-center gap-2">
              Variant
              <select
                value={variant}
                onChange={(e) => setVariant(e.target.value)}
                className="px-2 py-1.5 border border-gray-300 rounded-md text-sm bg-white"
              >
                {variants.map((v) => (
                  <option key={v.variant} value={v.variant}>
                    {v.variant} — F1 {v.f1?.toFixed(3) ?? '—'}
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
      </div>

      {/* Variant comparison */}
      {variants.length > 1 && (
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h2 className="text-base font-semibold text-gray-900">Variant Comparison</h2>
          <p className="text-xs text-gray-500 mt-0.5">F1 / AUC across every trained variant</p>
          <table className="mt-3 w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-200">
                <th className="text-left pb-2 font-medium">Variant</th>
                <th className="text-left pb-2 font-medium">Dataset</th>
                <th className="text-right pb-2 font-medium">F1</th>
                <th className="text-right pb-2 font-medium">AUC</th>
                <th className="text-right pb-2 font-medium">Edges</th>
                <th className="text-right pb-2 font-medium">Fraud rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {variants.map((v) => (
                <tr
                  key={v.variant}
                  onClick={() => setVariant(v.variant)}
                  className={`cursor-pointer hover:bg-gray-50 ${variant === v.variant ? 'bg-indigo-50' : ''}`}
                >
                  <td className="py-2 font-medium text-gray-800">{v.variant}</td>
                  <td className="py-2 text-gray-600 text-xs">{v.dataset_name || '—'}</td>
                  <td className="py-2 text-right tabular-nums">{v.f1?.toFixed(3) ?? '—'}</td>
                  <td className="py-2 text-right tabular-nums">{v.auc?.toFixed(3) ?? '—'}</td>
                  <td className="py-2 text-right tabular-nums">{v.n_edges?.toLocaleString('en-IN') ?? '—'}</td>
                  <td className="py-2 text-right tabular-nums">{((v.fraud_rate ?? 0) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="F1 Score" value={data.f1?.toFixed(3) ?? '—'} hint="harmonic mean of P/R" tone="indigo" />
        <MetricCard label="AUC-ROC" value={data.auc?.toFixed(3) ?? '—'} hint="threshold-free ranking quality" tone="emerald" />
        <MetricCard label="Precision" value={data.precision?.toFixed(3) ?? '—'} hint={`${tp} TP / ${tp + fp} predicted positive`} tone="amber" />
        <MetricCard label="Recall" value={data.recall?.toFixed(3) ?? '—'} hint={`${tp} TP / ${tp + fn} actual fraud`} tone="rose" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confusion matrix */}
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h2 className="text-base font-semibold text-gray-900">Confusion Matrix</h2>
          <p className="text-xs text-gray-500 mt-0.5">Rows = actual class, Cols = predicted class</p>
          <table className="mt-4 w-full text-sm border-collapse">
            <thead>
              <tr className="text-xs text-gray-500">
                <th className="p-2"></th>
                <th className="p-2 bg-gray-50">Pred: Normal</th>
                <th className="p-2 bg-gray-50">Pred: Fraud</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="p-2 text-xs text-gray-500 bg-gray-50 font-medium">Actual: Normal</td>
                <td className="p-3 text-center bg-emerald-50 text-emerald-800 font-bold text-lg">{tn}</td>
                <td className="p-3 text-center bg-rose-50 text-rose-800 font-bold text-lg">{fp}</td>
              </tr>
              <tr>
                <td className="p-2 text-xs text-gray-500 bg-gray-50 font-medium">Actual: Fraud</td>
                <td className="p-3 text-center bg-rose-50 text-rose-800 font-bold text-lg">{fn}</td>
                <td className="p-3 text-center bg-emerald-50 text-emerald-800 font-bold text-lg">{tp}</td>
              </tr>
            </tbody>
          </table>
          <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
            <div className="rounded-lg bg-gray-50 p-3">
              <p className="text-gray-500">False positives</p>
              <p className="font-bold text-gray-900">{fp}</p>
              <p className="text-gray-500 mt-0.5">Costs investigator time</p>
            </div>
            <div className="rounded-lg bg-gray-50 p-3">
              <p className="text-gray-500">False negatives</p>
              <p className="font-bold text-gray-900">{fn}</p>
              <p className="text-gray-500 mt-0.5">Fraud that slipped through</p>
            </div>
          </div>
        </div>

        {/* Dataset balance */}
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h2 className="text-base font-semibold text-gray-900">Training Data</h2>
          <p className="text-xs text-gray-500 mt-0.5">All counts at the edge level</p>
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div>
              <dt className="text-xs text-gray-500">Total edges</dt>
              <dd className="font-bold tabular-nums">{data.n_edges}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Features</dt>
              <dd className="font-bold tabular-nums">{data.n_features}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Train edges</dt>
              <dd className="font-bold tabular-nums">{data.n_train}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Test edges</dt>
              <dd className="font-bold tabular-nums">{data.n_test}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Train fraud</dt>
              <dd className="font-bold tabular-nums">{data.n_fraud_train}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Test fraud</dt>
              <dd className="font-bold tabular-nums">{data.n_fraud_test}</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Fraud rate</dt>
              <dd className="font-bold tabular-nums">{((data.fraud_rate ?? 0) * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt className="text-xs text-gray-500">Avg precision</dt>
              <dd className="font-bold tabular-nums">{data.average_precision?.toFixed(3)}</dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Feature importance */}
      {data.feature_importances?.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h2 className="text-base font-semibold text-gray-900">Top Predictive Features</h2>
          <p className="text-xs text-gray-500 mt-0.5">XGBoost gain-based importance</p>
          <div className="mt-3" style={{ height: 400 }}>
            <ResponsiveContainer>
              <BarChart data={data.feature_importances.slice(0, 15)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={210} />
                <Tooltip />
                <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* TGN metrics card */}
      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h2 className="text-base font-semibold text-gray-900">Temporal GNN (TGN)</h2>
        <p className="text-xs text-gray-500 mt-0.5">
          Temporal GNN (Rossi 2020) — predicts fraud from the graph&apos;s time-evolution.
        </p>
        {data.tgn ? (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-3">
            <div className="rounded-lg bg-indigo-50 border border-indigo-200 p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-indigo-700 opacity-70">AUPRC</p>
              <p className="text-2xl font-bold tabular-nums text-indigo-900 mt-1">
                {data.tgn.auprc?.toFixed(3) ?? '—'}
              </p>
            </div>
            <div className="rounded-lg bg-emerald-50 border border-emerald-200 p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-emerald-700 opacity-70">F2</p>
              <p className="text-2xl font-bold tabular-nums text-emerald-900 mt-1">
                {data.tgn.f2?.toFixed(3) ?? '—'}
              </p>
            </div>
            <div className="rounded-lg bg-amber-50 border border-amber-200 p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-amber-700 opacity-70">Precision</p>
              <p className="text-2xl font-bold tabular-nums text-amber-900 mt-1">
                {data.tgn.precision?.toFixed(3) ?? '—'}
              </p>
            </div>
            <div className="rounded-lg bg-rose-50 border border-rose-200 p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-rose-700 opacity-70">Recall</p>
              <p className="text-2xl font-bold tabular-nums text-rose-900 mt-1">
                {data.tgn.recall?.toFixed(3) ?? '—'}
              </p>
            </div>
            <div className="rounded-lg bg-gray-50 border border-gray-200 p-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-600 opacity-70">Threshold</p>
              <p className="text-2xl font-bold tabular-nums text-gray-900 mt-1">
                {data.tgn.threshold?.toFixed(3) ?? '—'}
              </p>
            </div>
          </div>
        ) : (
          <p className="mt-3 text-sm text-gray-400">TGN not trained for this variant.</p>
        )}
      </div>

      {/* Predicted future fraud panel */}
      <div className="bg-white border border-gray-200 rounded-xl p-5">
        <h2 className="text-base font-semibold text-gray-900">Predicted Future Fraud</h2>
        <p className="text-xs text-gray-500 mt-0.5">Top TGN predictions for unseen edges — edges the model flags as high-risk.</p>
        {tgnLoading ? (
          <p className="mt-3 text-sm text-gray-400">Loading predictions...</p>
        ) : !tgnPredictions?.trained ? (
          <p className="mt-3 text-sm text-gray-400">TGN not trained for this variant — no predictions available.</p>
        ) : tgnPredictions.predictions?.length === 0 ? (
          <p className="mt-3 text-sm text-gray-400">No predictions returned.</p>
        ) : (
          <table className="mt-3 w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-200">
                <th className="text-left pb-2 font-medium">Source</th>
                <th className="text-left pb-2 font-medium">Destination</th>
                <th className="text-right pb-2 font-medium">Probability</th>
                <th className="text-center pb-2 font-medium">Fraud?</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {tgnPredictions.predictions.map((p) => (
                <tr
                  key={`${p.src_id}->${p.dst_id}`}
                  className="hover:bg-gray-50"
                >
                  <td className="py-2 font-medium text-gray-800 truncate max-w-[160px]">{p.src_name}</td>
                  <td className="py-2 text-gray-700 truncate max-w-[160px]">{p.dst_name}</td>
                  <td className="py-2 text-right tabular-nums font-medium">
                    <span className={p.prob >= 0.7 ? 'text-rose-700' : p.prob >= 0.4 ? 'text-amber-700' : 'text-gray-600'}>
                      {(p.prob * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-2 text-center">
                    {p.is_fraud
                      ? <span className="text-rose-600 font-bold">✓</span>
                      : <span className="text-gray-400">✗</span>
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-900">
        <p className="font-semibold">A note on these numbers</p>
        {data.variant === 'ibm_aml' ? (
          <p className="mt-1">
            Metrics are on a stratified 100k sample of the IBM AML HI-Small public benchmark
            ({data.n_edges?.toLocaleString()} edges, {(data.fraud_rate * 100).toFixed(1)}% fraud rate).
            The ensemble adds GraphSAGE + GAT on top of XGBoost via a 3-fold OOF logistic-regression
            meta-learner — see the Ensemble tab for the per-base-model breakdown.
          </p>
        ) : (
          <p className="mt-1">
            Variant: {data.variant}. {data.n_edges?.toLocaleString()} edges,
            {' '}{(data.fraud_rate * 100).toFixed(2)}% fraud rate.
          </p>
        )}
      </div>
    </div>
  );
}
