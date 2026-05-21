import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { fetchAPI, postAPI } from '../api';

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

  function load() {
    setLoading(true);
    fetchAPI('/api/ml/metrics')
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  async function retrain() {
    setRetraining(true);
    try {
      const out = await postAPI('/api/ml/retrain', {});
      setData({ trained: true, ...(out.metrics || {}) });
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

  return (
    <div className="p-6 space-y-6 max-w-[1400px] mx-auto">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">ML Model — Edge-level Fraud Classifier</h1>
          <p className="text-sm text-gray-500 mt-1">
            {data.model_kind === 'xgboost' ? 'XGBoost' : 'Gradient Boosting'} trained on {data.n_features} engineered features
            from the fund-flow graph. Evaluated on a stratified 20% held-out split.
          </p>
        </div>
        <button onClick={retrain} disabled={retraining} className="px-3 py-2 text-sm border border-gray-300 rounded-md hover:bg-gray-50">
          {retraining ? 'Retraining...' : 'Retrain'}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="F1 Score" value={data.f1.toFixed(3)} hint="harmonic mean of P/R" tone="indigo" />
        <MetricCard label="AUC-ROC" value={data.auc.toFixed(3)} hint="threshold-free ranking quality" tone="emerald" />
        <MetricCard label="Precision" value={data.precision.toFixed(3)} hint={`${tp} TP / ${tp + fp} predicted positive`} tone="amber" />
        <MetricCard label="Recall" value={data.recall.toFixed(3)} hint={`${tp} TP / ${tp + fn} actual fraud`} tone="rose" />
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
              <dd className="font-bold tabular-nums">{(data.fraud_rate * 100).toFixed(1)}%</dd>
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

      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-blue-900">
        <p className="font-semibold">A note on these numbers</p>
        <p className="mt-1">
          The metrics above are computed on the synthetic dataset we generate locally. The fraud patterns
          we embed are clean by construction — circular rings have low amount variance, layering chains
          have monotonic flow, smurfing transactions cluster below ₹2L — so the model achieves very high
          scores. On the IBM AML-HI-Large public benchmark (2.1M nodes, 180M transactions), our planned
          FraudGT + BDH ensemble targets F1 = 0.72, vs the XGBoost baseline at ~0.42. This page is the
          internal-data baseline; the real comparison is against that public benchmark.
        </p>
      </div>
    </div>
  );
}
