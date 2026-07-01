export default function RcaReport({ dossier }) {
  if (!dossier) return null;
  if (dossier.error) return <div className="text-red-500 text-sm">{dossier.error}</div>;

  const { reconstruction: r, diagnosis: d, recommendations: rec, narrative } = dossier;

  return (
    <div className="mt-6 space-y-4 text-sm">
      <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-4">
        <h4 className="text-xs font-semibold text-indigo-800 uppercase tracking-wide mb-1">
          Root Cause Analysis
        </h4>
        <p className="italic text-gray-700">{narrative}</p>
      </div>

      <section className="rounded-lg border border-gray-200 bg-gray-50 p-4 space-y-1">
        <h4 className="font-semibold text-gray-800 mb-1">1. How it happened (forensic)</h4>
        <p className="text-gray-700">
          Method: <span className="font-medium">{r?.method?.fatf_typology}</span>{' '}
          <span className="text-gray-500">({r?.method?.fatf_code})</span>
        </p>
        <p className="text-gray-700">
          Origin: {r?.origin?.join(', ') || '—'} → Cash-out: {r?.cashout?.join(', ') || '—'}
        </p>
        <p className="text-gray-500 text-xs mt-1">
          {r?.signals?.n_txns} transactions · ₹{(r?.signals?.total_amount ?? 0).toLocaleString('en-IN')}
        </p>
      </section>

      <section className="rounded-lg border border-gray-200 bg-gray-50 p-4 space-y-1">
        <h4 className="font-semibold text-gray-800 mb-1">2. Why it succeeded (root cause)</h4>
        <p className="text-gray-700">{d?.control_gap}</p>
        <p className="text-gray-500 text-xs mt-1">
          Basis: {d?.basis} · Evidence: {d?.evidence}
        </p>
      </section>

      {dossier.foresight && dossier.foresight.next_targets && dossier.foresight.next_targets.length > 0 && (
        <section className="rounded-lg border border-indigo-200 bg-indigo-50/50 p-4 space-y-1">
          <div className="flex justify-between items-center mb-1">
            <h4 className="font-semibold text-indigo-900">3. Who's next (predictive analytics)</h4>
            <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-medium border border-indigo-200">
              GNN Latent Space k-NN
            </span>
          </div>
          <p className="text-indigo-800 text-sm mb-2 mt-1">
            Predicted targets based on structural similarity to known fraud cluster:
          </p>
          <ul className="space-y-2 mt-2">
            {dossier.foresight.next_targets.map(t => (
              <li key={t.entity_id} className="flex justify-between items-center bg-white p-2 rounded border border-indigo-100">
                <span className="font-mono text-sm text-gray-800">{t.entity_id}</span>
                <div className="flex items-center space-x-3">
                  <span className="text-xs text-red-600 font-medium">₹{t.exposure.toLocaleString('en-IN')} exposure</span>
                  <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-md">{(t.similarity * 100).toFixed(1)}% match</span>
                </div>
              </li>
            ))}
          </ul>
          <p className="text-xs text-indigo-600 mt-2 italic pt-1 border-t border-indigo-100">
            Total exposure at risk: ₹{dossier.foresight.exposure.toLocaleString('en-IN')}
          </p>
        </section>
      )}

      <section className="rounded-lg border border-gray-200 bg-gray-50 p-4">
        <h4 className="font-semibold text-gray-800 mb-2">{dossier.foresight ? '4' : '3'}. What to fix (recommendations)</h4>
        <ul className="list-disc ml-5 space-y-1 text-gray-700">
          {(rec?.policy_level || []).map((p) => (
            <li key={p.recommendation}>{p.recommendation}</li>
          ))}
        </ul>
        {rec?.account_level?.length > 0 && (
          <p className="text-gray-500 text-xs mt-2">
            Immediate: EDD + hold on{' '}
            {rec.account_level.map((a) => a.name).join(', ')}
          </p>
        )}
      </section>
    </div>
  );
}
