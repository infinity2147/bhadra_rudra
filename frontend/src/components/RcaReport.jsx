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
          {r?.signals?.n_txns} transactions · ₹{r?.signals?.total_amount?.toLocaleString('en-IN')}
        </p>
      </section>

      <section className="rounded-lg border border-gray-200 bg-gray-50 p-4 space-y-1">
        <h4 className="font-semibold text-gray-800 mb-1">2. Why it succeeded (root cause)</h4>
        <p className="text-gray-700">{d?.control_gap}</p>
        <p className="text-gray-500 text-xs mt-1">
          Basis: {d?.basis} · Evidence: {d?.evidence}
        </p>
      </section>

      <section className="rounded-lg border border-gray-200 bg-gray-50 p-4">
        <h4 className="font-semibold text-gray-800 mb-2">3. What to fix (recommendations)</h4>
        <ul className="list-disc ml-5 space-y-1 text-gray-700">
          {(rec?.policy_level || []).map((p, i) => (
            <li key={i}>{p.recommendation}</li>
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
