// Visualizes how the three base models (XGBoost / GraphSAGE / GAT) vote on an
// edge and how the stacked meta-learner reconciles them. Pure presentational —
// `scores` is the `ensemble` block from /api/simulate/score.
const BASE_MODELS = [
  { key: 'xgb', label: 'XGBoost' },
  { key: 'sage', label: 'GraphSAGE' },
  { key: 'gat', label: 'GAT' },
];

function ScoreBar({ label, value, fill, track = 'bg-gray-100', tall = false, bold = false }) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className={`text-xs ${bold ? 'font-semibold text-gray-900' : 'text-gray-600'}`}>{label}</span>
        <span className={`text-xs tabular-nums ${bold ? 'font-bold text-gray-900' : 'font-medium text-gray-700'}`}>
          {Number(value).toFixed(2)}
        </span>
      </div>
      <div className={`w-full ${track} rounded-full overflow-hidden ${tall ? 'h-3' : 'h-2'}`}>
        <div className={`${fill} ${tall ? 'h-3' : 'h-2'} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function EnsembleConsensus({ scores }) {
  if (!scores) {
    return (
      <div className="text-xs text-gray-400 italic">No ensemble score for this edge</div>
    );
  }

  const present = BASE_MODELS.filter(m => typeof scores[m.key] === 'number');
  const baseValues = present.map(m => scores[m.key]);
  const hasEnsemble = typeof scores.ensemble === 'number';

  // Agreement = how tightly the base models cluster. With <2 base scores there
  // is nothing to disagree about, so treat it as full agreement.
  let agreement = null;
  if (baseValues.length >= 2) {
    agreement = 1 - (Math.max(...baseValues) - Math.min(...baseValues));
  } else if (baseValues.length === 1) {
    agreement = 1;
  }

  const agreeTone =
    agreement == null ? 'text-gray-500'
    : agreement >= 0.8 ? 'text-green-600'
    : agreement >= 0.6 ? 'text-amber-600'
    : 'text-red-600';

  const agreeCaption =
    agreement == null ? null
    : agreement >= 0.8 ? 'High agreement — strong signal'
    : 'Models diverge — meta-learner adjudicates';

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-5">
      <h3 className="text-base font-semibold text-gray-900">Ensemble consensus</h3>
      <p className="text-xs text-gray-500 mt-0.5">How the base models vote, and where the meta-learner lands</p>

      <div className="mt-4 space-y-3">
        {present.length === 0 ? (
          <p className="text-xs text-gray-400 italic">No base-model scores available</p>
        ) : (
          present.map(m => (
            <ScoreBar key={m.key} label={m.label} value={scores[m.key]} fill="bg-indigo-500" />
          ))
        )}
      </div>

      {hasEnsemble && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <ScoreBar
            label="Ensemble (meta-learner)"
            value={scores.ensemble}
            fill="bg-indigo-700"
            tall
            bold
          />
        </div>
      )}

      {agreement != null && (
        <div className="mt-4">
          <p className="text-sm">
            <span className="text-gray-600">Models agree: </span>
            <span className={`font-bold tabular-nums ${agreeTone}`}>
              {(agreement * 100).toFixed(0)}%
            </span>
          </p>
          {agreeCaption && <p className={`text-xs mt-0.5 ${agreeTone}`}>{agreeCaption}</p>}
        </div>
      )}
    </div>
  );
}
