const PILL = 'text-[10px] font-semibold px-2 py-0.5 rounded';

// Classify a legal_basis string into a tone + short label.
function legalBasisTone(legalBasis) {
  if (!legalBasis) return null;
  if (legalBasis.includes('STR')) {
    return { tone: 'bg-red-100 text-red-700', label: 'STR — mandatory' };
  }
  if (legalBasis.includes('38') || /restrict/i.test(legalBasis)) {
    return { tone: 'bg-amber-100 text-amber-700', label: 'RBI §38 restrict' };
  }
  return { tone: 'bg-gray-100 text-gray-600', label: 'Internal EDD' };
}

export default function RegBadges({ alert }) {
  if (!alert || (!alert.fatf_code && !alert.legal_basis)) return null;

  const legal = legalBasisTone(alert.legal_basis);

  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      {alert.fatf_code && (
        <span className={`inline-flex items-center bg-indigo-100 text-indigo-700 ${PILL}`} title={alert.fatf_typology}>
          {alert.fatf_code}
        </span>
      )}
      {legal && (
        <span className={`inline-flex items-center ${legal.tone} ${PILL}`} title={alert.legal_basis}>
          {legal.label}
        </span>
      )}
    </span>
  );
}
