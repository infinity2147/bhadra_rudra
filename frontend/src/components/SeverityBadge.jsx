export default function SeverityBadge({ severity }) {
  if (!severity) return null;   // no severity → render nothing, not an empty pill
  const cls = {
    CRITICAL: 'bg-red-100 text-red-700',
    HIGH: 'bg-orange-100 text-orange-700',
    MEDIUM: 'bg-yellow-100 text-yellow-700',
    LOW: 'bg-green-100 text-green-700',
  }[severity] || 'bg-gray-100 text-gray-700';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${cls}`}>
      {severity}
    </span>
  );
}
