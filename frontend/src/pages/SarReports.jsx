import { useState } from 'react';
import { fetchAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';

function formatCurrency(value) {
  if (value == null) return '--';
  return '₹' + Number(value).toLocaleString('en-IN');
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function SarReports() {
  const [alertId, setAlertId] = useState('');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleGenerate(e) {
    e?.preventDefault();
    const id = alertId.trim();
    if (!id) return;
    setLoading(true);
    setReport(null);
    setError('');
    try {
      const data = await fetchAPI(`/api/sar/generate/${encodeURIComponent(id)}`);
      setReport(data);
    } catch {
      setReport(null);
      setError('Alert not found. Check the alert ID and try again.');
    } finally {
      setLoading(false);
    }
  }

  function handleDownloadTxt() {
    if (!report?.report_text) return;
    const filename = `SAR_${report.report_id ?? alertId}_${Date.now()}.txt`;
    downloadFile(filename, report.report_text, 'text/plain');
  }

  function handleDownloadJson() {
    if (!report) return;
    const filename = `SAR_${report.report_id ?? alertId}_${Date.now()}.json`;
    downloadFile(filename, JSON.stringify(report, null, 2), 'application/json');
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-4xl">
        <h1 className="text-2xl font-bold text-gray-900">SAR Report Generator</h1>
        <p className="text-sm text-gray-500 mt-1">
          Enter an alert ID to generate a Suspicious Activity Report on demand (nothing is pre-generated).
        </p>

        <form onSubmit={handleGenerate} className="mt-6 bg-white border border-gray-200 rounded-xl p-5">
          <label htmlFor="alert-id" className="block text-sm font-medium text-gray-700 mb-2">
            Alert ID
          </label>
          <div className="flex gap-3">
            <input
              id="alert-id"
              type="text"
              value={alertId}
              onChange={e => { setAlertId(e.target.value); setReport(null); setError(''); }}
              placeholder="e.g. ALERT_CIRC_0001"
              className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              autoComplete="off"
              spellCheck={false}
            />
            <button
              type="submit"
              disabled={!alertId.trim() || loading}
              className="px-6 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Generating...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  Generate SAR
                </>
              )}
            </button>
          </div>
          {error && <p className="text-sm text-red-600 mt-2">{error}</p>}
          <p className="text-xs text-gray-400 mt-2">
            Copy the alert ID from the Alerts page. Reports are created only for the ID you submit.
          </p>
        </form>

        {report && (
          <div className="mt-6 space-y-5">
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-lg font-bold text-gray-900 mb-4">Report Summary</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Report ID</p>
                  <p className="text-sm font-semibold mt-1">{report.report_id ?? '--'}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Alert ID</p>
                  <p className="text-sm font-semibold mt-1">{report.alert_id ?? alertId}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Severity</p>
                  <div className="mt-1"><SeverityBadge severity={report.severity} /></div>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Pattern Type</p>
                  <p className="text-sm font-semibold mt-1">{report.pattern_type ?? '--'}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Confidence</p>
                  <p className="text-sm font-semibold mt-1">{report.confidence != null ? `${Number(report.confidence).toFixed(1)}%` : '--'}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Total Flow</p>
                  <p className="text-sm font-semibold mt-1">{formatCurrency(report.total_flow)}</p>
                </div>
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-lg font-bold text-gray-900 mb-3">Report Text</h2>
              <pre className="bg-gray-900 text-gray-100 rounded-lg p-5 text-xs leading-relaxed overflow-auto max-h-[500px] whitespace-pre-wrap font-mono">
                {report.report_text ?? 'No report text available.'}
              </pre>
            </div>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleDownloadTxt}
                className="px-5 py-2.5 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
              >
                Download TXT
              </button>
              <button
                type="button"
                onClick={handleDownloadJson}
                className="px-5 py-2.5 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
              >
                Download JSON
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
