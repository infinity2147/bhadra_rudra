import { useState, useEffect } from 'react';
import { fetchAPI, postAPI } from '../api';
import SeverityBadge from '../components/SeverityBadge';
import RegBadges from '../components/RegBadges';

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
  const [alerts, setAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState('');
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [alertsLoading, setAlertsLoading] = useState(true);

  useEffect(() => {
    fetchAPI('/api/alerts')
      .then(data => {
        const list = Array.isArray(data) ? data : data.alerts ?? [];
        setAlerts(list);
      })
      .catch(() => setAlerts([]))
      .finally(() => setAlertsLoading(false));
  }, []);

  async function handleGenerate() {
    if (!selectedAlert) return;
    setLoading(true);
    setReport(null);
    try {
      const data = await postAPI(`/api/sar/generate/${selectedAlert}`, {});
      setReport(data);
    } catch {
      setReport(null);
    } finally {
      setLoading(false);
    }
  }

  function handleDownloadTxt() {
    if (!report?.report_text) return;
    const filename = `SAR_${report.report_id ?? selectedAlert}_${Date.now()}.txt`;
    downloadFile(filename, report.report_text, 'text/plain');
  }

  function handleDownloadJson() {
    if (!report) return;
    const filename = `SAR_${report.report_id ?? selectedAlert}_${Date.now()}.json`;
    downloadFile(filename, JSON.stringify(report, null, 2), 'application/json');
  }

  const selectedAlertObj = alerts.find(
    (a) => (a.alert_id ?? a.id) === selectedAlert,
  );
  const regRefs = selectedAlertObj?.regulatory_refs;
  const hasRegRefs = Array.isArray(regRefs) && regRefs.length > 0;
  const showRegSection =
    selectedAlertObj &&
    (selectedAlertObj.fatf_code ||
      selectedAlertObj.legal_basis ||
      hasRegRefs ||
      selectedAlertObj.pmla_section ||
      selectedAlertObj.rbi_ref);

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-4xl">
        {/* Header */}
        <h1 className="text-2xl font-bold text-gray-900">SAR Report Generator</h1>
        <p className="text-sm text-gray-500 mt-1">Generate Suspicious Activity Reports from fraud alerts</p>

        {/* Alert Selector */}
        <div className="mt-6 bg-white border border-gray-200 rounded-xl p-5">
          <label className="block text-sm font-medium text-gray-700 mb-2">Select Alert</label>
          <div className="flex gap-3">
            <select
              data-testid="sar-alert-select"
              value={selectedAlert}
              onChange={e => { setSelectedAlert(e.target.value); setReport(null); }}
              className="flex-1 px-4 py-2.5 border border-gray-300 rounded-lg text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="">-- Choose an alert --</option>
              {alerts.map(alert => (
                <option key={alert.alert_id ?? alert.id} value={alert.alert_id ?? alert.id}>
                  {(alert.alert_id ?? alert.id)} — {alert.pattern_type ?? 'Unknown'} — {alert.severity ?? 'N/A'}
                </option>
              ))}
            </select>
            <button
              onClick={handleGenerate}
              disabled={!selectedAlert || loading}
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
          {alertsLoading && <p className="text-xs text-gray-400 mt-2">Loading alerts...</p>}
        </div>

        {/* Regulatory context for the selected alert (additive) */}
        {showRegSection && (
          <div className="mt-6 bg-white border border-gray-200 rounded-xl p-5">
            <div className="flex items-center justify-between gap-3 flex-wrap">
              <h2 className="text-lg font-bold text-gray-900">Regulatory Context</h2>
              <RegBadges alert={selectedAlertObj} />
            </div>
            {selectedAlertObj.legal_basis && (
              <p className="text-sm text-gray-700 mt-3">
                <span className="font-semibold">Legal basis:</span> {selectedAlertObj.legal_basis}
              </p>
            )}
            {(hasRegRefs || selectedAlertObj.pmla_section || selectedAlertObj.rbi_ref) && (
              <div className="mt-3">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Regulatory references</p>
                <ul className="mt-1.5 space-y-1 text-sm text-gray-700 list-disc list-inside">
                  {hasRegRefs && regRefs.map((ref, i) => <li key={i}>{ref}</li>)}
                  {selectedAlertObj.pmla_section && <li>PMLA: {selectedAlertObj.pmla_section}</li>}
                  {selectedAlertObj.rbi_ref && <li>RBI: {selectedAlertObj.rbi_ref}</li>}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Report Output */}
        {report && (
          <div className="mt-6 space-y-5">
            {/* Report Summary */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-lg font-bold text-gray-900 mb-4">Report Summary</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Report ID</p>
                  <p className="text-sm font-semibold mt-1">{report.report_id ?? '--'}</p>
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

            {/* Report Text */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <h2 className="text-lg font-bold text-gray-900 mb-3">Report Text</h2>
              <pre className="bg-gray-900 text-gray-100 rounded-lg p-5 text-xs leading-relaxed overflow-auto max-h-[500px] whitespace-pre-wrap font-mono">
                {report.report_text ?? 'No report text available.'}
              </pre>
            </div>

            {/* Download Buttons */}
            <div className="flex gap-3">
              <button
                onClick={handleDownloadTxt}
                className="px-5 py-2.5 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download TXT
              </button>
              <button
                onClick={handleDownloadJson}
                className="px-5 py-2.5 bg-white border border-gray-300 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-50 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Download JSON
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
