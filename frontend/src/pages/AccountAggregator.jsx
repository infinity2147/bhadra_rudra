import { useState, useEffect } from 'react';
import { fetchAPI, postAPI } from '../api';

function formatINR(n) {
  if (n == null) return '--';
  return `₹${Number(n).toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
}

export default function AccountAggregator() {
  const [consents, setConsents] = useState([]);
  const [activeHandle, setActiveHandle] = useState(null);
  const [pulled, setPulled] = useState(null);
  const [creating, setCreating] = useState(false);
  const [pulling, setPulling] = useState(false);
  const [customer, setCustomer] = useState('CUST-DEMO-001');
  const [fips, setFips] = useState('FIP-HDFC,FIP-AXIS');
  const [purpose, setPurpose] = useState('103');

  // KYC screen
  const [screenName, setScreenName] = useState('Thunder Bolt Exports');
  const [screenType, setScreenType] = useState('shell_company');
  const [screenResult, setScreenResult] = useState(null);
  const [screening, setScreening] = useState(false);

  function refresh() {
    fetchAPI('/api/aa/consents').then((d) => setConsents(d.consents || []));
  }

  useEffect(refresh, []);

  async function createConsent() {
    setCreating(true);
    try {
      const consent = await postAPI('/api/aa/consent', {
        customer_id: customer,
        fip_ids: fips.split(',').map((s) => s.trim()).filter(Boolean),
        purpose_code: purpose,
        duration_days: 30,
      });
      setActiveHandle(consent.consent_handle);
      refresh();
    } finally {
      setCreating(false);
    }
  }

  async function pullData(handle) {
    setPulling(true); setActiveHandle(handle); setPulled(null);
    try {
      const d = await fetchAPI(`/api/aa/pull/${handle}?days_back=30`);
      setPulled(d);
    } finally {
      setPulling(false);
    }
  }

  async function revoke(handle) {
    await postAPI(`/api/aa/revoke/${handle}`, {});
    if (activeHandle === handle) setActiveHandle(null);
    refresh();
  }

  async function runScreen() {
    setScreening(true); setScreenResult(null);
    try {
      const params = new URLSearchParams({ name: screenName, entity_type: screenType });
      const r = await fetchAPI(`/api/kyc/screen?${params.toString()}`);
      setScreenResult(r);
    } finally {
      setScreening(false);
    }
  }

  return (
    <div className="p-6 max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">DPI Integrations</h1>
        <p className="text-sm text-gray-500 mt-1">
          Account Aggregator (consent-based financial data pull) + DiliSense KYC enrichment. Both are mocks — production calls the real Sahamati-licensed AA and DiliSense / Refinitiv APIs.
        </p>
      </div>

      {/* AA section */}
      <section className="bg-white border border-gray-200 rounded-xl p-5">
        <h2 className="text-lg font-semibold text-gray-900">Account Aggregator</h2>
        <p className="text-xs text-gray-500 mt-1">
          Issue a consent, pull data from FIPs, revoke when done. Real AA uses signed consent artefacts.
        </p>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
          <label className="text-sm">
            Customer ID
            <input
              value={customer} onChange={(e) => setCustomer(e.target.value)}
              className="mt-1 w-full px-3 py-1.5 border border-gray-300 rounded"
            />
          </label>
          <label className="text-sm">
            FIPs (comma sep)
            <input
              value={fips} onChange={(e) => setFips(e.target.value)}
              className="mt-1 w-full px-3 py-1.5 border border-gray-300 rounded"
            />
          </label>
          <label className="text-sm">
            Purpose
            <select
              value={purpose} onChange={(e) => setPurpose(e.target.value)}
              className="mt-1 w-full px-3 py-1.5 border border-gray-300 rounded bg-white"
            >
              <option value="101">101 Wealth management</option>
              <option value="102">102 Customer spending pattern</option>
              <option value="103">103 Aggregated statement</option>
              <option value="104">104 Explicit one-time access</option>
              <option value="105">105 Account verification</option>
            </select>
          </label>
        </div>

        <button
          onClick={createConsent}
          disabled={creating}
          className="mt-4 px-4 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
        >
          {creating ? 'Issuing...' : 'Issue Consent'}
        </button>

        {/* Consents list */}
        <div className="mt-6 space-y-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">Active Consents ({consents.length})</h3>
          {consents.length === 0 ? (
            <p className="text-sm text-gray-400">No consents yet. Issue one above.</p>
          ) : (
            consents.map((c) => (
              <div key={c.consent_handle} className="rounded-lg border border-gray-200 p-3 text-sm">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-xs text-gray-500">{c.consent_handle}</span>
                  <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                    c.status === 'ACTIVE' ? 'bg-emerald-100 text-emerald-700' : 'bg-gray-100 text-gray-600'
                  }`}>{c.status}</span>
                  <span className="text-xs text-gray-500">{c.purpose_description}</span>
                  <span className="ml-auto flex gap-2">
                    <button
                      onClick={() => pullData(c.consent_handle)}
                      disabled={c.status !== 'ACTIVE'}
                      className="text-xs px-2 py-1 bg-indigo-50 text-indigo-700 rounded hover:bg-indigo-100 disabled:opacity-50"
                    >
                      Pull data
                    </button>
                    {c.status === 'ACTIVE' && (
                      <button
                        onClick={() => revoke(c.consent_handle)}
                        className="text-xs px-2 py-1 bg-red-50 text-red-700 rounded hover:bg-red-100"
                      >
                        Revoke
                      </button>
                    )}
                  </span>
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  customer={c.customer_id} • fips={c.fip_ids.join(', ')} • expires {c.expires_at?.slice(0, 10)}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Pulled data */}
        {pulled && (
          <div className="mt-6 border border-gray-200 rounded-lg overflow-hidden">
            <div className="px-4 py-2 bg-gray-50 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold">Pulled data — {pulled.transaction_count} txns</h4>
                <p className="text-[11px] text-gray-500">{pulled._mock_disclaimer}</p>
              </div>
            </div>
            <div className="max-h-72 overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="bg-gray-50 sticky top-0">
                  <tr className="text-left text-gray-500">
                    <th className="px-3 py-1.5">Time</th>
                    <th className="px-3 py-1.5">Type</th>
                    <th className="px-3 py-1.5 text-right">Amount</th>
                    <th className="px-3 py-1.5">Narration</th>
                    <th className="px-3 py-1.5 text-right">Balance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {(pulled.transactions || []).map((t) => (
                    <tr key={t.txn_id}>
                      <td className="px-3 py-1 font-mono text-gray-500 whitespace-nowrap">{t.timestamp}</td>
                      <td className={`px-3 py-1 ${t.type === 'CREDIT' ? 'text-emerald-700' : 'text-rose-700'}`}>{t.type}</td>
                      <td className="px-3 py-1 text-right font-medium">{formatINR(t.amount)}</td>
                      <td className="px-3 py-1 text-gray-700">{t.narration}</td>
                      <td className="px-3 py-1 text-right text-gray-500">{formatINR(t.balance_after)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* KYC screen */}
      <section className="bg-white border border-gray-200 rounded-xl p-5">
        <h2 className="text-lg font-semibold text-gray-900">DiliSense KYC Screen</h2>
        <p className="text-xs text-gray-500 mt-1">
          Sanctions / PEP / adverse-media check. Deterministic per name in mock mode — same query → same result.
        </p>
        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="text-sm">
            Name
            <input
              value={screenName} onChange={(e) => setScreenName(e.target.value)}
              className="mt-1 w-72 px-3 py-1.5 border border-gray-300 rounded"
            />
          </label>
          <label className="text-sm">
            Entity Type
            <select
              value={screenType} onChange={(e) => setScreenType(e.target.value)}
              className="mt-1 w-48 px-3 py-1.5 border border-gray-300 rounded bg-white"
            >
              <option value="individual">individual</option>
              <option value="business">business</option>
              <option value="shell_company">shell_company</option>
            </select>
          </label>
          <button
            onClick={runScreen} disabled={screening}
            className="px-4 py-1.5 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50"
          >
            {screening ? 'Screening...' : 'Run Screen'}
          </button>
        </div>

        {screenResult && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="rounded-lg p-4 bg-rose-50 border border-rose-200">
              <p className="text-[11px] uppercase text-rose-700">Risk</p>
              <p className="text-2xl font-bold text-rose-900">{screenResult.risk}</p>
              <p className="text-xs text-rose-700 mt-1">score: {screenResult.risk_score_0_100}/100</p>
            </div>
            <div className="md:col-span-2 rounded-lg p-4 bg-gray-50 border border-gray-200">
              <p className="text-[11px] uppercase text-gray-700 mb-2">Hits</p>
              {(screenResult.hits || []).length === 0 ? (
                <p className="text-sm text-gray-500">No hits found.</p>
              ) : (
                <ul className="space-y-1.5">
                  {screenResult.hits.map((h, i) => (
                    <li key={i} className="text-sm">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-100 text-red-700 mr-2">
                        {h.type}
                      </span>
                      {h.source} — <span className="text-gray-600">{h.notes}</span>
                    </li>
                  ))}
                </ul>
              )}
              <p className="text-[10px] text-gray-400 mt-3">{screenResult._mock_disclaimer}</p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
