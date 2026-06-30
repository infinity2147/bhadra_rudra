import { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { getCollusionRings } from '../api';

function RingGraph({ ring }) {
  const nodes = ring.account_ids.map((id) => ({ id, kind: 'account' }));
  const links = [];
  (ring.shared_identifiers || []).forEach((s) => {
    const hubId = `${s.type}:${s.value}`;
    nodes.push({ id: hubId, kind: 'identifier' });
    ring.account_ids.forEach((a) => links.push({ source: a, target: hubId }));
  });
  return (
    <div className="h-56 border border-gray-200 rounded bg-gray-50">
      <ForceGraph2D
        graphData={{ nodes, links }}
        width={420}
        height={220}
        nodeRelSize={5}
        nodeLabel="id"
        nodeColor={(n) => (n.kind === 'identifier' ? '#dc2626' : '#4f46e5')}
        linkColor={() => '#cbd5e1'}
      />
    </div>
  );
}

export default function Collusion() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    getCollusionRings().then(setData).catch(() => setErr('Failed to load collusion rings'));
  }, []);

  if (err) return <div className="p-6 text-red-600">{err}</div>;
  if (!data) return <div className="p-6 text-gray-500">Loading…</div>;

  return (
    <div className="p-6 space-y-4">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Collusion Rings</h2>
        <p className="text-sm text-gray-600">
          Accounts secretly linked by a shared device/IP or KYC document — caught even when
          they never transact with each other.
        </p>
      </div>
      <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
        Synthetic identity demo — this lane runs on a generated identity dataset
        ({data.n_accounts} accounts). The rest of RUDRA runs on the IBM AML dataset.
      </div>

      {data.rings.length === 0 && (
        <div className="text-gray-500">No collusion rings detected.</div>
      )}

      <div className="space-y-6">
        {data.rings.map((ring) => (
          <div key={ring.ring_id} className="border border-gray-200 rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">{ring.ring_id}</h3>
              <span className="text-xs text-gray-500">{ring.size} accounts</span>
            </div>
            <ul className="mt-1 text-sm text-gray-700 list-disc ml-5">
              {ring.shared_identifiers.map((s, i) => (
                <li key={i}>
                  {s.count} accounts share <span className="font-mono">{s.type}</span> ={' '}
                  <span className="font-mono">{s.value}</span>
                </li>
              ))}
            </ul>
            <div className="mt-3">
              <RingGraph ring={ring} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
