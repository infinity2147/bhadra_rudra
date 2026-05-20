import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { fetchAPI } from '../api';

const NODE_COLORS = {
  individual: '#3b82f6',
  business: '#22c55e',
  shell_company: '#ef4444',
};

const EDGE_COLORS = {
  fraud: '#ef4444',
  normal: '#d1d5db',
};

function formatINR(value) {
  return `₹${Number(value).toLocaleString('en-IN')}`;
}

export default function Graph() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Controls
  const [fraudOnly, setFraudOnly] = useState(false);
  const [highRiskOnly, setHighRiskOnly] = useState(false);
  const [minAmount, setMinAmount] = useState(0);
  const [selectedEntity, setSelectedEntity] = useState('');

  // Node selection & detail panel
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetails, setNodeDetails] = useState(null);

  // Subgraph
  const [subgraphData, setSubgraphData] = useState(null);
  const [subgraphLoading, setSubgraphLoading] = useState(false);

  const mainGraphRef = useRef();
  const subGraphRef = useRef();

  // Build entity options from nodes
  const entityOptions = useMemo(() => {
    return graphData.nodes.map(n => ({ id: n.id, name: n.name || n.id }));
  }, [graphData.nodes]);

  // Fetch main graph
  useEffect(() => {
    setLoading(true);
    setError(null);
    const params = new URLSearchParams();
    if (fraudOnly) params.set('fraud_only', 'true');
    if (highRiskOnly) params.set('high_risk_only', 'true');
    if (minAmount > 0) params.set('min_amount', String(minAmount));

    const qs = params.toString();
    const url = `/api/graph${qs ? `?${qs}` : ''}`;

    fetchAPI(url)
      .then(data => {
        // Normalize: API may return { nodes, links } or { nodes, edges }
        const links = data.links || data.edges || [];
        setGraphData({ nodes: data.nodes || [], links });
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [fraudOnly, highRiskOnly, minAmount]);

  // Clear subgraph and selection when controls change
  useEffect(() => {
    setSelectedNode(null);
    setNodeDetails(null);
    setSubgraphData(null);
  }, [fraudOnly, highRiskOnly, minAmount]);

  const handleNodeClick = useCallback((node) => {
    setSelectedNode(node);
    // Build detail from node properties
    const inflow = (node.inflow || 0);
    const outflow = (node.outflow || 0);
    setNodeDetails({
      id: node.id,
      name: node.name || node.id,
      type: node.type || 'unknown',
      branch: node.branch || 'N/A',
      risk_score: node.risk_score ?? 'N/A',
      degree: node.degree ?? node.val ?? 'N/A',
      inflow,
      outflow,
    });
  }, []);

  const handleLoadSubgraph = useCallback(async (entityId) => {
    setSubgraphLoading(true);
    setSubgraphData(null);
    try {
      const data = await fetchAPI(`/api/graph/${entityId}`);
      const links = data.links || data.edges || [];
      setSubgraphData({ nodes: data.nodes || [], links });
    } catch (err) {
      console.error('Failed to load subgraph:', err);
    } finally {
      setSubgraphLoading(false);
    }
  }, []);

  // Custom node canvas object
  const paintNode = useCallback((node, ctx, globalScale) => {
    const label = node.name || node.id;
    const fontSize = Math.max(12 / globalScale, 2);
    const nodeSize = Math.max(4, Math.min(12, (node.val || 3)));

    // Draw circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeSize, 0, 2 * Math.PI);
    ctx.fillStyle = NODE_COLORS[node.type] || '#6366f1';
    ctx.fill();
    ctx.strokeStyle = selectedNode?.id === node.id ? '#1e1b4b' : 'rgba(255,255,255,0.8)';
    ctx.lineWidth = selectedNode?.id === node.id ? 2 / globalScale : 0.5 / globalScale;
    ctx.stroke();

    // Draw label
    ctx.font = `${fontSize}px Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#374151';
    ctx.fillText(label, node.x, node.y + nodeSize + 1);
  }, [selectedNode]);

  // Custom link canvas object
  const paintLink = useCallback((link, ctx, globalScale) => {
    const isFraud = link.is_fraud || link.fraud;
    const amount = link.amount || 1;
    const width = Math.max(0.5, Math.log(amount + 1) * 0.4);

    ctx.beginPath();
    ctx.moveTo(link.source.x, link.source.y);
    ctx.lineTo(link.target.x, link.target.y);
    ctx.strokeStyle = isFraud ? EDGE_COLORS.fraud : EDGE_COLORS.normal;
    ctx.lineWidth = width / globalScale;
    ctx.stroke();
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 pt-6 pb-2">
        <h1 className="text-2xl font-bold text-gray-900">Fund Flow Graph</h1>
        <p className="text-sm text-gray-500 mt-1">
          Interactive visualization of entity relationships and money flows
        </p>
      </div>

      {/* Controls */}
      <div className="px-6 py-3 flex items-center gap-6 flex-wrap border-b border-gray-200 bg-white">
        <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
          <input
            type="checkbox"
            checked={fraudOnly}
            onChange={e => setFraudOnly(e.target.checked)}
            className="rounded border-gray-300 text-red-600 focus:ring-red-500"
          />
          Fraud Only
        </label>

        <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
          <input
            type="checkbox"
            checked={highRiskOnly}
            onChange={e => setHighRiskOnly(e.target.checked)}
            className="rounded border-gray-300 text-amber-600 focus:ring-amber-500"
          />
          High Risk Only
        </label>

        <label className="flex items-center gap-2 text-sm text-gray-700">
          Min Amount
          <input
            type="number"
            value={minAmount}
            onChange={e => setMinAmount(Math.max(0, Number(e.target.value)))}
            min={0}
            step={100000}
            placeholder="0"
            className="w-28 rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          />
        </label>

        <label className="flex items-center gap-2 text-sm text-gray-700">
          Subgraph Entity
          <select
            value={selectedEntity}
            onChange={e => setSelectedEntity(e.target.value)}
            className="rounded-md border border-gray-300 px-2 py-1 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
          >
            <option value="">-- Select --</option>
            {entityOptions.map(opt => (
              <option key={opt.id} value={opt.id}>
                {opt.name}
              </option>
            ))}
          </select>
        </label>

        {selectedEntity && (
          <button
            onClick={() => handleLoadSubgraph(selectedEntity)}
            className="px-3 py-1 text-sm bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
          >
            Load Subgraph
          </button>
        )}

        {/* Legend */}
        <div className="ml-auto flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-blue-500 inline-block" /> Individual
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> Business
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> Shell Company
          </span>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex flex-1 min-h-0">
        {/* Graph Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-pulse text-gray-400 text-lg">Loading graph...</div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-red-600 text-lg">Error: {error}</div>
            </div>
          ) : (
            <div className="flex-1 relative">
              <ForceGraph2D
                ref={mainGraphRef}
                graphData={graphData}
                nodeVal="val"
                nodeCanvasObject={paintNode}
                nodeCanvasObjectMode={() => 'replace'}
                linkCanvasObject={paintLink}
                linkCanvasObjectMode={() => 'replace'}
                linkDirectionalArrowLength={3}
                linkDirectionalArrowRelPos={1}
                onNodeClick={handleNodeClick}
                backgroundColor="#fafafa"
                cooldownTicks={100}
                enableNodeDrag={true}
                enableZoomInteraction={true}
                enablePanInteraction={true}
              />
            </div>
          )}

          {/* Subgraph */}
          {(subgraphData || subgraphLoading) && (
            <div className="h-64 border-t border-gray-200 bg-white">
              <div className="px-4 py-2 border-b border-gray-100 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Subgraph View</h3>
                <button
                  onClick={() => setSubgraphData(null)}
                  className="text-xs text-gray-400 hover:text-gray-600"
                >
                  Close
                </button>
              </div>
              {subgraphLoading ? (
                <div className="flex items-center justify-center h-48">
                  <div className="animate-pulse text-gray-400 text-sm">Loading subgraph...</div>
                </div>
              ) : subgraphData ? (
                <ForceGraph2D
                  ref={subGraphRef}
                  graphData={subgraphData}
                  nodeVal="val"
                  nodeCanvasObject={paintNode}
                  nodeCanvasObjectMode={() => 'replace'}
                  linkCanvasObject={paintLink}
                  linkCanvasObjectMode={() => 'replace'}
                  linkDirectionalArrowLength={3}
                  linkDirectionalArrowRelPos={1}
                  backgroundColor="#ffffff"
                  cooldownTicks={50}
                  enableNodeDrag={true}
                  enableZoomInteraction={true}
                  enablePanInteraction={true}
                />
              ) : null}
            </div>
          )}
        </div>

        {/* Detail Panel */}
        {nodeDetails && (
          <div className="w-72 bg-white border-l border-gray-200 p-4 overflow-y-auto shrink-0">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-gray-900">Entity Details</h3>
              <button
                onClick={() => {
                  setSelectedNode(null);
                  setNodeDetails(null);
                }}
                className="text-gray-400 hover:text-gray-600 text-lg leading-none"
              >
                &times;
              </button>
            </div>

            <dl className="space-y-3 text-sm">
              <div>
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Name</dt>
                <dd className="text-gray-900 font-medium mt-0.5">{nodeDetails.name}</dd>
              </div>
              <div>
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Type</dt>
                <dd className="mt-0.5">
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                      nodeDetails.type === 'shell_company'
                        ? 'bg-red-100 text-red-700'
                        : nodeDetails.type === 'business'
                        ? 'bg-green-100 text-green-700'
                        : 'bg-blue-100 text-blue-700'
                    }`}
                  >
                    {nodeDetails.type.replace('_', ' ')}
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Branch</dt>
                <dd className="text-gray-900 mt-0.5">{nodeDetails.branch}</dd>
              </div>
              <div>
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Risk Score</dt>
                <dd className="text-gray-900 font-medium mt-0.5">{nodeDetails.risk_score}</dd>
              </div>
              <div>
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Connections</dt>
                <dd className="text-gray-900 mt-0.5">{nodeDetails.degree}</dd>
              </div>

              <div className="border-t border-gray-100 pt-3">
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Inflow</dt>
                <dd className="text-green-700 font-medium mt-0.5">
                  {typeof nodeDetails.inflow === 'number' ? formatINR(nodeDetails.inflow) : nodeDetails.inflow}
                </dd>
              </div>
              <div>
                <dt className="text-gray-500 text-xs uppercase tracking-wide">Outflow</dt>
                <dd className="text-red-700 font-medium mt-0.5">
                  {typeof nodeDetails.outflow === 'number' ? formatINR(nodeDetails.outflow) : nodeDetails.outflow}
                </dd>
              </div>
            </dl>

            <button
              onClick={() => handleLoadSubgraph(nodeDetails.id)}
              disabled={subgraphLoading}
              className="mt-5 w-full px-3 py-2 text-sm bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {subgraphLoading ? 'Loading...' : 'View Subgraph'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
