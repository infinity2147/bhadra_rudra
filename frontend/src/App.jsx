import { Component, useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Cases from './pages/Cases';
import Incidents from './pages/Incidents';
import Journey from './pages/Journey';
import Graph from './pages/Graph';
import Analytics from './pages/Analytics';
import Live from './pages/Live';
import ModelMetrics from './pages/ModelMetrics';
import Patterns from './pages/Patterns';
import Entities from './pages/Entities';
import Copilot from './pages/Copilot';
import SarReports from './pages/SarReports';
import Settings from './pages/Settings';
import AccountAggregator from './pages/AccountAggregator';
import SimulationStudio from './pages/SimulationStudio';
import GeoMap from './pages/GeoMap';
import Replay from './pages/Replay';
import Collusion from './pages/Collusion';
import { getRole, setRole } from './api';

class ErrorBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div className="p-8">
          <h1 className="text-xl font-bold text-red-600 mb-2">React Error</h1>
          <pre className="bg-red-50 p-4 rounded text-sm overflow-auto whitespace-pre-wrap">
            {this.state.error.message}{'\n\n'}{this.state.error.stack}
          </pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const ICONS = {
  dashboard: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1',
  incidents: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
  cases: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2',
  journey: 'M13 7l5 5m0 0l-5 5m5-5H6',
  graph: 'M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1',
  analytics: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
  live: 'M13 10V3L4 14h7v7l9-11h-7z',
  model: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z',
  patterns: 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z',
  entities: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z',
  copilot: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z',
  sar: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z',
  settings: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065zM15 12a3 3 0 11-6 0 3 3 0 016 0z',
  aa: 'M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4',
  simulate: 'M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5',
  map: 'M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7',
  replay: 'M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
  collusion: 'M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z',
};

const NAV_GROUPS = [
  {
    title: 'Investigate',
    items: [
      { to: '/', label: 'Dashboard', icon: ICONS.dashboard },
      { to: '/incidents', label: 'Incidents', icon: ICONS.incidents },
      { to: '/cases', label: 'Case Workbench', icon: ICONS.cases },
      { to: '/journey', label: 'Fund Journey', icon: ICONS.journey },
      { to: '/replay', label: 'Temporal Replay', icon: ICONS.replay },
      { to: '/graph', label: 'Network Graph', icon: ICONS.graph },
    ],
  },
  {
    title: 'Analyse',
    items: [
      { to: '/analytics', label: 'Channel / Branch', icon: ICONS.analytics },
      { to: '/map', label: 'Geo Map', icon: ICONS.map },
      { to: '/patterns', label: 'Pattern Library', icon: ICONS.patterns },
      { to: '/entities', label: 'Entity Explorer', icon: ICONS.entities },
      { to: '/collusion', label: 'Collusion Rings', icon: ICONS.collusion },
      { to: '/model', label: 'ML Models', icon: ICONS.model },
    ],
  },
  {
    title: 'Act',
    items: [
      { to: '/live', label: 'Live Stream', icon: ICONS.live },
      { to: '/simulate', label: 'Simulation Studio', icon: ICONS.simulate },
      { to: '/copilot', label: 'AI Copilot', icon: ICONS.copilot },
      { to: '/sar', label: 'SAR Reports', icon: ICONS.sar },
    ],
  },
  {
    title: 'Integrate',
    items: [
      { to: '/aa', label: 'Account Aggregator', icon: ICONS.aa },
      { to: '/settings', label: 'Detector Settings', icon: ICONS.settings },
    ],
  },
];

const ROLES = ['INVESTIGATOR', 'SUPERVISOR', 'ADMIN'];

function RoleSwitcher() {
  const [role, setLocalRole] = useState(getRole());
  function change(r) {
    setRole(r);
    setLocalRole(r);
    // Reload to refetch under the new role's permissions
    window.location.reload();
  }
  return (
    <div className="px-3 py-2 border-t border-gray-200">
      <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Role</p>
      <select
        data-testid="role-switcher"
        value={role}
        onChange={(e) => change(e.target.value)}
        className="w-full text-xs px-2 py-1.5 border border-gray-300 rounded bg-white"
      >
        {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
      </select>
      <p className="text-[10px] text-gray-400 mt-1">Demo gate — production uses IDP</p>
    </div>
  );
}

function NotFound() {
  return (
    <div className="p-12 text-center">
      <h1 className="text-xl font-bold text-gray-900">Page not found</h1>
      <p className="text-sm text-gray-500 mt-2">This route is not registered in the RUDRA app.</p>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="flex h-screen bg-gray-50 text-gray-900">
          <aside className="w-60 bg-white border-r border-gray-200 flex flex-col shrink-0">
            <div className="p-4 border-b border-gray-200">
              <h1 className="text-xl font-bold text-indigo-900">RUDRA</h1>
              <p className="text-xs text-gray-500 mt-0.5">Shield Against Deception</p>
              <p className="text-[10px] text-gray-400">Fund Flow Intelligence System</p>
            </div>
            <nav className="flex-1 overflow-y-auto py-2 space-y-3">
              {NAV_GROUPS.map(group => (
                <div key={group.title} className="px-2">
                  <p className="px-2 mb-1 text-[10px] font-semibold text-gray-400 uppercase tracking-wider">
                    {group.title}
                  </p>
                  {group.items.map(item => (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.to === '/'}
                      className={({ isActive }) =>
                        `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                          isActive ? 'bg-indigo-50 text-indigo-700 font-medium' : 'text-gray-600 hover:bg-gray-100'
                        }`
                      }
                    >
                      <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
                      </svg>
                      {item.label}
                    </NavLink>
                  ))}
                </div>
              ))}
            </nav>
            <RoleSwitcher />
            <div className="p-3 border-t border-gray-200 text-center">
              <p className="text-[10px] text-gray-400">PSBs Hackathon 2026</p>
              <p className="text-[10px] text-gray-500 font-medium">Team Bhadra</p>
            </div>
          </aside>
          <main className="flex-1 min-h-0 overflow-hidden">
            <div className="h-full overflow-y-auto overflow-x-hidden">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/incidents" element={<Incidents />} />
              <Route path="/cases" element={<Cases />} />
              <Route path="/journey" element={<Journey />} />
              <Route path="/graph" element={<Graph />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/patterns" element={<Patterns />} />
              <Route path="/entities" element={<Entities />} />
              <Route path="/model" element={<ModelMetrics />} />
              <Route path="/live" element={<Live />} />
              <Route path="/copilot" element={<Copilot />} />
              <Route path="/sar" element={<SarReports />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/aa" element={<AccountAggregator />} />
              <Route path="/simulate" element={<SimulationStudio />} />
              <Route path="/map" element={<GeoMap />} />
              <Route path="/replay" element={<Replay />} />
              <Route path="/collusion" element={<Collusion />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
