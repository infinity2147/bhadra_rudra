import { Component } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Graph from './pages/Graph';
import Alerts from './pages/Alerts';
import Patterns from './pages/Patterns';
import Entities from './pages/Entities';
import Copilot from './pages/Copilot';
import SarReports from './pages/SarReports';

class ErrorBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div className="p-8">
          <h1 className="text-xl font-bold text-red-600 mb-2">React Error</h1>
          <pre className="bg-red-50 p-4 rounded text-sm overflow-auto whitespace-pre-wrap">{this.state.error.message}\n\n{this.state.error.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

const NAV = [
  { to: '/', label: 'Dashboard', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0a1 1 0 01-1-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1' },
  { to: '/graph', label: 'Fund Flow Graph', icon: 'M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1' },
  { to: '/alerts', label: 'Fraud Alerts', icon: 'M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9' },
  { to: '/patterns', label: 'Pattern Analysis', icon: 'M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z' },
  { to: '/entities', label: 'Entity Explorer', icon: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z' },
  { to: '/copilot', label: 'AI Copilot', icon: 'M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z' },
  { to: '/sar', label: 'SAR Reports', icon: 'M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
];

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
          <nav className="flex-1 p-2 space-y-0.5">
            {NAV.map(item => (
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
          </nav>
          <div className="p-3 border-t border-gray-200 text-center">
            <p className="text-[10px] text-gray-400">PSBs Hackathon 2026</p>
            <p className="text-[10px] text-gray-500 font-medium">Team Bhadra</p>
          </div>
        </aside>
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/graph" element={<Graph />} />
            <Route path="/alerts" element={<Alerts />} />
            <Route path="/patterns" element={<Patterns />} />
            <Route path="/entities" element={<Entities />} />
            <Route path="/copilot" element={<Copilot />} />
            <Route path="/sar" element={<SarReports />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
    </ErrorBoundary>
  );
}
