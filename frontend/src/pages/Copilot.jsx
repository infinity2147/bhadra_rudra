import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { postAPI } from '../api';

const QUICK_ACTIONS = [
  { label: 'Overview', query: 'Give me an overview of the current fraud detection status' },
  { label: 'High Risk', query: 'Which entities are at the highest risk right now?' },
  { label: 'Find Cycles', query: 'Find any circular fund flow patterns or cycles in the transaction network' },
  { label: 'Active Alerts', query: 'Summarize all currently active fraud alerts' },
];

export default function Copilot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  // Track the most recent response's mode so the header can switch between
  // "AI Copilot (Claude)" and "Quick Commands (no LLM)" honestly — we don't
  // claim AI when the backend is doing keyword routing.
  const [mode, setMode] = useState(null);
  const [modeLabel, setModeLabel] = useState(null);
  const [fallbackReason, setFallbackReason] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function sendQuery(query) {
    const trimmed = query.trim();
    if (!trimmed || sending) return;

    setMessages(prev => [...prev, { role: 'user', content: trimmed }]);
    setInput('');
    setSending(true);

    try {
      const data = await postAPI('/api/copilot/query', { query: trimmed });
      setMessages(prev => [...prev, { role: 'assistant', content: data.response ?? data.message ?? data.content ?? JSON.stringify(data) }]);
      setMode(data.mode ?? null);
      setModeLabel(data.mode_label ?? null);
      setFallbackReason(data.fallback_reason ?? null);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error processing your request. Please try again.' }]);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    sendQuery(input);
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 pb-4">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-gray-900">
            {mode === 'quick_commands' ? 'RUDRA Quick Commands' : 'RUDRA AI Copilot'}
          </h1>
          {modeLabel && (
            <span
              className={`text-xs px-2 py-1 rounded font-medium ${
                mode === 'ai_copilot'
                  ? 'bg-indigo-100 text-indigo-700'
                  : 'bg-amber-100 text-amber-700'
              }`}
              title={fallbackReason || ''}
            >
              {modeLabel}
            </span>
          )}
        </div>
        <p className="text-sm text-gray-500 mt-1">
          {mode === 'quick_commands'
            ? 'Keyword-routed quick commands — set ANTHROPIC_API_KEY to enable LLM-driven natural-language understanding.'
            : 'Ask questions about fraud patterns, entities, and alerts in natural language.'}
        </p>
      </div>

      {/* Quick Actions */}
      <div className="px-6 pb-4 flex gap-2 flex-wrap">
        {QUICK_ACTIONS.map(action => (
          <button
            key={action.label}
            onClick={() => sendQuery(action.query)}
            disabled={sending}
            className="px-4 py-2 text-sm font-medium rounded-lg border border-indigo-200 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {action.label}
          </button>
        ))}
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-6 pb-4 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
            <svg className="w-16 h-16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <p className="text-sm">Ask a question or use a quick action to get started</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-br-md'
                  : 'bg-gray-100 text-gray-800 rounded-bl-md'
              }`}
            >
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              ) : (
                <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:my-2 prose-pre:bg-gray-800 prose-pre:text-gray-100">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}

        {sending && (
          <div className="flex justify-start">
            <div className="bg-gray-100 text-gray-500 rounded-2xl rounded-bl-md px-4 py-3 text-sm flex items-center gap-2">
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Thinking...
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input Row */}
      <form onSubmit={handleSubmit} className="px-6 py-4 border-t border-gray-200 bg-white">
        <div className="flex gap-3">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about fraud patterns, entities, alerts..."
            className="flex-1 px-4 py-2.5 border border-gray-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            disabled={sending}
          />
          <button
            type="submit"
            disabled={sending || !input.trim()}
            className="px-5 py-2.5 bg-indigo-600 text-white text-sm font-medium rounded-xl hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
