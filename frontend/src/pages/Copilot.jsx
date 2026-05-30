import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { postAPI } from '../api';

const QUICK_ACTIONS = [
  { label: 'Overview', query: 'Give me an overview of the current fraud detection status' },
  { label: 'High Risk', query: 'Which entities are at the highest risk right now?' },
  { label: 'Find Cycles', query: 'Find any circular fund flow patterns or cycles in the transaction network' },
  { label: 'Active Alerts', query: 'Summarize all currently active fraud alerts' },
];

function sourceLabel(source) {
  if (!source) return 'RUDRA AI Copilot';
  if (source.startsWith('local')) return 'Local fallback (real tools, no Gemini)';
  if (source.startsWith('gemini')) return 'Gemini 2.0 Flash + tools';
  return source;
}

export default function Copilot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [lastSource, setLastSource] = useState(null);
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
      setLastSource(data.source || null);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response ?? data.message ?? data.content ?? JSON.stringify(data),
        source: data.source,
        toolCalls: data.tool_calls || [],
      }]);
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
      <div className="p-6 pb-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{sourceLabel(lastSource)}</h1>
            <p className="text-sm text-gray-500 mt-1">
              Ask questions about fraud patterns, entities, and alerts. Tool results match the Journey and Cases APIs.
            </p>
          </div>
          {lastSource?.startsWith('local') && (
            <span className="shrink-0 text-xs px-2 py-1 rounded bg-amber-50 text-amber-800 border border-amber-200">
              No GEMINI_API_KEY — keyword router active
            </span>
          )}
        </div>
      </div>

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
                <>
                  <div className="prose prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:my-2 prose-pre:bg-gray-800 prose-pre:text-gray-100">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                  {msg.toolCalls?.length > 0 && (
                    <details className="mt-2 text-xs text-gray-500">
                      <summary className="cursor-pointer">{msg.toolCalls.length} tool call(s)</summary>
                      <ul className="mt-1 space-y-1 font-mono">
                        {msg.toolCalls.map((tc, j) => (
                          <li key={j}>{tc.tool}({JSON.stringify(tc.args || {})})</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </>
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
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
