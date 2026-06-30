import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';

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
  const [streaming, setStreaming] = useState(false);
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
    setStreaming(false);

    try {
      // Call the backend directly to bypass the Vite proxy, which buffers
      // streaming responses and defeats the token-by-token effect.
      const streamUrl = import.meta.env.DEV
        ? 'http://localhost:8000/api/copilot/stream'
        : '/api/copilot/stream';
      const response = await fetch(streamUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-Role': 'INVESTIGATOR',
        },
        body: JSON.stringify({ query: trimmed }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assistantMsgAdded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // hold incomplete line for next chunk

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let data;
          try { data = JSON.parse(line.slice(6)); } catch { continue; }

          if (data.token) {
            if (!assistantMsgAdded) {
              setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
              assistantMsgAdded = true;
              setStreaming(true);
            }
            setMessages(prev => {
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              msgs[msgs.length - 1] = { ...last, content: last.content + data.token };
              return msgs;
            });
          }

          if (data.done) {
            setMode(data.mode ?? null);
            setModeLabel(data.mode_label ?? null);
            setFallbackReason(data.fallback_reason ?? null);
          }
        }
      }

      // If backend returned nothing (empty response), add fallback message
      if (!assistantMsgAdded) {
        setMessages(prev => [...prev, { role: 'assistant', content: 'No response received. Please try again.' }]);
      }
    } catch {
      setMessages(prev => {
        const msgs = [...prev];
        const last = msgs[msgs.length - 1];
        // If we already started streaming, append error to that message; otherwise add a new one
        if (last?.role === 'assistant') {
          msgs[msgs.length - 1] = { ...last, content: last.content || 'Error — please try again.' };
        } else {
          msgs.push({ role: 'assistant', content: 'Error — please try again.' });
        }
        return msgs;
      });
    } finally {
      setSending(false);
      setStreaming(false);
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
            {msg.role === 'assistant' && (
              <div className="w-7 h-7 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 shrink-0">R</div>
            )}
            <div
              className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'max-w-[60%] bg-indigo-600 text-white rounded-br-md'
                  : 'max-w-[85%] bg-white border border-gray-200 shadow-sm text-gray-800 rounded-bl-md'
              }`}
            >
              {msg.role === 'user' ? (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              ) : (
                <div className="prose prose-sm max-w-none
                  prose-p:my-1.5 prose-p:leading-relaxed
                  prose-ul:my-2 prose-ul:pl-4
                  prose-ol:my-2 prose-ol:pl-4
                  prose-li:my-1
                  prose-headings:font-semibold prose-headings:text-gray-900 prose-headings:my-2
                  prose-h3:text-sm prose-h2:text-base
                  prose-strong:text-gray-900 prose-strong:font-semibold
                  prose-code:bg-gray-100 prose-code:px-1 prose-code:rounded prose-code:text-xs
                  prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-pre:text-xs
                  prose-hr:border-gray-200 prose-hr:my-3">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                  {/* Blinking cursor while this message is still streaming */}
                  {streaming && i === messages.length - 1 && (
                    <span className="inline-block w-0.5 h-4 bg-indigo-500 ml-0.5 align-middle animate-pulse" />
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Thinking indicator — only before first token arrives */}
        {sending && !streaming && (
          <div className="flex justify-start">
            <div className="w-7 h-7 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-1 shrink-0">R</div>
            <div className="bg-white border border-gray-200 shadow-sm text-gray-500 rounded-2xl rounded-bl-md px-4 py-3 text-sm flex items-center gap-2">
              <span className="flex gap-1">
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </span>
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
