/**
 * App.jsx — Main application component
 * Wires together all components and manages global state
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { Bot } from 'lucide-react';

import DocumentPanel from './components/DocumentPanel';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import StatusBar from './components/StatusBar';
import { askQuestion, getDocuments, getHealth } from './services/api';

export default function App() {
  // ── State ──────────────────────────────────────────────────────
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const messagesEndRef = useRef(null);

  const loadDocuments = useCallback(async () => {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to load documents:', err);
    }
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealth(h);
    } catch (err) {
      console.error('Health check failed:', err);
    }
  }, []);

  // ── Load initial data ──────────────────────────────────────────
  // Define functions BEFORE useEffect that uses them
  useEffect(() => {
    loadDocuments();
    loadHealth();
  }, []);

  // ── Auto-scroll to latest message ─────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── Send a query ───────────────────────────────────────────────
  const handleSend = async (query) => {
    if (!query.trim() || isLoading) return;

    const userMessage = { role: 'user', content: query };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await askQuestion(query);

      const aiMessage = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        total_time: response.total_time,
        chunks_retrieved: response.chunks_retrieved,
        success: response.success,
      };
      setMessages(prev => [...prev, aiMessage]);

    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message || 'Request failed';
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${errorMsg}`,
        success: false,
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // ── Welcome screen ─────────────────────────────────────────────
  const WelcomeScreen = () => (
    <div className="flex flex-col items-center justify-center h-full gap-4 px-8">
      <div className="w-16 h-16 rounded-2xl bg-primary-600/20 flex items-center justify-center">
        <Bot size={32} className="text-primary-400" />
      </div>
      <div className="text-center">
        <h2 className="text-xl font-semibold text-slate-200 mb-2">
          Local RAG System
        </h2>
        <p className="text-sm text-slate-500 max-w-sm leading-relaxed">
          Upload PDF or DOCX documents on the left, then ask questions
          in natural language. All processing runs locally on your machine.
        </p>
      </div>
      {documents.length === 0 ? (
        <div className="mt-2 px-4 py-2 bg-slate-800 border border-slate-700
                        rounded-lg text-xs text-slate-500 text-center">
          👈 Start by uploading a document
        </div>
      ) : (
        <div className="mt-2 px-4 py-2 bg-primary-500/10 border border-primary-500/30
                        rounded-lg text-xs text-primary-400 text-center">
          {documents.length} document{documents.length > 1 ? 's' : ''} ready.
          Ask a question below!
        </div>
      )}
    </div>
  );

  // ── Render ─────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-screen bg-slate-900">

      {/* Top Header */}
      <header className="flex items-center gap-3 px-6 py-4
                          border-b border-slate-700/50 bg-slate-900">
        <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center">
          <Bot size={18} className="text-white" />
        </div>
        <div>
          <h1 className="text-base font-semibold text-slate-100">LocalRAG</h1>
          <p className="text-xs text-slate-500">AI Document Search — Offline</p>
        </div>
      </header>

      {/* Status Bar */}
      <StatusBar health={health} />

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">

        {/* Left Panel — Documents */}
        <aside className="w-72 border-r border-slate-700/50 bg-slate-900
                           flex flex-col overflow-hidden shrink-0">
          <DocumentPanel
            documents={documents}
            onDocumentsChange={loadDocuments}
          />
        </aside>

        {/* Right Panel — Chat */}
        <main className="flex flex-col flex-1 overflow-hidden">

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-6 py-4">
            {messages.length === 0
              ? <WelcomeScreen />
              : (
                <div className="flex flex-col gap-6 max-w-3xl mx-auto">
                  {messages.map((msg, idx) => (
                    <ChatMessage key={idx} message={msg} />
                  ))}

                  {/* Loading indicator */}
                  {isLoading && (
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded-full bg-slate-700
                                      flex items-center justify-center shrink-0">
                        <Bot size={16} className="text-primary-400" />
                      </div>
                      <div className="px-4 py-3 bg-slate-800 border border-slate-700
                                      rounded-2xl rounded-tl-sm">
                        <div className="flex gap-1 items-center h-5">
                          <div className="w-2 h-2 bg-primary-500 rounded-full
                                          animate-bounce [animation-delay:-0.3s]" />
                          <div className="w-2 h-2 bg-primary-500 rounded-full
                                          animate-bounce [animation-delay:-0.15s]" />
                          <div className="w-2 h-2 bg-primary-500 rounded-full
                                          animate-bounce" />
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )
            }
          </div>

          {/* Input Bar */}
          <ChatInput
            onSend={handleSend}
            disabled={isLoading || documents.length === 0}
            placeholder={
              documents.length === 0
                ? 'Upload a document first...'
                : 'Ask a question about your documents...'
            }
          />
        </main>
      </div>
    </div>
  );
}