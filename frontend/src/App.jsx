/**
 * App.jsx — Main application with chat history
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { Bot, History, X, Trash2 } from 'lucide-react';

import DocumentPanel from './components/DocumentPanel';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import StatusBar from './components/StatusBar';
import ChatHistory from './components/ChatHistory';
import {
    loadChatSessions,
    createSession,
    saveSession,
    deleteSession,
    clearAllHistory,
    generateTitle,
} from './services/chatHistory';
import { askQuestion, getDocuments, getHealth } from './services/api';

export default function App() {
    // ── Core State ─────────────────────────────────────────────
    const [documents, setDocuments] = useState([]);
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [health, setHealth] = useState(null);
    const messagesEndRef = useRef(null);

    // ── Chat History State ──────────────────────────────────────
    const [sessions, setSessions] = useState([]);
    const [currentSession, setCurrentSession] = useState(null);
    const [showHistory, setShowHistory] = useState(false);

    // ── Initialize ──────────────────────────────────────────────
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

    useEffect(() => {
        loadDocuments();
        loadHealth();
        // Load chat history from localStorage
        const saved = loadChatSessions();
        setSessions(saved);
        // Start with a fresh session
        const session = createSession();
        setCurrentSession(session);
    }, []);

    // ── Auto-scroll ─────────────────────────────────────────────
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // ── Save session whenever messages change ───────────────────
    useEffect(() => {
        if (!currentSession || messages.length === 0) return;

        const firstUserMsg = messages.find(m => m.role === 'user');
        const title = generateTitle(firstUserMsg);

        saveSession(currentSession.id, messages, title);

        // Refresh session list
        setSessions(loadChatSessions());
    }, [messages, currentSession]);

    // ── New Chat ────────────────────────────────────────────────
    const handleNewChat = () => {
        const session = createSession();
        setCurrentSession(session);
        setMessages([]);
        setShowHistory(false);
    };

    // ── Load Session ────────────────────────────────────────────
    const handleSelectSession = (session) => {
        setCurrentSession(session);
        setMessages(session.messages || []);
        setShowHistory(false);
    };

    // ── Delete Session ──────────────────────────────────────────
    const handleDeleteSession = (sessionId) => {
        deleteSession(sessionId);
        setSessions(loadChatSessions());

        // If deleting current session, start new chat
        if (currentSession?.id === sessionId) {
            handleNewChat();
        }
    };

    // ── Clear All History ───────────────────────────────────────
    const handleClearHistory = () => {
        if (!confirm('Clear all chat history? This cannot be undone.')) return;
        clearAllHistory();
        setSessions([]);
        handleNewChat();
    };

    // ── Send Query ──────────────────────────────────────────────
    const handleSend = async (query) => {
        if (!query.trim() || isLoading) return;

        const userMessage = { role: 'user', content: query };
        const updatedMessages = [...messages, userMessage];
        setMessages(updatedMessages);
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
            const errorMsg = err.response?.data?.detail
                || err.message
                || 'Request failed';
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Error: ${errorMsg}`,
                success: false,
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    // ── Welcome Screen ──────────────────────────────────────────
    const WelcomeScreen = () => (
        <div className="flex flex-col items-center justify-center h-full gap-4 px-8">
            <div className="w-16 h-16 rounded-2xl bg-primary-600/20
                            flex items-center justify-center">
                <Bot size={32} className="text-primary-400" />
            </div>
            <div className="text-center">
                <h2 className="text-xl font-semibold text-slate-200 mb-2">
                    Local RAG System
                </h2>
                <p className="text-sm text-slate-500 max-w-sm leading-relaxed">
                    Upload PDF or DOCX documents on the left, then ask questions
                    in natural language. All processing runs locally.
                </p>
            </div>
            {documents.length === 0 ? (
                <div className="px-4 py-2 bg-slate-800 border border-slate-700
                                rounded-lg text-xs text-slate-500 text-center">
                    👈 Start by uploading a document
                </div>
            ) : (
                <div className="px-4 py-2 bg-primary-500/10 border border-primary-500/30
                                rounded-lg text-xs text-primary-400 text-center">
                    {documents.length} document{documents.length > 1 ? 's' : ''} ready —
                    ask a question below!
                </div>
            )}
        </div>
    );

    // ── Render ──────────────────────────────────────────────────
    return (
        <div className="flex flex-col h-screen bg-slate-900">

            {/* Header */}
            <header className="flex items-center gap-3 px-6 py-3
                                border-b border-slate-700/50 bg-slate-900 shrink-0">
                <div className="w-8 h-8 rounded-lg bg-primary-600
                                flex items-center justify-center">
                    <Bot size={18} className="text-white" />
                </div>
                <div className="flex-1">
                    <h1 className="text-base font-semibold text-slate-100">
                        LocalRAG
                    </h1>
                    <p className="text-xs text-slate-500">
                        AI Document Search — Offline
                    </p>
                </div>

                {/* History toggle button */}
                <button
                    onClick={() => setShowHistory(!showHistory)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg
                                text-xs transition-colors
                                ${showHistory
                                    ? 'bg-primary-600/20 text-primary-400 border border-primary-600/30'
                                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                                }`}
                >
                    <History size={14} />
                    <span>History</span>
                    {sessions.length > 0 && (
                        <span className="bg-slate-700 text-slate-300 px-1.5 py-0.5
                                         rounded-full text-xs">
                            {sessions.length}
                        </span>
                    )}
                </button>
            </header>

            {/* Status Bar */}
            <StatusBar health={health} />

            {/* Main Content */}
            <div className="flex flex-1 overflow-hidden">

                {/* Left Panel — Documents */}
                <aside className="w-64 border-r border-slate-700/50 bg-slate-900
                                   flex flex-col overflow-hidden shrink-0">
                    <DocumentPanel
                        documents={documents}
                        onDocumentsChange={loadDocuments}
                    />
                </aside>

                {/* Center Panel — Chat */}
                <main className="flex flex-col flex-1 overflow-hidden relative">

                    {/* Messages area */}
                    <div className="flex-1 overflow-y-auto px-6 py-4">
                        {messages.length === 0
                            ? <WelcomeScreen />
                            : (
                                <div className="flex flex-col gap-6 max-w-3xl mx-auto">
                                    {messages.map((msg, idx) => (
                                        <ChatMessage key={idx} message={msg} />
                                    ))}

                                    {/* Loading dots */}
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

                    {/* Input bar */}
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

                {/* Right Panel — Chat History (slide in) */}
                {showHistory && (
                    <aside className="w-64 border-l border-slate-700/50 bg-slate-900
                                       flex flex-col overflow-hidden shrink-0">

                        {/* History header */}
                        <div className="flex items-center justify-between
                                        px-4 py-3 border-b border-slate-700/50">
                            <div className="flex items-center gap-2">
                                <History size={14} className="text-slate-400" />
                                <span className="text-sm font-medium text-slate-300">
                                    History
                                </span>
                            </div>
                            <div className="flex items-center gap-1">
                                {sessions.length > 0 && (
                                    <button
                                        onClick={handleClearHistory}
                                        className="p-1.5 rounded hover:bg-red-500/20
                                                   text-slate-500 hover:text-red-400
                                                   transition-colors"
                                        title="Clear all history"
                                    >
                                        <Trash2 size={13} />
                                    </button>
                                )}
                                <button
                                    onClick={() => setShowHistory(false)}
                                    className="p-1.5 rounded hover:bg-slate-800
                                               text-slate-500 hover:text-slate-300
                                               transition-colors"
                                >
                                    <X size={13} />
                                </button>
                            </div>
                        </div>

                        {/* History list */}
                        <div className="flex-1 overflow-hidden">
                            <ChatHistory
                                sessions={sessions}
                                currentSessionId={currentSession?.id}
                                onSelectSession={handleSelectSession}
                                onNewChat={handleNewChat}
                                onDeleteSession={handleDeleteSession}
                            />
                        </div>
                    </aside>
                )}
            </div>
        </div>
    );
}