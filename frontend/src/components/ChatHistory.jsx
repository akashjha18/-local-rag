/**
 * ChatHistory.jsx — Sidebar showing past conversations
 */

import { Trash2, MessageSquare, Plus, Clock } from 'lucide-react';

export default function ChatHistory({
    sessions,
    currentSessionId,
    onSelectSession,
    onNewChat,
    onDeleteSession,
}) {
    // Format timestamp to readable string
    const formatTime = (isoString) => {
        const date = new Date(isoString);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    };

    return (
        <div className="flex flex-col h-full">

            {/* New Chat Button */}
            <div className="p-3 border-b border-slate-700/50">
                <button
                    onClick={onNewChat}
                    className="w-full flex items-center gap-2 px-3 py-2
                               bg-primary-600 hover:bg-primary-500
                               text-white text-sm font-medium rounded-lg
                               transition-colors"
                >
                    <Plus size={16} />
                    New Chat
                </button>
            </div>

            {/* Session List */}
            <div className="flex-1 overflow-y-auto p-2">
                {sessions.length === 0 ? (
                    <div className="text-center py-8">
                        <MessageSquare size={24} className="text-slate-600 mx-auto mb-2" />
                        <p className="text-xs text-slate-600">No chat history yet</p>
                    </div>
                ) : (
                    // Show most recent first
                    [...sessions].reverse().map((session) => (
                        <div
                            key={session.id}
                            onClick={() => onSelectSession(session)}
                            className={`
                                group flex items-start gap-2 p-2.5 rounded-lg
                                cursor-pointer mb-1 transition-colors
                                ${currentSessionId === session.id
                                    ? 'bg-primary-600/20 border border-primary-600/30'
                                    : 'hover:bg-slate-800 border border-transparent'
                                }
                            `}
                        >
                            {/* Chat icon */}
                            <MessageSquare
                                size={14}
                                className={`shrink-0 mt-0.5 ${
                                    currentSessionId === session.id
                                        ? 'text-primary-400'
                                        : 'text-slate-500'
                                }`}
                            />

                            {/* Session info */}
                            <div className="flex-1 min-w-0">
                                <p className={`text-xs font-medium truncate ${
                                    currentSessionId === session.id
                                        ? 'text-primary-300'
                                        : 'text-slate-300'
                                }`}>
                                    {session.title}
                                </p>
                                <div className="flex items-center gap-1 mt-0.5">
                                    <Clock size={9} className="text-slate-600" />
                                    <p className="text-xs text-slate-600">
                                        {formatTime(session.updatedAt)}
                                    </p>
                                    <span className="text-slate-700">·</span>
                                    <p className="text-xs text-slate-600">
                                        {Math.floor(session.messages.length / 2)} Q&A
                                    </p>
                                </div>
                            </div>

                            {/* Delete button */}
                            <button
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteSession(session.id);
                                }}
                                className="opacity-0 group-hover:opacity-100
                                           p-1 rounded hover:bg-red-500/20
                                           text-slate-500 hover:text-red-400
                                           transition-all shrink-0"
                            >
                                <Trash2 size={12} />
                            </button>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}