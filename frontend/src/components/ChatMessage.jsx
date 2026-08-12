/**
 * ChatMessage.jsx — Single message bubble with copy button
 */

import { useState } from 'react';
import { User, Bot, Clock, FileText, AlertCircle, Copy, Check } from 'lucide-react';

export default function ChatMessage({ message }) {
    const isUser = message.role === 'user';
    const isError = message.success === false;
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(message.content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Copy failed:', err);
        }
    };

    // Color code confidence scores
    const getConfidenceColor = (percent) => {
        if (percent >= 70) return 'text-green-400 bg-green-400/10 border-green-400/30';
        if (percent >= 50) return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30';
        if (percent >= 30) return 'text-orange-400 bg-orange-400/10 border-orange-400/30';
        return 'text-slate-400 bg-slate-700/50 border-slate-600';
    };

    return (
        <div className={`flex gap-3 group ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>

            {/* Avatar */}
            <div className={`
                shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                ${isUser
                    ? 'bg-primary-600'
                    : isError
                        ? 'bg-red-500/20'
                        : 'bg-slate-700'
                }
            `}>
                {isUser
                    ? <User size={16} className="text-white" />
                    : <Bot size={16} className={isError ? 'text-red-400' : 'text-primary-400'} />
                }
            </div>

            {/* Message content */}
            <div className={`
                max-w-[80%] flex flex-col gap-2
                ${isUser ? 'items-end' : 'items-start'}
            `}>

                {/* Text bubble */}
                <div className={`
                    relative px-4 py-3 rounded-2xl text-sm leading-relaxed
                    ${isUser
                        ? 'bg-primary-600 text-white rounded-tr-sm'
                        : isError
                            ? 'bg-red-500/10 border border-red-500/30 text-red-300 rounded-tl-sm'
                            : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-tl-sm'
                    }
                `}>
                    {/* Error badge */}
                    {isError && (
                        <div className="flex items-center gap-2 mb-2">
                            <AlertCircle size={14} className="text-red-400" />
                            <span className="text-xs text-red-400 font-medium">Error</span>
                        </div>
                    )}

                    <p className="whitespace-pre-wrap">{message.content}</p>

                    {/* Copy button for AI messages */}
                    {!isUser && (
                        <button
                            onClick={handleCopy}
                            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100
                                       p-1 rounded bg-slate-700/50 hover:bg-slate-600
                                       text-slate-400 hover:text-slate-200
                                       transition-all"
                            title="Copy answer"
                        >
                            {copied
                                ? <Check size={12} className="text-green-400" />
                                : <Copy size={12} />
                            }
                        </button>
                    )}
                </div>

                {/* Sources */}
                {!isUser && message.sources && message.sources.length > 0 && (
                    <div className="flex flex-col gap-1.5 w-full">
                        <p className="text-xs text-slate-500 font-medium px-1">
                            Sources:
                        </p>
                        <div className="flex flex-wrap gap-2">
                            {message.sources.map((src, idx) => (
                                <div
                                    key={idx}
                                    className="flex items-center gap-1.5 px-2.5 py-1.5
                                               bg-slate-800/80 border border-slate-700
                                               rounded-lg text-xs text-slate-400"
                                >
                                    <FileText size={11} className="text-primary-500 shrink-0" />
                                    <span
                                        className="truncate max-w-[140px]"
                                        title={src.filename}
                                    >
                                        {src.filename}
                                    </span>
                                    <span className="text-slate-600">·</span>
                                    <span>p.{src.page_number}</span>
                                    <span className="text-slate-600">·</span>
                                    <span className={`
                                        text-xs font-medium px-1.5 py-0.5 rounded
                                        border ${getConfidenceColor(src.confidence_percent)}
                                    `}>
                                        {src.confidence_percent}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Timing info */}
                {!isUser && message.total_time !== undefined && (
                    <div className="flex items-center gap-1 px-1">
                        <Clock size={10} className="text-slate-600" />
                        <span className="text-xs text-slate-600">
                            {message.total_time.toFixed(1)}s
                            {message.chunks_retrieved > 0 &&
                                ` · ${message.chunks_retrieved} chunks retrieved`
                            }
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
}