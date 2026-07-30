/**
 * ChatMessage — Single message bubble in the chat
 * Handles both user messages and AI responses with sources
 */

import { User, Bot, Clock, FileText, AlertCircle } from 'lucide-react';

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const isError = message.success === false;

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>

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

      {/* Message bubble */}
      <div className={`
        max-w-[80%] flex flex-col gap-2
        ${isUser ? 'items-end' : 'items-start'}
      `}>

        {/* Text content */}
        <div className={`
          px-4 py-3 rounded-2xl text-sm leading-relaxed
          ${isUser
            ? 'bg-primary-600 text-white rounded-tr-sm'
            : isError
              ? 'bg-red-500/10 border border-red-500/30 text-red-300 rounded-tl-sm'
              : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-tl-sm'
          }
        `}>
          {isError && (
            <div className="flex items-center gap-2 mb-2">
              <AlertCircle size={14} className="text-red-400" />
              <span className="text-xs text-red-400 font-medium">Error</span>
            </div>
          )}
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>

        {/* Sources (only for AI messages with sources) */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="flex flex-col gap-1.5 w-full">
            <p className="text-xs text-slate-500 font-medium px-1">Sources:</p>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((src, idx) => (
                <div
                  key={idx}
                  className="flex items-center gap-1.5 px-2.5 py-1.5
                             bg-slate-800/80 border border-slate-700
                             rounded-lg text-xs text-slate-400"
                >
                  <FileText size={11} className="text-primary-500 shrink-0" />
                  <span className="truncate max-w-[150px]">{src.filename}</span>
                  <span className="text-slate-600">·</span>
                  <span>p.{src.page_number}</span>
                  <span className="text-slate-600">·</span>
                  <span className={`font-medium ${
                    src.confidence_percent >= 70
                      ? 'text-green-400'
                      : src.confidence_percent >= 50
                        ? 'text-yellow-400'
                        : 'text-slate-500'
                  }`}>
                    {src.confidence_percent}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Timing info for AI messages */}
        {!isUser && message.total_time && (
          <div className="flex items-center gap-1 px-1">
            <Clock size={10} className="text-slate-600" />
            <span className="text-xs text-slate-600">
              {message.total_time.toFixed(1)}s
              {message.chunks_retrieved > 0 &&
                ` · ${message.chunks_retrieved} chunks`
              }
            </span>
          </div>
        )}
      </div>
    </div>
  );
}