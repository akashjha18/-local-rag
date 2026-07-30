/**
 * ChatInput — Message input bar at the bottom of the chat
 */

import { useState, useRef } from 'react';
import { Send, Loader2 } from 'lucide-react';

export default function ChatInput({ onSend, disabled, placeholder }) {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  const handleSend = () => {
    const text = input.trim();
    if (!text || disabled) return;
    onSend(text);
    setInput('');
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    // Send on Enter, new line on Shift+Enter
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e) => {
    setInput(e.target.value);
    // Auto-resize textarea
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
    }
  };

  return (
    <div className="flex items-end gap-3 p-4
                    border-t border-slate-700/50 bg-slate-900">
      <textarea
        ref={textareaRef}
        value={input}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={placeholder || 'Ask a question about your documents...'}
        rows={1}
        className="
          flex-1 resize-none bg-slate-800 border border-slate-700
          rounded-xl px-4 py-3 text-sm text-slate-200
          placeholder-slate-500 outline-none
          focus:border-primary-500/50 focus:ring-1 focus:ring-primary-500/20
          disabled:opacity-50 disabled:cursor-not-allowed
          max-h-[120px] overflow-y-auto
        "
      />
      <button
        onClick={handleSend}
        disabled={disabled || !input.trim()}
        className="
          shrink-0 w-10 h-10 flex items-center justify-center
          bg-primary-600 hover:bg-primary-500
          disabled:bg-slate-700 disabled:cursor-not-allowed
          rounded-xl transition-colors
        "
      >
        {disabled
          ? <Loader2 size={18} className="text-slate-400 animate-spin" />
          : <Send size={18} className="text-white" />
        }
      </button>
    </div>
  );
}