/**
 * StatusBar — Top bar showing system status
 */

import { Activity, Database, Cpu } from 'lucide-react';

export default function StatusBar({ health }) {
  if (!health) return null;

  return (
    <div className="flex items-center gap-4 px-4 py-2
                    bg-slate-800/50 border-b border-slate-700/50
                    text-xs text-slate-500">

      {/* Status dot */}
      <div className="flex items-center gap-1.5">
        <div className={`w-1.5 h-1.5 rounded-full ${
          health.status === 'ok' ? 'bg-green-400' : 'bg-yellow-400'
        }`} />
        <span className={
          health.status === 'ok' ? 'text-green-400' : 'text-yellow-400'
        }>
          {health.status === 'ok' ? 'System Ready' : 'Degraded'}
        </span>
      </div>

      <span className="text-slate-700">|</span>

      {/* LLM model */}
      <div className="flex items-center gap-1">
        <Cpu size={11} />
        <span>{health.llm_model}</span>
      </div>

      <span className="text-slate-700">|</span>

      {/* Embedding model */}
      <div className="flex items-center gap-1">
        <Activity size={11} />
        <span>{health.embedding_model}</span>
      </div>

      <span className="text-slate-700">|</span>

      {/* Vector count */}
      <div className="flex items-center gap-1">
        <Database size={11} />
        <span>{health.total_vectors} vectors</span>
      </div>
    </div>
  );
}