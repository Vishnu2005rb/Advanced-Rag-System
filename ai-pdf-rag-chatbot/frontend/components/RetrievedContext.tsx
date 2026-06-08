import React, { useState } from "react";
import { Search, FileText, CheckCircle2, ChevronDown, ChevronUp, BarChart3, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { RagasScores, SourceReference } from "../lib/api";

interface RetrievedContextProps {
  sources: SourceReference[];
  retrievedChunks: SourceReference[];
  activeDocument: {
    filename: string;
    size: string;
    pages: number;
    chunks: number;
    status: string;
  } | null;
  ragasScores?: RagasScores | null;
}

export default function RetrievedContext({
  sources,
  retrievedChunks,
  activeDocument,
  ragasScores,
}: RetrievedContextProps) {
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  const getRankColors = (index: number) => {
    switch (index) {
      case 0:
        return {
          dotBg: "bg-emerald-500",
          scoreBg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
          border: "hover:border-emerald-500/20",
        };
      case 1:
        return {
          dotBg: "bg-blue-500",
          scoreBg: "bg-blue-500/10 text-blue-400 border-blue-500/20",
          border: "hover:border-blue-500/20",
        };
      default:
        return {
          dotBg: "bg-purple-500",
          scoreBg: "bg-purple-500/10 text-purple-400 border-purple-500/20",
          border: "hover:border-purple-500/20",
        };
    }
  };

  const handleToggleChunk = (chunkId: string) => {
    setExpandedChunkId((prev) => (prev === chunkId ? null : chunkId));
  };

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      {/* 1. RAGAS EVALUATION METRICS PANEL */}
      {ragasScores && (
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-[#0f172a]/80 border-2 border-indigo-500/35 rounded-2xl p-5 shadow-[0_0_20px_rgba(99,102,241,0.15)] backdrop-blur-md"
        >
          <div className="flex items-center gap-2 mb-3.5">
            <BarChart3 className="w-4 h-4 text-indigo-400" />
            <h4 className="text-xs font-black text-white uppercase tracking-wider">
              RAGAS Audit Metrics
            </h4>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center">
            {/* Faithfulness */}
            <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-2.5 flex flex-col justify-between">
              <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Faithfulness</span>
              <span className={`text-sm font-black mt-1 ${ragasScores.faithfulness >= 0.8 ? "text-emerald-400" : "text-amber-400"}`}>
                {(ragasScores.faithfulness * 100).toFixed(0)}%
              </span>
              <div className="w-full bg-slate-900 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${ragasScores.faithfulness * 100}%` }} />
              </div>
            </div>

            {/* Answer Relevance */}
            <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-2.5 flex flex-col justify-between">
              <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Relevance</span>
              <span className={`text-sm font-black mt-1 ${ragasScores.answer_relevance >= 0.8 ? "text-emerald-400" : "text-amber-400"}`}>
                {(ragasScores.answer_relevance * 100).toFixed(0)}%
              </span>
              <div className="w-full bg-slate-900 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${ragasScores.answer_relevance * 100}%` }} />
              </div>
            </div>

            {/* Context Precision */}
            <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-2.5 flex flex-col justify-between">
              <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Precision</span>
              <span className={`text-sm font-black mt-1 ${ragasScores.context_precision >= 0.8 ? "text-emerald-400" : "text-amber-400"}`}>
                {(ragasScores.context_precision * 100).toFixed(0)}%
              </span>
              <div className="w-full bg-slate-900 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${ragasScores.context_precision * 100}%` }} />
              </div>
            </div>

            {/* Context Recall */}
            <div className="bg-slate-950/40 border border-slate-900 rounded-xl p-2.5 flex flex-col justify-between">
              <span className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">Recall</span>
              <span className={`text-sm font-black mt-1 ${ragasScores.context_recall >= 0.8 ? "text-emerald-400" : "text-amber-400"}`}>
                {(ragasScores.context_recall * 100).toFixed(0)}%
              </span>
              <div className="w-full bg-slate-900 h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${ragasScores.context_recall * 100}%` }} />
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* 2. SIMILARITY RESULTS PANEL */}
      <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md flex-1 overflow-y-auto max-h-[400px] scrollbar-thin">
        <div className="flex items-center gap-2 mb-4">
          <Search className="w-4 h-4 text-cyan-400" />
          <h4 className="text-xs font-bold text-white uppercase tracking-wider">
            Retrieved Chunks ({sources.length})
          </h4>
        </div>

        {sources && sources.length > 0 ? (
          <div className="flex flex-col gap-3">
            {sources.map((src, idx) => {
              const theme = getRankColors(idx);
              const chunkId = src.chunk_id || `chunk_${idx}`;
              const isExpanded = expandedChunkId === chunkId;
              
              // Find the corresponding full chunk text
              const fullChunk = retrievedChunks.find((c) => c.chunk_id === chunkId) || src;

              return (
                <motion.div
                  key={idx}
                  layout
                  className={`bg-[#0b0d19]/40 border border-slate-900 rounded-xl p-3 flex flex-col gap-2 transition-all duration-200 ${theme.border}`}
                >
                  <div className="flex justify-between items-center cursor-pointer" onClick={() => handleToggleChunk(chunkId)}>
                    {/* Rank Dot */}
                    <div className="flex items-center gap-2 min-w-0">
                      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-extrabold text-white shrink-0 ${theme.dotBg}`}>
                        {idx + 1}
                      </div>
                      <span className="text-[10px] text-slate-500 font-bold truncate max-w-[120px]" title={src.source}>
                        {src.source}
                      </span>
                    </div>

                    {/* Score and Toggle Action */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      <div className={`text-[9px] font-black px-2 py-0.5 rounded-full border ${theme.scoreBg}`}>
                        {src.score > 1.0 ? "Rerank: " : "Score: "}{src.score.toFixed(2)}
                      </div>
                      {isExpanded ? (
                        <ChevronUp className="w-3.5 h-3.5 text-slate-500" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
                      )}
                    </div>
                  </div>

                  {/* Content Text Preview / Full */}
                  <div className="flex flex-col">
                    <p className="text-[11px] font-medium leading-relaxed text-slate-300">
                      {isExpanded ? `"${fullChunk.content}"` : `"${src.content}"`}
                    </p>
                    
                    <div className="flex justify-between items-center mt-2">
                      <span className="text-[9px] font-bold text-slate-500">
                        Page {src.page}
                      </span>
                      {src.chunk_id && (
                        <span className="text-[8px] font-mono text-slate-600 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-900">
                          {src.chunk_id}
                        </span>
                      )}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center border border-dashed border-slate-800/60 rounded-xl p-8 text-center h-[200px]">
            <Search className="w-6 h-6 text-slate-700 mb-2" />
            <p className="text-xs font-semibold text-slate-500">
              No context retrieved yet
            </p>
            <p className="text-[10px] text-slate-600 mt-1 max-w-[160px]">
              Ask a question to run advanced similarity retrieval
            </p>
          </div>
        )}
      </div>

      {/* 3. SOURCE DOCUMENT INFO PANEL */}
      <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-2 mb-3">
          <FileText className="w-4 h-4 text-purple-400" />
          <h4 className="text-xs font-bold text-white uppercase tracking-wider">
            Source Document Info
          </h4>
        </div>

        {activeDocument ? (
          <div className="bg-[#0b0d19]/40 border border-slate-900 rounded-xl p-3 flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/20 shrink-0">
              <FileText className="w-5 h-5" />
            </div>
            <div className="flex flex-col min-w-0 flex-1">
              <span className="text-xs font-bold text-white truncate" title={activeDocument.filename}>
                {activeDocument.filename}
              </span>
              <span className="text-[9px] text-slate-500 mt-0.5">
                {activeDocument.size} • {activeDocument.pages} pages • {activeDocument.chunks} chunks
              </span>
            </div>
            <div className="flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[9px] font-bold px-2 py-0.5 rounded-full shrink-0">
              <CheckCircle2 className="w-3 h-3" />
              <span>Processed</span>
            </div>
          </div>
        ) : (
          <div className="border border-dashed border-slate-800/60 rounded-xl py-5 text-center text-xs font-medium text-slate-500">
            No active document index
          </div>
        )}
      </div>
    </div>
  );
}
