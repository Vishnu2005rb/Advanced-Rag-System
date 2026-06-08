import React from "react";
import {
  Bot,
  Home,
  MessageSquare,
  FileText,
  Settings,
  Layers,
  Dna,
  Binary,
  Cpu,
  RefreshCw,
} from "lucide-react";
import { motion } from "framer-motion";
import UploadPDF from "./UploadPDF";

interface DocumentInfo {
  filename: string;
  size: string;
  pages: number;
  chunks: number;
  vector_dimension: number;
  status: string;
}

interface SidebarProps {
  activePage: string;
  setActivePage: (page: string) => void;
  uploadedDoc: DocumentInfo | null;
  onUploadStart: () => void;
  onUploadSuccess: (filename: string, pages: number, chunks: number, vectorDim: number) => void;
  onUploadError: (error: string) => void;
  isUploading: boolean;
  totalChunks: number;
  vectorDimension: number;
  topK: number;
  llmModelName: string;
  onClearAll: () => void;
}

export default function Sidebar({
  activePage,
  setActivePage,
  uploadedDoc,
  onUploadStart,
  onUploadSuccess,
  onUploadError,
  isUploading,
  totalChunks,
  vectorDimension,
  topK,
  llmModelName,
  onClearAll,
}: SidebarProps) {
  const navItems = [
    { id: "Dashboard", label: "Dashboard", icon: <Home className="w-4 h-4" /> },
    { id: "Chat", label: "Chat", icon: <MessageSquare className="w-4 h-4" /> },
    { id: "Documents", label: "Documents", icon: <FileText className="w-4 h-4" /> },
    { id: "Settings", label: "Settings", icon: <Settings className="w-4 h-4" /> },
  ];

  return (
    <aside className="w-64 border-r border-slate-800/80 bg-[#070B18]/90 flex flex-col h-screen overflow-y-auto px-4 py-6 shrink-0">
      {/* Branding Header */}
      <div className="flex items-center gap-3 mb-6 px-1">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 shadow-[0_0_15px_rgba(124,58,237,0.15)]">
          <Bot className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <h1 className="text-sm font-extrabold text-white tracking-tight leading-tight">
            RAG PDF Chatbot
          </h1>
          <p className="text-[10px] font-semibold text-slate-400">
            Powered by Groq + Llama 3.3
          </p>
        </div>
      </div>

      {/* Upload PDF Section */}
      <div className="mb-6">
        <UploadPDF
          isUploading={isUploading}
          onUploadStart={onUploadStart}
          onUploadSuccess={onUploadSuccess}
          onUploadError={onUploadError}
        />

        {/* Uploaded File Detail Card */}
        {uploadedDoc ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-3 flex items-center justify-between bg-slate-900/60 border border-slate-800 rounded-lg p-2.5"
          >
            <div className="flex items-center gap-2 min-w-0">
              <div className="flex items-center justify-center w-8 h-8 rounded-md bg-purple-500/10 text-purple-400 shrink-0">
                <FileText className="w-4.5 h-4.5" />
              </div>
              <div className="flex flex-col min-w-0">
                <span
                  className="text-xs font-semibold text-slate-100 truncate"
                  title={uploadedDoc.filename}
                >
                  {uploadedDoc.filename}
                </span>
                <span className="text-[9px] text-slate-500">
                  {uploadedDoc.size} • {uploadedDoc.pages} pages
                </span>
              </div>
            </div>
            <div className="flex items-center justify-center w-4 h-4 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold shrink-0">
              ✓
            </div>
          </motion.div>
        ) : (
          <div className="mt-3 border border-dashed border-slate-800/80 rounded-lg py-3 text-center text-[10px] text-slate-500">
            No document loaded
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="mb-6">
        <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1 mb-2">
          Navigation
        </h2>
        <nav className="flex flex-col gap-1">
          {navItems.map((item) => {
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-semibold transition-all text-left ${
                  isActive
                    ? "bg-purple-500/10 border border-purple-500/20 text-purple-300 shadow-[0_0_10px_rgba(124,58,237,0.05)]"
                    : "text-slate-400 hover:bg-slate-900/40 hover:text-slate-200"
                }`}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Comparison Navigation Button */}
      <div className="mb-6">
        <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1 mb-2">
          Architecture Comparison
        </h2>
        <button
          onClick={() => setActivePage("Comparison")}
          className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-bold transition-all text-left relative overflow-hidden group ${
            activePage === "Comparison"
              ? "bg-gradient-to-r from-purple-600/20 to-indigo-600/20 border-2 border-purple-500 text-purple-200 shadow-[0_0_15px_rgba(168,85,247,0.3)]"
              : "bg-slate-900/45 border border-purple-500/35 text-purple-400 hover:border-purple-500/60 hover:text-purple-300 hover:bg-purple-500/5 shadow-[0_0_10px_rgba(168,85,247,0.05)]"
          }`}
        >
          {/* Glowing background animation on hover */}
          <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 to-indigo-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          
          <div className="flex items-center gap-2 relative z-10">
            <Bot className="w-4 h-4 animate-bounce shrink-0 text-purple-400" style={{ animationDuration: "3s" }} />
            <span>Normal vs Eng. RAG</span>
          </div>
          <span className="relative z-10 text-[9px] uppercase tracking-widest font-black bg-purple-500 text-white px-1.5 py-0.5 rounded-md animate-pulse shadow-[0_0_8px_rgba(168,85,247,0.5)]">
            New
          </span>
        </button>
      </div>

      {/* Vector DB Stats */}
      <div className="mb-6">
        <div className="flex items-center justify-between px-1 mb-2">
          <h2 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
            Stats
          </h2>
          {uploadedDoc && (
            <button
              onClick={onClearAll}
              title="Clear Database"
              className="text-[9px] text-red-400 hover:text-red-300 font-bold transition-colors flex items-center gap-1"
            >
              <RefreshCw className="w-2.5 h-2.5" /> Clear
            </button>
          )}
        </div>

        <div className="flex flex-col gap-2">
          {/* Total Chunks */}
          <div className="flex items-center gap-3 bg-slate-950/40 border border-slate-900 rounded-lg p-2.5">
            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-blue-500/10 text-blue-400 shrink-0">
              <Layers className="w-3.5 h-3.5" />
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-blue-400">{totalChunks}</span>
              <span className="text-[9px] text-slate-500 font-medium">Total Chunks</span>
            </div>
          </div>

          {/* Vector Dimension */}
          <div className="flex items-center gap-3 bg-slate-950/40 border border-slate-900 rounded-lg p-2.5">
            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-amber-500/10 text-amber-400 shrink-0">
              <Binary className="w-3.5 h-3.5" />
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-amber-400">{vectorDimension}</span>
              <span className="text-[9px] text-slate-500 font-medium">Vector Dimension</span>
            </div>
          </div>

          {/* Top K results */}
          <div className="flex items-center gap-3 bg-slate-950/40 border border-slate-900 rounded-lg p-2.5">
            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-cyan-500/10 text-cyan-400 shrink-0">
              <Dna className="w-3.5 h-3.5" />
            </div>
            <div className="flex flex-col">
              <span className="text-xs font-bold text-cyan-400">{topK}</span>
              <span className="text-[9px] text-slate-500 font-medium">Top K Results</span>
            </div>
          </div>

          {/* LLM Model Name */}
          <div className="flex items-center gap-3 bg-slate-950/40 border border-slate-900 rounded-lg p-2.5">
            <div className="flex items-center justify-center w-7 h-7 rounded-md bg-emerald-500/10 text-emerald-400 shrink-0">
              <Cpu className="w-3.5 h-3.5" />
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-bold text-emerald-400 truncate max-w-[120px]" title={llmModelName}>
                {llmModelName}
              </span>
              <span className="text-[9px] text-slate-500 font-medium">LLM Model</span>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Branding */}
      <div className="mt-auto pt-4 border-t border-slate-900 text-[9px] text-slate-600 leading-relaxed text-center px-1">
        Built with ❤️ using <br />
        Next.js 15 • LangChain • FastAPI • ChromaDB • Groq
      </div>
    </aside>
  );
}
