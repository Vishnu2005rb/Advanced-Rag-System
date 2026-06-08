import React, { useState, useRef, useEffect } from "react";
import { Send, MessageSquare, Sparkles, User, Bot, Loader2, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import AnswerCard from "./AnswerCard";
import { SourceReference } from "../lib/api";

export interface ChatMessage {
  id: string;
  sender: "user" | "ai";
  text: string;
  timestamp: string;
  timeTaken?: number;
  modelUsed?: string;
  sources?: SourceReference[];
}

interface ChatBoxProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  viewMode: "dashboard" | "chat"; // "dashboard" is the compact view, "chat" is the full screen bubble thread
}

export default function ChatBox({
  messages,
  onSendMessage,
  isLoading,
  viewMode,
}: ChatBoxProps) {
  const [question, setQuestion] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat thread when new messages arrive
  useEffect(() => {
    if (chatEndRef.current && viewMode === "chat") {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading, viewMode]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    onSendMessage(question.trim());
    setQuestion("");
  };

  // Find the latest AI message to show in dashboard mode
  const latestAiMessage = [...messages]
    .reverse()
    .find((msg) => msg.sender === "ai");

  if (viewMode === "dashboard") {
    return (
      <div className="flex flex-col gap-4 w-full">
        {/* Input Box Card */}
        <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md">
          <div className="flex items-center gap-2 mb-3">
            <MessageSquare className="w-4.5 h-4.5 text-purple-400" />
            <h4 className="text-xs font-bold text-white uppercase tracking-wider">
              Ask Anything from Your PDF
            </h4>
          </div>

          <form onSubmit={handleSubmit} className="relative flex items-center bg-[#0b0d19] border border-slate-800 rounded-xl p-1.5 focus-within:border-purple-500/40 transition-colors">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question..."
              disabled={isLoading}
              className="flex-1 bg-transparent border-0 text-slate-100 text-xs font-medium placeholder-slate-600 focus:outline-none focus:ring-0 px-3 py-2 disabled:opacity-60"
            />
            <motion.button
              type="submit"
              disabled={isLoading || !question.trim()}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg hover:shadow-purple-500/20 disabled:opacity-40 disabled:pointer-events-none cursor-pointer"
            >
              {isLoading ? (
                <Loader2 className="w-4.5 h-4.5 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </motion.button>
          </form>
        </div>

        {/* Answer Box Card */}
        {latestAiMessage ? (
          <AnswerCard
            answer={latestAiMessage.text}
            timeTaken={latestAiMessage.timeTaken || 0}
            modelUsed={latestAiMessage.modelUsed || "Llama 3.3 70B"}
            sources={latestAiMessage.sources}
          />
        ) : isLoading ? (
          <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md">
            <div className="flex items-center gap-2 mb-3">
              <Bot className="w-4.5 h-4.5 text-purple-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                AI Answer
              </h4>
            </div>
            <div className="bg-[#0b0d19]/80 border border-slate-900 rounded-xl p-6 flex flex-col items-center justify-center h-[160px] text-center">
              <Loader2 className="w-8 h-8 text-purple-500 animate-spin mb-3" />
              <p className="text-xs font-semibold text-slate-400">
                Retrieving chunks & generating response...
              </p>
              <p className="text-[10px] text-slate-600 mt-1">
                Llama 3.3 70B is formulating answers via Groq API
              </p>
            </div>
          </div>
        ) : (
          <div className="bg-[#0f172a]/80 border border-slate-800/80 rounded-2xl p-5 shadow-xl backdrop-blur-md">
            <div className="flex items-center gap-2 mb-3">
              <Bot className="w-4.5 h-4.5 text-purple-400" />
              <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                AI Answer
              </h4>
            </div>
            <div className="border border-dashed border-slate-800/60 rounded-xl py-10 text-center text-xs font-medium text-slate-500">
              Submit a question above to see the AI's generated response
            </div>
          </div>
        )}
      </div>
    );
  }

  // ChatGPT Style Full Conversation View
  return (
    <div className="flex flex-col h-[calc(100vh-140px)] w-full">
      {/* Scrollable Messages Stream */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 mb-4 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <Sparkles className="w-10 h-10 text-purple-500/20 mb-3 animate-pulse" />
            <h3 className="text-sm font-bold text-white">RAG Chat System Active</h3>
            <p className="text-[11px] text-slate-500 max-w-[240px] mt-1.5 leading-relaxed">
              Ask anything about the uploaded PDF. The Llama 3.3 model will reference page-specific document context.
            </p>
          </div>
        ) : (
          messages.map((msg) => {
            const isAi = msg.sender === "ai";
            
            // Generate local citations mapping for bubble display
            const citations: Record<string, Set<number>> = {};
            if (isAi && msg.sources) {
              msg.sources.forEach((src) => {
                const name = src.source || "Document";
                if (!citations[name]) citations[name] = new Set<number>();
                if (src.page) citations[name].add(src.page);
              });
            }

            return (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-[85%] ${
                  isAi ? "self-start" : "self-end flex-row-reverse ml-auto"
                }`}
              >
                {/* Avatar */}
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 border ${
                    isAi
                      ? "bg-purple-500/10 text-purple-400 border-purple-500/20"
                      : "bg-slate-800 text-slate-300 border-slate-700"
                  }`}
                >
                  {isAi ? <Bot className="w-4.5 h-4.5" /> : <User className="w-4 h-4" />}
                </div>

                {/* Message Bubble */}
                <div className="flex flex-col gap-1">
                  <div
                    className={`rounded-2xl px-4 py-2.5 text-xs leading-relaxed shadow-md ${
                      isAi
                        ? "bg-slate-900/90 border border-slate-800 text-slate-200"
                        : "bg-gradient-to-br from-purple-600/90 to-indigo-600/90 border border-purple-500/20 text-white"
                    }`}
                  >
                    {isAi ? (
                      <div className="prose prose-invert max-w-none text-xs font-medium space-y-1.5">
                        <ReactMarkdown
                          components={{
                            p: ({ node, ...props }) => <p className="mb-1.5 last:mb-0" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc pl-4 mb-1.5 space-y-0.5" {...props} />,
                            ol: ({ node, ...props }) => <ol className="list-decimal pl-4 mb-1.5 space-y-0.5" {...props} />,
                            li: ({ node, ...props }) => <li className="mb-0.5" {...props} />,
                            strong: ({ node, ...props }) => <strong className="font-bold text-white" {...props} />,
                          }}
                        >
                          {msg.text}
                        </ReactMarkdown>

                        {/* Citations block inside Chat bubble */}
                        {Object.keys(citations).length > 0 && (
                          <div className="mt-2.5 border-t border-slate-800/80 pt-2 text-[9px] text-slate-400">
                            <span className="font-extrabold text-slate-400 uppercase tracking-widest flex items-center gap-1 mb-1">
                              <Info className="w-3 h-3 text-purple-400" /> Grounding Citations
                            </span>
                            <ul className="list-none pl-0 space-y-0.5">
                              {Object.entries(citations).map(([filename, pages]) => (
                                <li key={filename} className="flex items-center gap-1 font-semibold">
                                  <span className="text-purple-400">●</span>
                                  <span>
                                    PDF: <span className="text-slate-300">{filename}</span> | Pages:{" "}
                                    <span className="text-cyan-400 font-extrabold">
                                      {Array.from(pages)
                                        .sort((a, b) => a - b)
                                        .join(", ")}
                                    </span>
                                  </span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    ) : (
                      msg.text
                    )}
                  </div>
                  {/* Speed Tag (for AI) */}
                  {isAi && msg.timeTaken !== undefined && (
                    <span className="text-[9px] text-slate-500 px-2 font-semibold">
                      {msg.modelUsed || "Llama 3.3"} • {msg.timeTaken}s
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}

        {/* AI Typing Indicator */}
        {isLoading && (
          <div className="flex gap-3 max-w-[85%] self-start">
            <div className="w-7 h-7 rounded-full flex items-center justify-center bg-purple-500/10 text-purple-400 border border-purple-500/20 shrink-0">
              <Bot className="w-4.5 h-4.5" />
            </div>
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl px-4 py-3 shadow-md flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce" />
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Form Footer */}
      <form onSubmit={handleSubmit} className="flex items-center bg-[#0b0d19] border border-slate-800 rounded-xl p-1.5 focus-within:border-purple-500/40 transition-colors">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask anything from PDF..."
          disabled={isLoading}
          className="flex-1 bg-transparent border-0 text-slate-100 text-xs font-medium placeholder-slate-600 focus:outline-none focus:ring-0 px-3 py-2 disabled:opacity-60"
        />
        <motion.button
          type="submit"
          disabled={isLoading || !question.trim()}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          className="flex items-center justify-center w-8 h-8 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg disabled:opacity-40 disabled:pointer-events-none cursor-pointer"
        >
          {isLoading ? (
            <Loader2 className="w-4.5 h-4.5 animate-spin" />
          ) : (
            <Send className="w-4 h-4" />
          )}
        </motion.button>
      </form>
    </div>
  );
}
