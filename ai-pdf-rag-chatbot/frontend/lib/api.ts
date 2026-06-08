const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface DocumentInfo {
  filename: string;
  size: string;
  pages: number;
  chunks: number;
  embedding_model: string;
  vector_count: number;
  status: string;
  timestamp: number;
}

export interface GlobalStats {
  total_documents: number;
  total_pages: number;
  total_chunks: number;
  vector_dimension: number;
  embedding_model: string;
  llm_model: string;
}

export interface SourceReference {
  page: number;
  content: string;
  score: number;
  source: string;
  chunk_id?: string;
}

export interface RagasScores {
  faithfulness: number;
  answer_relevance: number;
  context_precision: number;
  context_recall: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceReference[];
  retrieved_chunks: SourceReference[];
  scores?: RagasScores;
  rewritten_query?: string;
  time_taken: number;
  model_used: string;
}

export interface ChatConfig {
  sessionId?: string;
  embeddingModel?: string;
  retrieverType?: string; // similarity, mmr, hybrid
  scoreThreshold?: number;
  enableQueryRewrite?: boolean;
  enableReranking?: boolean;
  enableMemory?: boolean;
  runEvaluation?: boolean;
  topK?: number;
  fetchK?: number;
}

export const api = {
  getDocuments: async (): Promise<DocumentInfo[]> => {
    const res = await fetch(`${API_BASE_URL}/documents`);
    if (!res.ok) throw new Error("Failed to fetch documents");
    return res.json();
  },

  getStats: async (modelName: string = "BAAI/bge-small-en-v1.5"): Promise<GlobalStats> => {
    const res = await fetch(`${API_BASE_URL}/stats?model_name=${encodeURIComponent(modelName)}`);
    if (!res.ok) throw new Error("Failed to fetch stats");
    return res.json();
  },

  uploadPDF: async (
    file: File,
    chunkSize: number = 1000,
    chunkOverlap: number = 200,
    chunkerStrategy: string = "recursive",
    embeddingModel: string = "BAAI/bge-small-en-v1.5"
  ): Promise<{ filename: string; pages: number; chunks: number; embedding_model: string; vector_count: number }> => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("chunk_size", chunkSize.toString());
    formData.append("chunk_overlap", chunkOverlap.toString());
    formData.append("chunker_strategy", chunkerStrategy);
    formData.append("embedding_model", embeddingModel);

    const res = await fetch(`${API_BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error occurred" }));
      throw new Error(err.detail || "Ingestion failed");
    }
    return res.json();
  },

  askQuestion: async (
    question: string,
    filenameFilter?: string,
    modelName: string = "llama-3.3-70b-versatile",
    temperature: number = 0.3,
    apiKey?: string,
    config: ChatConfig = {}
  ): Promise<ChatResponse> => {
    const body: any = {
      question,
      model_name: modelName,
      temperature,
      filename_filter: filenameFilter || null,
      api_key: apiKey || null,
      session_id: config.sessionId || "default_session",
      embedding_model: config.embeddingModel || "BAAI/bge-small-en-v1.5",
      retriever_type: config.retrieverType || "hybrid",
      score_threshold: config.scoreThreshold !== undefined ? config.scoreThreshold : 0.70,
      enable_query_rewrite: config.enableQueryRewrite !== undefined ? config.enableQueryRewrite : true,
      enable_reranking: config.enableReranking !== undefined ? config.enableReranking : true,
      enable_memory: config.enableMemory !== undefined ? config.enableMemory : true,
      run_evaluation: config.runEvaluation !== undefined ? config.runEvaluation : true,
      top_k: config.topK || 5,
      fetch_k: config.fetchK || 20,
    };

    const res = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error occurred" }));
      throw new Error(err.detail || "Chat query failed");
    }
    return res.json();
  },

  clearAll: async (sessionId: string = "default_session"): Promise<{ status: string; message: string }> => {
    const res = await fetch(`${API_BASE_URL}/clear?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Failed to reset storage");
    return res.json();
  },

  clearHistory: async (sessionId: string = "default_session"): Promise<{ status: string; message: string }> => {
    const res = await fetch(`${API_BASE_URL}/clear_history?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    });
    if (!res.ok) throw new Error("Failed to clear chat memory");
    return res.json();
  },

  deleteDocument: async (filename: string, embeddingModel: string = "BAAI/bge-small-en-v1.5"): Promise<{ status: string; message: string }> => {
    const res = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(filename)}?embedding_model=${encodeURIComponent(embeddingModel)}`, {
      method: "DELETE",
    });
    if (!res.ok) throw new Error("Failed to delete document from index");
    return res.json();
  },
};
