import React, { useState, useRef } from "react";
import { Upload, FileText, Loader2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface UploadPDFProps {
  onUploadSuccess: (filename: string, pages: number, chunks: number, vectorDim: number) => void;
  onUploadStart: () => void;
  onUploadError: (error: string) => void;
  isUploading: boolean;
}

export default function UploadPDF({
  onUploadSuccess,
  onUploadStart,
  onUploadError,
  isUploading,
}: UploadPDFProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type !== "application/pdf") {
        onUploadError("Only PDF documents are supported.");
        return;
      }
      await uploadFile(file);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      await uploadFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const uploadFile = async (file: File) => {
    onUploadStart();
    const { api } = await import("../lib/api");
    try {
      const response = await api.uploadPDF(file);
      onUploadSuccess(
        response.filename,
        response.pages,
        response.chunks,
        response.vector_dimension
      );
    } catch (err: any) {
      onUploadError(err.message || "Failed to parse PDF document.");
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={handleFileChange}
        disabled={isUploading}
      />

      <motion.div
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={triggerFileInput}
        whileHover={{ scale: isUploading ? 1 : 1.01 }}
        whileTap={{ scale: isUploading ? 1 : 0.99 }}
        className={`relative flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-5 cursor-pointer text-center transition-all ${
          isDragActive
            ? "border-cyan-400 bg-cyan-950/10 shadow-[0_0_15px_rgba(6,182,212,0.15)]"
            : "border-purple-500/30 bg-slate-950/40 hover:border-purple-500/50 hover:bg-purple-950/5"
        } ${isUploading ? "pointer-events-none opacity-80" : ""}`}
      >
        <AnimatePresence mode="wait">
          {isUploading ? (
            <motion.div
              key="uploading"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="flex flex-col items-center py-2"
            >
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin mb-2" />
              <p className="text-xs font-semibold text-cyan-400">
                Extracting & Indexing...
              </p>
              <p className="text-[10px] text-slate-500 mt-1">
                PyPDF text parsing + ChromaDB creation
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="flex flex-col items-center py-1"
            >
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-purple-500/10 text-purple-400 mb-2">
                <Upload className="w-5 h-5" />
              </div>
              <p className="text-xs font-bold text-slate-200">
                Drag & Drop PDF
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                or click to browse from files
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
}
