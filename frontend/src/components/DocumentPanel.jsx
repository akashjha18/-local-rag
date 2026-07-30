/**
 * DocumentPanel — Left sidebar showing uploaded documents
 * Handles file upload via drag-and-drop or click
 */

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FileText, Trash2, Upload, Loader2, AlertCircle } from 'lucide-react';
import { uploadDocument, deleteDocument } from '../services/api';

export default function DocumentPanel({ documents, onDocumentsChange }) {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  // ── Drag and Drop Handler ──────────────────────────────────────
  const onDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];

    // Validate file type
    const allowed = ['.pdf', '.docx'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
      setError(`Unsupported file type: ${ext}. Use PDF or DOCX.`);
      return;
    }

    setError(null);
    setUploading(true);
    setUploadProgress(0);

    try {
      await uploadDocument(file, setUploadProgress);
      await onDocumentsChange(); // Refresh document list
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Upload failed';
      setError(msg);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [onDocumentsChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  // ── Delete Handler ─────────────────────────────────────────────
  const handleDelete = async (documentId, filename) => {
    if (!confirm(`Delete "${filename}" from the index?`)) return;

    setDeletingId(documentId);
    try {
      await deleteDocument(documentId);
      await onDocumentsChange();
    } catch (err) {
      setError(`Delete failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex flex-col h-full p-4 gap-4">

      {/* Header */}
      <div className="flex items-center gap-2">
        <FileText size={20} className="text-primary-500" />
        <h2 className="text-lg font-semibold text-slate-200">Documents</h2>
        <span className="ml-auto text-xs text-slate-500 bg-slate-800 px-2 py-1 rounded-full">
          {documents.length}
        </span>
      </div>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-4 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive
            ? 'border-primary-500 bg-primary-500/10'
            : 'border-slate-600 hover:border-primary-500/50 hover:bg-slate-800/50'
          }
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        {uploading ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 size={24} className="text-primary-500 animate-spin" />
            <p className="text-sm text-slate-400">
              Uploading... {uploadProgress}%
            </p>
            {/* Progress bar */}
            <div className="w-full bg-slate-700 rounded-full h-1.5">
              <div
                className="bg-primary-500 h-1.5 rounded-full transition-all"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload size={24} className="text-slate-500" />
            <p className="text-sm text-slate-400">
              {isDragActive
                ? 'Drop file here...'
                : 'Drop PDF or DOCX here'
              }
            </p>
            <p className="text-xs text-slate-600">or click to browse</p>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && (
        <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <AlertCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      {/* Document List */}
      <div className="flex flex-col gap-2 overflow-y-auto flex-1">
        {documents.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-sm text-slate-600">No documents yet</p>
            <p className="text-xs text-slate-700 mt-1">Upload a PDF or DOCX to get started</p>
          </div>
        ) : (
          documents.map((doc) => (
            <div
              key={doc.document_id}
              className="flex items-start gap-3 p-3 bg-slate-800 rounded-lg
                         border border-slate-700 hover:border-slate-600 group"
            >
              {/* File icon */}
              <div className="p-2 bg-primary-500/10 rounded-lg shrink-0">
                <FileText size={16} className="text-primary-500" />
              </div>

              {/* File info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-slate-200 truncate font-medium">
                  {doc.filename}
                </p>
                <p className="text-xs text-slate-500 mt-0.5">
                  {doc.chunk_count} chunks · {doc.document_type.toUpperCase()}
                </p>
              </div>

              {/* Delete button */}
              <button
                onClick={() => handleDelete(doc.document_id, doc.filename)}
                disabled={deletingId === doc.document_id}
                className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg
                           hover:bg-red-500/20 text-slate-500 hover:text-red-400
                           transition-all disabled:opacity-50"
                title="Remove document"
              >
                {deletingId === doc.document_id
                  ? <Loader2 size={14} className="animate-spin" />
                  : <Trash2 size={14} />
                }
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}