/**
 * api.js — Backend API Client
 * All communication with FastAPI backend goes through here.
 * Base URL points to our backend server.
 */

import axios from 'axios';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 minutes — LLM can be slow on CPU
});

// ── Document Operations ────────────────────────────────────────────

/**
 * Upload and index a document file (PDF or DOCX)
 * @param {File} file - The file object from input/dropzone
 * @param {Function} onProgress - Progress callback (0-100)
 */
export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/v1/documents', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded * 100) / e.total));
      }
    },
  });
  return response.data;
};

/**
 * Get all indexed documents
 */
export const getDocuments = async () => {
  const response = await api.get('/api/v1/documents');
  return response.data;
};

/**
 * Delete a document from the index
 * @param {string} documentId
 */
export const deleteDocument = async (documentId) => {
  const response = await api.delete(`/api/v1/documents/${documentId}`);
  return response.data;
};

// ── Query Operations ───────────────────────────────────────────────

/**
 * Ask a question using the RAG pipeline
 * @param {string} query - The user's question
 * @param {Object} options - top_k, score_threshold, temperature
 */
export const askQuestion = async (query, options = {}) => {
  const response = await api.post('/api/v1/query', {
    query,
    top_k: options.top_k ?? 5,
    score_threshold: options.score_threshold ?? 0.1,
    temperature: options.temperature ?? 0.1,
  });
  return response.data;
};

// ── System Operations ──────────────────────────────────────────────

/**
 * Get system health status
 * 
 */
export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Get detailed system statistics
 */
export const getStats = async () => {
  const response = await api.get('/api/v1/stats');
  return response.data;
};