"""
llm.py — Ollama LLM Integration
=================================
Connects to a locally running Ollama server and generates
answers using retrieved document chunks as context.

Ollama must be running: `ollama serve` (auto-starts on Windows).
Model must be pulled: `ollama pull phi3:mini`

Key responsibilities:
  1. Build structured prompts from chunks + query
  2. Call Ollama's chat API
  3. Parse and validate the response
  4. Return a typed LLMResponse with metadata
  5. Handle timeouts and connection errors gracefully

Design: Stateless — no memory between calls.
        Each generate_answer() is a fresh LLM conversation.
        Chat history is handled at a higher layer (Phase 15).
"""

import time
from typing import Optional

import ollama
from ollama import Client, ResponseError

from backend.models.document_models import LLMResponse, SearchResult
from backend.config import get_settings
from backend.utils.logger import logger


# ── Prompt Templates ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions \
based strictly on the provided document context.

Rules you must follow:
1. Answer ONLY using information from the provided context.
2. If the context does not contain enough information to answer \
the question, say: "I cannot find this information in the provided documents."
3. Do NOT make up facts, statistics, or details not in the context.
4. When possible, mention which document or page the information came from.
5. Be concise and direct. Avoid unnecessary preamble.
6. If the question is ambiguous, answer the most likely interpretation."""

CONTEXT_TEMPLATE = """Context from documents:
{context_text}

Question: {query}

Answer based only on the context above:"""


class OllamaLLM:
    """
    Interface to a locally running Ollama LLM.

    Args:
        model:    Ollama model name (e.g., "phi3:mini", "mistral", "llama3")
        base_url: Ollama server URL (default: http://localhost:11434)
        timeout:  Request timeout in seconds

    Usage:
        llm = OllamaLLM()
        response = llm.generate_answer(
            query="What is the main topic?",
            context_chunks=retrieved_results,
        )
        print(response.answer)
        print(response.sources)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.timeout = timeout or settings.ollama_timeout

        # Create Ollama client pointed at our local server
        self._client = Client(host=self.base_url)

        logger.info(
            f"OllamaLLM initialized | "
            f"model={self.model} | "
            f"url={self.base_url}"
        )

    def generate_answer(
        self,
        query: str,
        context_chunks: list[SearchResult],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Generate an answer using retrieved chunks as context.

        Args:
            query:          The user's question
            context_chunks: Retrieved SearchResult list from Retriever
            system_prompt:  Override default system prompt if needed
            temperature:    LLM creativity (0.0=deterministic, 1.0=creative)
                            0.1 is ideal for factual RAG — consistent answers

        Returns:
            LLMResponse with answer, sources, and token usage.
            Never raises — errors are captured in LLMResponse.success=False.
        """
        start = time.time()

        if not query.strip():
            return self._error_response(
                query, context_chunks, "Query cannot be empty", start
            )

        if not context_chunks:
            return LLMResponse(
                answer=(
                    "I cannot answer this question because no relevant "
                    "document context was found. Please upload documents "
                    "related to your question first."
                ),
                model=self.model,
                sources=[],
                generation_time=time.time() - start,
                success=True,
            )

        # ── Build the prompt ───────────────────────────────────────
        context_text = self._build_context_text(context_chunks)
        user_message = CONTEXT_TEMPLATE.format(
            context_text=context_text,
            query=query.strip(),
        )

        system = system_prompt or SYSTEM_PROMPT

        # ── Build messages in OpenAI/Ollama chat format ────────────
        # Ollama uses the same message format as OpenAI:
        #   role: "system" | "user" | "assistant"
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ]

        logger.info(
            f"Calling Ollama | model={self.model} | "
            f"context_chunks={len(context_chunks)} | "
            f"query='{query[:50]}...'"
        )

        # ── Call Ollama ────────────────────────────────────────────
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={
                    # temperature: controls randomness
                    # Low = more factual, deterministic answers
                    "temperature": temperature,
                    # num_predict: max tokens to generate
                    # 512 is enough for most answers
                    "num_predict": 512,
                    # top_p: nucleus sampling threshold
                    "top_p": 0.9,
                },
            )

            # ── Extract answer text ────────────────────────────────
            # Handle both dict and object response formats.
            # ollama Python library >= 0.2.x returns a plain dict.
            # Older versions return an object with .message attribute.
            if isinstance(response, dict):
                answer = response["message"]["content"].strip()
                prompt_tokens = response.get("prompt_eval_count", 0) or 0
                completion_tokens = response.get("eval_count", 0) or 0
            else:
                answer = response.message.content.strip()
                prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
                completion_tokens = getattr(response, "eval_count", 0) or 0

            generation_time = time.time() - start

            logger.info(
                f"Ollama response received | "
                f"tokens={prompt_tokens}+{completion_tokens} | "
                f"time={generation_time:.2f}s"
            )

            return LLMResponse(
                answer=answer,
                model=self.model,
                sources=context_chunks,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                generation_time=generation_time,
                success=True,
            )

        except ResponseError as e:
            # Ollama-specific error (model not found, etc.)
            error_msg = f"Ollama error: {e}"
            logger.error(error_msg)
            return self._error_response(
                query, context_chunks, error_msg, start
            )

        except ConnectionError as e:
            error_msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Run: ollama serve"
            )
            logger.error(error_msg)
            return self._error_response(
                query, context_chunks, error_msg, start
            )

        except Exception as e:
            error_msg = f"Unexpected LLM error: {str(e)}"
            logger.error(error_msg)
            return self._error_response(
                query, context_chunks, error_msg, start
            )

    def _build_context_text(
        self, chunks: list[SearchResult]
    ) -> str:
        """
        Format retrieved chunks into a readable context block.

        Each chunk is labeled with its source for citation purposes.
        The LLM sees these labels and can reference them in its answer.

        Format:
            [Source 1: filename.pdf, Page 3] (Relevance: 87%)
            ...chunk text...

            [Source 2: notes.docx, Page 1] (Relevance: 72%)
            ...chunk text...

        Args:
            chunks: SearchResult list, already sorted by score

        Returns:
            Formatted string ready to insert into the prompt
        """
        parts = []

        for i, chunk in enumerate(chunks, start=1):
            source_label = (
                f"[Source {i}: {chunk.filename}, "
                f"Page {chunk.page_number}] "
                f"(Relevance: {chunk.confidence_percent}%)"
            )
            parts.append(f"{source_label}\n{chunk.text}")

        # Separate chunks with clear divider
        return "\n\n---\n\n".join(parts)

    def is_available(self) -> bool:
        """
        Check if Ollama is running and the model is available.

        Returns:
            True if Ollama responds and model is loaded.
            False otherwise (no exception raised).

        Used by the /health API endpoint.
        """
        try:
            response = self._client.list()

            # Handle both dict and object response formats
            # ollama >= 0.2.x returns a dict, older returns an object
            if isinstance(response, dict):
                model_list = response.get("models", [])
                model_names = [
                    m.get("name", "") or m.get("model", "")
                    for m in model_list
                ]
            else:
                model_names = [m.model for m in response.models]

            # Strip ":latest" for flexible matching
            # "phi3" matches "phi3:mini", "mistral" matches "mistral:latest"
            target = self.model.replace(":latest", "").split(":")[0]
            available = any(target in name for name in model_names)

            if not available:
                logger.warning(
                    f"Model '{self.model}' not found. "
                    f"Available: {model_names}"
                )
            return available

        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def list_models(self) -> list[str]:
        """
        Return list of locally available Ollama model names.
        Used by the /models API endpoint.
        """
        try:
            response = self._client.list()

            if isinstance(response, dict):
                model_list = response.get("models", [])
                return [
                    m.get("name", "") or m.get("model", "")
                    for m in model_list
                ]
            else:
                return [m.model for m in response.models]

        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    def _error_response(
        self,
        query: str,
        chunks: list[SearchResult],
        error: str,
        start: float,
    ) -> LLMResponse:
        """Build a consistent error LLMResponse."""
        return LLMResponse(
            answer=(
                "I encountered an error while generating an answer. "
                f"Details: {error}"
            ),
            model=self.model,
            sources=chunks,
            generation_time=time.time() - start,
            success=False,
            error=error,
        )