"""
query_processor.py — Query Preprocessing
==========================================
Cleans and normalizes user queries before embedding.

Why preprocess queries?
  Raw user input often has:
  - Extra whitespace: "what  is   this?"
  - No punctuation:   "what is rag"
  - Trailing spaces:  "  explain this  "
  - Very short queries: "?" or "ok"
  
  Preprocessing improves embedding quality and
  gives better search results.
"""

import re
from backend.utils.logger import logger


# Minimum query length after cleaning
MIN_QUERY_LENGTH = 3

# Maximum query length (prevent abuse)
MAX_QUERY_LENGTH = 2000


def preprocess_query(query: str) -> str:
    """
    Clean and normalize a user query for better embedding.

    Steps:
    1. Strip leading/trailing whitespace
    2. Normalize internal whitespace
    3. Remove special characters that confuse embeddings
    4. Ensure query ends with punctuation (helps sentence transformers)
    5. Truncate if too long

    Args:
        query: Raw user input

    Returns:
        Cleaned query string

    Raises:
        ValueError: If query is empty or too short after cleaning
    """
    if not query:
        raise ValueError("Query cannot be empty")

    # Step 1: Strip whitespace
    cleaned = query.strip()

    # Step 2: Normalize internal whitespace
    # "what  is   this" → "what is this"
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Step 3: Remove null bytes and control characters
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned)

    # Step 4: Truncate if too long
    if len(cleaned) > MAX_QUERY_LENGTH:
        cleaned = cleaned[:MAX_QUERY_LENGTH]
        logger.warning(
            f"Query truncated to {MAX_QUERY_LENGTH} chars"
        )

    # Step 5: Validate minimum length
    if len(cleaned) < MIN_QUERY_LENGTH:
        raise ValueError(
            f"Query too short ({len(cleaned)} chars). "
            f"Please ask a more specific question."
        )

    # Step 6: Add question mark if query looks like a question
    # but has no ending punctuation
    # This helps sentence transformers understand it's a question
    question_words = (
        'what', 'who', 'where', 'when', 'why', 'how',
        'which', 'is', 'are', 'was', 'were', 'can', 'could',
        'would', 'should', 'does', 'did', 'do'
    )
    first_word = cleaned.split()[0].lower().rstrip('?')
    if (first_word in question_words and
            not cleaned[-1] in '.!?'):
        cleaned = cleaned + '?'

    logger.debug(f"Query preprocessed: '{query[:30]}' → '{cleaned[:30]}'")
    return cleaned


def extract_keywords(query: str) -> list[str]:
    """
    Extract key terms from a query for logging/debugging.

    Removes common stop words and returns significant terms.
    Used for logging what topics users are searching for.

    Args:
        query: The user's question

    Returns:
        List of significant words
    """
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'shall', 'can', 'need', 'dare', 'ought', 'used',
        'what', 'who', 'where', 'when', 'why', 'how', 'which',
        'this', 'that', 'these', 'those', 'it', 'its',
        'of', 'in', 'to', 'for', 'on', 'at', 'by', 'from',
        'with', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'then', 'once',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both',
        'either', 'neither', 'not', 'only', 'own', 'same',
        'than', 'too', 'very', 'just', 'me', 'my', 'myself',
        'we', 'our', 'you', 'your', 'he', 'she', 'they', 'their',
    }

    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords