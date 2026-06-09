"""
Text Chunking Service
Handles splitting documents into smaller chunks for embedding and retrieval.
"""
from __future__ import annotations

from typing import NamedTuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunk(NamedTuple):
    """Represents a chunk of text with metadata."""
    content: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict


class DocumentChunker:
    """
    Splits documents into chunks using recursive character-based splitting.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            separators: Custom separators for splitting (optional)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators,
        )

    def split_text(
        self,
        text: str,
        source: str = "",
        extra_metadata: dict | None = None
    ) -> list[TextChunk]:
        """
        Split text into chunks with metadata.

        Args:
            text: The text content to split
            source: Source identifier (e.g., filename)
            extra_metadata: Additional metadata to include with each chunk

        Returns:
            List of TextChunk objects with content and metadata
        """
        if not text.strip():
            return []

        # Use LangChain splitter
        chunks = self._splitter.split_text(text)

        result = []
        current_pos = 0

        for i, chunk_content in enumerate(chunks):
            # Find the actual position in the original text
            # This is approximate due to overlap handling
            start_pos = text.find(chunk_content[:50], current_pos)
            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(chunk_content)

            metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(extra_metadata or {})
            }

            result.append(TextChunk(
                content=chunk_content,
                chunk_index=i,
                char_start=start_pos,
                char_end=end_pos,
                metadata=metadata
            ))

            # Update position for next search (accounting for overlap)
            current_pos = max(start_pos + len(chunk_content) - self.chunk_overlap, current_pos + 1)

        return result

    def estimate_chunk_count(self, text: str) -> int:
        """
        Estimate the number of chunks without actually splitting.

        Args:
            text: The text to estimate

        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0

        text_length = len(text)
        effective_chunk = self.chunk_size - self.chunk_overlap

        if effective_chunk <= 0:
            return 1

        return max(1, (text_length + effective_chunk - 1) // effective_chunk)


# Default chunker instance
default_chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)


def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[TextChunk]:
    """
    Convenience function to chunk text with default or custom settings.

    Args:
        text: Text to chunk
        source: Source identifier
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlapping characters between chunks

    Returns:
        List of TextChunk objects
    """
    if chunk_size == 500 and chunk_overlap == 50:
        return default_chunker.split_text(text, source)

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_text(text, source)
