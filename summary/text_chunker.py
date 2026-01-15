"""Text chunking utilities for splitting large texts into manageable chunks."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from spacy.lang.xx import MultiLanguage
from spacy.language import Language


@dataclass
class ChunkWithSentences:
    """A text chunk with associated sentence IDs."""

    text: str
    sentence_ids: list[int]


class TextChunker:
    """Splits text into sentences and groups them into chunks."""

    def __init__(self, max_chunk_length: int = 2000, batch_size: int = 50000):
        """Initialize the text chunker.

        Args:
            max_chunk_length: Maximum character length for each chunk
            batch_size: Size of text batch to process with spacy at once
        """
        self.max_chunk_length = max_chunk_length
        self.batch_size = batch_size
        self._nlp = self._load_language_model()
        self._sentence_map: dict[int, str] = {}  # sentence_id -> sentence_text
        self._next_sentence_id: int = 1

    def _load_language_model(self) -> Language:
        """Load the spacy language model with sentencizer."""
        nlp: Language = MultiLanguage()
        nlp.add_pipe("sentencizer")
        return nlp

    def _generate_text_batches(self, file_path: Path) -> Generator[str, None, None]:
        """Generate text batches from a file for streaming processing.

        Args:
            file_path: Path to the text file

        Yields:
            Text batches of approximately batch_size characters
        """
        text_buffer = ""
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                # FIXME: 这里按行读的，碰到巨长行还是会爆炸！
                text_buffer += line

                # Yield batch when buffer reaches batch_size
                if len(text_buffer) >= self.batch_size:
                    yield text_buffer
                    text_buffer = ""

        # Yield remaining text
        if text_buffer:
            yield text_buffer

    def stream_chunks_from_file(self, file_path: Path) -> Generator[ChunkWithSentences, None, None]:
        """Stream text chunks from a file using spacy's pipe() for efficient processing.

        Args:
            file_path: Path to the text file

        Yields:
            ChunkWithSentences objects with text and sentence IDs
        """
        current_chunk = []
        current_chunk_length = 0
        current_sentence_ids = []

        # Use spacy's pipe() for efficient batch processing
        for doc in self._nlp.pipe(self._generate_text_batches(file_path), batch_size=10):
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue

                sentence_length = len(sentence_text)

                # Yield chunk if adding sentence would exceed limit
                if current_chunk and current_chunk_length + sentence_length > self.max_chunk_length:
                    yield ChunkWithSentences(
                        text="".join(current_chunk),
                        sentence_ids=current_sentence_ids.copy(),
                    )
                    current_chunk = []
                    current_chunk_length = 0
                    current_sentence_ids = []

                # Assign sentence ID and save to map
                sentence_id = self._next_sentence_id
                self._sentence_map[sentence_id] = sentence_text
                self._next_sentence_id += 1

                current_chunk.append(sentence_text)
                current_chunk_length += sentence_length
                current_sentence_ids.append(sentence_id)

        # Yield the last chunk if it exists
        if current_chunk:
            yield ChunkWithSentences(
                text="".join(current_chunk),
                sentence_ids=current_sentence_ids.copy(),
            )

    def get_sentence_map(self) -> dict[int, str]:
        """Get the sentence ID to sentence text mapping.

        Returns:
            Dictionary mapping sentence IDs to sentence text
        """
        return self._sentence_map.copy()

    def reset_sentence_tracking(self) -> None:
        """Reset sentence ID tracking and clear sentence map.

        Call this before processing a new file if you want fresh sentence IDs.
        """
        self._sentence_map.clear()
        self._next_sentence_id = 1
