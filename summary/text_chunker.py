"""Text chunking utilities for splitting large texts into manageable chunks."""

from collections.abc import Generator
from pathlib import Path

from spacy.lang.xx import MultiLanguage
from spacy.language import Language


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

    def stream_chunks_from_file(self, file_path: Path) -> Generator[str, None, None]:
        """Stream text chunks from a file using spacy's pipe() for efficient processing.

        Args:
            file_path: Path to the text file

        Yields:
            Text chunks with length <= max_chunk_length
        """
        current_chunk = []
        current_chunk_length = 0

        # Use spacy's pipe() for efficient batch processing
        for doc in self._nlp.pipe(self._generate_text_batches(file_path), batch_size=10):
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue

                sentence_length = len(sentence_text)

                # Yield chunk if adding sentence would exceed limit
                if current_chunk and current_chunk_length + sentence_length > self.max_chunk_length:
                    yield "".join(current_chunk)
                    current_chunk = []
                    current_chunk_length = 0

                current_chunk.append(sentence_text)
                current_chunk_length += sentence_length

        # Yield the last chunk if it exists
        if current_chunk:
            yield "".join(current_chunk)
