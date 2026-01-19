"""Text fragmentation utilities for splitting large texts into manageable fragments."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from spacy.lang.xx import MultiLanguage
from spacy.language import Language
from tiktoken import Encoding

from .fragment import FragmentWriter, SentenceId


@dataclass
class FragmentWithSentences:
    """A text fragment with associated sentence IDs."""

    text: str
    sentence_ids: list[SentenceId]
    sentence_texts: list[str]  # Sentence texts corresponding to sentence_ids
    sentence_token_counts: list[int]  # Token counts corresponding to sentence_ids


class TextFragmenter:
    """Splits text into sentences and groups them into fragments.

    Integrates with FragmentWriter to store sentences in workspace fragments.
    """

    def __init__(
        self,
        fragment_writer: FragmentWriter,
        encoding: Encoding,
        max_fragment_tokens: int = 800,
        batch_size: int = 50000,
    ):
        """Initialize the text fragmenter.

        Args:
            fragment_writer: FragmentWriter for storing sentences
            encoding: Tiktoken encoding for token counting
            max_fragment_tokens: Maximum token count for each fragment
            batch_size: Size of text batch to process with spacy at once
        """
        self.fragment_writer = fragment_writer
        self.encoding = encoding
        self.max_fragment_tokens = max_fragment_tokens
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

    def stream_fragments_from_file(self, file_path: Path) -> Generator[FragmentWithSentences, None, None]:
        """Stream text fragments from a file using spacy's pipe() for efficient processing.

        Args:
            file_path: Path to the text file

        Yields:
            FragmentWithSentences objects with text and sentence IDs
        """
        current_fragment = []
        current_fragment_tokens = 0
        current_sentence_ids = []
        current_sentence_texts = []
        current_sentence_token_counts = []

        # Use spacy's pipe() for efficient batch processing
        for doc in self._nlp.pipe(self._generate_text_batches(file_path), batch_size=10):
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue

                # Calculate token count for this sentence
                sentence_token_count = len(self.encoding.encode(sentence_text))

                # Yield fragment if adding sentence would exceed limit
                if current_fragment and current_fragment_tokens + sentence_token_count > self.max_fragment_tokens:
                    # End current fragment and yield
                    self.fragment_writer.end_fragment()
                    yield FragmentWithSentences(
                        text="".join(current_fragment),
                        sentence_ids=current_sentence_ids.copy(),
                        sentence_texts=current_sentence_texts.copy(),
                        sentence_token_counts=current_sentence_token_counts.copy(),
                    )
                    current_fragment = []
                    current_fragment_tokens = 0
                    current_sentence_ids = []
                    current_sentence_texts = []
                    current_sentence_token_counts = []

                # Start new fragment if needed (first sentence of new fragment)
                if not current_fragment:
                    self.fragment_writer.start_fragment()

                # Add sentence to fragment writer with token count and get sentence ID
                sentence_id = self.fragment_writer.add_sentence(sentence_text, sentence_token_count)

                current_fragment.append(sentence_text)
                current_fragment_tokens += sentence_token_count
                current_sentence_ids.append(sentence_id)
                current_sentence_texts.append(sentence_text)
                current_sentence_token_counts.append(sentence_token_count)

        # Yield the last fragment if it exists
        if current_fragment:
            self.fragment_writer.end_fragment()
            yield FragmentWithSentences(
                text="".join(current_fragment),
                sentence_ids=current_sentence_ids.copy(),
                sentence_texts=current_sentence_texts.copy(),
                sentence_token_counts=current_sentence_token_counts.copy(),
            )

    def finalize(self):
        """Finalize text fragmentation and flush remaining fragments."""
        self.fragment_writer.finalize()
