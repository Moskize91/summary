"""Text fragmentation utilities for splitting large texts into manageable fragments."""

from collections.abc import Generator, Iterable
from dataclasses import dataclass

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
    """Splits pre-processed sentences into fragments for a single chapter.

    Integrates with FragmentWriter to store sentences in workspace fragments.
    """

    def __init__(
        self,
        fragment_writer: FragmentWriter,
        encoding: Encoding,
        max_fragment_tokens: int = 800,
    ):
        """Initialize the text fragmenter.

        Args:
            fragment_writer: FragmentWriter for storing sentences (already initialized with chapter_id)
            encoding: Tiktoken encoding for token counting (not used, kept for compatibility)
            max_fragment_tokens: Maximum token count for each fragment
        """
        self.fragment_writer = fragment_writer
        self.encoding = encoding
        self.max_fragment_tokens = max_fragment_tokens

    def stream_fragments(
        self, sentences: Iterable[tuple[int, str]]
    ) -> Generator[FragmentWithSentences, None, None]:
        """Stream text fragments from pre-processed sentences for a single chapter.

        NOTE: Fragments are yielded with fragment_writer still open (fragment not ended yet).
        The caller should call set_summary() if needed, then the next iteration will end it.
        The last fragment must be ended by calling finalize().

        Args:
            sentences: Iterable of (token_count, sentence_text) tuples for this chapter

        Yields:
            FragmentWithSentences objects with chapter-aware sentence IDs
        """
        fragment_pending = False  # Track if there's a fragment waiting to be ended

        current_fragment = []
        current_fragment_tokens = 0
        current_sentence_ids = []
        current_sentence_texts = []
        current_sentence_token_counts = []

        for token_count, sentence_text in sentences:
            sentence_text = sentence_text.strip()
            if not sentence_text:
                continue

            # Yield fragment if adding sentence would exceed limit
            if current_fragment and current_fragment_tokens + token_count > self.max_fragment_tokens:
                # Yield current fragment (WITHOUT ending it yet - caller can still set_summary)
                yield FragmentWithSentences(
                    text="".join(current_fragment),
                    sentence_ids=current_sentence_ids.copy(),
                    sentence_texts=current_sentence_texts.copy(),
                    sentence_token_counts=current_sentence_token_counts.copy(),
                )
                fragment_pending = True
                current_fragment = []
                current_fragment_tokens = 0
                current_sentence_ids = []
                current_sentence_texts = []
                current_sentence_token_counts = []

            # Start new fragment if needed (first sentence of new fragment)
            if not current_fragment:
                # End previous fragment if it's still pending
                if fragment_pending:
                    self.fragment_writer.end_fragment()
                    fragment_pending = False
                self.fragment_writer.start_fragment()

            # Add sentence to fragment writer and get sentence ID (3-tuple with chapter_id)
            sentence_id = self.fragment_writer.add_sentence(sentence_text, token_count)

            current_fragment.append(sentence_text)
            current_fragment_tokens += token_count
            current_sentence_ids.append(sentence_id)
            current_sentence_texts.append(sentence_text)
            current_sentence_token_counts.append(token_count)

        # Yield the last fragment if it exists (WITHOUT ending it yet)
        if current_fragment:
            yield FragmentWithSentences(
                text="".join(current_fragment),
                sentence_ids=current_sentence_ids.copy(),
                sentence_texts=current_sentence_texts.copy(),
                sentence_token_counts=current_sentence_token_counts.copy(),
            )
            fragment_pending = True  # Mark fragment as pending for finalize()

    def finalize(self):
        """Finalize text fragmentation and flush remaining fragments."""
        self.fragment_writer.finalize()
