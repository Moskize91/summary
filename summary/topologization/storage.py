"""Fragment storage management for topologization workspace."""

import json
from pathlib import Path

# Type alias for sentence ID: (fragment_id, sentence_index)
SentenceId = tuple[int, int]


class FragmentWriter:
    """Manages fragment storage during text processing.

    Each fragment corresponds to one LLM request and is stored as:
    fragments/fragment_1.json, fragments/fragment_2.json, etc.
    """

    def __init__(self, workspace_path: Path):
        """Initialize fragment writer.

        Args:
            workspace_path: Root workspace directory
        """
        self.workspace_path = workspace_path
        self.fragments_dir = workspace_path / "fragments"
        self.next_fragment_id = 1  # Start from 1
        self.current_sentences: list[str] = []
        self.is_fragment_open = False

        # Ensure fragments directory exists
        self.fragments_dir.mkdir(parents=True, exist_ok=True)

    def start_fragment(self):
        """Start a new fragment for collecting sentences."""
        if self.is_fragment_open:
            raise RuntimeError("Cannot start new fragment: previous fragment not ended")
        self.current_sentences = []
        self.is_fragment_open = True

    def add_sentence(self, text: str) -> SentenceId:
        """Add sentence to current fragment and return its ID.

        Args:
            text: Sentence text to store

        Returns:
            Sentence ID as (fragment_id, sentence_index) tuple

        Raises:
            RuntimeError: If no fragment is currently open
        """
        if not self.is_fragment_open:
            raise RuntimeError("Cannot add sentence: no fragment is open. Call start_fragment() first")

        sentence_index = len(self.current_sentences)
        self.current_sentences.append(text)

        # Use next_fragment_id as the current fragment ID
        # (will be written when end_fragment is called)
        return (self.next_fragment_id, sentence_index)

    def end_fragment(self):
        """Write current fragment to disk and prepare for next fragment."""
        if not self.is_fragment_open:
            raise RuntimeError("Cannot end fragment: no fragment is open")

        if not self.current_sentences:
            # Empty fragment, just close it without writing
            self.is_fragment_open = False
            return

        # Write fragment_N.json directly in fragments directory
        fragment_path = self.fragments_dir / f"fragment_{self.next_fragment_id}.json"
        with open(fragment_path, "w", encoding="utf-8") as f:
            json.dump(self.current_sentences, f, ensure_ascii=False, indent=2)

        # Move to next fragment
        self.next_fragment_id += 1
        self.current_sentences = []
        self.is_fragment_open = False

    def finalize(self):
        """Finalize writing. Closes any open fragment."""
        if self.is_fragment_open:
            self.end_fragment()


class FragmentReader:
    """Reads sentence text from fragment files.

    Provides lazy loading of sentence text from workspace fragments.
    """

    def __init__(self, workspace_path: Path):
        """Initialize fragment reader.

        Args:
            workspace_path: Root workspace directory
        """
        self.workspace_path = workspace_path
        self.fragments_dir = workspace_path / "fragments"

    def get_sentence(self, sentence_id: SentenceId) -> str:
        """Load sentence text from fragment file.

        Args:
            sentence_id: (fragment_id, sentence_index) tuple

        Returns:
            Sentence text

        Raises:
            FileNotFoundError: If fragment file doesn't exist
            IndexError: If sentence_index is out of range
        """
        fragment_id, sentence_index = sentence_id
        fragment_path = self.fragments_dir / f"fragment_{fragment_id}.json"

        with open(fragment_path, encoding="utf-8") as f:
            sentences = json.load(f)

        return sentences[sentence_index]
