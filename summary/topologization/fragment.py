"""Fragment storage management for topologization workspace."""

import json
from dataclasses import dataclass
from pathlib import Path

# Type alias for sentence ID: (chapter_id, fragment_id, sentence_index)
SentenceId = tuple[int, int, int]


@dataclass
class Sentence:
    """Sentence with text and token count."""

    text: str
    token_count: int


class FragmentWriter:
    """Manages fragment storage during text processing.

    Each fragment corresponds to one LLM request and is stored as:
    fragments/chapter-N/fragment-M.json where N is chapter_id and M is fragment_id.
    Fragment IDs are scoped per-chapter (start from 0 for each chapter).
    """

    def __init__(self, workspace_path: Path):
        """Initialize fragment writer.

        Args:
            workspace_path: Root workspace directory
        """
        self.workspace_path = workspace_path
        self.fragments_dir = workspace_path / "fragments"
        self.current_chapter_id = 0  # Current chapter being processed
        self.next_fragment_id = 0  # Fragment ID within current chapter
        self.current_sentences: list[Sentence] = []
        self.current_summary: str | None = None  # Summary for current fragment
        self.is_fragment_open = False

        # Ensure fragments directory exists
        self.fragments_dir.mkdir(parents=True, exist_ok=True)

    def start_fragment(self):
        """Start a new fragment for collecting sentences."""
        if self.is_fragment_open:
            raise RuntimeError("Cannot start new fragment: previous fragment not ended")
        self.current_sentences = []
        self.current_summary = None
        self.is_fragment_open = True

    def start_chapter(self, chapter_id: int):
        """Start processing a new chapter. Resets fragment_id to 0.

        Args:
            chapter_id: Chapter ID (usually from enumeration)

        Raises:
            RuntimeError: If a fragment is currently open
        """
        if self.is_fragment_open:
            raise RuntimeError("Cannot start new chapter: current fragment not ended")

        self.current_chapter_id = chapter_id
        self.next_fragment_id = 0  # Reset fragment ID for new chapter

        # Create chapter directory
        chapter_dir = self.fragments_dir / f"chapter-{chapter_id}"
        chapter_dir.mkdir(parents=True, exist_ok=True)

    def set_summary(self, summary: str):
        """Set summary for current fragment.

        Args:
            summary: Summary text for this fragment

        Raises:
            RuntimeError: If no fragment is currently open
        """
        if not self.is_fragment_open:
            raise RuntimeError("Cannot set summary: no fragment is open. Call start_fragment() first")
        self.current_summary = summary

    def add_sentence(self, text: str, token_count: int) -> SentenceId:
        """Add sentence to current fragment and return its ID.

        Args:
            text: Sentence text to store
            token_count: Token count for this sentence

        Returns:
            Sentence ID as (chapter_id, fragment_id, sentence_index) tuple

        Raises:
            RuntimeError: If no fragment is currently open
        """
        if not self.is_fragment_open:
            raise RuntimeError("Cannot add sentence: no fragment is open. Call start_fragment() first")

        sentence_index = len(self.current_sentences)
        self.current_sentences.append(Sentence(text=text, token_count=token_count))

        # Return 3-tuple with chapter_id
        return (self.current_chapter_id, self.next_fragment_id, sentence_index)

    def end_fragment(self):
        """Write current fragment to disk and prepare for next fragment."""
        if not self.is_fragment_open:
            raise RuntimeError("Cannot end fragment: no fragment is open")

        if not self.current_sentences:
            # Empty fragment, just close it without writing
            self.is_fragment_open = False
            self.current_summary = None
            return

        # Write fragment file in chapter subdirectory
        chapter_dir = self.fragments_dir / f"chapter-{self.current_chapter_id}"
        fragment_path = chapter_dir / f"fragment_{self.next_fragment_id}.json"

        # Build fragment data structure
        fragment_data = {
            "summary": self.current_summary or "",  # Empty string if no summary provided
            "sentences": [{"text": s.text, "token_count": s.token_count} for s in self.current_sentences],
        }

        with open(fragment_path, "w", encoding="utf-8") as f:
            json.dump(fragment_data, f, ensure_ascii=False, indent=2)

        # Move to next fragment
        self.next_fragment_id += 1
        self.current_sentences = []
        self.current_summary = None
        self.is_fragment_open = False

    def finalize(self):
        """Finalize writing. Closes any open fragment."""
        if self.is_fragment_open:
            self.end_fragment()


class FragmentReader:
    """Reads sentence text and summaries from fragment files.

    Provides lazy loading of sentence text and fragment summaries from workspace fragments.
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
            sentence_id: (chapter_id, fragment_id, sentence_index) tuple

        Returns:
            Sentence text

        Raises:
            FileNotFoundError: If fragment file doesn't exist
            IndexError: If sentence_index is out of range
        """
        chapter_id, fragment_id, sentence_index = sentence_id
        chapter_dir = self.fragments_dir / f"chapter-{chapter_id}"
        fragment_path = chapter_dir / f"fragment_{fragment_id}.json"

        with open(fragment_path, encoding="utf-8") as f:
            fragment_data = json.load(f)

        # Handle both old format (list of sentences) and new format (dict with summary + sentences)
        if isinstance(fragment_data, list):
            # Old format: [{"text": ..., "token_count": ...}, ...]
            return fragment_data[sentence_index]["text"]
        else:
            # New format: {"summary": "...", "sentences": [...]}
            return fragment_data["sentences"][sentence_index]["text"]

    def get_summary(self, chapter_id: int, fragment_id: int) -> str:
        """Load fragment summary.

        Args:
            chapter_id: Chapter ID
            fragment_id: Fragment ID within chapter

        Returns:
            Summary text (empty string if not available)

        Raises:
            FileNotFoundError: If fragment file doesn't exist
        """
        chapter_dir = self.fragments_dir / f"chapter-{chapter_id}"
        fragment_path = chapter_dir / f"fragment_{fragment_id}.json"

        with open(fragment_path, encoding="utf-8") as f:
            fragment_data = json.load(f)

        # Handle both old format (list of sentences) and new format (dict with summary + sentences)
        if isinstance(fragment_data, list):
            # Old format: no summary available
            return ""
        else:
            # New format: {"summary": "...", "sentences": [...]}
            return fragment_data.get("summary", "")
