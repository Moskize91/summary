"""Cognitive chunk extractor using LLM."""

import json
import re
from pathlib import Path

from ..llm import LLM
from ..text import normalize_text
from .cognitive_chunk import ChunkBatch, CognitiveChunk
from .fragment import SentenceId
from .working_memory import WorkingMemory


class ChunkExtractor:
    """Extracts cognitive chunks from text using LLM with two-stage extraction."""

    def __init__(self, llm: LLM, extraction_guidance: str):
        """Initialize the extractor.

        Args:
            llm: LLM client instance
            extraction_guidance: Pre-generated retention strategy from user intention
        """
        self.llm = llm
        self.extraction_guidance = extraction_guidance

        # Find prompt templates internally (relative to summary/data/)
        data_dir = Path(__file__).parent.parent / "data" / "topologization"
        self.user_focused_template = data_dir / "user_focused_extraction.jinja"
        self.book_coherence_template = data_dir / "book_coherence_extraction.jinja"

    def extract_user_focused(
        self,
        text: str,
        working_memory: WorkingMemory,
        chunk_sentence_ids: list[SentenceId],
        chunk_sentence_texts: list[str],
    ) -> ChunkBatch | None:
        """Extract user-focused chunks from text (Stage 1).

        Args:
            text: Text segment to process
            working_memory: Current working memory state
            chunk_sentence_ids: List of sentence IDs for this text chunk
            chunk_sentence_texts: List of sentence texts corresponding to sentence IDs

        Returns:
            ChunkBatch containing user-focused chunks
            None if extraction failed
        """
        # Build prompt
        system_prompt = self.llm.load_system_prompt(
            self.user_focused_template,
            extraction_guidance=self.extraction_guidance,
            working_memory=working_memory.format_for_prompt(),
        )

        # Call LLM
        response = self.llm.request(
            system_prompt=system_prompt,
            user_message=text,
            temperature=0.3,
        )

        if not response:
            return None

        # Parse and process
        return self._parse_and_process_chunks(
            response=response,
            chunk_sentence_ids=chunk_sentence_ids,
            chunk_sentence_texts=chunk_sentence_texts,
            expected_type="user_focused",
            metadata_field="retention",
        )

    def extract_book_coherence(
        self,
        text: str,
        working_memory: WorkingMemory,
        user_focused_chunks: list[CognitiveChunk],
        chunk_sentence_ids: list[SentenceId],
        chunk_sentence_texts: list[str],
    ) -> ChunkBatch | None:
        """Extract book-coherence chunks from text (Stage 2).

        Args:
            text: Text segment to process (same as Stage 1)
            working_memory: Current working memory state (includes Stage 1 chunks)
            user_focused_chunks: Chunks extracted in Stage 1 (with assigned integer IDs)
            chunk_sentence_ids: List of sentence IDs for this text chunk
            chunk_sentence_texts: List of sentence texts corresponding to sentence IDs

        Returns:
            ChunkBatch containing book-coherence chunks
            None if extraction failed
        """
        # Format user_focused chunks for template (show with integer IDs)
        user_focused_for_template = []
        for chunk in user_focused_chunks:
            user_focused_for_template.append(
                {
                    "id": chunk.id,  # Show as integer ID (1, 2, 3, ...)
                    "label": chunk.label,
                    "content": chunk.content,
                }
            )

        # Build prompt
        system_prompt = self.llm.load_system_prompt(
            self.book_coherence_template,
            user_focused_chunks=user_focused_for_template,
            working_memory=working_memory.format_for_prompt(),
        )

        # Call LLM
        response = self.llm.request(
            system_prompt=system_prompt,
            user_message=text,
            temperature=0.3,
        )

        if not response:
            return None

        # Parse and process (temp_id should continue from last user_focused letter)
        return self._parse_and_process_chunks(
            response=response,
            chunk_sentence_ids=chunk_sentence_ids,
            chunk_sentence_texts=chunk_sentence_texts,
            expected_type="book_coherence",
            metadata_field="importance",
        )

    def _parse_and_process_chunks(
        self,
        response: str,
        chunk_sentence_ids: list[SentenceId],
        chunk_sentence_texts: list[str],
        expected_type: str,
        metadata_field: str,
    ) -> ChunkBatch | None:
        """Parse LLM response and process chunks.

        Args:
            response: LLM response text
            chunk_sentence_ids: List of sentence IDs
            chunk_sentence_texts: List of sentence texts
            expected_type: Expected chunk type ("user_focused" or "book_coherence")
            metadata_field: Metadata field to extract ("retention" or "importance")

        Returns:
            ChunkBatch or None if parsing failed
        """
        # Parse JSON response
        try:
            parsed_data = self._parse_json_response(response)

            # Check if order is correct (chunks before links)
            order_correct = self._check_json_order(response)

            chunks_data = parsed_data.get("chunks", [])
            links_data = parsed_data.get("links", [])

            chunks = []
            temp_ids = []

            # Build sentence text to ID mapping for efficient lookup
            sentence_text_to_id = {text: sid for text, sid in zip(chunk_sentence_texts, chunk_sentence_ids)}

            for data in chunks_data:
                # Parse source_sentences to find matching sentence IDs
                source_sentences = data.get("source_sentences", [])
                matched_sentence_ids = []
                seen_ids = set()  # Track seen sentence IDs to avoid duplicates

                for source_sent in source_sentences:
                    # Try exact match first
                    if source_sent in sentence_text_to_id:
                        sid = sentence_text_to_id[source_sent]
                        if sid not in seen_ids:
                            matched_sentence_ids.append(sid)
                            seen_ids.add(sid)
                    else:
                        # If no exact match, try fuzzy matching
                        matched_ids = self._fuzzy_match_sentence(source_sent, chunk_sentence_texts, sentence_text_to_id)
                        for matched_id in matched_ids:
                            if matched_id not in seen_ids:
                                matched_sentence_ids.append(matched_id)
                                seen_ids.add(matched_id)

                # If no sentences matched, use minimum sentence ID as fallback
                if not matched_sentence_ids:
                    fallback_id = min(chunk_sentence_ids) if chunk_sentence_ids else (0, 0)
                    matched_sentence_ids = [fallback_id]
                    print(f"[[WARNING]] Failed to match source_sentences for chunk '{data.get('label', 'unknown')}'")
                    print(f"  Source sentences: {source_sentences[:1]}...")
                    print(f"  Using fallback sentence_id: {fallback_id}")

                # Use first matched sentence ID as primary sentence_id (for sorting)
                primary_sentence_id = matched_sentence_ids[0]

                # Parse type field (should match expected_type)
                chunk_type_str = data.get("type", expected_type)
                chunk_type = 1 if chunk_type_str == "user_focused" else 2

                # Extract metadata (retention or importance)
                retention = data.get("retention") if metadata_field == "retention" else None
                importance = data.get("importance") if metadata_field == "importance" else None

                chunk = CognitiveChunk(
                    id=0,  # Will be assigned by WorkingMemory
                    generation=0,  # Will be assigned by WorkingMemory
                    sentence_id=primary_sentence_id,
                    sentence_ids=matched_sentence_ids,  # Store all matched sentence IDs
                    label=data.get("label", ""),
                    content=data.get("content", ""),
                    chunk_type=chunk_type,
                    links=[],  # Will be populated after ID assignment
                    retention=retention,
                    importance=importance,
                )
                chunks.append(chunk)
                temp_ids.append(data.get("temp_id", ""))

            return ChunkBatch(
                chunks=chunks,
                temp_ids=temp_ids,
                links=links_data,  # Keep strength field from LLM
                order_correct=order_correct,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response[:200]}...")
            return None

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response.

        Handles cases where LLM includes extra text around JSON.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON data as dict with "chunks" and "links" keys
        """
        # Try to extract JSON object from response
        # Look for JSON object pattern
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            # Ensure keys exist and enforce order
            if "chunks" not in parsed or "links" not in parsed:
                raise ValueError("Missing 'chunks' or 'links' in response")

            # Return with enforced key order (Python 3.7+ preserves insertion order)
            return {
                "chunks": parsed["chunks"],
                "links": parsed["links"],
            }

        # If no object found, try parsing the whole response
        parsed = json.loads(response)
        return {
            "chunks": parsed["chunks"],
            "links": parsed["links"],
        }

    def _check_json_order(self, response: str) -> bool:
        """Check if JSON keys appear in correct order (chunks before links).

        Args:
            response: Raw LLM response text

        Returns:
            True if chunks appears before links, False otherwise
        """
        # Extract the JSON portion
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # Try the whole response
            json_str = response
        else:
            json_str = json_match.group(0)

        # Find positions of "chunks" and "links" keys in the raw JSON string
        chunks_pos = json_str.find('"chunks"')
        links_pos = json_str.find('"links"')

        if chunks_pos == -1 or links_pos == -1:
            return False

        return chunks_pos < links_pos

    def _fuzzy_match_sentence(
        self,
        source_sent: str,
        candidate_texts: list[str],
        text_to_id: dict[str, SentenceId],
    ) -> list[SentenceId]:
        """Find all matching sentences using fuzzy matching.

        Args:
            source_sent: Source sentence to match (may contain multiple actual sentences)
            candidate_texts: List of candidate sentence texts
            text_to_id: Mapping from sentence text to sentence ID

        Returns:
            List of matched sentence IDs (may be multiple if source contains multiple sentences)
        """
        # Normalize using robust text normalization (handles punctuation, spaces, accents, etc.)
        source_clean = normalize_text(source_sent)

        matched_ids = []

        for candidate in candidate_texts:
            candidate_clean = normalize_text(candidate)

            # Check if they are exactly the same after normalization
            if source_clean == candidate_clean:
                return [text_to_id[candidate]]

            # Skip if either is too short
            if len(source_clean) < 20 or len(candidate_clean) < 20:
                continue

            # Check substring containment (both directions)
            if source_clean in candidate_clean or candidate_clean in source_clean:
                matched_ids.append(text_to_id[candidate])
                continue

            # Use longest common substring for fuzzy matching
            match_len = 0
            for i in range(len(source_clean)):
                for j in range(len(candidate_clean)):
                    k = 0
                    while (
                        i + k < len(source_clean)
                        and j + k < len(candidate_clean)
                        and source_clean[i + k] == candidate_clean[j + k]
                    ):
                        k += 1
                    match_len = max(match_len, k)

            # Calculate similarity against the shorter string (more lenient)
            min_len = min(len(source_clean), len(candidate_clean))
            similarity = match_len / min_len if min_len > 0 else 0

            if similarity >= 0.8:  # 80% similarity threshold
                matched_ids.append(text_to_id[candidate])

        return matched_ids if matched_ids else []
