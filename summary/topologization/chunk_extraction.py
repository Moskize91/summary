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
        chunk_sentence_token_counts: list[int],
    ) -> tuple[ChunkBatch | None, str | None]:
        """Extract user-focused chunks from text (Stage 1).

        Args:
            text: Text segment to process
            working_memory: Current working memory state
            chunk_sentence_ids: List of sentence IDs for this text chunk
            chunk_sentence_texts: List of sentence texts corresponding to sentence IDs
            chunk_sentence_token_counts: List of token counts corresponding to sentence IDs

        Returns:
            Tuple of (ChunkBatch, fragment_summary):
            - ChunkBatch containing user-focused chunks (None if extraction failed)
            - fragment_summary string for this fragment (None if extraction failed)
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
            return None, None

        # Parse and process
        chunk_batch, fragment_summary = self._parse_and_process_chunks(
            response=response,
            chunk_sentence_ids=chunk_sentence_ids,
            chunk_sentence_texts=chunk_sentence_texts,
            chunk_sentence_token_counts=chunk_sentence_token_counts,
            expected_type="user_focused",
            metadata_field="retention",
        )
        return chunk_batch, fragment_summary

    def extract_book_coherence(
        self,
        text: str,
        working_memory: WorkingMemory,
        user_focused_chunks: list[CognitiveChunk],
        chunk_sentence_ids: list[SentenceId],
        chunk_sentence_texts: list[str],
        chunk_sentence_token_counts: list[int],
    ) -> ChunkBatch | None:
        """Extract book-coherence chunks from text (Stage 2).

        Args:
            text: Text segment to process (same as Stage 1)
            working_memory: Current working memory state (includes Stage 1 chunks)
            user_focused_chunks: Chunks extracted in Stage 1 (with assigned integer IDs)
            chunk_sentence_ids: List of sentence IDs for this text chunk
            chunk_sentence_texts: List of sentence texts corresponding to sentence IDs
            chunk_sentence_token_counts: List of token counts corresponding to sentence IDs

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
            working_memory=working_memory.format_for_prompt(include_current_fragment=False),
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
        chunk_batch, _ = self._parse_and_process_chunks(
            response=response,
            chunk_sentence_ids=chunk_sentence_ids,
            chunk_sentence_texts=chunk_sentence_texts,
            chunk_sentence_token_counts=chunk_sentence_token_counts,
            expected_type="book_coherence",
            metadata_field="importance",
        )
        return chunk_batch

    def _parse_and_process_chunks(
        self,
        response: str,
        chunk_sentence_ids: list[SentenceId],
        chunk_sentence_texts: list[str],
        chunk_sentence_token_counts: list[int],
        expected_type: str,
        metadata_field: str,
    ) -> tuple[ChunkBatch | None, str | None]:
        """Parse LLM response and process chunks.

        Args:
            response: LLM response text
            chunk_sentence_ids: List of sentence IDs
            chunk_sentence_texts: List of sentence texts
            chunk_sentence_token_counts: List of token counts corresponding to sentence IDs
            expected_type: Expected chunk type ("user_focused" or "book_coherence")
            metadata_field: Metadata field to extract ("retention" or "importance")

        Returns:
            Tuple of (ChunkBatch, fragment_summary):
            - ChunkBatch or None if parsing failed
            - fragment_summary string (only for user_focused, None for book_coherence)
        """
        # Parse JSON response
        try:
            parsed_data = self._parse_json_response(response, expected_type)

            # Check if order is correct
            order_correct = self._check_json_order(response, expected_type)

            chunks_data = parsed_data.get("chunks", [])
            links_data = parsed_data.get("links", [])
            importance_annotations = parsed_data.get("importance_annotations")  # Only present in Stage 2
            fragment_summary = parsed_data.get("fragment_summary")  # Only present in user_focused (Stage 1)

            chunks = []
            temp_ids = []

            # Build sentence text to ID mapping and ID to token count mapping for efficient lookup
            sentence_text_to_id = {text: sid for text, sid in zip(chunk_sentence_texts, chunk_sentence_ids)}
            sentence_id_to_tokens = {
                sid: tokens for sid, tokens in zip(chunk_sentence_ids, chunk_sentence_token_counts)
            }

            for data in chunks_data:
                # Parse source_sentences to find matching sentence IDs
                # Handle common AI typos: "source_sences" (missing 't'), "source_sentances", etc.
                source_sentences = (
                    data.get("source_sentences")
                    or data.get("source_sences")  # Common typo: missing 't'
                    or data.get("source_sentances")  # Common typo: wrong spelling
                    or []
                )
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
                    fallback_id = min(chunk_sentence_ids) if chunk_sentence_ids else (0, 0, 0)
                    matched_sentence_ids = [fallback_id]
                    print(f"[[WARNING]] Failed to match source_sentences for chunk '{data.get('label', 'unknown')}'")
                    print(f"  Source sentences: {source_sentences[:1]}...")
                    print(f"  Using fallback sentence_id: {fallback_id}")

                # Use first matched sentence ID as primary sentence_id (for sorting)
                primary_sentence_id = matched_sentence_ids[0]

                # Calculate total tokens from all matched sentences
                total_tokens = sum(sentence_id_to_tokens.get(sid, 0) for sid in matched_sentence_ids)

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
                    links=[],  # Will be populated after ID assignment
                    retention=retention,
                    importance=importance,
                    tokens=total_tokens,
                )
                chunks.append(chunk)
                temp_ids.append(data.get("temp_id", ""))

            chunk_batch = ChunkBatch(
                chunks=chunks,
                temp_ids=temp_ids,
                links=links_data,  # Keep strength field from LLM
                order_correct=order_correct,
                importance_annotations=importance_annotations,
            )
            return chunk_batch, fragment_summary

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response[:200]}...")
            return None, None

    def _parse_json_response(self, response: str, expected_type: str) -> dict:
        """Parse JSON from LLM response.

        Handles cases where LLM includes extra text around JSON.

        Args:
            response: LLM response text
            expected_type: Expected chunk type ("user_focused" or "book_coherence")

        Returns:
            Parsed JSON data as dict:
            - For user_focused: {"fragment_summary", "chunks", "links"}
            - For book_coherence: {"importance_annotations", "chunks", "links"}
        """
        # Try to extract JSON object from response
        # Look for JSON object pattern
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)

            # For user_focused: ensure fragment_summary, chunks and links exist
            if expected_type == "user_focused":
                if "fragment_summary" not in parsed or "chunks" not in parsed or "links" not in parsed:
                    raise ValueError("Missing 'fragment_summary', 'chunks', or 'links' in user_focused response")
                return {
                    "fragment_summary": parsed["fragment_summary"],
                    "chunks": parsed["chunks"],
                    "links": parsed["links"],
                }
            # For book_coherence: ensure importance_annotations, chunks, and links exist
            else:
                if "importance_annotations" not in parsed or "chunks" not in parsed or "links" not in parsed:
                    raise ValueError(
                        "Missing 'importance_annotations', 'chunks', or 'links' in book_coherence response"
                    )
                return {
                    "importance_annotations": parsed["importance_annotations"],
                    "chunks": parsed["chunks"],
                    "links": parsed["links"],
                }

        # If no object found, try parsing the whole response
        parsed = json.loads(response)
        if expected_type == "user_focused":
            return {
                "fragment_summary": parsed["fragment_summary"],
                "chunks": parsed["chunks"],
                "links": parsed["links"],
            }
        else:
            return {
                "importance_annotations": parsed["importance_annotations"],
                "chunks": parsed["chunks"],
                "links": parsed["links"],
            }

    def _check_json_order(self, response: str, expected_type: str) -> bool:
        """Check if JSON keys appear in correct order.

        For user_focused: fragment_summary before chunks before links
        For book_coherence: importance_annotations before chunks before links

        Args:
            response: Raw LLM response text
            expected_type: Expected chunk type ("user_focused" or "book_coherence")

        Returns:
            True if keys are in correct order, False otherwise
        """
        # Extract the JSON portion
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            # Try the whole response
            json_str = response
        else:
            json_str = json_match.group(0)

        if expected_type == "user_focused":
            # Find positions of "fragment_summary", "chunks" and "links" keys
            summary_pos = json_str.find('"fragment_summary"')
            chunks_pos = json_str.find('"chunks"')
            links_pos = json_str.find('"links"')

            if summary_pos == -1 or chunks_pos == -1 or links_pos == -1:
                return False

            return summary_pos < chunks_pos < links_pos
        else:
            # For book_coherence: check importance_annotations -> chunks -> links
            importance_pos = json_str.find('"importance_annotations"')
            chunks_pos = json_str.find('"chunks"')
            links_pos = json_str.find('"links"')

            if importance_pos == -1 or chunks_pos == -1 or links_pos == -1:
                return False

            return importance_pos < chunks_pos < links_pos

    def _fuzzy_match_sentence(
        self,
        source_sent: str,
        candidate_texts: list[str],
        text_to_id: dict[str, SentenceId],
    ) -> list[SentenceId]:
        """Find all matching sentences using fuzzy matching.

        Supports two modes:
        1. Exact/fuzzy matching for original text
        2. Ellipsis matching for format like "prefix...suffix"

        Args:
            source_sent: Source sentence to match (may contain ellipsis or be original text)
            candidate_texts: List of candidate sentence texts
            text_to_id: Mapping from sentence text to sentence ID

        Returns:
            List of matched sentence IDs
        """
        # Normalize using robust text normalization (handles punctuation, spaces, accents, etc.)
        source_clean = normalize_text(source_sent)

        matched_ids = []

        # Phase 1: Try normal matching (exact + fuzzy)
        for candidate in candidate_texts:
            candidate_clean = normalize_text(candidate)

            # Check if they are exactly the same after normalization
            if source_clean == candidate_clean:
                return [text_to_id[candidate]]

            # Skip if either is too short (reduced from 20 to 10 to handle shorter sentences)
            if len(source_clean) < 10 or len(candidate_clean) < 10:
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

        # If normal matching succeeded, return results
        if matched_ids:
            return matched_ids

        # Phase 2: Try ellipsis matching (only if Phase 1 failed and contains "...")
        if "..." in source_sent:
            ellipsis_matches = self._match_with_ellipsis(source_sent, candidate_texts, text_to_id)
            if ellipsis_matches:
                return ellipsis_matches

        # Phase 3: Try splitting by sentence boundaries (。or .)
        # Sometimes LLM merges multiple sentences together
        split_matches = self._match_by_splitting(source_sent, candidate_texts, text_to_id)
        if split_matches:
            return split_matches

        return []

    def _match_with_ellipsis(
        self,
        source_sent: str,
        candidate_texts: list[str],
        text_to_id: dict[str, SentenceId],
    ) -> list[SentenceId]:
        """Match sentence using ellipsis format: "prefix...suffix".

        Args:
            source_sent: Source sentence with ellipsis (e.g., "前面...后面")
            candidate_texts: List of candidate sentence texts
            text_to_id: Mapping from sentence text to sentence ID

        Returns:
            List of matched sentence IDs
        """
        # Split by "..." and strip whitespace
        parts = [p.strip() for p in source_sent.split("...")]

        # Validate: should have exactly 2 parts (prefix and suffix)
        if len(parts) != 2:
            return []

        prefix, suffix = parts

        # Normalize prefix and suffix
        prefix_clean = normalize_text(prefix)
        suffix_clean = normalize_text(suffix)

        # Both parts should be non-empty and reasonably long
        if len(prefix_clean) < 5 or len(suffix_clean) < 5:
            return []

        matched_ids = []

        for candidate in candidate_texts:
            candidate_clean = normalize_text(candidate)

            # Check if candidate starts with prefix and ends with suffix
            if candidate_clean.startswith(prefix_clean) and candidate_clean.endswith(suffix_clean):
                matched_ids.append(text_to_id[candidate])

        return matched_ids

    def _match_by_splitting(
        self,
        source_sent: str,
        candidate_texts: list[str],
        text_to_id: dict[str, SentenceId],
    ) -> list[SentenceId]:
        """Match by splitting source sentence into parts.

        Sometimes LLM merges multiple sentences together. Try splitting by
        Chinese/English period and match each part separately.

        Args:
            source_sent: Source sentence that may contain multiple sentences
            candidate_texts: List of candidate sentence texts
            text_to_id: Mapping from sentence text to sentence ID

        Returns:
            List of matched sentence IDs (in order)
        """
        import re

        # Split by Chinese period (。) or English period followed by space/CJK char
        # This regex splits on: 。 or . followed by space or CJK character
        parts = re.split(r"[。]|(?<=\.)\s*(?=[\u4e00-\u9fff])", source_sent)
        parts = [p.strip() for p in parts if p.strip()]

        # Need at least 2 parts to consider it as merged sentences
        if len(parts) < 2:
            return []

        matched_ids = []
        seen_ids = set()

        for part in parts:
            part_clean = normalize_text(part)
            if len(part_clean) < 5:  # Skip very short fragments
                continue

            # Try to match this part
            for candidate in candidate_texts:
                candidate_clean = normalize_text(candidate)

                # Check exact match
                if part_clean == candidate_clean:
                    sid = text_to_id[candidate]
                    if sid not in seen_ids:
                        matched_ids.append(sid)
                        seen_ids.add(sid)
                    break

                # Check if part is contained in candidate or vice versa
                if len(part_clean) >= 10 and len(candidate_clean) >= 10:
                    if part_clean in candidate_clean or candidate_clean in part_clean:
                        sid = text_to_id[candidate]
                        if sid not in seen_ids:
                            matched_ids.append(sid)
                            seen_ids.add(sid)
                        break

        return matched_ids
