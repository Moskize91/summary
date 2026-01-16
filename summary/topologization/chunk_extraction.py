"""Cognitive chunk extractor using LLM."""

import json
import re
from pathlib import Path

from ..llm import LLM
from .cognitive_chunk import ChunkBatch, CognitiveChunk
from .storage import SentenceId
from .working_memory import WorkingMemory


class ChunkExtractor:
    """Extracts cognitive chunks from text using LLM."""

    def __init__(self, llm: LLM, extraction_guidance: str):
        """Initialize the extractor.

        Args:
            llm: LLM client instance
            extraction_guidance: Pre-generated guidance from user intention
        """
        self.llm = llm
        self.extraction_guidance = extraction_guidance

        # Find prompt template internally (relative to summary/data/)
        self.prompt_template_path = Path(__file__).parent.parent / "data" / "topologization" / "chunk_extraction.jinja"

    def extract_chunks(
        self,
        text: str,
        working_memory: WorkingMemory,
        chunk_sentence_ids: list[SentenceId],
        chunk_sentence_texts: list[str],
    ) -> ChunkBatch | None:
        """Extract cognitive chunks from text.

        Args:
            text: Text segment to process
            working_memory: Current working memory state
            chunk_sentence_ids: List of sentence IDs for this text chunk
            chunk_sentence_texts: List of sentence texts corresponding to sentence IDs

        Returns:
            ChunkBatch containing chunks, temp_ids, links, and order correctness
            None if extraction failed
        """
        # Build prompt
        system_prompt = self.llm.load_system_prompt(
            self.prompt_template_path,
            extraction_guidance=self.extraction_guidance,
            working_memory=working_memory.format_for_prompt(),
        )

        # Call LLM
        response = self.llm.request(
            system_prompt=system_prompt,
            user_message=text,
            temperature=0.3,  # Lower temperature for more consistent extraction
        )

        if not response:
            return None

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
                        matched_id = self._fuzzy_match_sentence(source_sent, chunk_sentence_texts, sentence_text_to_id)
                        if matched_id and matched_id not in seen_ids:
                            matched_sentence_ids.append(matched_id)
                            seen_ids.add(matched_id)

                # If no sentences matched, use minimum sentence ID as fallback
                if not matched_sentence_ids:
                    matched_sentence_ids = [min(chunk_sentence_ids)] if chunk_sentence_ids else [(0, 0)]

                # Use first matched sentence ID as primary sentence_id (for sorting)
                primary_sentence_id = matched_sentence_ids[0]

                chunk = CognitiveChunk(
                    id=0,  # Will be assigned by WorkingMemory
                    generation=0,  # Will be assigned by WorkingMemory
                    sentence_id=primary_sentence_id,
                    sentence_ids=matched_sentence_ids,  # Store all matched sentence IDs
                    label=data.get("label", ""),
                    content=data.get("content", ""),
                    links=[],  # Will be populated after ID assignment
                )
                chunks.append(chunk)
                temp_ids.append(data.get("temp_id", ""))

            return ChunkBatch(
                chunks=chunks,
                temp_ids=temp_ids,
                links=links_data,
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
    ) -> SentenceId | None:
        """Find best matching sentence using fuzzy matching.

        Args:
            source_sent: Source sentence to match
            candidate_texts: List of candidate sentence texts
            text_to_id: Mapping from sentence text to sentence ID

        Returns:
            Matched sentence ID, or None if no good match found
        """
        # Normalize source sentence (strip whitespace, convert to lowercase)
        normalized_source = source_sent.strip().lower()

        # Try to find best match
        best_match = None
        best_score = 0.0

        for candidate in candidate_texts:
            normalized_candidate = candidate.strip().lower()

            # Simple similarity: ratio of common characters
            # This is a simple approach; could use Levenshtein distance or other algorithms
            if normalized_source == normalized_candidate:
                return text_to_id[candidate]

            # Calculate overlap ratio
            if len(normalized_source) > 0:
                # Count how many characters from source appear in candidate
                common_chars = sum(1 for c in normalized_source if c in normalized_candidate)
                score = common_chars / len(normalized_source)

                if score > best_score:
                    best_score = score
                    best_match = candidate

        # Return best match if score is above threshold (80%)
        if best_score > 0.8 and best_match:
            return text_to_id[best_match]

        return None
