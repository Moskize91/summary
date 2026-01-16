"""Cognitive chunk extractor using LLM."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLM
from .cognitive_chunk import CognitiveChunk
from .working_memory import WorkingMemory


@dataclass
class ExtractionResult:
    """Result of chunk extraction containing chunks and their relationships."""

    chunks: list[CognitiveChunk]  # Extracted chunks (id=0, to be assigned)
    temp_ids: list[str]  # Temporary IDs corresponding to chunks
    links: list[dict]  # Raw link data: [{"from": ..., "to": ...}]
    order_correct: bool  # Whether JSON key order was correct


class ChunkExtractor:
    """Extracts cognitive chunks from text using LLM."""

    def __init__(self, llm: LLM, prompt_template_path: Path):
        """Initialize the extractor.

        Args:
            llm: LLM client instance
            prompt_template_path: Path to the extraction prompt template
        """
        self.llm = llm
        self.prompt_template_path = prompt_template_path

    def extract_chunks(
        self, text: str, working_memory: WorkingMemory, sentence_map: dict[int, str]
    ) -> ExtractionResult | None:
        """Extract cognitive chunks from text.

        Args:
            text: Text segment to process
            working_memory: Current working memory state
            sentence_map: Mapping from sentence ID to sentence text

        Returns:
            ExtractionResult containing chunks, temp_ids, links, and order correctness
            None if extraction failed
        """
        # Build prompt
        system_prompt = self.llm.load_system_prompt(
            self.prompt_template_path,
            working_memory=working_memory.format_for_prompt(),
            new_text=text,
        )

        # Call LLM
        response = self.llm.request(
            system_prompt=system_prompt,
            user_message="Please extract key information in JSON format.",
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

            for data in chunks_data:
                # Find sentence_id from source_sentences
                source_sentences = data.get("source_sentences", [])
                sentence_id = self._find_sentence_id(source_sentences, sentence_map)

                chunk = CognitiveChunk(
                    id=0,  # Will be assigned by WorkingMemory
                    generation=0,  # Will be assigned by WorkingMemory
                    sentence_id=sentence_id,
                    label=data.get("label", ""),
                    content=data.get("content", ""),
                    links=[],  # Will be populated after ID assignment
                )
                chunks.append(chunk)
                temp_ids.append(data.get("temp_id", ""))

            return ExtractionResult(
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

    def _find_sentence_id(self, source_sentences: list[str], sentence_map: dict[int, str]) -> int:
        """Find the minimum sentence ID from source sentences using substring matching.

        Args:
            source_sentences: List of sentence strings from LLM output
            sentence_map: Mapping from sentence ID to sentence text

        Returns:
            Minimum sentence ID found, or 0 if no match found
        """
        found_ids = []

        for source_sent in source_sentences:
            source_sent_stripped = source_sent.strip()
            if not source_sent_stripped:
                continue

            # Try to find this sentence in sentence_map
            for sent_id, sent_text in sentence_map.items():
                # Use substring matching (source_sent might be truncated by LLM)
                if source_sent_stripped in sent_text or sent_text in source_sent_stripped:
                    found_ids.append(sent_id)
                    break

        if not found_ids:
            print(f"Warning: Could not find sentence IDs for source_sentences: {source_sentences[:2]}...")
            return 0

        return min(found_ids)
