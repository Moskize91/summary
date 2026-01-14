"""Cognitive chunk extractor using LLM."""

import json
import re
from pathlib import Path

from .cognitive_chunk import CognitiveChunk
from .llm import LLM
from .working_memory import WorkingMemory


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

    def extract_chunks(self, text: str, working_memory: WorkingMemory) -> list[CognitiveChunk]:
        """Extract cognitive chunks from text.

        Args:
            text: Text segment to process
            working_memory: Current working memory state

        Returns:
            List of extracted chunks
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
            return []

        # Parse JSON response
        try:
            chunks_data = self._parse_json_response(response)
            chunks = []
            for data in chunks_data:
                chunk = CognitiveChunk(
                    id=0,  # Will be assigned by WorkingMemory
                    generation=0,  # Will be assigned by WorkingMemory
                    label=data.get("label", ""),
                    content=data.get("content", ""),
                    links=data.get("links", []),
                )
                chunks.append(chunk)
            return chunks
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            print(f"Response was: {response[:200]}...")
            return []

    def _parse_json_response(self, response: str) -> list[dict]:
        """Parse JSON from LLM response.

        Handles cases where LLM includes extra text around JSON.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON data
        """
        # Try to extract JSON from response
        # Look for JSON array pattern
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)

        # If no array found, try parsing the whole response
        return json.loads(response)
