"""LLM client with logging and caching support."""

import asyncio
import datetime
import hashlib
import json
import threading
from dataclasses import dataclass
from io import StringIO
from logging import DEBUG, FileHandler, Formatter, Logger, getLogger
from pathlib import Path
from typing import cast

import aiofiles
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .template import create_env

# Global state for logger filename generation
_LOGGER_LOCK = threading.Lock()
_LAST_TIMESTAMP: str | None = None
_LOGGER_SUFFIX_ID: int = 1


@dataclass
class LLMessage:
    role: str
    content: str


class LLM:
    """LLM client with configuration, logging, caching, and retry support."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        data_dir_path: Path,
        log_dir_path: Path | None = None,
        cache_dir_path: Path | None = None,
        concurrent: int = 1,
        timeout: float = 360.0,
        temperature: float = 0.6,
        top_p: float = 0.6,
        retry_times: int = 5,
        retry_interval_seconds: float = 6.0,
    ):
        """Initialize the LLM client.

        Args:
            api_key: OpenAI API key
            base_url: OpenAI API base URL
            model: Model name (e.g., "gpt-4")
            data_dir_path: Path to the data directory containing Jinja templates
            log_dir_path: Directory path for saving logs
            cache_dir_path: Directory path for caching responses
            timeout: Request timeout in seconds (default: 360.0)
            temperature: Sampling temperature (default: 0.6)
            top_p: Nucleus sampling parameter (default: 0.6)
            retry_times: Number of retry attempts on failure
            retry_interval_seconds: Wait time between retries
        """
        # Store configuration
        self.config = {
            "key": api_key,
            "url": base_url,
            "model": model,
            "timeout": timeout,
            "temperature": temperature,
            "top_p": top_p,
        }

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.retry_times = retry_times
        self.retry_interval_seconds = retry_interval_seconds

        # Setup concurrency control
        self._semaphore = asyncio.Semaphore(concurrent)

        # Setup Jinja environment
        self._data_dir_path = data_dir_path.resolve()
        self.jinja_env = create_env(data_dir_path)

        # Setup logging and caching
        self._log_dir_path = self._ensure_dir_path(log_dir_path)
        self._cache_dir_path = self._ensure_dir_path(cache_dir_path)

    async def request(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str | None:
        """Send a request to the LLM with retry logic, logging, and caching.

        Args:
            system_prompt: System prompt for the LLM
            user_message: User message content
            temperature: Temperature parameter (uses config default if None)
            top_p: Top-p parameter (uses config default if None)

        Returns:
            LLM response text, or None if request failed
        """
        messages = [
            LLMessage(role="system", content=system_prompt),
            LLMessage(role="user", content=user_message),
        ]
        return await self.request_with_history(messages, temperature, top_p)

    async def request_with_history(
        self,
        messages: list[LLMessage],
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str | None:
        """Send a request to the LLM with conversation history.

        Args:
            messages: List of LLMessage objects
                     Example: [
                         LLMessage(role="system", content="..."),
                         LLMessage(role="user", content="..."),
                         LLMessage(role="assistant", content="..."),
                         LLMessage(role="user", content="...")
                     ]
            temperature: Temperature parameter (uses config default if None)
            top_p: Top-p parameter (uses config default if None)

        Returns:
            LLM response text, or None if request failed
        """
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p

        # Compute cache key from messages
        cache_key = None
        if self._cache_dir_path is not None:
            cache_data = {
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "temperature": temperature,
                "top_p": top_p,
                "model": self.model,
            }
            cache_json = json.dumps(cache_data, ensure_ascii=False, sort_keys=True)
            cache_key = hashlib.sha512(cache_json.encode("utf-8")).hexdigest()

        # Create logger and log request
        logger = self._create_logger()
        if logger is not None:
            log_params = f"[[Parameters]]:\n\ttemperature={temperature}\n\ttop_p={top_p}\n"
            if cache_key is not None:
                log_params += f"\tcache_key={cache_key}\n"
            logger.debug(log_params)

            # Log messages
            buffer = StringIO()
            for msg in messages:
                buffer.write(f"{msg.role.capitalize()}:\n{msg.content}\n\n")
            logger.debug("[[Request]]:\n%s", buffer.getvalue())

        # Check cache after logging
        if cache_key is not None and self._cache_dir_path is not None:
            cache_file = self._cache_dir_path / f"{cache_key}.txt"
            if cache_file.exists():
                async with aiofiles.open(cache_file, encoding="utf-8") as f:
                    cached_response = await f.read()
                print(f"[Cache Hit] Using cached response (key: {cache_key[:12]}...)")
                if logger is not None:
                    logger.debug("[[Response]] (from cache):\n%s\n", cached_response)
                return cached_response

        response: str = ""
        last_error: Exception | None = None
        did_success = False

        try:
            for i in range(self.retry_times + 1):
                try:
                    # Convert LLMessage objects to dict format for OpenAI API
                    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

                    # Use semaphore to limit concurrent API calls
                    async with self._semaphore:
                        completion = await self.client.chat.completions.create(
                            model=self.model,
                            messages=cast(list[ChatCompletionMessageParam], messages_dict),
                            temperature=temperature,
                            top_p=top_p,
                        )

                    response = completion.choices[0].message.content or ""

                    if logger is not None:
                        logger.debug("[[Response]]:\n%s\n", response)

                    did_success = True
                    break

                except Exception as err:
                    last_error = err
                    if not self._is_retry_error(err):
                        if logger is not None:
                            logger.error("[[Error]]:\n%s\n", err)
                        raise err

                    if logger is not None:
                        logger.warning("Request failed with connection error, retrying... (%s times)", i + 1)

                    if self.retry_interval_seconds > 0.0 and i < self.retry_times:
                        await asyncio.sleep(self.retry_interval_seconds)
                    continue

        except KeyboardInterrupt as err:
            if last_error is not None and logger is not None:
                logger.debug("[[Error]]:\n%s\n", last_error)
            raise err

        if not did_success:
            if logger is not None and last_error is not None:
                logger.error("[[Error]]:\n%s\n", last_error)
            return None

        # Save to cache (only cache non-empty responses)
        if self._cache_dir_path is not None and cache_key is not None and response:
            cache_file = self._cache_dir_path / f"{cache_key}.txt"
            async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                await f.write(response)

        return response

    def load_system_prompt(self, prompt_template_path: Path, **kwargs) -> str:
        """Load and render the system prompt from a Jinja template.

        Args:
            prompt_template_path: Path to the prompt.jinja file
            **kwargs: Variables to pass to the template

        Returns:
            Rendered system prompt
        """
        # Calculate relative path from data directory
        # The template path should be relative to data_dir for jinja to find it
        try:
            # Try to get relative path from data dir
            relative_path = prompt_template_path.relative_to(self._data_dir_path)
            template_name = str(relative_path)
        except ValueError:
            # If not relative to data_dir, just use the name (backward compatibility)
            template_name = prompt_template_path.name

        template = self.jinja_env.get_template(template_name)
        return template.render(**kwargs)

    def _is_retry_error(self, error: Exception) -> bool:
        """Check if error should trigger a retry."""
        error_str = str(error).lower()
        retry_keywords = ["connection", "timeout", "network", "rate limit"]
        return any(keyword in error_str for keyword in retry_keywords)

    def _ensure_dir_path(self, path: Path | None) -> Path | None:
        """Ensure directory exists and return resolved path."""
        if path is None:
            return None
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            return None
        return path.resolve()

    def _create_logger(self) -> Logger | None:
        """Create a logger for this request with timestamped filename."""
        # pylint: disable=global-statement
        global _LAST_TIMESTAMP, _LOGGER_SUFFIX_ID

        if self._log_dir_path is None:
            return None

        now = datetime.datetime.now(datetime.UTC)
        timestamp_key = now.strftime("%Y-%m-%d %H-%M-%S")

        with _LOGGER_LOCK:
            if _LAST_TIMESTAMP == timestamp_key:
                _LOGGER_SUFFIX_ID += 1
                suffix_id = _LOGGER_SUFFIX_ID
            else:
                _LAST_TIMESTAMP = timestamp_key
                _LOGGER_SUFFIX_ID = 1
                suffix_id = 1

        if suffix_id == 1:
            file_name = f"request {timestamp_key}.log"
            logger_name = f"LLM Request {timestamp_key}"
        else:
            file_name = f"request {timestamp_key}_{suffix_id}.log"
            logger_name = f"LLM Request {timestamp_key}_{suffix_id}"

        file_path = self._log_dir_path / file_name
        logger = getLogger(logger_name)
        logger.setLevel(DEBUG)
        handler = FileHandler(file_path, encoding="utf-8")
        handler.setLevel(DEBUG)
        handler.setFormatter(Formatter("%(asctime)s    %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)

        return logger
