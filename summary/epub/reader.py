"""EPUB reading utilities with sentence-level streaming."""

import xml.etree.ElementTree as ET
from collections.abc import Callable, Generator
from pathlib import Path

import ebooklib
from ebooklib import epub
from spacy.lang.xx import MultiLanguage
from tiktoken import Encoding


def _extract_chapter_title(html_bytes: bytes, fallback_name: str) -> str:
    """Extract chapter title from HTML content.

    Tries to extract title from:
    1. <title> tag
    2. First <h1> or <h2> tag
    3. Falls back to provided name

    Args:
        html_bytes: Raw HTML/XHTML content as bytes
        fallback_name: Fallback name if no title found

    Returns:
        Chapter title string
    """
    try:
        root = ET.fromstring(html_bytes)
    except ET.ParseError:
        return fallback_name

    # Try to find <title> tag
    for elem in root.iter():
        if elem.tag.endswith("title") and elem.text:
            return elem.text.strip()

    # Try to find first <h1> or <h2> tag
    for elem in root.iter():
        if elem.tag.endswith(("h1", "h2")) and elem.text:
            return elem.text.strip()

    # Fallback to provided name
    return fallback_name


def _extract_text_from_html(html_bytes: bytes) -> Generator[str, None, None]:
    """Extract text from HTML/XHTML content by traversing all elements.

    Yields text in chunks as it traverses the DOM tree.

    Args:
        html_bytes: Raw HTML/XHTML content as bytes

    Yields:
        Text chunks extracted from HTML elements
    """
    try:
        # Parse HTML as XML (EPUB uses XHTML which is well-formed)
        root = ET.fromstring(html_bytes)
    except ET.ParseError:
        # If parsing fails, skip this chapter
        return

    # Traverse all elements and collect text + tail
    for element in root.iter():
        if element.text:
            yield element.text
        if element.tail:
            yield element.tail


def _process_chapter_to_sentences(
    html_bytes: bytes,
    nlp: MultiLanguage,
    encoding: Encoding,
    batch_size: int = 15000,
) -> Generator[tuple[int, str], None, None]:
    """Process a chapter's HTML content into sentences with token counts.

    Args:
        html_bytes: Raw HTML content
        nlp: spaCy NLP model with sentencizer
        encoding: Tiktoken encoding for token counting
        batch_size: Characters to accumulate before processing with spaCy

    Yields:
        Tuples of (token_count, sentence_text)
    """
    text_buffer = ""

    # Stream text chunks from HTML
    for text_chunk in _extract_text_from_html(html_bytes):
        text_buffer += text_chunk

        # Process batch when buffer reaches batch_size
        if len(text_buffer) >= batch_size:
            doc = nlp(text_buffer)
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if sentence_text:
                    token_count = len(encoding.encode(sentence_text))
                    yield (token_count, sentence_text)
            text_buffer = ""

    # Process remaining text
    if text_buffer.strip():
        doc = nlp(text_buffer)
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if sentence_text:
                token_count = len(encoding.encode(sentence_text))
                yield (token_count, sentence_text)


def read_epub_sentences(
    epub_path: Path | str,
    encoding: Encoding,
    batch_size: int = 15000,
) -> Generator[tuple[str, Callable[[], Generator[tuple[int, str], None, None]]], None, None]:
    """Read EPUB file and yield chapter info with lazy sentence generators.

    This function returns a two-level generator structure:
    - First level: yields (chapter_title, sentence_generator_factory) tuples for each chapter
    - Second level: sentence_generator_factory() returns a generator that yields (token_count, sentence_text)

    The sentence generator is lazy - it only reads and processes the chapter when called.
    This avoids loading all chapters into memory at once.

    Args:
        epub_path: Path to EPUB file
        encoding: Tiktoken encoding for token counting
        batch_size: Characters to accumulate before processing with spaCy

    Yields:
        Tuples of (chapter_title, sentence_generator_factory) for each chapter

    Example:
        >>> encoding = get_encoding("o200k_base")
        >>> for chapter_title, get_sentences in read_epub_sentences("book.epub", encoding):
        ...     print(f"Chapter: {chapter_title}")
        ...     for token_count, sentence in get_sentences():  # Call to get the generator
        ...         print(f"{token_count} tokens: {sentence}")
    """
    # Load spaCy multilingual model with sentencizer
    nlp = MultiLanguage()
    nlp.add_pipe("sentencizer")

    # Read EPUB file
    book = epub.read_epub(epub_path)

    # Get chapters in spine order (reading order)
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        # Use item name as title (avoid loading HTML content until needed)
        chapter_title = item.get_name()

        # Create a callable that loads HTML and returns the sentence generator when called
        def get_sentences(epub_item=item, nlp_model=nlp, enc=encoding, bs=batch_size):
            html_bytes = epub_item.get_content()  # Load HTML only when called!
            return _process_chapter_to_sentences(html_bytes, nlp_model, enc, bs)

        # Yield (title, sentence_generator_factory) tuple
        yield (chapter_title, get_sentences)
