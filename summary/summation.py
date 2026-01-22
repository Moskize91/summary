"""High-level API for text summarization pipeline."""

from os import PathLike
from pathlib import Path

from tiktoken import get_encoding

from .editor import compress_text
from .epub import read_epub_sentences, write_epub
from .llm import LLM
from .topologization import Topologization


async def summary(
    intention: str,
    input_epub_file: PathLike | str,
    output_epub_file: PathLike | str,
    llm_api_key: str,
    llm_base_url: str,
    llm_model: str,
    workspace_path: PathLike | str | None = None,
    log_dir: PathLike | str | None = None,
    cache_dir: PathLike | str | None = None,
    llm_timeout: float = 360.0,
    llm_temperature: float = 0.6,
    llm_top_p: float = 0.6,
    fragment_tokens: int = 800,
    memory_capacity: int = 7,
    snake_tokens: int = 700,
    group_tokens_count: int = 9600,
    compression_ratio: float = 0.2,
    max_iterations: int = 10,
    max_clues: int = 6,
) -> None:
    """Compress EPUB file according to user intention.

    Args:
        intention: User's reading intention/goal for extraction guidance
        input_epub_file: Input EPUB file path
        output_epub_file: Output compressed EPUB file path
        llm_api_key: OpenAI API key
        llm_base_url: OpenAI API base URL
        llm_model: Model name (e.g., "gpt-4")
        workspace_path: Workspace directory for fragments and database (default: "workspace")
        log_dir: Log directory for LLM logs (default: "logs/")
        cache_dir: Cache directory for LLM cache (default: "cache")
        llm_timeout: LLM request timeout in seconds (default: 360.0)
        llm_temperature: LLM sampling temperature (default: 0.6)
        llm_top_p: LLM nucleus sampling parameter (default: 0.6)
        fragment_tokens: Maximum token count per text fragment (default: 800)
        memory_capacity: Working memory capacity in chunks (default: 7)
        snake_tokens: Maximum tokens allowed in a snake (default: 700)
        group_tokens_count: Maximum tokens per fragment group (default: 9600)
        compression_ratio: Target compression ratio (default: 0.2)
        max_iterations: Maximum compression iterations (default: 10)
        max_clues: Maximum clues per iteration (default: 6)

    Raises:
        FileNotFoundError: If input file not found
        Exception: If pipeline fails
    """
    # Convert paths
    input_path = Path(input_epub_file)
    output_path = Path(output_epub_file)
    workspace_path = Path(workspace_path) if workspace_path is not None else Path("workspace")

    # Setup log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)

    # Setup data directory for templates
    data_dir = Path(__file__).parent / "data"

    # Create LLM instance
    llm = LLM(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
        data_dir_path=data_dir,
        log_dir_path=log_dir / "llm",
        cache_dir_path=Path(cache_dir) if cache_dir is not None else None,
        timeout=llm_timeout,
        temperature=llm_temperature,
        top_p=llm_top_p,
    )

    # Prepare encoding
    encoding = get_encoding("o200k_base")
    print(f"Processing input EPUB: {input_path}")

    # Create Topologization instance
    topologization = Topologization(
        workspace_path=workspace_path,
        llm=llm,
        encoding=encoding,
        max_fragment_tokens=fragment_tokens,
        working_memory_capacity=memory_capacity,
        snake_tokens=snake_tokens,
        group_tokens_count=group_tokens_count,
    )

    # Read EPUB and incrementally load chapters
    chapter_info = []  # List of (chapter_id, chapter_title)

    for chapter_title, chapter_sentences in read_epub_sentences(input_path, encoding):
        sentences_list = list(chapter_sentences)
        print(f"  Loading: {chapter_title} ({len(sentences_list)} sentences)")

        # Load chapter and get its ID
        chapter_id = await topologization.load(sentences_list, intention)
        chapter_info.append((chapter_id, chapter_title))

    print(f"Total chapters loaded: {len(chapter_info)}")

    # Run compression for each chapter and group
    print("\n" + "=" * 60)
    print("=== Compression Stage ===")
    print("=" * 60)

    print(f"\nFound {len(chapter_info)} chapters to compress")

    # Collect compressed chapters for EPUB output
    compressed_chapters = []

    for chapter_id, chapter_title in chapter_info:
        group_ids = topologization.get_group_ids_for_chapter(chapter_id)
        print(f"\nChapter {chapter_id}: {chapter_title}")
        print(f"  Groups: {len(group_ids)}")

        # Compress all groups in this chapter
        chapter_compressed_texts = []

        for group_id in group_ids:
            print(f"  Compressing Group {group_id}...")

            # Compress this group
            compressed_text = await compress_text(
                topologization=topologization,
                chapter_id=chapter_id,
                group_id=group_id,
                intention=intention,
                llm=llm,
                compression_ratio=compression_ratio,
                max_iterations=max_iterations,
                max_clues=max_clues,
                log_dir_path=log_dir / "compression",
            )

            chapter_compressed_texts.append(compressed_text)
            print(f"    ✓ Length: {len(compressed_text)} characters")

        # Merge all groups in this chapter with double newlines
        full_chapter_text = "\n\n".join(chapter_compressed_texts)
        compressed_chapters.append((chapter_title, [full_chapter_text]))

    # Write output EPUB
    print("\n" + "=" * 60)
    print("=== Writing Output EPUB ===")
    print("=" * 60)

    # Extract book title from input filename (remove .epub extension)
    book_title = input_path.stem + " (summarized)"

    write_epub(
        chapters=compressed_chapters,
        output_path=output_path,
        book_title=book_title,
        author="AI Summarizer",
        language="en",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("=== Pipeline Complete ===")
    print("=" * 60)
    print(f"Workspace: {workspace_path}")
    print(f"Output EPUB: {output_path}")
    print(f"Total chapters: {len(compressed_chapters)}")
