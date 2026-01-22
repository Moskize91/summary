"""EPUB writing utilities."""

from collections.abc import Iterable
from pathlib import Path

from ebooklib import epub


def write_epub(
    chapters: Iterable[tuple[str, Iterable[str]]],
    output_path: Path | str,
    book_title: str = "Untitled",
    author: str = "Unknown Author",
    language: str = "zh",
) -> None:
    """Write chapters to an EPUB file.

    Takes an iterable of (chapter_title, sentences) and creates an EPUB file.
    Sentences are concatenated and split by newlines into paragraphs.

    Args:
        chapters: Iterable of (chapter_title, sentences_iterable) tuples
        output_path: Path to output EPUB file
        book_title: Title of the book
        author: Author name
        language: Language code (e.g., "zh", "en")

    Example:
        >>> chapters = [
        ...     ("Chapter 1", ["Sentence 1.", "Sentence 2.\\n\\nSentence 3."]),
        ...     ("Chapter 2", ["Another sentence.", "More text."])
        ... ]
        >>> write_epub(chapters, "output.epub", "My Book", "Author Name")
    """
    # Create EPUB book
    book = epub.EpubBook()

    # Set metadata
    book.set_identifier(f"id_{book_title}")
    book.set_title(book_title)
    book.set_language(language)
    book.add_author(author)

    # Create CSS for styling
    style = """
        body {
            font-family: serif;
            line-height: 1.6;
            margin: 2em;
        }
        h1 {
            font-size: 2em;
            font-weight: bold;
            margin-top: 1em;
            margin-bottom: 1em;
            text-align: center;
        }
        p {
            margin-top: 0.5em;
            margin-bottom: 0.5em;
            text-indent: 2em;
        }
    """

    # Create CSS item
    nav_css = epub.EpubItem(
        uid="style_nav",
        file_name="style/nav.css",
        media_type="text/css",
        content=style.encode("utf-8"),
    )
    book.add_item(nav_css)

    # Process chapters
    epub_chapters = []
    spine = ["nav"]

    for chapter_index, (chapter_title, sentences) in enumerate(chapters, start=1):
        # Concatenate all sentences
        full_text = "".join(sentences)

        # Split into paragraphs by newlines
        paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]

        # Generate XHTML content
        chapter_content = _generate_chapter_xhtml(chapter_title, paragraphs)

        # Create EPUB chapter
        chapter_file_name = f"chapter_{chapter_index}.xhtml"
        epub_chapter = epub.EpubHtml(
            title=chapter_title,
            file_name=chapter_file_name,
            lang=language,
        )
        epub_chapter.content = chapter_content.encode("utf-8")
        epub_chapter.add_item(nav_css)

        # Add to book
        book.add_item(epub_chapter)
        epub_chapters.append(epub_chapter)
        spine.append(epub_chapter)

    # Create table of contents
    book.toc = tuple(epub_chapters)

    # Add navigation files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Set spine (reading order)
    book.spine = spine

    # Write EPUB file
    epub.write_epub(output_path, book)


def _generate_chapter_xhtml(chapter_title: str, paragraphs: list[str]) -> str:
    """Generate XHTML content for a chapter.

    Args:
        chapter_title: Title of the chapter
        paragraphs: List of paragraph texts

    Returns:
        XHTML string
    """

    # Escape HTML special characters
    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    # Build paragraphs HTML
    paragraphs_html = "\n".join(f"    <p>{escape_html(p)}</p>" for p in paragraphs)

    # Generate complete XHTML
    xhtml = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{escape_html(chapter_title)}</title>
</head>
<body>
    <h1>{escape_html(chapter_title)}</h1>
{paragraphs_html}
</body>
</html>"""

    return xhtml
