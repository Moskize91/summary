"""SQLite database operations for topologization workspace."""

import sqlite3
from pathlib import Path

from .fragment import SentenceId


def create_schema(conn: sqlite3.Connection):
    """Create all database tables for topologization workspace.

    Args:
        conn: SQLite database connection
    """
    cursor = conn.cursor()

    # Chunks table (knowledge graph nodes)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            generation INTEGER NOT NULL,
            chapter_id INTEGER NOT NULL,
            fragment_id INTEGER NOT NULL,
            sentence_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            content TEXT NOT NULL,
            retention TEXT,
            importance TEXT,
            tokens INTEGER NOT NULL DEFAULT 0,
            weight REAL NOT NULL DEFAULT 0.0
        )
    """)

    # Index for sentence ID lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_sentence
        ON chunks(chapter_id, fragment_id, sentence_index)
    """)

    # Chunk sentences table (many-to-many: chunks can span multiple sentences)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunk_sentences (
            chunk_id INTEGER NOT NULL,
            chapter_id INTEGER NOT NULL,
            fragment_id INTEGER NOT NULL,
            sentence_index INTEGER NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id),
            PRIMARY KEY (chunk_id, chapter_id, fragment_id, sentence_index)
        )
    """)

    # Knowledge edges table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            from_id INTEGER NOT NULL,
            to_id INTEGER NOT NULL,
            strength TEXT,
            weight REAL NOT NULL DEFAULT 0.1,
            PRIMARY KEY (from_id, to_id),
            FOREIGN KEY (from_id) REFERENCES chunks(id),
            FOREIGN KEY (to_id) REFERENCES chunks(id)
        )
    """)

    # Snakes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            local_snake_id INTEGER NOT NULL,
            size INTEGER NOT NULL,
            first_label TEXT NOT NULL,
            last_label TEXT NOT NULL,
            tokens INTEGER NOT NULL DEFAULT 0,
            weight REAL NOT NULL DEFAULT 0.0,
            UNIQUE(chapter_id, group_id, local_snake_id)
        )
    """)

    # Snake chunks junction table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snake_chunks (
            snake_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            FOREIGN KEY (snake_id) REFERENCES snakes(id),
            FOREIGN KEY (chunk_id) REFERENCES chunks(id),
            PRIMARY KEY (snake_id, chunk_id)
        )
    """)

    # Snake edges table (inter-snake connections)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snake_edges (
            from_snake_id INTEGER NOT NULL,
            to_snake_id INTEGER NOT NULL,
            weight REAL NOT NULL DEFAULT 0.1,
            PRIMARY KEY (from_snake_id, to_snake_id),
            FOREIGN KEY (from_snake_id) REFERENCES snakes(id),
            FOREIGN KEY (to_snake_id) REFERENCES snakes(id)
        )
    """)

    # Fragment groups table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fragment_groups (
            chapter_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            fragment_id INTEGER NOT NULL,
            PRIMARY KEY (chapter_id, group_id, fragment_id)
        )
    """)

    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    chunk_id: int,
    generation: int,
    sentence_id: SentenceId,
    label: str,
    content: str,
    sentence_ids: list[SentenceId],
    retention: str | None = None,
    importance: str | None = None,
    tokens: int = 0,
    weight: float = 0.0,
):
    """Insert chunk and its sentences.

    Args:
        conn: SQLite database connection
        chunk_id: Chunk ID
        generation: Generation number
        sentence_id: Primary sentence ID (usually min sentence)
        label: Chunk label
        content: AI-generated summary content
        sentence_ids: All sentence IDs comprising this chunk's content
        retention: Retention level (verbatim/detailed/focused/relevant)
        importance: Importance level (critical/important/helpful)
        tokens: Total token count of original source sentences
        weight: Node weight (computed from retention + importance)
    """
    cursor = conn.cursor()

    # Insert chunk metadata
    chapter_id, fragment_id, sentence_index = sentence_id
    cursor.execute(
        """
        INSERT INTO chunks (id, generation, chapter_id, fragment_id, sentence_index,
                           label, content, retention, importance, tokens, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_id,
            generation,
            chapter_id,
            fragment_id,
            sentence_index,
            label,
            content,
            retention,
            importance,
            tokens,
            weight,
        ),
    )

    # Insert chunk-sentence associations
    for sid in sentence_ids:
        cid, fid, sidx = sid
        cursor.execute(
            """
            INSERT INTO chunk_sentences (chunk_id, chapter_id, fragment_id, sentence_index)
            VALUES (?, ?, ?, ?)
            """,
            (chunk_id, cid, fid, sidx),
        )

    conn.commit()


def insert_edge(conn: sqlite3.Connection, from_id: int, to_id: int, strength: str | None = None, weight: float = 0.1):
    """Insert knowledge edge.

    Args:
        conn: SQLite database connection
        from_id: Source chunk ID
        to_id: Target chunk ID
        strength: Link strength (critical/important/helpful)
        weight: Edge weight (default: 0.1)
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO knowledge_edges (from_id, to_id, strength, weight)
        VALUES (?, ?, ?, ?)
        """,
        (from_id, to_id, strength, weight),
    )
    conn.commit()


def insert_snake(
    conn: sqlite3.Connection,
    snake_id: int,
    size: int,
    first_label: str,
    last_label: str,
    tokens: int = 0,
    weight: float = 0.0,
):
    """Insert snake metadata.

    Args:
        conn: SQLite database connection
        snake_id: Snake ID
        size: Number of chunks in snake
        first_label: Label of first chunk
        last_label: Label of last chunk
        tokens: Total tokens in snake (sum of chunk tokens)
        weight: Total weight of snake (sum of chunk weights)
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO snakes (snake_id, size, first_label, last_label, tokens, weight)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (snake_id, size, first_label, last_label, tokens, weight),
    )
    conn.commit()


def insert_snake_chunk(
    conn: sqlite3.Connection,
    snake_id: int,
    chunk_id: int,
    position: int,
):
    """Insert snake-chunk association.

    Args:
        conn: SQLite database connection
        snake_id: Snake ID
        chunk_id: Chunk ID
        position: Position of chunk within snake (0-indexed)
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO snake_chunks (snake_id, chunk_id, position)
        VALUES (?, ?, ?)
        """,
        (snake_id, chunk_id, position),
    )
    conn.commit()


def insert_snake_edge(
    conn: sqlite3.Connection,
    from_snake: int,
    to_snake: int,
    weight: float,
):
    """Insert snake edge (inter-snake connection).

    Args:
        conn: SQLite database connection
        from_snake: Source snake ID
        to_snake: Target snake ID
        weight: Total weight of edges between snakes (sum of chunk edge weights)
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO snake_edges (from_snake, to_snake, weight)
        VALUES (?, ?, ?)
        """,
        (from_snake, to_snake, weight),
    )
    conn.commit()


def initialize_database(db_path: Path) -> sqlite3.Connection:
    """Initialize database with schema.

    Args:
        db_path: Path to database file

    Returns:
        SQLite connection
    """
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    return conn
