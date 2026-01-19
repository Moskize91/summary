"""Enumeration types for chunk and edge metadata."""

from enum import IntEnum


class RetentionLevel(IntEnum):
    """Retention level for user-focused chunks.

    Higher values indicate higher priority for retention.
    Values are designed to support future weight calculations.
    """

    VERBATIM = 27  # Entire passages preserved word-for-word
    DETAILED = 9  # Complete information retained without compression
    FOCUSED = 3  # User's primary interest (can be moderately summarized)
    RELEVANT = 1  # User finds this useful but not essential (can be dropped if needed)

    @classmethod
    def from_string(cls, value: str | None) -> "RetentionLevel | None":
        """Convert string to RetentionLevel enum.

        Args:
            value: String value ("verbatim", "detailed", "focused", "relevant")

        Returns:
            RetentionLevel enum value or None if value is None

        Raises:
            ValueError: If value is not a valid retention level
        """
        if value is None:
            return None

        mapping = {
            "verbatim": cls.VERBATIM,
            "detailed": cls.DETAILED,
            "focused": cls.FOCUSED,
            "relevant": cls.RELEVANT,
        }

        if value not in mapping:
            raise ValueError(f"Invalid retention level: {value}")

        return mapping[value]

    def to_string(self) -> str:
        """Convert enum to string representation.

        Returns:
            String value ("verbatim", "detailed", "focused", "relevant")
        """
        mapping = {
            self.VERBATIM: "verbatim",
            self.DETAILED: "detailed",
            self.FOCUSED: "focused",
            self.RELEVANT: "relevant",
        }
        return mapping[self]


class ImportanceLevel(IntEnum):
    """Importance level for book-coherence chunks.

    Higher values indicate higher importance for narrative coherence.
    Values are designed to support future weight calculations.
    """

    CRITICAL = 9  # Essential setup, major turning points, or causal links
    IMPORTANT = 3  # Key context, definitions, or transitions
    HELPFUL = 1  # Supplementary background or minor connective tissue

    @classmethod
    def from_string(cls, value: str | None) -> "ImportanceLevel | None":
        """Convert string to ImportanceLevel enum.

        Args:
            value: String value ("critical", "important", "helpful")

        Returns:
            ImportanceLevel enum value or None if value is None

        Raises:
            ValueError: If value is not a valid importance level
        """
        if value is None:
            return None

        mapping = {
            "critical": cls.CRITICAL,
            "important": cls.IMPORTANT,
            "helpful": cls.HELPFUL,
        }

        if value not in mapping:
            raise ValueError(f"Invalid importance level: {value}")

        return mapping[value]

    def to_string(self) -> str:
        """Convert enum to string representation.

        Returns:
            String value ("critical", "important", "helpful")
        """
        mapping = {
            self.CRITICAL: "critical",
            self.IMPORTANT: "important",
            self.HELPFUL: "helpful",
        }
        return mapping[self]


class LinkStrength(IntEnum):
    """Link strength for connections between chunks.

    Higher values indicate stronger dependencies.
    Values are designed to support future weight calculations.
    """

    CRITICAL = 9  # Must understand FROM to understand TO
    IMPORTANT = 3  # FROM provides essential context for TO
    HELPFUL = 1  # FROM and TO are related but both can stand alone

    @classmethod
    def from_string(cls, value: str | None) -> "LinkStrength | None":
        """Convert string to LinkStrength enum.

        Args:
            value: String value ("critical", "important", "helpful")

        Returns:
            LinkStrength enum value or None if value is None

        Raises:
            ValueError: If value is not a valid link strength
        """
        if value is None:
            return None

        mapping = {
            "critical": cls.CRITICAL,
            "important": cls.IMPORTANT,
            "helpful": cls.HELPFUL,
        }

        if value not in mapping:
            raise ValueError(f"Invalid link strength: {value}")

        return mapping[value]

    def to_string(self) -> str:
        """Convert enum to string representation.

        Returns:
            String value ("critical", "important", "helpful")
        """
        mapping = {
            self.CRITICAL: "critical",
            self.IMPORTANT: "important",
            self.HELPFUL: "helpful",
        }
        return mapping[self]
