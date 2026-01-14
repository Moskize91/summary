"""
Example test module demonstrating basic test structure.
Replace this with your actual tests.
"""


def test_example_assertion():
    """Example test that demonstrates a simple assertion."""
    assert 1 + 1 == 2


def test_example_string():
    """Example test that demonstrates string operations."""
    result = "hello".upper()
    assert result == "HELLO"


def test_example_list():
    """Example test that demonstrates list operations."""
    items = [1, 2, 3]
    items.append(4)
    assert len(items) == 4
    assert items[-1] == 4
