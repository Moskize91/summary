import sys

import pytest

try:
    exit_code = pytest.main(
        [
            "tests/",
            "-v",
            "--tb=short",
        ]
    )
    sys.exit(exit_code)

except Exception as error:
    print(error)
    sys.exit(1)
