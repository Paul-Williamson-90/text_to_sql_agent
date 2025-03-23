STOP_AFTER_ATTEMPT: int = 5

WAIT_FIXED: int = 1

REPLACEMENT_PATTERNS: list[tuple[str, str]] = [
    (r":\s*true", ": True"),
    (r":\s*false", ": False"),
    (
        r":\s*null",
        ": None",
    ),
]
