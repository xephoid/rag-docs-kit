"""Make untrusted text safe to print to a terminal.

Document text, filenames, QA questions and model output can contain terminal
control sequences (for example ``ESC [ 2 J``) that would clear, rewrite or spoof
what the user sees. :func:`terminal_safe` replaces every control character
except newline and tab, plus Unicode bidirectional overrides, with a visible
escape such as ``\\x1b``.
"""

from __future__ import annotations

import unicodedata

_KEEP = {"\n", "\t"}
_BIDI = {chr(c) for c in (*range(0x202A, 0x202F), *range(0x2066, 0x206A), 0x200E, 0x200F)}


def _escape(ch: str) -> str:
    code = ord(ch)
    return f"\\x{code:02x}" if code <= 0xFF else f"\\u{code:04x}"


def terminal_safe(text: str) -> str:
    """Escape control characters (C0, DEL, C1) and bidi overrides; keep \\n and \\t."""
    return "".join(
        _escape(ch) if ch not in _KEEP and (unicodedata.category(ch) == "Cc" or ch in _BIDI) else ch
        for ch in text
    )
