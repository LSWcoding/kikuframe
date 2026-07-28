from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(value: str) -> str:
    lines: list[str] = []
    for raw_line in unicodedata.normalize("NFKC", value).splitlines():
        clean = _WHITESPACE.sub(" ", raw_line).strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines)
