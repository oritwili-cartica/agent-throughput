"""Names of the skills, plugins, MCP servers and output styles a session used.

Only names and call counts leave this module. Free text (a Skill call's
``input.args``, prompts, replies, tool output) is never read.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Dict, List, Optional, Set

SIGNALS_VERSION = 1
MAX_NAMES_PER_KEY = 50
MAX_NAME_LENGTH = 64
# Half of the 64 KiB row cap: non-ASCII names grow up to 6x once JSON-escaped.
MAX_SIGNALS_JSON_CHARS = 32 * 1024
COUNTED_KEYS = ("skills", "plugins", "mcp_servers")
_STYLE_KEY = "output_styles"
_ATTRIBUTION_FIELDS = (
    ("attributionSkill", "skills"),
    ("attributionPlugin", "plugins"),
    ("attributionMcpServer", "mcp_servers"),
)


def clean_name(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name or any(unicodedata.category(ch) == "Cc" for ch in name):
        return None
    return name[:MAX_NAME_LENGTH].strip() or None


class SessionSignalCollector:
    def __init__(self) -> None:
        self._seen: Dict[str, Dict[str, Set[str]]] = {
            key: {} for key in COUNTED_KEYS + (_STYLE_KEY,)
        }

    def _add(self, key: str, value, call_id: str) -> None:
        name = clean_name(value)
        if name is not None:
            self._seen[key].setdefault(name, set()).add(call_id)

    def observe_assistant(self, rec: dict, content, call_id: str) -> None:
        for field, key in _ATTRIBUTION_FIELDS:
            self._add(key, rec.get(field), call_id)
        if not isinstance(content, list):
            return
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == "Skill"
                and isinstance(block.get("input"), dict)
            ):
                self._add("skills", block["input"].get("skill"), call_id)

    def observe_attachment(self, rec: dict, call_id: str) -> None:
        attachment = rec.get("attachment")
        if isinstance(attachment, dict) and attachment.get("type") == "output_style":
            self._add(_STYLE_KEY, attachment.get("style"), call_id)

    def _counts(self, key: str) -> Dict[str, int]:
        ranked = sorted(
            ((name, len(calls)) for name, calls in self._seen[key].items()),
            key=lambda item: (-item[1], item[0]),
        )
        return dict(ranked[:MAX_NAMES_PER_KEY])

    def metadata(self) -> dict:
        counts = {key: self._counts(key) for key in COUNTED_KEYS}
        styles = sorted(self._seen[_STYLE_KEY])[:MAX_NAMES_PER_KEY]
        while True:
            signals = {key: value for key, value in counts.items() if value}
            if styles:
                signals[_STYLE_KEY] = styles
            if len(json.dumps(signals, sort_keys=True)) < MAX_SIGNALS_JSON_CHARS:
                break
            key, name = self._cheapest_to_drop(counts)
            del counts[key][name]
        signals["signals_version"] = SIGNALS_VERSION
        return signals

    @staticmethod
    def _cheapest_to_drop(counts: Dict[str, Dict[str, int]]):
        candidates: List = [
            (count, name, key) for key, names in counts.items() for name, count in names.items()
        ]
        if not candidates:
            raise ValueError("output style names alone exceed the signals byte budget")
        lowest = min(count for count, _, _ in candidates)
        _, name, key = max(c for c in candidates if c[0] == lowest)[0:3]
        return key, name
