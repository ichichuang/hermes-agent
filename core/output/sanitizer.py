from dataclasses import dataclass
import re
from typing import Any


_TOOL_NOISE_PATTERNS = [
    r"\s*to=(?:multi_tool_use|functions)\..*$",
    r"RTLU to=.*",
]

_INTERNAL_ASSISTANT_FIELDS = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "tool_calls",
    "codex_reasoning_items",
)


@dataclass(frozen=True)
class SanitizedAssistantAssertion:
    cleaned: Any
    violated: bool
    warning: str | None = None


_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*[{}\[\]\",:]+\s*$"),
    re.compile(r"^\s*RTLU\b.*$"),
    re.compile(r"^\s*(?:to=(?:multi_tool_use|functions)\..*)$"),
    re.compile(r'.*"(?:tool_uses|recipient_name|parameters|notify_on_complete|workdir)".*'),
    re.compile(
        r'^\s*"?(?:tool_uses|recipient_name|parameters|output|reasoning|finish_reason|'
        r'commentary|status|success|error|background|command|notify_on_complete|'
        r'pty|timeout|max_output_tokens|max_output_chars|yield_time_ms|workdir)'
        r'"?\s*:?.*$'
    ),
]


def strip_tool_noise(text: str) -> str:
    if not text:
        return text

    cleaned = text
    for pattern in _TOOL_NOISE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE)
    filtered_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if any(pattern.match(stripped) for pattern in _NOISE_LINE_PATTERNS):
            continue
        filtered_lines.append(line.rstrip())
    cleaned = "\n".join(filtered_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_assistant_message(msg: dict) -> dict:
    if not isinstance(msg, dict):
        return msg

    cleaned = dict(msg)
    for field in _INTERNAL_ASSISTANT_FIELDS:
        cleaned.pop(field, None)
    if cleaned.get("finish_reason") == "tool_calls":
        cleaned.pop("finish_reason", None)

    if isinstance(cleaned.get("content"), str):
        cleaned["content"] = strip_tool_noise(cleaned["content"])

    return cleaned


def sanitize_assistant_text(text: Any) -> Any:
    if isinstance(text, str):
        return strip_tool_noise(text)
    return text


def assert_sanitized_assistant_output(data: Any) -> SanitizedAssistantAssertion:
    if isinstance(data, dict):
        cleaned = sanitize_assistant_message(data)
    else:
        cleaned = sanitize_assistant_text(data)
    violated = cleaned != data
    return SanitizedAssistantAssertion(
        cleaned=cleaned,
        violated=violated,
        warning="sanitizer_reapplied" if violated else None,
    )
