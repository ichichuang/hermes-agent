"""Lightweight intent classification and adaptive tool/model policy for gateway turns."""

from __future__ import annotations

import re
from typing import Mapping
from typing import Iterable, Sequence

from .model_router import RoutingDecision, route_model


_NORMALIZE_RE = re.compile(r"[\s`'\"“”‘’.,，。!?！？:：;；()\[\]{}<>《》\-_/\\]+")
_QUESTION_HINTS = (
    "?",
    "？",
    "为什么",
    "为何",
    "怎么",
    "怎样",
    "what",
    "why",
    "how",
    "when",
    "will",
    "会不会",
)
_RESET_EXACT_PHRASES = {
    "清理上下文",
    "清除上下文",
    "清空上下文",
    "重置上下文",
    "清理所有上下文",
    "清空所有上下文",
    "开始新任务",
    "开启新任务",
    "重新开始",
    "重新开始对话",
    "新任务开始",
    "resetcontext",
    "resetsession",
    "clearcontext",
    "clearsession",
    "startnewsession",
    "startanewsession",
    "startnewtask",
    "freshstart",
    "清理所有上下文准备开始新任务",
}
_RESET_VERBS = ("清理", "清除", "清空", "重置", "reset", "clear")
_RESET_OBJECTS = ("上下文", "context", "session", "会话")
_RESET_FRESH_START = ("新任务", "新会话", "重新开始", "freshstart", "startnewtask", "startnewsession")

_HEAVY_TERMS = (
    "terminal",
    "shell",
    "browser",
    "delegate",
    "subagent",
    "automation",
    "执行命令",
    "运行命令",
    "执行脚本",
    "打开浏览器",
    "调用agent",
    "调用工具",
    "子任务",
    "自动化",
    "压缩记录",
)
_TOOL_TERMS = (
    "/users/",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    "日志",
    "文件",
    "代码",
    "路径",
    "目录",
    "命令",
    "搜索",
    "检索",
    "读取",
    "查看",
    "分析这个session",
    "tool",
    "tools",
    "prompt token",
    "grep",
    "rg ",
    "read ",
    "write ",
    "patch",
)
def _normalize_message(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return ""
    return _NORMALIZE_RE.sub("", lowered)


IntentDecision = RoutingDecision


def looks_like_explicit_reset_request(text: str) -> bool:
    """Match only short, explicit reset intents to avoid accidental resets."""
    raw = str(text or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if any(hint in lowered for hint in _QUESTION_HINTS):
        return False

    normalized = _normalize_message(raw)
    if normalized in _RESET_EXACT_PHRASES:
        return True

    if len(normalized) > 48:
        return False

    has_reset_verb = any(term in normalized for term in _RESET_VERBS)
    has_reset_object = any(term in normalized for term in _RESET_OBJECTS)
    has_fresh_start = any(term in normalized for term in _RESET_FRESH_START)

    return (has_reset_verb and has_reset_object) or (has_reset_verb and has_fresh_start)


def classify_message_intent(message: str) -> str:
    """Classify a turn into lightweight chat/reason/tool/heavy buckets."""
    raw = str(message or "").strip()
    lowered = raw.lower()
    normalized = _normalize_message(raw)

    if not normalized:
        return "L0_CHAT"

    if any(term in lowered for term in _HEAVY_TERMS):
        return "L3_HEAVY"

    if raw.startswith("/") or any(term in lowered for term in _TOOL_TERMS):
        return "L2_TOOL"

    word_count = len(raw.split())
    if len(raw) <= 48 and word_count <= 12:
        return "L0_CHAT"

    return "L1_REASON"


def decide_tool_policy(
    message: str,
    configured_toolsets: Iterable[str],
    *,
    model: str = "gpt-5.4",
    lightweight_model: str | None = None,
    runtime_feedback: Mapping[str, object] | None = None,
    model_map: Mapping[str, str] | None = None,
    allow_model_tiering: bool = True,
) -> IntentDecision:
    """Map a message to a concrete tool policy for this turn."""
    intent = classify_message_intent(message)
    return route_model(
        intent=intent,
        input_text=message,
        configured_toolsets=configured_toolsets,
        model=model,
        lightweight_model=lightweight_model,
        runtime_feedback=runtime_feedback,
        model_map=model_map,
        allow_model_tiering=allow_model_tiering,
    )


def summarize_tool_decision(decision: IntentDecision, configured_toolsets: Sequence[str]) -> str:
    configured = list(configured_toolsets or [])
    enabled = list(decision.enabled_toolsets)
    return (
        f"intent={decision.effective_level} classifier={decision.intent} "
        f"model={decision.model} tier={decision.tier} reasoning={decision.reasoning_effort} "
        f"complexity={decision.complexity:.2f} max_tokens={decision.max_tokens} "
        f"tool_mode={decision.tool_mode} configured={configured} enabled={enabled} "
        f"heavy_prompts={decision.load_heavy_prompts} budget={decision.budget_tokens}"
    )
