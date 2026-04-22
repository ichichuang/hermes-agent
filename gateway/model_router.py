"""Dynamic model routing for gateway turns."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable
from typing import Mapping

from .complexity_analyzer import analyze_complexity

MODEL_MAP = {
    "mini": "gpt-5.4-mini",
    "standard": "gpt-5.1",
    "high": "gpt-5.2",
    "xhigh": "gpt-5.4",
}

_TIER_BOUNDARIES = {
    "mini": 0.2,
    "standard": 0.4,
    "high": 0.7,
}
_HYSTERESIS_MARGIN = 0.05

_INTENT_COMPLEXITY_FLOOR = {
    "L0_CHAT": 0.0,
    "L1_REASON": 0.2,
    "L1_REASONING": 0.2,
    "L2_TOOL": 0.4,
    "L3_HEAVY": 0.7,
}

_LIGHT_TOOLSET_ORDER = ("web", "file", "memory", "session_search", "skills", "vision", "homeassistant")


@dataclass(frozen=True)
class RoutingDecision:
    intent: str
    effective_level: str
    tier: str
    complexity: float
    model: str
    reasoning_effort: str
    tools_enabled: bool
    enabled_toolsets: tuple[str, ...]
    tool_mode: str
    load_heavy_prompts: bool
    max_tokens: int
    temperature: float
    cacheable: bool
    early_stopping: bool
    budget_tokens: int
    reason: str


def _strip_model_namespace(model: str) -> str:
    raw = str(model or "").strip()
    if "/" not in raw:
        return raw
    return raw.rsplit("/", 1)[-1].strip()


def _preserve_model_namespace(base_model: str, routed_model: str) -> str:
    base = str(base_model or "").strip()
    routed = str(routed_model or "").strip()
    if not base or not routed or "/" not in base:
        return routed

    prefix, suffix = base.rsplit("/", 1)
    if not suffix.lower().startswith("gpt-5"):
        return routed
    return f"{prefix}/{routed}"


def _resolve_model_map(
    *,
    base_model: str,
    lightweight_model: str | None = None,
    model_map: Mapping[str, str] | None = None,
    allow_model_tiering: bool = True,
) -> dict[str, str]:
    normalized_base = _strip_model_namespace(base_model).lower()
    resolved = {
        "mini": str(lightweight_model or base_model or "").strip(),
        "standard": str(base_model or "").strip(),
        "high": str(base_model or "").strip(),
        "xhigh": str(base_model or "").strip(),
    }

    if allow_model_tiering and normalized_base == "gpt-5.4":
        resolved.update(
            {
                tier: _preserve_model_namespace(base_model, routed_model)
                for tier, routed_model in MODEL_MAP.items()
            }
        )
        if lightweight_model:
            resolved["mini"] = str(lightweight_model).strip()

    for tier in MODEL_MAP:
        configured = str((model_map or {}).get(tier, "") or "").strip()
        if configured:
            resolved[tier] = configured

    if not resolved["mini"]:
        resolved["mini"] = resolved["standard"]
    return resolved


def _base_tier_for_complexity(intent: str, complexity: float) -> str:
    if intent == "L0_CHAT" and complexity < _TIER_BOUNDARIES["mini"]:
        return "mini"
    if complexity < _TIER_BOUNDARIES["standard"]:
        return "standard"
    if complexity < _TIER_BOUNDARIES["high"]:
        return "high"
    return "xhigh"


def _stabilize_tier_with_hysteresis(
    *,
    intent: str,
    complexity: float,
    candidate_tier: str,
    previous_tier: str | None,
) -> tuple[str, bool]:
    previous = str(previous_tier or "").strip().lower()
    if previous not in MODEL_MAP:
        return candidate_tier, False
    if previous == candidate_tier:
        return candidate_tier, False

    pair = frozenset((previous, candidate_tier))
    boundary = None
    if pair == frozenset(("mini", "standard")) and intent == "L0_CHAT":
        boundary = _TIER_BOUNDARIES["mini"]
    elif pair == frozenset(("standard", "high")):
        boundary = _TIER_BOUNDARIES["standard"]
    elif pair == frozenset(("high", "xhigh")):
        boundary = _TIER_BOUNDARIES["high"]

    if boundary is None:
        return candidate_tier, False
    if abs(complexity - boundary) > _HYSTERESIS_MARGIN:
        return candidate_tier, False
    return previous, True


def _scale_token_limit(
    *,
    complexity: float,
    lower_bound: float,
    upper_bound: float,
    min_tokens: int,
    max_tokens: int,
    step: int,
) -> int:
    if max_tokens <= min_tokens:
        return int(max_tokens)
    if upper_bound <= lower_bound:
        return int(max_tokens)

    normalized = max(0.0, min((complexity - lower_bound) / (upper_bound - lower_bound), 1.0))
    raw_tokens = float(min_tokens) + (float(max_tokens - min_tokens) * normalized)
    quantized = int(math.floor((raw_tokens + (step / 2.0)) / step) * step)
    return max(int(min_tokens), min(int(max_tokens), quantized))


def route_model(
    *,
    intent: str,
    input_text: str,
    configured_toolsets: Iterable[str],
    model: str = "gpt-5.4",
    lightweight_model: str | None = None,
    runtime_feedback: Mapping[str, object] | None = None,
    model_map: Mapping[str, str] | None = None,
    allow_model_tiering: bool = True,
) -> RoutingDecision:
    """Return the routing config for a gateway turn."""
    configured = tuple(sorted(str(toolset) for toolset in (configured_toolsets or [])))
    raw_complexity = analyze_complexity(input_text, runtime_feedback=runtime_feedback)
    effective_complexity = max(raw_complexity, _INTENT_COMPLEXITY_FLOOR.get(intent, 0.0))
    if not str(lightweight_model or "").strip() and "gpt-5.4" in str(model or "").lower():
        lightweight_model = MODEL_MAP["mini"]
    resolved_model_map = _resolve_model_map(
        base_model=model,
        lightweight_model=lightweight_model,
        model_map=model_map,
        allow_model_tiering=allow_model_tiering,
    )
    previous_tier = str((runtime_feedback or {}).get("previous_tier", "") or "").strip()
    candidate_tier = _base_tier_for_complexity(intent, effective_complexity)
    tier, hysteresis_applied = _stabilize_tier_with_hysteresis(
        intent=intent,
        complexity=effective_complexity,
        candidate_tier=candidate_tier,
        previous_tier=previous_tier,
    )
    l0_model = resolved_model_map["mini"]
    standard_model = resolved_model_map["standard"]
    has_dedicated_lightweight_model = bool(l0_model) and l0_model != standard_model
    l0_max_tokens = _scale_token_limit(
        complexity=effective_complexity,
        lower_bound=0.0,
        upper_bound=_TIER_BOUNDARIES["mini"],
        min_tokens=64 if has_dedicated_lightweight_model else 48,
        max_tokens=96 if has_dedicated_lightweight_model else 80,
        step=8,
    )
    l1_max_tokens = _scale_token_limit(
        complexity=effective_complexity,
        lower_bound=_INTENT_COMPLEXITY_FLOOR["L1_REASONING"],
        upper_bound=_TIER_BOUNDARIES["standard"],
        min_tokens=256,
        max_tokens=480,
        step=32,
    )
    l2_max_tokens = _scale_token_limit(
        complexity=effective_complexity,
        lower_bound=_INTENT_COMPLEXITY_FLOOR["L2_TOOL"],
        upper_bound=_TIER_BOUNDARIES["high"],
        min_tokens=1024,
        max_tokens=1536,
        step=64,
    )

    if intent == "L0_CHAT" and tier == "mini":
        return RoutingDecision(
            intent=intent,
            effective_level="L0_CHAT",
            tier="mini",
            complexity=effective_complexity,
            model=l0_model,
            reasoning_effort="low",
            tools_enabled=False,
            enabled_toolsets=(),
            tool_mode="none",
            load_heavy_prompts=False,
            max_tokens=l0_max_tokens,
            temperature=0.7,
            cacheable=True,
            early_stopping=not has_dedicated_lightweight_model,
            budget_tokens=1200,
            reason=(
                "ultra-simple conversational turn"
                if not hysteresis_applied
                else f"tier hysteresis held at {tier} near boundary"
            ) if has_dedicated_lightweight_model else "ultra-simple turn on fallback primary model",
        )

    if tier == "standard":
        return RoutingDecision(
            intent=intent,
            effective_level="L1_REASONING",
            tier="standard",
            complexity=effective_complexity,
            model=resolved_model_map["standard"],
            reasoning_effort="medium",
            tools_enabled=False,
            enabled_toolsets=(),
            tool_mode="none",
            load_heavy_prompts=False,
            max_tokens=l1_max_tokens,
            temperature=0.5,
            cacheable=False,
            early_stopping=False,
            budget_tokens=3200,
            reason="tier hysteresis held at standard near boundary" if hysteresis_applied else "medium reasoning turn",
        )

    if tier == "high":
        light = tuple(ts for ts in _LIGHT_TOOLSET_ORDER if ts in configured)
        enabled = light or configured
        return RoutingDecision(
            intent=intent,
            effective_level="L2_TOOL",
            tier="high",
            complexity=effective_complexity,
            model=resolved_model_map["high"],
            reasoning_effort="high",
            tools_enabled=True,
            enabled_toolsets=enabled,
            tool_mode="light" if enabled else "none",
            load_heavy_prompts=bool(enabled),
            max_tokens=l2_max_tokens,
            temperature=0.3,
            cacheable=False,
            early_stopping=False,
            budget_tokens=5200,
            reason="tier hysteresis held at high near boundary" if hysteresis_applied else "tool-capable high-complexity turn",
        )

    return RoutingDecision(
        intent=intent,
        effective_level="L3_HEAVY",
        tier="xhigh",
        complexity=effective_complexity,
        model=resolved_model_map["xhigh"],
        reasoning_effort="xhigh",
        tools_enabled=True,
        enabled_toolsets=configured,
        tool_mode="full" if configured else "none",
        load_heavy_prompts=bool(configured),
        max_tokens=3000,
        temperature=0.2,
        cacheable=False,
        early_stopping=False,
        budget_tokens=10000,
        reason="tier hysteresis held at xhigh near boundary" if hysteresis_applied else "heavy execution or agentic turn",
    )


def route_v2_5(
    *,
    intent: str,
    input_text: str,
    configured_toolsets: Iterable[str],
    model: str = "gpt-5.4",
    lightweight_model: str | None = None,
    runtime_feedback: Mapping[str, object] | None = None,
    model_map: Mapping[str, str] | None = None,
    allow_model_tiering: bool = True,
) -> RoutingDecision:
    return route_model(
        intent=intent,
        input_text=input_text,
        configured_toolsets=configured_toolsets,
        model=model,
        lightweight_model=lightweight_model,
        runtime_feedback=runtime_feedback,
        model_map=model_map,
        allow_model_tiering=allow_model_tiering,
    )


__all__ = ["MODEL_MAP", "RoutingDecision", "route_model", "route_v2_5"]
