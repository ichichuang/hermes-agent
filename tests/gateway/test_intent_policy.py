"""Tests for gateway intent classification, complexity scoring, and model routing."""

from gateway.complexity_analyzer import analyze_complexity
from gateway.intent_policy import decide_tool_policy, looks_like_explicit_reset_request


def test_explicit_reset_phrase_is_detected():
    assert looks_like_explicit_reset_request("清理所有上下文 准备开始新任务") is True


def test_reset_question_is_not_misclassified():
    assert looks_like_explicit_reset_request("为什么清理上下文之后 token 还是很多？") is False


def test_complexity_analyzer_scores_code_and_logic():
    score = analyze_complexity("def route(x):\n    if x:\n        explain architecture")
    assert score >= 0.5


def test_light_chat_routes_to_low_effort_without_tools():
    decision = decide_tool_policy("你好", ["web", "file", "memory"])

    assert decision.intent == "L0_CHAT"
    assert decision.effective_level == "L0_CHAT"
    assert decision.tier == "mini"
    assert decision.model == "gpt-5.4-mini"
    assert decision.tools_enabled is False
    assert decision.enabled_toolsets == ()
    assert decision.reasoning_effort == "low"
    assert decision.max_tokens == 64
    assert decision.temperature == 0.7
    assert decision.cacheable is True


def test_medium_reasoning_routes_without_tools():
    decision = decide_tool_policy(
        "Please explain why this design uses a service layer and how it helps maintainability.",
        ["web", "file", "memory"],
    )

    assert decision.intent == "L1_REASON"
    assert decision.effective_level == "L1_REASONING"
    assert decision.tier == "standard"
    assert decision.model == "gpt-5.1"
    assert decision.tools_enabled is False
    assert decision.reasoning_effort == "medium"
    assert decision.max_tokens == 448
    assert decision.temperature == 0.5


def test_runtime_feedback_escalates_complexity():
    decision = decide_tool_policy(
        "hello",
        ["web", "file", "memory"],
        runtime_feedback={
            "used_tools_previously": True,
            "retry_count": 1,
            "previous_failure": True,
        },
    )

    assert decision.effective_level == "L2_TOOL"
    assert decision.tier == "high"
    assert decision.reasoning_effort == "high"
    assert decision.max_tokens == 1216


def test_router_hysteresis_keeps_standard_near_boundary():
    decision = decide_tool_policy(
        "x" * 205,
        ["web", "file", "memory"],
        runtime_feedback={"previous_tier": "standard"},
    )

    assert decision.tier == "standard"
    assert decision.reasoning_effort == "medium"


def test_l1_max_tokens_floor_is_quantized_for_low_complexity_reasoning():
    decision = decide_tool_policy(
        "Summarize this module in one short paragraph for the team handoff notes.",
        ["web", "file", "memory"],
    )

    assert decision.tier == "standard"
    assert decision.max_tokens == 256


def test_complexity_analyzer_uses_decayed_penalty():
    score = analyze_complexity(
        "hello",
        runtime_feedback={
            "used_tools_previously": False,
            "retry_count": 0,
            "previous_failure": False,
            "decayed_penalty": 0.4,
        },
    )

    assert score >= 0.4


def test_file_analysis_uses_light_tool_subset():
    decision = decide_tool_policy(
        "请读取 /Users/cc/.hermes/session.json 并分析日志",
        ["web", "file", "memory", "session_search", "exa"],
    )

    assert decision.intent == "L2_TOOL"
    assert decision.effective_level == "L2_TOOL"
    assert decision.tier == "high"
    assert decision.model == "gpt-5.2"
    assert decision.tool_mode == "light"
    assert decision.tools_enabled is True
    assert decision.enabled_toolsets == ("web", "file", "memory", "session_search")
    assert decision.reasoning_effort == "high"
    assert decision.max_tokens == 1024
    assert decision.temperature == 0.3


def test_heavy_execution_keeps_full_toolset():
    decision = decide_tool_policy(
        "运行命令并调用agent处理这个任务",
        ["web", "file", "memory", "exa"],
    )

    assert decision.intent == "L3_HEAVY"
    assert decision.effective_level == "L3_HEAVY"
    assert decision.tier == "xhigh"
    assert decision.model == "gpt-5.4"
    assert decision.tool_mode == "full"
    assert decision.tools_enabled is True
    assert decision.enabled_toolsets == ("exa", "file", "memory", "web")
    assert decision.reasoning_effort == "xhigh"
    assert decision.max_tokens == 3000
    assert decision.temperature == 0.2


def test_explicit_model_map_can_use_gpt53_when_configured():
    decision = decide_tool_policy(
        "请读取这个文件并 explain architecture decisions in the code path",
        ["web", "file", "memory"],
        model="gpt-5.4",
        model_map={
            "mini": "gpt-5.4-mini",
            "standard": "gpt-5.1",
            "high": "gpt-5.3-codex",
            "xhigh": "gpt-5.4",
        },
    )

    assert decision.tier == "high"
    assert decision.model == "gpt-5.3-codex"
