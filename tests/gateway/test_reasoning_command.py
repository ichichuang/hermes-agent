"""Tests for gateway /reasoning command and hot reload behavior."""

import asyncio
import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._routing_feedback = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


class _CapturingAgent:
    """Fake agent that records init kwargs for assertions."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


class TestReasoningCommand:
    @pytest.mark.asyncio
    async def test_reasoning_in_help_output(self):
        runner = _make_runner()
        event = _make_event(text="/help")

        result = await runner._handle_help_command(event)

        assert "/reasoning [level|show|hide]" in result

    def test_reasoning_is_known_command(self):
        source = inspect.getsource(gateway_run.GatewayRunner._handle_message)
        assert '"reasoning"' in source

    @pytest.mark.asyncio
    async def test_reasoning_command_reloads_current_state_from_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "agent:\n  reasoning_effort: none\ndisplay:\n  show_reasoning: true\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "xhigh"}
        runner._show_reasoning = False

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert "**Effort:** `none (disabled)`" in result
        assert "**Display:** on ✓" in result
        assert runner._reasoning_config == {"enabled": False}
        assert runner._show_reasoning is True

    @pytest.mark.asyncio
    async def test_handle_reasoning_command_updates_config_and_cache(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}

        result = await runner._handle_reasoning_command(_make_event("/reasoning low"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "low"
        assert runner._reasoning_config == {"enabled": True, "effort": "low"}
        assert "takes effect on next message" in result

    def test_run_agent_reloads_reasoning_config_per_message(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "xhigh"}

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "low"}
        assert _CapturingAgent.last_init["model"] == "gpt-5.4-mini"
        assert _CapturingAgent.last_init["max_tokens"] == 64
        assert _CapturingAgent.last_init["request_overrides"]["temperature"] == 0.7

    def test_run_agent_includes_enabled_mcp_servers_in_gateway_toolsets(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n"
            "mcp_servers:\n"
            "  exa:\n"
            "    url: https://mcp.exa.ai/mcp\n"
            "  web-search-prime:\n"
            "    url: https://api.z.ai/api/mcp/web_search_prime/mcp\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="请读取 session 日志并分析这个文件",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        enabled_toolsets = set(_CapturingAgent.last_init["enabled_toolsets"])
        assert "web" in enabled_toolsets
        assert "memory" in enabled_toolsets
        assert "exa" not in enabled_toolsets
        assert "web-search-prime" not in enabled_toolsets

    def test_run_agent_heavy_turn_keeps_full_gateway_toolsets(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n"
            "mcp_servers:\n"
            "  exa:\n"
            "    url: https://mcp.exa.ai/mcp\n"
            "  web-search-prime:\n"
            "    url: https://api.z.ai/api/mcp/web_search_prime/mcp\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="运行命令并调用agent处理这个任务",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        enabled_toolsets = set(_CapturingAgent.last_init["enabled_toolsets"])
        assert "web" in enabled_toolsets
        assert "memory" in enabled_toolsets
        assert "exa" in enabled_toolsets
        assert "web-search-prime" in enabled_toolsets

    def test_run_agent_homeassistant_uses_default_platform_toolset(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.HOMEASSISTANT,
            chat_id="ha",
            chat_name="Home Assistant",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="请读取当前 Home Assistant 状态",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:homeassistant:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert "homeassistant" in set(_CapturingAgent.last_init["enabled_toolsets"])

    def test_run_agent_light_chat_disables_gateway_toolsets(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="你好",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["model"] == "gpt-5.4-mini"
        assert _CapturingAgent.last_init["enabled_toolsets"] == []
        assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "low"}
        assert _CapturingAgent.last_init["prefill_messages"] is None
        assert _CapturingAgent.last_init["max_tokens"] == 64
        assert _CapturingAgent.last_init["request_overrides"]["temperature"] == 0.7
        assert result["model"] == "gpt-5.4-mini"
        assert result["routing_tier"] == "mini"
        assert result["reasoning_effort"] == "low"
        assert result["max_tokens"] == 64

    def test_run_agent_tool_turn_routes_high_reasoning_and_loads_heavy_prompts(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()
        runner._prefill_messages = [{"role": "user", "content": "prime"}]

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="请读取这个文件并分析日志",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "high"}
        assert _CapturingAgent.last_init["prefill_messages"] == [{"role": "user", "content": "prime"}]
        assert _CapturingAgent.last_init["max_tokens"] == 1024
        assert _CapturingAgent.last_init["request_overrides"]["temperature"] == 0.3

    def test_run_agent_cacheable_turn_can_bypass_llm_with_fast_cache(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()
        runner._fast_response_cache = gateway_run.OrderedDict(
            {
                "L0_CHAT|mini|low|false|hello": {
                    "input_text": "hello",
                    "response": "cached hello",
                    "routing_bucket": "L0_CHAT|mini|low|false",
                    "vector": gateway_run.vectorize_text("hello"),
                }
            }
        )

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="hello",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "cached hello"
        assert result["api_calls"] == 0
        assert result["cache_hit"] is True
        assert result["model"] == "gpt-5.4-mini"
        assert result["routing_tier"] == "mini"
        assert result["reasoning_effort"] == "low"
        assert _CapturingAgent.last_init is None

    def test_build_session_meta_entry_prefers_routed_model_over_default(self):
        source = SessionSource(
            platform=Platform.FEISHU,
            chat_id="chat-1",
            chat_name="Feishu",
            chat_type="dm",
            user_id="user-1",
        )

        entry = gateway_run.GatewayRunner._build_session_meta_entry(
            agent_result={
                "tools": [],
                "model": "gpt-5.4-mini",
                "routing_tier": "mini",
                "reasoning_effort": "low",
                "max_tokens": 64,
                "intent_level": "L0_CHAT",
                "tool_mode": "none",
                "enabled_toolsets": [],
                "routing_complexity": 0.02,
                "cache_hit": True,
            },
            source=source,
            timestamp="2026-04-22T21:28:34.029079",
            fallback_model="gpt-5.4",
        )

        assert entry["model"] == "gpt-5.4-mini"
        assert entry["routing_tier"] == "mini"
        assert entry["reasoning_effort"] == "low"
        assert entry["max_tokens"] == 64
        assert entry["intent_level"] == "L0_CHAT"
        assert entry["cache_hit"] is True

    def test_run_agent_cacheable_turn_uses_semantic_cache(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "test-key",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()
        runner._fast_response_cache = gateway_run.OrderedDict(
            {
                "L0_CHAT|mini|low|false|hello": {
                    "input_text": "hello",
                    "response": "cached hello",
                    "routing_bucket": "L0_CHAT|mini|low|false",
                    "vector": gateway_run.vectorize_text("hello"),
                }
            }
        )

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="hey",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "cached hello"
        assert result["cache_hit"] is True
        assert _CapturingAgent.last_init is None

    def test_agent_cache_signature_includes_routing_context(self):
        runtime = {
            "api_key": "secret",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        }

        sig_l0 = gateway_run.GatewayRunner._agent_config_signature(
            "gpt-5.4-mini",
            runtime,
            [],
            "",
            "L0_CHAT",
            "mini",
            "low",
            False,
            "hello",
            120,
            {"temperature": 0.7},
        )
        sig_l3 = gateway_run.GatewayRunner._agent_config_signature(
            "gpt-5.4",
            runtime,
            ["web"],
            "",
            "L3_HEAVY",
            "xhigh",
            "xhigh",
            True,
            "hello",
            3000,
            {"temperature": 0.2},
        )

        assert sig_l0 != sig_l3

    def test_agent_cache_signature_ignores_input_text(self):
        runtime = {
            "api_key": "secret",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        }

        sig1 = gateway_run.GatewayRunner._agent_config_signature(
            "gpt-5.4-mini",
            runtime,
            [],
            "",
            "L0_CHAT",
            "mini",
            "low",
            False,
            "hello",
            120,
            {"temperature": 0.7},
        )
        sig2 = gateway_run.GatewayRunner._agent_config_signature(
            "gpt-5.4-mini",
            runtime,
            [],
            "",
            "L0_CHAT",
            "mini",
            "low",
            False,
            "hey there",
            4000,
            {"temperature": 0.2},
        )

        assert sig1 == sig2

    def test_routing_feedback_penalty_decays_per_turn(self):
        runner = _make_runner()

        runner._update_routing_feedback(
            session_key="agent:main:local:dm",
            user_id="user-1",
            used_tools=True,
            failed=True,
            routed_tier="high",
        )
        first = runner._routing_feedback_for("agent:main:local:dm", "user-1")

        runner._update_routing_feedback(
            session_key="agent:main:local:dm",
            user_id="user-1",
            used_tools=False,
            failed=False,
            routed_tier="standard",
        )
        second = runner._routing_feedback_for("agent:main:local:dm", "user-1")

        assert first["decayed_penalty"] == pytest.approx(0.5)
        assert second["decayed_penalty"] == pytest.approx(0.4)
        assert second["previous_tier"] == "standard"
