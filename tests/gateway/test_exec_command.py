"""Tests for gateway /exec and /gh direct execution commands."""

import json

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source, message_id="msg-1")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    return runner


def test_parse_direct_exec_args_without_flags():
    parsed = gateway_run.GatewayRunner._parse_direct_exec_args("git status")

    assert parsed["command"] == "git status"
    assert parsed["background"] is False
    assert parsed["notify_on_complete"] is False
    assert parsed["pty"] is False
    assert parsed["timeout"] is None
    assert parsed["workdir"] is None


def test_parse_direct_exec_args_preserves_command_double_dash_options():
    parsed = gateway_run.GatewayRunner._parse_direct_exec_args("git status --short --branch")

    assert parsed["command"] == "git status --short --branch"
    assert parsed["timeout"] is None
    assert parsed["workdir"] is None


def test_parse_direct_exec_args_with_flags():
    parsed = gateway_run.GatewayRunner._parse_direct_exec_args(
        "--background --notify --pty --timeout 90 --workdir /Users/cc/.hermes -- git status"
    )

    assert parsed["command"] == "git status"
    assert parsed["background"] is True
    assert parsed["notify_on_complete"] is True
    assert parsed["pty"] is True
    assert parsed["timeout"] == 90
    assert parsed["workdir"] == "/Users/cc/.hermes"


@pytest.mark.asyncio
async def test_handle_exec_command_usage_when_missing_args():
    runner = _make_runner()

    result = await runner._handle_exec_command(_make_event("/exec"))

    assert "Usage: /exec" in result


@pytest.mark.asyncio
async def test_handle_exec_command_executes_terminal_tool(monkeypatch):
    runner = _make_runner()
    captured = {}

    import tools.terminal_tool as terminal_tool_module

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "On branch main", "exit_code": 0})

    async def run_inline(func, *args):
        return func(*args)

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)
    runner._run_in_executor_with_context = run_inline

    result = await runner._handle_exec_command(
        _make_event("/exec --timeout 45 --workdir /Users/cc/.hermes -- git status --short --branch")
    )

    assert captured["command"] == "git status --short --branch"
    assert captured["timeout"] == 45
    assert captured["workdir"] == "/Users/cc/.hermes"
    assert "Command completed" in result
    assert "On branch main" in result


@pytest.mark.asyncio
async def test_handle_gh_command_prefixes_gh(monkeypatch):
    runner = _make_runner()
    captured = {}

    import tools.terminal_tool as terminal_tool_module

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "Logged in to github.com", "exit_code": 0})

    async def run_inline(func, *args):
        return func(*args)

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)
    runner._run_in_executor_with_context = run_inline

    result = await runner._handle_gh_command(
        _make_event("/gh --workdir /Users/cc/.hermes -- pr status")
    )

    assert captured["command"] == "gh pr status"
    assert captured["workdir"] == "/Users/cc/.hermes"
    assert "gh pr status" in result
    assert "Logged in to github.com" in result


@pytest.mark.asyncio
async def test_handle_gh_command_allows_native_gh_flags_without_separator(monkeypatch):
    runner = _make_runner()
    captured = {}

    import tools.terminal_tool as terminal_tool_module

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return json.dumps({"output": "gh version 2.x", "exit_code": 0})

    async def run_inline(func, *args):
        return func(*args)

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)
    runner._run_in_executor_with_context = run_inline

    result = await runner._handle_gh_command(_make_event("/gh --version"))

    assert captured["command"] == "gh --version"
    assert "gh version 2.x" in result


@pytest.mark.asyncio
async def test_handle_exec_command_background_formats_session(monkeypatch):
    runner = _make_runner()

    import tools.terminal_tool as terminal_tool_module

    def fake_terminal_tool(**kwargs):
        return json.dumps(
            {
                "output": "Background process started",
                "session_id": "proc-123",
                "exit_code": 0,
                "notify_on_complete": True,
            }
        )

    async def run_inline(func, *args):
        return func(*args)

    monkeypatch.setattr(terminal_tool_module, "terminal_tool", fake_terminal_tool)
    runner._run_in_executor_with_context = run_inline

    result = await runner._handle_exec_command(
        _make_event("/exec --background --notify -- python3 ~/.hermes/scripts/auto_safe_upgrade.py upgrade")
    )

    assert "Started" in result
    assert "proc-123" in result
    assert "Completion notification: enabled" in result
