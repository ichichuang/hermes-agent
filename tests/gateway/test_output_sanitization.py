from types import SimpleNamespace

import pytest

from gateway.display_config import resolve_display_setting
from gateway.run import GatewayRunner
from run_agent import AIAgent


class _CaptureAdapter:
    def __init__(self):
        self.messages = []

    async def send(self, chat_id, content, metadata=None):
        self.messages.append((chat_id, content, metadata))
        return SimpleNamespace(success=True, message_id="msg-1")


@pytest.mark.asyncio
async def test_deliver_gateway_message_sanitizes_payload():
    runner = object.__new__(GatewayRunner)
    adapter = _CaptureAdapter()

    await runner._deliver_gateway_message(
        adapter,
        "chat-1",
        'Clean reply\nto=functions.exec_command {"cmd":"whoami"}',
        metadata={"thread_id": "t1"},
    )

    assert adapter.messages == [
        ("chat-1", "Clean reply", {"thread_id": "t1", "_hermes_sanitizer_warning": "sanitizer_reapplied"})
    ]


def test_fire_stream_delta_strips_tool_noise():
    agent = object.__new__(AIAgent)
    seen = []
    agent.stream_delta_callback = seen.append
    agent._stream_callback = None
    agent._stream_needs_break = False
    agent._current_streamed_assistant_text = ""

    AIAgent._fire_stream_delta(agent, 'Visible\nto=multi_tool_use.parallel {"tool_uses":[]}')

    assert seen == ["Visible"]


def test_emit_interim_assistant_message_strips_internal_text():
    agent = object.__new__(AIAgent)
    seen = []
    agent.interim_assistant_callback = lambda text, **kwargs: seen.append((text, kwargs))
    agent._current_streamed_assistant_text = ""

    AIAgent._emit_interim_assistant_message(
        agent,
        {"role": "assistant", "content": 'Visible update\n"reasoning": "hidden"'},
    )

    assert seen == [("Visible update", {"already_streamed": False})]


def test_feishu_quiet_delivery_settings_resolve_off():
    config = {
        "display": {
            "platforms": {
                "feishu": {
                    "tool_progress": "off",
                    "interim_assistant_messages": False,
                }
            }
        }
    }

    assert resolve_display_setting(config, "feishu", "tool_progress") == "off"
    assert resolve_display_setting(config, "feishu", "interim_assistant_messages", True) is False


@pytest.mark.asyncio
async def test_deliver_gateway_message_preserves_clean_payload_metadata():
    runner = object.__new__(GatewayRunner)
    adapter = _CaptureAdapter()

    await runner._deliver_gateway_message(
        adapter,
        "chat-1",
        "Already clean",
        metadata={"thread_id": "t1"},
    )

    assert adapter.messages == [("chat-1", "Already clean", {"thread_id": "t1"})]
