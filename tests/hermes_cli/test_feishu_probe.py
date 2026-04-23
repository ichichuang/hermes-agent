import asyncio
import sys
import types
from types import SimpleNamespace

import hermes_cli.gateway as gateway_cli


def test_feishu_probe_requires_external_probe_env(monkeypatch):
    monkeypatch.delenv("ENABLE_EXTERNAL_PROBES", raising=False)
    errors = []
    monkeypatch.setattr(gateway_cli, "print_error", errors.append)

    rc = asyncio.run(gateway_cli._run_feishu_probe(confirm=True))

    assert rc == 1
    assert any("ENABLE_EXTERNAL_PROBES=true" in message for message in errors)


def test_feishu_probe_requires_confirmation_for_live_send(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTERNAL_PROBES", "true")
    errors = []
    monkeypatch.setattr(gateway_cli, "print_error", errors.append)

    rc = asyncio.run(gateway_cli._run_feishu_probe(confirm=False, dry_run=False))

    assert rc == 1
    assert any("--feishu-probe-confirm" in message for message in errors)


def test_feishu_probe_dry_run_prints_payloads_without_network(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTERNAL_PROBES", "true")
    infos = []
    warnings = []
    monkeypatch.setattr(gateway_cli, "print_info", infos.append)
    monkeypatch.setattr(gateway_cli, "print_warning", warnings.append)
    monkeypatch.setattr(gateway_cli, "_build_feishu_probe_payloads", lambda chat_id: ("raw", "clean", {"chat_id": chat_id, "content": "clean", "metadata": {"probe": "feishu"}}))

    from gateway.config import Platform
    fake_config = SimpleNamespace(
        platforms={
            Platform.FEISHU: SimpleNamespace(
                enabled=True,
                home_channel=SimpleNamespace(chat_id="chat_123"),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: fake_config)

    class ForbiddenAdapter:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("dry-run should not create a live adapter")

    fake_module = types.SimpleNamespace(FeishuAdapter=ForbiddenAdapter)
    monkeypatch.setitem(sys.modules, "gateway.platforms.feishu", fake_module)

    rc = asyncio.run(gateway_cli._run_feishu_probe(confirm=False, dry_run=True))

    assert rc == 0
    assert any("payload before send" in message for message in infos)
    assert any("payload after sanitizer" in message for message in infos)
    assert any("final payload" in message for message in infos)
    assert any("No network call was made" in message for message in warnings)


def test_feishu_probe_live_send_uses_final_payload(monkeypatch):
    monkeypatch.setenv("ENABLE_EXTERNAL_PROBES", "true")
    infos = []
    warnings = []
    monkeypatch.setattr(gateway_cli, "print_info", infos.append)
    monkeypatch.setattr(gateway_cli, "print_warning", warnings.append)
    monkeypatch.setattr(gateway_cli, "_build_feishu_probe_payloads", lambda chat_id: ("raw", "clean", {"chat_id": chat_id, "content": "final-clean", "metadata": {"probe": "feishu", "bypass_tools": True}}))

    from gateway.config import Platform
    fake_config = SimpleNamespace(
        platforms={
            Platform.FEISHU: SimpleNamespace(
                enabled=True,
                home_channel=SimpleNamespace(chat_id="chat_live"),
            )
        }
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: fake_config)

    sent = {}

    class FakeAdapter:
        def __init__(self, _config):
            pass

        async def connect(self):
            return True

        async def send(self, chat_id, content, metadata=None):
            sent["chat_id"] = chat_id
            sent["content"] = content
            sent["metadata"] = metadata
            return SimpleNamespace(success=True, message_id="m1", error=None)

        async def disconnect(self):
            sent["disconnected"] = True

    monkeypatch.setitem(sys.modules, "gateway.platforms.feishu", types.SimpleNamespace(FeishuAdapter=FakeAdapter))

    rc = asyncio.run(gateway_cli._run_feishu_probe(confirm=True, dry_run=False))

    assert rc == 0
    assert sent == {
        "chat_id": "chat_live",
        "content": "final-clean",
        "metadata": {"probe": "feishu", "bypass_tools": True},
        "disconnected": True,
    }
    assert any("External messaging will occur" in message for message in warnings)
    assert any("adapter output" in message for message in infos)
