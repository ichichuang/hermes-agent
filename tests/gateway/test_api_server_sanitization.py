from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app._state["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


@pytest.mark.asyncio
async def test_chat_completions_sanitizes_final_assistant_text():
    adapter = _make_adapter()
    app = _create_app(adapter)

    mock_result = {
        "final_response": 'Visible answer\nto=functions.exec_command {"cmd":"whoami"}\n"reasoning": "hidden"',
        "messages": [],
        "api_calls": 1,
    }

    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
            )

        assert resp.status == 200
        data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "Visible answer"
        assert data["_hermes"]["sanitizer_warning"] is True
        assert resp.headers["X-Hermes-Sanitizer-Warning"] == "1"


@pytest.mark.asyncio
async def test_responses_stream_sanitizes_deltas_and_terminal_text():
    adapter = _make_adapter()
    app = _create_app(adapter)

    async def _mock_run_agent(**kwargs):
        cb = kwargs.get("stream_delta_callback")
        if cb:
            cb('Visible\nto=multi_tool_use.parallel {"tool_uses":[]}')
            cb('\n"reasoning": "hidden"')
        return (
            {
                "final_response": 'Visible\nto=functions.exec_command {"cmd":"whoami"}',
                "messages": [],
                "api_calls": 1,
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            resp = await cli.post(
                "/v1/responses",
                json={"model": "hermes-agent", "input": "hi", "stream": True},
            )

        assert resp.status == 200
        body = await resp.text()
        assert "Visible" in body
        assert "to=functions.exec_command" not in body
        assert "to=multi_tool_use.parallel" not in body
        assert '"reasoning": "hidden"' not in body
        assert "x_hermes_sanitizer_warning" in body
