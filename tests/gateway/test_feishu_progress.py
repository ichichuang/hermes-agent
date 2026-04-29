import json

from gateway.feishu_progress import (
    build_feishu_progress_card,
    phase_for_tool,
    progress_for_update_count,
)


def test_feishu_progress_card_hides_raw_tool_details():
    phase = phase_for_tool("skill_view")
    card = build_feishu_progress_card(
        phase=phase,
        progress_percent=progress_for_update_count(3),
        elapsed_seconds=12,
        update_count=4,
    )

    payload = json.dumps(card, ensure_ascii=False)

    assert phase == "读取能力上下文"
    assert "skill_view" not in payload
    assert "hermes-safe-auto-upgrade" not in payload
    assert "auto_safe_upgrade.py" not in payload
    assert "使用 skill" not in payload
    assert "生成变更" not in payload
    assert "📚" not in payload
    assert "🐍" not in payload
    assert "→" not in payload
    assert "#" not in payload
    assert "`" not in payload


def test_feishu_progress_card_uses_native_chart_progress():
    card = build_feishu_progress_card(
        phase="整理变更",
        progress_percent=72,
        elapsed_seconds=75,
        update_count=6,
        done=True,
    )

    payload = json.dumps(card, ensure_ascii=False)

    assert "Hermes 任务进度" in payload
    assert "准备发送答复" in payload
    assert "100%" in payload
    assert '"schema": "2.0"' in payload
    assert '"tag": "chart"' in payload
    assert '"type": "bar"' in payload
    assert '"animation": true' in payload
    assert "#" not in payload
    assert "`" not in payload
