"""Feishu progress-card helpers.

The gateway's normal tool progress is useful in terminal-like surfaces, but it
is too noisy for Feishu.  This module renders a single high-level card and
intentionally avoids exposing tool names, skill names, file previews, or raw
arguments.
"""

from __future__ import annotations

from typing import Any


_TOOL_PHASES = {
    "read_file": "读取项目上下文",
    "search_files": "检索相关代码",
    "terminal": "执行本地操作",
    "execute_code": "执行本地操作",
    "web_search": "检索外部信息",
    "write_file": "整理变更",
    "edit_file": "整理变更",
    "apply_patch": "整理变更",
    "todo": "更新任务状态",
    "memory": "整理上下文",
    "skill": "读取能力上下文",
}


def phase_for_tool(tool_name: str | None) -> str:
    """Map internal tool activity to a user-safe Feishu progress phase."""
    name = (tool_name or "").lower()
    for key, phase in _TOOL_PHASES.items():
        if key in name:
            return phase
    return "处理请求"


def progress_for_update_count(update_count: int) -> int:
    """Return a smooth synthetic progress percentage for long-running turns."""
    if update_count <= 0:
        return 6
    return min(96, 12 + update_count * 7)


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m"


def _progress_chart(percent: int, *, done: bool) -> dict[str, Any]:
    """Return a compact VChart spec that renders as a real progress bar."""
    percent = max(0, min(100, int(percent)))
    remaining = max(0, 100 - percent)
    fill = "rgba(34,197,94,1)" if done else "rgba(37,99,235,1)"
    track = "rgba(37,99,235,0.12)"

    return {
        "type": "bar",
        "direction": "horizontal",
        "data": [
            {
                "id": "progress",
                "values": [
                    {"item": "progress", "segment": "done", "value": percent},
                    {"item": "progress", "segment": "remaining", "value": remaining},
                ],
            }
        ],
        "xField": "value",
        "yField": "item",
        "seriesField": "segment",
        "stack": True,
        "color": [fill, track],
        "padding": {"top": 0, "right": 0, "bottom": 0, "left": 0},
        "barWidth": 16,
        "bar": {"style": {"cornerRadius": 8}},
        "axes": [
            {"orient": "bottom", "visible": False},
            {"orient": "left", "visible": False},
        ],
        "legends": {"visible": False},
        "tooltip": {"visible": False},
        "label": {"visible": False},
        "animation": True,
    }


def build_feishu_progress_card(
    *,
    phase: str,
    progress_percent: int,
    elapsed_seconds: float,
    update_count: int,
    done: bool = False,
) -> dict[str, Any]:
    """Build a polished Feishu interactive card without raw tool details."""
    percent = 100 if done else max(0, min(99, int(progress_percent)))
    title = "Hermes 任务进度"
    status = "已完成" if done else "正在处理"
    template = "green" if done else "blue"
    phase_text = "准备发送答复" if done else (phase or "处理请求")

    return {
        "schema": "2.0",
        "config": {
            "wide_screen_mode": True,
            "update_multi": True,
            "summary": {"content": f"{title}：{status} {percent}%"},
        },
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": [
                {
                    "tag": "column_set",
                    "horizontal_spacing": "8px",
                    "columns": [
                        {
                            "tag": "column",
                            "width": "weighted",
                            "weight": 4,
                            "elements": [
                                {
                                    "tag": "markdown",
                                    "content": f"**{status}**\n当前阶段：{phase_text}",
                                    "text_size": "normal",
                                }
                            ],
                        },
                        {
                            "tag": "column",
                            "width": "auto",
                            "elements": [
                                {
                                    "tag": "markdown",
                                    "content": f"**{percent}%**",
                                    "text_align": "right",
                                    "text_size": "heading",
                                }
                            ],
                        },
                    ],
                },
                {
                    "tag": "chart",
                    "aspect_ratio": "16:2",
                    "chart_spec": _progress_chart(percent, done=done),
                },
                {
                    "tag": "note",
                    "elements": [
                        {
                            "tag": "plain_text",
                            "content": f"已用 {format_elapsed(elapsed_seconds)} · 第 {max(1, update_count)} 次更新",
                        }
                    ],
                },
            ],
        },
    }
