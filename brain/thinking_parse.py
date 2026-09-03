# -*- coding: utf-8 -*-
"""五段式回答的解析器。**只依赖标准库**——本仓库的 scripts/thinking_ingest.py
和跑在另一套环境里的采集守护都要用它,不能引入任何第三方依赖。

采集端的提示词固定要求模型输出这五段:
  ## 任务类型 / ## 拆解步骤 / ## 工具计划 / ## 坑与假设 / ## 验收标准
工具计划行格式:`步骤N | 工具 | 参数`
"""
import re

_SECTION_RE = re.compile(r"^#{1,4}\s*(.+?)\s*$", re.M)
_STEP_RE = re.compile(r"^\s*[-*]?\s*(\d+)[.、]\s*(.+?)\s*$", re.M)
# 容忍全角竖线和 markdown 表格包裹
_TOOL_RE = re.compile(
    r"步骤\s*(\d+)\s*[|｜]\s*([A-Za-z_]+)\s*[|｜]\s*(.+?)\s*$", re.M)
KNOWN_TOOLS = {"read_file", "list_dir", "grep", "bash", "git",
               "web_search", "run_python", "ask_user"}


def split_sections(text):
    out, marks = {}, list(_SECTION_RE.finditer(text or ""))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip()] = text[m.end():end].strip()
    return out


def pick(sections, *names):
    for n in names:
        for k, v in sections.items():
            if n in k:
                return v
    return ""


def bullets(body, limit=8):
    """一段正文 → 条目列表:优先 markdown 列表项,否则按非空行。"""
    items = []
    for line in (body or "").splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        s = re.sub(r"^[-*+]\s*", "", s)
        s = re.sub(r"^\d+[.、]\s*", "", s)
        s = s.strip("| ").strip()
        if s:
            items.append(s)
        if len(items) >= limit:
            break
    return items


def parse_answer(answer):
    """→ (task_type, steps, tool_plan, pitfalls, acceptance)"""
    sec = split_sections(answer)
    task_type = " ".join(bullets(pick(sec, "任务类型"), 1)) or None

    steps = []
    for m in _STEP_RE.finditer(pick(sec, "拆解步骤", "步骤")):
        raw = m.group(2).strip()
        action, _, why = raw.partition("——")
        if not why:
            action, _, why = raw.partition("--")
        steps.append({"n": int(m.group(1)), "action": action.strip(" |"),
                      "why": why.strip()})

    tool_plan = []
    for m in _TOOL_RE.finditer(answer or ""):
        tool = m.group(2).strip().lower()
        tool_plan.append({"step": int(m.group(1)), "tool": tool,
                          "args": m.group(3).strip(" |`"),
                          "known": tool in KNOWN_TOOLS})

    return (task_type, steps, tool_plan,
            bullets(pick(sec, "坑与假设", "坑"), 6),
            bullets(pick(sec, "验收标准", "验收"), 5))
