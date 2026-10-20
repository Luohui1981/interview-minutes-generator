#!/usr/bin/env python3
"""Check an interview-minutes coverage matrix and flag abnormal brevity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ALLOWED_FINAL_STATUSES = {"covered", "excluded"}
ALLOWED_EXCLUDED_REASONS = {
    "寒暄",
    "纯应和",
    "流程话术",
    "没有新增信息的重复表述",
}


class CoverageError(RuntimeError):
    pass


def read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CoverageError(f"File not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise CoverageError(f"File is not valid UTF-8: {path}") from exc


def read_matrix(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_utf8(path))
    except json.JSONDecodeError as exc:
        raise CoverageError(f"Invalid coverage matrix JSON: {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("content_units"), list):
        raise CoverageError("Coverage matrix must contain a content_units array.")
    return value


def count_content_characters(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))


def transcript_answer_text(markdown: str) -> str:
    answer_parts: list[str] = []
    current_is_answer = False
    saw_speaker_labels = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if re.match(r"^访谈者(?:[^：:]*)[：:]", line):
            current_is_answer = False
            saw_speaker_labels = True
            continue
        match = re.match(r"^受访者(?:[^：:]*)[：:]\s*(.*)$", line)
        if match:
            current_is_answer = True
            saw_speaker_labels = True
            if match.group(1):
                answer_parts.append(match.group(1))
            continue
        if current_is_answer and line and not line.startswith("#") and not line.startswith("**") and line != "---":
            answer_parts.append(line)
    if saw_speaker_labels:
        return "\n".join(answer_parts)
    body = markdown.split("---", 1)[-1]
    return "\n".join(
        line for line in body.splitlines() if line.strip() and not line.lstrip().startswith(("#", "**"))
    )


def minutes_body_text(markdown: str) -> str:
    body = markdown.split("---", 1)[-1]
    parts: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or re.match(r"^\*\*问题\d+\*\*", line):
            continue
        if re.match(r"^【受访者】", line):
            continue
        parts.append(line)
    return "\n".join(parts)


def validate_matrix(matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    units = matrix["content_units"]
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            errors.append(f"content_units[{index}] is not an object")
            continue
        unit_id = str(unit.get("id") or "").strip()
        label = unit_id or f"content_units[{index}]"
        if not unit_id:
            errors.append(f"{label}: missing id")
        elif unit_id in seen_ids:
            errors.append(f"{label}: duplicate id")
        seen_ids.add(unit_id)
        if not unit.get("source_ranges"):
            errors.append(f"{label}: missing source_ranges")
        status = str(unit.get("status") or "").strip()
        if status not in ALLOWED_FINAL_STATUSES:
            errors.append(f"{label}: unresolved status {status or '<empty>'}")
            continue
        if status == "covered":
            if not unit.get("question_id"):
                errors.append(f"{label}: covered unit missing question_id")
            if not unit.get("minutes_location"):
                errors.append(f"{label}: covered unit missing minutes_location")
        if status == "excluded":
            reason = str(unit.get("excluded_reason") or "").strip()
            if reason not in ALLOWED_EXCLUDED_REASONS:
                errors.append(f"{label}: invalid excluded_reason {reason or '<empty>'}")
    if not units:
        errors.append("coverage matrix has no content units")
    return errors


def run_check(transcript: Path, minutes: Path, matrix_path: Path, threshold: float) -> tuple[int, dict[str, Any]]:
    transcript_text = read_utf8(transcript)
    minutes_text = read_utf8(minutes)
    matrix = read_matrix(matrix_path)
    matrix_errors = validate_matrix(matrix)
    transcript_chars = count_content_characters(transcript_answer_text(transcript_text))
    minutes_chars = count_content_characters(minutes_body_text(minutes_text))
    ratio = (minutes_chars / transcript_chars) if transcript_chars else None
    brevity_warning = ratio is not None and ratio < threshold
    report = {
        "matrix_valid": not matrix_errors,
        "matrix_errors": matrix_errors,
        "transcript_answer_characters": transcript_chars,
        "minutes_body_characters": minutes_chars,
        "minutes_to_transcript_ratio": round(ratio, 4) if ratio is not None else None,
        "brevity_threshold": threshold,
        "brevity_warning": brevity_warning,
        "interpretation": (
            "Run a second coverage review; do not mechanically expand text."
            if brevity_warning
            else "No abnormal brevity signal."
        ),
    }
    if matrix_errors:
        return 2, report
    if brevity_warning:
        return 1, report
    return 0, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a completed interview coverage matrix and flag an abnormally concise memo."
    )
    parser.add_argument("--transcript", required=True, help="访谈实录 Markdown path")
    parser.add_argument("--minutes", required=True, help="访谈纪要 Markdown path")
    parser.add_argument("--matrix", required=True, help="纪要覆盖矩阵 JSON path")
    parser.add_argument("--brevity-threshold", type=float, default=0.5, help="Second-review warning threshold. Default: 0.5")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.brevity_threshold <= 1:
        print("ERROR: --brevity-threshold must be between 0 and 1", file=sys.stderr)
        return 2
    try:
        code, report = run_check(
            Path(args.transcript).expanduser().resolve(),
            Path(args.minutes).expanduser().resolve(),
            Path(args.matrix).expanduser().resolve(),
            args.brevity_threshold,
        )
    except CoverageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Matrix valid: {report['matrix_valid']}")
        for error in report["matrix_errors"]:
            print(f"- {error}")
        print(f"Transcript answer characters: {report['transcript_answer_characters']}")
        print(f"Minutes body characters: {report['minutes_body_characters']}")
        print(f"Minutes/transcript ratio: {report['minutes_to_transcript_ratio']}")
        if report["brevity_warning"]:
            print("WARNING: Run a second coverage review; do not mechanically expand text.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
