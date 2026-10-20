#!/usr/bin/env python3
"""Transcribe a local interview audio file with Doubao ASR.

The script keeps the source audio untouched. It can normalize and split long
audio into temporary MP3 chunks, submit each chunk to the Doubao recording-file
flash API, checkpoint every successful result, and merge the utterances back
onto the original timeline.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


FLASH_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
DEFAULT_FLASH_RESOURCE_ID = "volc.bigasr.auc_turbo"
SUCCESS_CODES = {"0", "20000000"}
BUSY_CODES = {"55000031"}
DIRECT_EXTENSIONS = {".mp3", ".wav", ".ogg", ".opus"}
DIRECT_MAX_BYTES = 18 * 1024 * 1024
MAX_FLASH_DURATION_SECONDS = 2 * 60 * 60
DEFAULT_CHUNK_SECONDS = 30 * 60
SILENCE_SEARCH_WINDOW_SECONDS = 60
PRIORITY_MAP = {"high": 0, "medium": 10, "low": 20}


class TranscriptionError(RuntimeError):
    """A safe-to-report transcription error."""


class AmbiguousRequestError(TranscriptionError):
    """The request may have reached the server but no definitive result arrived."""


@dataclass
class Chunk:
    index: int
    path: Path
    start_seconds: float
    duration_seconds: float
    source_is_original: bool = False

    @property
    def offset_ms(self) -> int:
        return int(round(self.start_seconds * 1000))

    @property
    def duration_ms(self) -> int:
        return int(round(self.duration_seconds * 1000))


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def require_credentials() -> tuple[str | None, str | None, str | None, str]:
    app_key = env_first("DOUBAO_STT_APP_ID", "VOLC_ASR_APP_KEY", "VOLC_ASR_APP_ID")
    access_key = env_first("DOUBAO_STT_ACCESS_KEY", "VOLC_ASR_ACCESS_KEY")
    api_key = env_first("DOUBAO_STT_API_KEY", "VOLC_ASR_API_KEY")
    resource_id = env_first("DOUBAO_STT_FLASH_RESOURCE_ID", "VOLC_ASR_FLASH_RESOURCE_ID") or DEFAULT_FLASH_RESOURCE_ID
    if not api_key and not (app_key and access_key):
        raise TranscriptionError(
            "Missing Doubao ASR credentials. Configure DOUBAO_STT_API_KEY (new console), "
            "or DOUBAO_STT_APP_ID plus DOUBAO_STT_ACCESS_KEY (old console)."
        )
    return app_key, access_key, api_key, resource_id


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TranscriptionError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"Invalid UTF-8 JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TranscriptionError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_filename(value: str, *, max_len: int = 80) -> str:
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", str(value or "").strip())
    value = re.sub(r"\s+", "_", value).strip(" ._")
    return (value or "访谈音频")[:max_len]


def normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(20\d{2})[-/.年]?(\d{1,2})[-/.月]?(\d{1,2})", text)
    if not match:
        return "日期待确认"
    return f"{int(match.group(1)):04d}{int(match.group(2)):02d}{int(match.group(3)):02d}"


def primary_interviewee(metadata: dict[str, Any]) -> str:
    candidates = metadata.get("interviewees") or metadata.get("受访者") or metadata.get("interviewee")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            return str(first.get("name") or first.get("姓名") or "").strip()
        return str(first).strip()
    if isinstance(candidates, dict):
        return str(candidates.get("name") or candidates.get("姓名") or "").strip()
    return str(candidates or "").strip()


def output_prefix(audio_path: Path, metadata: dict[str, Any]) -> str:
    person = primary_interviewee(metadata)
    date = normalize_date(metadata.get("date") or metadata.get("访谈日期"))
    if person:
        return sanitize_filename(f"{person}_{date}")
    return sanitize_filename(f"{audio_path.stem}_{date}")


def run_command(command: list[str], *, purpose: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-4000:]
        raise TranscriptionError(f"{purpose} failed (exit {result.returncode}).\n{detail}")
    return result


def require_media_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise TranscriptionError("ffmpeg and ffprobe are required for local audio processing.")
    return ffmpeg, ffprobe


def probe_audio(path: Path, ffprobe: str) -> dict[str, Any]:
    result = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size,format_name:stream=index,codec_type,codec_name,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        purpose="Audio inspection",
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("ffprobe returned invalid JSON.") from exc
    streams = data.get("streams") if isinstance(data, dict) else None
    audio_streams = [item for item in streams or [] if isinstance(item, dict) and item.get("codec_type") == "audio"]
    if not audio_streams:
        raise TranscriptionError(f"No audio stream found in: {path}")
    format_data = data.get("format") if isinstance(data.get("format"), dict) else {}
    try:
        duration = float(format_data.get("duration"))
    except (TypeError, ValueError) as exc:
        raise TranscriptionError(f"Could not determine audio duration: {path}") from exc
    if duration <= 0:
        raise TranscriptionError(f"Audio duration must be positive: {path}")
    return {
        "duration_seconds": duration,
        "size_bytes": path.stat().st_size,
        "format_name": format_data.get("format_name", ""),
        "audio_stream": audio_streams[0],
    }


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)


def normalize_term(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" ,，。;；、|/\\\"'`[]【】()（）")
    return text[:80]


def normalize_context(
    raw_context: dict[str, Any],
    metadata: dict[str, Any],
    *,
    max_terms: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    order = 0

    def add_term(term: Any, *, category: str, priority: int, source: str, aliases: Any = None) -> None:
        nonlocal order
        canonical = normalize_term(term)
        if not canonical:
            return
        key = canonical.casefold()
        alias_values = [normalize_term(item) for item in flatten_strings(aliases)] if aliases is not None else []
        alias_values = [item for item in alias_values if item and item.casefold() != key]
        if key in seen:
            for item in candidates:
                if item["canonical"].casefold() == key:
                    item["aliases"] = list(dict.fromkeys(item.get("aliases", []) + alias_values))
                    item["priority"] = min(item["priority"], priority)
                    return
        seen.add(key)
        candidates.append(
            {
                "canonical": canonical,
                "category": category or "术语",
                "aliases": alias_values,
                "priority": priority,
                "source": source,
                "order": order,
            }
        )
        order += 1

    raw_terms = raw_context.get("terms") or raw_context.get("术语") or []
    if isinstance(raw_terms, (str, dict)):
        raw_terms = [raw_terms]
    for item in raw_terms if isinstance(raw_terms, list) else []:
        if isinstance(item, dict):
            priority_value = item.get("priority", "high")
            if isinstance(priority_value, str):
                priority = PRIORITY_MAP.get(priority_value.lower(), 0)
            else:
                try:
                    priority = int(priority_value)
                except (TypeError, ValueError):
                    priority = 0
            add_term(
                item.get("canonical") or item.get("term") or item.get("规范写法"),
                category=str(item.get("category") or item.get("类别") or "术语"),
                priority=priority,
                source=str(item.get("source") or item.get("来源") or "用户上下文"),
                aliases=item.get("aliases") or item.get("别名"),
            )
        else:
            add_term(item, category="术语", priority=0, source="用户上下文")

    corrections_raw = raw_context.get("corrections") or raw_context.get("纠错") or {}
    corrections: dict[str, str] = {}
    if isinstance(corrections_raw, dict):
        for wrong, correct in corrections_raw.items():
            wrong_term = normalize_term(wrong)
            correct_term = normalize_term(correct)
            if wrong_term and correct_term and wrong_term != correct_term:
                corrections[wrong_term] = correct_term
                add_term(correct_term, category="明确纠错", priority=-10, source="用户纠错", aliases=[wrong_term])

    people_keys = ("interviewers", "interviewees", "访谈者", "受访者")
    for key in people_keys:
        people = metadata.get(key)
        if not people:
            continue
        if not isinstance(people, list):
            people = [people]
        for person in people:
            if isinstance(person, dict):
                name = person.get("name") or person.get("姓名")
                organization = person.get("organization") or person.get("unit") or person.get("单位")
                title = person.get("title") or person.get("职位")
                add_term(name, category="人名", priority=5, source="访谈元信息")
                add_term(organization, category="机构", priority=6, source="访谈元信息")
                add_term(title, category="职位", priority=15, source="访谈元信息")
            else:
                add_term(person, category="人物", priority=5, source="访谈元信息")

    metadata_term_fields = {
        "companies": "公司",
        "products": "产品",
        "technologies": "技术",
        "units": "单位口径",
        "context_terms": "补充术语",
        "公司": "公司",
        "产品": "产品",
        "技术": "技术",
        "单位口径": "单位口径",
    }
    for field, category in metadata_term_fields.items():
        for term in flatten_strings(metadata.get(field)):
            add_term(term, category=category, priority=10, source="访谈元信息")

    for field, category in (("industry", "行业"), ("涉及行业", "行业")):
        for term in flatten_strings(metadata.get(field)):
            add_term(term, category=category, priority=15, source="访谈元信息")

    candidates.sort(key=lambda item: (item["priority"], item["order"]))
    selected = candidates[: max(0, max_terms)]
    excluded = candidates[max(0, max_terms) :]
    for item in selected + excluded:
        item.pop("order", None)

    background_values: list[str] = []
    for value in flatten_strings(raw_context.get("background") or raw_context.get("背景") or []):
        normalized = re.sub(r"\s+", " ", value).strip()
        if normalized and normalized not in background_values:
            background_values.append(normalized[:500])
    if not background_values:
        title = str(metadata.get("title") or metadata.get("访谈主题") or "").strip()
        industry = str(metadata.get("industry") or metadata.get("涉及行业") or "").strip()
        if title or industry:
            background_values.append(f"访谈主题：{title}；涉及行业：{industry}。".strip())

    context_data: list[dict[str, str]] = []
    for background in background_values[:3]:
        context_data.append({"text": f"以下内容仅用于语音识别背景提示：{background}"})

    terms: list[str] = []
    term_keys: set[str] = set()
    for item in selected:
        for value in [item["canonical"], *item.get("aliases", [])]:
            key = value.casefold()
            if key not in term_keys:
                term_keys.add(key)
                terms.append(value)
    current: list[str] = []
    current_len = 0
    for term in terms:
        if current and current_len + len(term) + 1 > 450:
            context_data.append({"text": "请优先正确识别这些专有名词、英文缩写和单位口径：" + "、".join(current) + "。"})
            current = []
            current_len = 0
        current.append(term)
        current_len += len(term) + 1
    if current:
        context_data.append({"text": "请优先正确识别这些专有名词、英文缩写和单位口径：" + "、".join(current) + "。"})

    if corrections:
        correction_text = "；".join(f"{wrong}应识别为{correct}" for wrong, correct in corrections.items())
        context_data.append({"text": "明确纠错关系：" + correction_text[:1000] + "。"})

    sources = list(dict.fromkeys(flatten_strings(raw_context.get("sources") or raw_context.get("来源") or [])))
    structured_context: dict[str, Any] = {}
    if terms:
        structured_context["hotwords"] = [{"word": term} for term in terms]
    if corrections:
        structured_context["correct_words"] = corrections
    if context_data:
        structured_context["context_data"] = context_data
    text_context = {"context_data": context_data} if context_data else {}

    return {
        "enabled": bool(structured_context),
        "max_terms": max_terms,
        "selected_terms": selected,
        "excluded_terms": excluded,
        "corrections": corrections,
        "background": background_values,
        "sources": sources,
        "context_data": context_data,
        "structured_context": structured_context,
        "text_context": text_context,
        "requested_mode": "structured" if structured_context else "none",
        "applied_modes": [],
    }


def find_silence_endpoints(audio_path: Path, ffmpeg: str) -> list[float]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(audio_path),
        "-af",
        "silencedetect=noise=-35dB:d=0.5",
        "-f",
        "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", result.stderr)]


def plan_boundaries(
    duration_seconds: float,
    silence_endpoints: list[float],
    *,
    target_seconds: int = DEFAULT_CHUNK_SECONDS,
    search_window_seconds: int = SILENCE_SEARCH_WINDOW_SECONDS,
) -> list[float]:
    if duration_seconds <= target_seconds:
        return [0.0, duration_seconds]
    boundaries = [0.0]
    target = float(target_seconds)
    while target < duration_seconds:
        candidates = [
            value
            for value in silence_endpoints
            if abs(value - target) <= search_window_seconds and value - boundaries[-1] >= 60
        ]
        boundary = min(candidates, key=lambda value: abs(value - target)) if candidates else target
        if duration_seconds - boundary < 60:
            break
        boundaries.append(boundary)
        target = boundary + target_seconds
    boundaries.append(duration_seconds)
    return boundaries


def safe_remove(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise TranscriptionError(f"Refusing to remove path outside output directory: {resolved}")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def chunks_from_state(state: dict[str, Any], source_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for item in state.get("chunks", []):
        if not isinstance(item, dict):
            continue
        stored_path = Path(str(item.get("path") or ""))
        source_is_original = bool(item.get("source_is_original"))
        if source_is_original:
            stored_path = source_path
        if not stored_path.exists():
            return []
        chunks.append(
            Chunk(
                index=int(item["index"]),
                path=stored_path,
                start_seconds=float(item["start_seconds"]),
                duration_seconds=float(item["duration_seconds"]),
                source_is_original=source_is_original,
            )
        )
    return chunks


def prepare_chunks(
    source_path: Path,
    probe: dict[str, Any],
    source_sha256: str,
    work_dir: Path,
    ffmpeg: str,
    *,
    force: bool,
) -> tuple[list[Chunk], dict[str, Any]]:
    state_path = work_dir / "chunks.json"
    if state_path.exists() and not force:
        state = read_json(state_path)
        if state.get("source_sha256") == source_sha256:
            existing = chunks_from_state(state, source_path)
            if existing:
                return existing, state

    work_dir.mkdir(parents=True, exist_ok=True)
    duration = float(probe["duration_seconds"])
    direct = (
        source_path.suffix.lower() in DIRECT_EXTENSIONS
        and duration <= MAX_FLASH_DURATION_SECONDS
        and int(probe["size_bytes"]) <= DIRECT_MAX_BYTES
    )
    chunks: list[Chunk] = []
    normalization = "direct"
    silence_endpoints: list[float] = []
    if direct:
        chunks.append(Chunk(1, source_path, 0.0, duration, True))
    else:
        normalization = "16kHz mono 64kbps MP3"
        if duration > DEFAULT_CHUNK_SECONDS:
            silence_endpoints = find_silence_endpoints(source_path, ffmpeg)
        boundaries = plan_boundaries(duration, silence_endpoints)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            output = work_dir / f"chunk_{index:03d}.mp3"
            if not output.exists() or output.stat().st_size == 0 or force:
                run_command(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{start:.3f}",
                        "-t",
                        f"{end - start:.3f}",
                        "-i",
                        str(source_path),
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "libmp3lame",
                        "-b:a",
                        "64k",
                        str(output),
                    ],
                    purpose=f"Preparing audio chunk {index}",
                )
            chunks.append(Chunk(index, output, start, end - start, False))

    state = {
        "source_sha256": source_sha256,
        "normalization": normalization,
        "silence_endpoints_found": len(silence_endpoints),
        "chunks": [
            {
                "index": chunk.index,
                "path": str(chunk.path),
                "start_seconds": chunk.start_seconds,
                "duration_seconds": chunk.duration_seconds,
                "source_is_original": chunk.source_is_original,
                "size_bytes": chunk.path.stat().st_size,
                "sha256": sha256_file(chunk.path),
            }
            for chunk in chunks
        ],
    }
    write_json(state_path, state)
    return chunks, state


def api_headers(
    app_key: str | None,
    access_key: str | None,
    api_key: str | None,
    resource_id: str,
    request_id: str,
) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": request_id,
        "X-Api-Sequence": "-1",
    }
    if app_key and access_key:
        headers["X-Api-App-Key"] = app_key
        headers["X-Api-Access-Key"] = access_key
    if api_key:
        headers["X-Api-Key"] = api_key
    return headers


def request_bytes(
    url: str,
    *,
    data: bytes,
    headers: dict[str, str],
    timeout_seconds: int,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            try:
                raw = response.read()
            except http.client.IncompleteRead as exc:
                raise AmbiguousRequestError(
                    "The ASR response ended unexpectedly. The request may already have been processed."
                ) from exc
            return response.status, dict(response.headers.items()), raw
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()), exc.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise AmbiguousRequestError(
            f"The ASR connection ended without a definitive response: {type(exc).__name__}: {exc}"
        ) from exc


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
) -> tuple[int, dict[str, str], Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    status, response_headers, raw = request_bytes(
        url,
        data=data,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    text = raw.decode("utf-8", errors="replace")
    try:
        body: Any = json.loads(text) if text else {}
    except json.JSONDecodeError:
        body = {"raw_text": text[:4000]}
    return status, response_headers, body


def status_from(headers: dict[str, str], body: Any) -> tuple[str | None, str, str | None]:
    lowered = {key.lower(): value for key, value in headers.items()}
    code = lowered.get("x-api-status-code")
    message = lowered.get("x-api-message") or ""
    log_id = lowered.get("x-tt-logid")
    if isinstance(body, dict):
        header = body.get("header") if isinstance(body.get("header"), dict) else {}
        code = code or str(body.get("code") or body.get("status_code") or header.get("code") or "")
        message = message or str(body.get("message") or body.get("msg") or header.get("message") or "")
    return code or None, message, log_id


def has_transcript(body: Any) -> bool:
    result = find_result(body)
    return bool(result.get("text") or result.get("utterances") or result.get("utterance"))


def is_context_rejection(code: str | None, message: str, body: Any) -> bool:
    detail = f"{message} {json.dumps(body, ensure_ascii=False)[:1500]}".lower()
    markers = ("corpus", "context", "hotword", "correct_word", "unsupported", "invalid parameter", "invalid param")
    return any(marker in detail for marker in markers) or (code == "45000001" and not message)


def context_modes(context: dict[str, Any], disabled: bool) -> list[tuple[str, dict[str, Any] | None]]:
    if disabled or not context.get("enabled"):
        return [("none", None)]
    modes: list[tuple[str, dict[str, Any] | None]] = []
    structured = context.get("structured_context")
    text_only = context.get("text_context")
    if isinstance(structured, dict) and structured:
        modes.append(("structured", structured))
    if isinstance(text_only, dict) and text_only and text_only != structured:
        modes.append(("context_data", text_only))
    modes.append(("none", None))
    return modes


def build_payload(audio_path: Path, context_value: dict[str, Any] | None) -> dict[str, Any]:
    audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    request_data: dict[str, Any] = {
        "model_name": "bigmodel",
        "enable_itn": True,
        "enable_punc": True,
        "enable_ddc": True,
        "show_utterances": True,
        "enable_speaker_info": True,
    }
    if context_value:
        request_data["corpus"] = {"context": json.dumps(context_value, ensure_ascii=False)}
    return {
        "user": {"uid": "codex-local-user"},
        "audio": {"data": audio_data, "format": audio_path.suffix.lower().lstrip(".")},
        "request": request_data,
    }


def recognize_chunk(
    chunk: Chunk,
    *,
    app_key: str | None,
    access_key: str | None,
    api_key: str | None,
    resource_id: str,
    context: dict[str, Any],
    no_asr_context: bool,
    retry_ambiguous: bool,
    timeout_seconds: int,
    record_attempt: Callable[[dict[str, Any]], None],
) -> tuple[Any, str, str]:
    modes = context_modes(context, no_asr_context)
    for mode_name, context_value in modes:
        busy_retries = 0
        ambiguous_retries = 0
        while True:
            request_id = str(uuid.uuid4())
            attempt = {
                "request_id": request_id,
                "context_mode": mode_name,
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                "status": "submitting",
            }
            record_attempt(attempt)
            try:
                headers = api_headers(app_key, access_key, api_key, resource_id, request_id)
                payload = build_payload(chunk.path, context_value)
                http_status, response_headers, body = post_json(
                    FLASH_URL,
                    headers,
                    payload,
                    timeout_seconds=timeout_seconds,
                )
            except AmbiguousRequestError as exc:
                attempt.update(
                    {
                        "status": "ambiguous",
                        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "error": str(exc),
                    }
                )
                record_attempt(attempt)
                if retry_ambiguous and ambiguous_retries < 1:
                    ambiguous_retries += 1
                    continue
                raise AmbiguousRequestError(
                    f"Chunk {chunk.index} has an ambiguous ASR request (request_id={request_id}). "
                    "Re-run with --retry-ambiguous only if repeating this possibly billable request is acceptable."
                ) from exc

            code, message, log_id = status_from(response_headers, body)
            attempt.update(
                {
                    "http_status": http_status,
                    "api_status": code,
                    "message": message,
                    "log_id": log_id,
                    "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
            if http_status < 400 and (not code or code in SUCCESS_CODES) and has_transcript(body):
                attempt["status"] = "success"
                record_attempt(attempt)
                return body, mode_name, request_id

            if http_status < 400 and (not code or code in SUCCESS_CODES) and not has_transcript(body):
                attempt["status"] = "failed"
                attempt["message"] = "ASR returned success without transcript text or utterances"
                record_attempt(attempt)
                raise TranscriptionError(
                    f"ASR returned success without transcript content for chunk {chunk.index}. "
                    "Keep the raw response and check whether the API response shape changed."
                )

            if code in BUSY_CODES and busy_retries < 3:
                attempt["status"] = "busy_retry"
                record_attempt(attempt)
                delay = (5, 15, 30)[busy_retries]
                busy_retries += 1
                time.sleep(delay)
                continue

            if context_value and is_context_rejection(code, message, body):
                attempt["status"] = "context_rejected"
                record_attempt(attempt)
                break

            attempt["status"] = "failed"
            record_attempt(attempt)
            safe_body = json.dumps(body, ensure_ascii=False)[:3000]
            hint = ""
            if "requested resource not granted" in f"{message} {safe_body}".lower() or code == "45000030":
                hint = (
                    " Activate Doubao recording-file flash recognition and use resource ID "
                    f"{DEFAULT_FLASH_RESOURCE_ID}."
                )
            raise TranscriptionError(
                f"ASR failed for chunk {chunk.index}. HTTP={http_status}; API={code or 'unknown'} {message}."
                f"{hint}\nResponse: {safe_body}"
            )
    raise TranscriptionError(f"ASR context fallback exhausted for chunk {chunk.index}.")


def find_result(body: Any) -> dict[str, Any]:
    if isinstance(body, dict):
        for key in ("result", "data"):
            if isinstance(body.get(key), dict):
                return body[key]
        return body
    return {}


def normalize_utterances(result: dict[str, Any]) -> list[dict[str, Any]]:
    utterances = result.get("utterances") or result.get("utterance") or []
    if isinstance(utterances, dict):
        utterances = utterances.get("items") or utterances.get("utterances") or []
    return [item for item in utterances if isinstance(item, dict)] if isinstance(utterances, list) else []


def number_ms(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0


def timestamp_ms(milliseconds: int) -> str:
    total_seconds = max(0, int(round(milliseconds / 1000)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def speaker_value(item: dict[str, Any]) -> str:
    raw: Any = None
    for key in ("speaker", "speaker_id", "spk", "role"):
        if key in item and item.get(key) not in (None, ""):
            raw = item.get(key)
            break
    additions = item.get("additions")
    if raw in (None, "") and isinstance(additions, dict):
        raw = additions.get("speaker")
    return str(raw) if raw not in (None, "") else "不明"


def apply_corrections(text: str, corrections: dict[str, str], counts: dict[str, int]) -> str:
    result = text
    for wrong in sorted(corrections, key=len, reverse=True):
        count = result.count(wrong)
        if count:
            result = result.replace(wrong, corrections[wrong])
            counts[wrong] = counts.get(wrong, 0) + count
    return result


def result_duration_ms(body: Any) -> int:
    if isinstance(body, dict):
        audio_info = body.get("audio_info") if isinstance(body.get("audio_info"), dict) else {}
        value = audio_info.get("duration")
        if value not in (None, ""):
            return number_ms(value)
    result = find_result(body)
    additions = result.get("additions") if isinstance(result.get("additions"), dict) else {}
    return number_ms(additions.get("duration"))


def merge_results(
    wrappers: list[dict[str, Any]],
    corrections: dict[str, str],
) -> tuple[list[dict[str, Any]], str, dict[str, int], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    utterance_correction_counts: dict[str, int] = {}
    full_text_correction_counts: dict[str, int] = {}
    duration_checks: list[dict[str, Any]] = []
    multiple_chunks = len(wrappers) > 1
    for wrapper in wrappers:
        chunk_index = int(wrapper["chunk_index"])
        offset_ms = int(wrapper["offset_ms"])
        expected_duration_ms = int(wrapper["duration_ms"])
        body = wrapper["response"]
        result = find_result(body)
        recognized_duration_ms = result_duration_ms(body)
        utterances = normalize_utterances(result)
        for item in utterances:
            text = apply_corrections(
                str(item.get("text") or item.get("utterance") or "").strip(),
                corrections,
                utterance_correction_counts,
            )
            if not text:
                continue
            start = offset_ms + number_ms(item.get("start_time", item.get("start", item.get("begin_time", 0))))
            end = offset_ms + number_ms(item.get("end_time", item.get("end", item.get("stop_time", 0))))
            speaker = speaker_value(item)
            label = f"片段{chunk_index:02d}-说话人{speaker}" if multiple_chunks else f"说话人{speaker}"
            merged.append({"start_ms": start, "end_ms": end, "speaker": label, "text": text, "chunk_index": chunk_index})
        text = apply_corrections(
            str(result.get("text") or "").strip(),
            corrections,
            full_text_correction_counts,
        )
        if text:
            full_text_parts.append(text)
        duration_checks.append(
            {
                "chunk_index": chunk_index,
                "expected_duration_ms": expected_duration_ms,
                "recognized_duration_ms": recognized_duration_ms,
                "difference_ms": abs(expected_duration_ms - recognized_duration_ms) if recognized_duration_ms else None,
            }
        )
    merged.sort(key=lambda item: (item["start_ms"], item["end_ms"]))
    final_end_ms = max((item["end_ms"] for item in merged), default=0)
    expected_total_ms = max((int(item["offset_ms"]) + int(item["duration_ms"]) for item in wrappers), default=0)
    validation = {
        "duration_checks": duration_checks,
        "expected_total_ms": expected_total_ms,
        "last_utterance_end_ms": final_end_ms,
        "tail_gap_ms": max(0, expected_total_ms - final_end_ms),
    }
    correction_counts = full_text_correction_counts if full_text_parts else utterance_correction_counts
    return merged, "\n".join(full_text_parts), correction_counts, validation


def export_transcript(
    output: Path,
    *,
    audio_path: Path,
    source_sha256: str,
    metadata: dict[str, Any],
    resource_id: str,
    wrappers: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    full_text: str,
    correction_counts: dict[str, int],
    validation: dict[str, Any],
) -> None:
    lines = [
        "# 转写原稿",
        "",
        f"**来源音频**：{audio_path}",
        f"**源文件 SHA-256**：{source_sha256}",
        f"**访谈主题**：{metadata.get('title') or metadata.get('访谈主题') or ''}",
        f"**访谈日期**：{metadata.get('date') or metadata.get('访谈日期') or ''}",
        f"**涉及行业**：{metadata.get('industry') or metadata.get('涉及行业') or ''}",
        f"**识别模型资源**：{resource_id}",
        f"**音频片段数**：{len(wrappers)}",
        f"**最后有效时间戳**：{timestamp_ms(int(validation['last_utterance_end_ms']))}",
        f"**导出时间**：{dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "> 本文件的自动纠错仅来自用户明确提供的纠错映射。背景资料只用于提升识别准确率，不是访谈内容来源。",
        "",
        "---",
        "",
        "## 分段转写",
        "",
    ]
    for item in utterances:
        lines.append(
            f"[{timestamp_ms(int(item['start_ms']))}-{timestamp_ms(int(item['end_ms']))}] "
            f"{item['speaker']}：{item['text']}"
        )
        lines.append("")
    if full_text:
        lines.extend(["## 完整文本", "", full_text, ""])
    lines.extend(["## 自动纠错记录", ""])
    if correction_counts:
        for wrong, count in correction_counts.items():
            lines.append(f"- `{wrong}`：替换 {count} 处")
    else:
        lines.append("- 未应用自动纠错。")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def process(args: argparse.Namespace) -> int:
    source_path = Path(args.audio_file).expanduser().resolve()
    if not source_path.is_file():
        raise TranscriptionError(f"Audio file not found: {source_path}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_json(Path(args.metadata_file).expanduser().resolve()) if args.metadata_file else {}
    raw_context = read_json(Path(args.context_file).expanduser().resolve()) if args.context_file else {}
    prefix = output_prefix(source_path, metadata)
    paths = {
        "audio_metadata": out_dir / f"音频元信息_{prefix}.json",
        "context": out_dir / f"ASR上下文_{prefix}.json",
        "task": out_dir / f"豆包ASR任务_{prefix}.json",
        "raw": out_dir / f"豆包ASR原始结果_{prefix}.json",
        "transcript": out_dir / f"转写原稿_{prefix}.md",
        "manifest": out_dir / f"处理清单_{prefix}.json",
        "work_dir": out_dir / f".audio_to_transcript_work_{prefix}",
    }

    if args.force:
        for key, path in paths.items():
            if path.exists():
                safe_remove(path, out_dir)

    ffmpeg, ffprobe = require_media_tools()
    probe = probe_audio(source_path, ffprobe)
    source_sha256 = sha256_file(source_path)
    audio_metadata = {
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_size_bytes": source_path.stat().st_size,
        "source_preserved": True,
        "probe": probe,
        "interview_metadata": metadata,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    write_json(paths["audio_metadata"], audio_metadata)

    context = normalize_context(raw_context, metadata, max_terms=max(0, args.max_context_terms))
    context_for_disk = {key: value for key, value in context.items() if key not in {"structured_context", "text_context"}}
    context_for_disk["request_variants"] = {
        "structured": context.get("structured_context", {}),
        "context_data": context.get("text_context", {}),
    }
    write_json(paths["context"], context_for_disk)

    try:
        chunks, chunk_state = prepare_chunks(
            source_path,
            probe,
            source_sha256,
            paths["work_dir"],
            ffmpeg,
            force=args.force,
        )

        task_state: dict[str, Any]
        if paths["task"].exists() and not args.force:
            task_state = read_json(paths["task"])
            if task_state.get("source_sha256") != source_sha256:
                raise TranscriptionError(
                    "Existing task state belongs to a different source audio. Use --force to rebuild derived artifacts."
                )
        else:
            task_state = {
                "source_sha256": source_sha256,
                "resource_id": None,
                "chunks": {},
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            }

        app_key, access_key, api_key, resource_id = require_credentials()
        task_state["resource_id"] = resource_id
        write_json(paths["task"], task_state)
    except Exception as exc:
        status = "ambiguous" if isinstance(exc, AmbiguousRequestError) else "failed"
        write_json(
            paths["manifest"],
            {
                "status": status,
                "source_audio": str(source_path),
                "source_sha256": source_sha256,
                "source_preserved": source_path.exists(),
                "error": str(exc),
                "temporary_files_retained": paths["work_dir"].exists(),
                "paths": {key: str(value) for key, value in paths.items()},
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
        raise

    wrappers: list[dict[str, Any]] = []
    try:
        for chunk in chunks:
            key = str(chunk.index)
            checkpoint = paths["work_dir"] / f"raw_chunk_{chunk.index:03d}.json"
            chunk_task = task_state["chunks"].setdefault(key, {"attempts": []})
            if chunk_task.get("status") == "ambiguous" and not args.retry_ambiguous:
                request_id = chunk_task.get("last_request_id") or "unknown"
                raise AmbiguousRequestError(
                    f"Chunk {chunk.index} previously ended ambiguously (request_id={request_id}). "
                    "Use --retry-ambiguous to repeat it."
                )
            if checkpoint.exists() and chunk_task.get("status") == "success":
                wrapper = read_json(checkpoint)
                wrappers.append(wrapper)
                print(f"Reused completed chunk {chunk.index}/{len(chunks)}")
                continue

            def record_attempt(attempt: dict[str, Any], *, chunk_key: str = key) -> None:
                current = task_state["chunks"].setdefault(chunk_key, {"attempts": []})
                attempts = current.setdefault("attempts", [])
                request_id = attempt.get("request_id")
                replaced = False
                for index, existing in enumerate(attempts):
                    if existing.get("request_id") == request_id:
                        attempts[index] = dict(attempt)
                        replaced = True
                        break
                if not replaced:
                    attempts.append(dict(attempt))
                current["status"] = attempt.get("status")
                current["last_request_id"] = request_id
                write_json(paths["task"], task_state)

            print(f"Transcribing chunk {chunk.index}/{len(chunks)}")
            body, applied_mode, request_id = recognize_chunk(
                chunk,
                app_key=app_key,
                access_key=access_key,
                api_key=api_key,
                resource_id=resource_id,
                context=context,
                no_asr_context=args.no_asr_context,
                retry_ambiguous=args.retry_ambiguous,
                timeout_seconds=args.timeout_seconds,
                record_attempt=record_attempt,
            )
            wrapper = {
                "chunk_index": chunk.index,
                "chunk_sha256": sha256_file(chunk.path),
                "offset_ms": chunk.offset_ms,
                "duration_ms": chunk.duration_ms,
                "request_id": request_id,
                "resource_id": resource_id,
                "context_mode": applied_mode,
                "response": body,
            }
            write_json(checkpoint, wrapper)
            chunk_task = task_state["chunks"].setdefault(key, {"attempts": []})
            chunk_task.update({"status": "success", "request_id": request_id, "context_mode": applied_mode})
            write_json(paths["task"], task_state)
            wrappers.append(wrapper)

        wrappers.sort(key=lambda item: int(item["chunk_index"]))
        applied_modes = [str(item.get("context_mode") or "none") for item in wrappers]
        context["applied_modes"] = applied_modes
        context_for_disk["applied_modes"] = applied_modes
        write_json(paths["context"], context_for_disk)
        write_json(
            paths["raw"],
            {
                "source_audio": str(source_path),
                "source_sha256": source_sha256,
                "resource_id": resource_id,
                "chunks": wrappers,
            },
        )
        utterances, full_text, correction_counts, validation = merge_results(
            wrappers,
            context.get("corrections", {}),
        )
        export_transcript(
            paths["transcript"],
            audio_path=source_path,
            source_sha256=source_sha256,
            metadata=metadata,
            resource_id=resource_id,
            wrappers=wrappers,
            utterances=utterances,
            full_text=full_text,
            correction_counts=correction_counts,
            validation=validation,
        )

        temp_deleted = False
        if not args.keep_temp and paths["work_dir"].exists():
            safe_remove(paths["work_dir"], out_dir)
            temp_deleted = True
        manifest = {
            "status": "transcribed",
            "source_audio": str(source_path),
            "source_sha256": source_sha256,
            "source_preserved": source_path.exists() and sha256_file(source_path) == source_sha256,
            "chunk_count": len(chunks),
            "normalization": chunk_state.get("normalization"),
            "context_modes": applied_modes,
            "correction_counts": correction_counts,
            "validation": validation,
            "temporary_files_deleted": temp_deleted,
            "paths": {key: str(value) for key, value in paths.items()},
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        write_json(paths["manifest"], manifest)
        print(f"Wrote transcript: {paths['transcript']}")
        print(f"Wrote manifest: {paths['manifest']}")
        return 0
    except Exception as exc:
        status = "ambiguous" if isinstance(exc, AmbiguousRequestError) else "failed"
        write_json(
            paths["manifest"],
            {
                "status": status,
                "source_audio": str(source_path),
                "source_sha256": source_sha256,
                "source_preserved": source_path.exists(),
                "error": str(exc),
                "temporary_files_retained": paths["work_dir"].exists(),
                "paths": {key: str(value) for key, value in paths.items()},
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe a local interview audio file with Doubao ASR and preserve resumable artifacts."
    )
    parser.add_argument("audio_file", help="Local audio file. The source file is never deleted or overwritten.")
    parser.add_argument("--out-dir", default=str(Path.cwd() / "访谈纪要"), help="Exact output directory for this interview.")
    parser.add_argument("--metadata-file", help="Optional UTF-8 JSON file with interview metadata and context terms.")
    parser.add_argument("--context-file", help="Optional UTF-8 JSON file with background, terms, aliases, and corrections.")
    parser.add_argument("--max-context-terms", type=int, default=80, help="Maximum selected ASR terms. Default: 80.")
    parser.add_argument("--timeout-seconds", type=int, default=1800, help="Timeout for one flash ASR request. Default: 1800.")
    parser.add_argument("--force", action="store_true", help="Rebuild derived artifacts for this exact audio; never delete the source audio.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep normalized chunks and checkpoints after success.")
    parser.add_argument("--retry-ambiguous", action="store_true", help="Retry a chunk whose previous request may already have been billed.")
    parser.add_argument("--no-asr-context", action="store_true", help="Do not send background, hotwords, or corrections to ASR.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return process(args)
    except TranscriptionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
