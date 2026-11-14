import json
import sys
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from llm_tells import regen_fix
from llm_tells.generate import LMS, seed_for

REAL_CLIENT = httpx.Client
ENGLISH = "A plain English first attempt that passes the new rule."
SECOND = "A plain English second attempt kept from the original run."
BEFORE = "2026-11-07T12:00:00"


def record(prompt_id: str, **fields: Any) -> dict[str, Any]:
    return {
        "text_id": f"{prompt_id}|gemma",
        "prompt_id": prompt_id,
        "source": "gemma",
        "generated_at": "2026-11-07T09:00:00-0500",
        "attempt": 0,
        "failed": False,
        "raw": SECOND,
        "seed": 1,
        **fields,
    }


@pytest.fixture
def setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    commands: list[list[str]] = []
    bodies: list[dict[str, Any]] = []

    def run(args: list[Any], **kwargs: Any) -> None:
        commands.append([str(arg) for arg in args])

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        prompt = body["messages"][0]["content"]
        content = "Это ответ не на английском языке." if "bad" in prompt else ENGLISH
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": content}, "finish_reason": "length"}
                ]
            },
        )

    def factory(**kwargs: Any) -> httpx.Client:
        return REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(regen_fix.subprocess, "run", run)
    monkeypatch.setattr(httpx, "Client", factory)
    pl.DataFrame(
        {
            "prompt_id": [f"hn:{i}" for i in range(6)],
            "prompt": ["good", "bad", "bad", "good", "good", "good"],
            "n_words": [40] * 6,
        }
    ).write_parquet(tmp_path / "prompts.parquet")
    return commands, bodies


def run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = [
        "regen_fix",
        str(tmp_path / "prompts.parquet"),
        str(tmp_path / "gen.jsonl"),
        str(tmp_path / "patch.jsonl"),
        "--model",
        "gemma",
        "--before",
        BEFORE,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    regen_fix.main()


def write_records(tmp_path: Path, records: list[dict[str, Any]]) -> None:
    (tmp_path / "gen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_restores_passing_first_attempts_and_keeps_the_rest(
    setup: tuple[list[list[str]], list[dict[str, Any]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    commands, bodies = setup
    write_records(
        tmp_path,
        [
            record("hn:0", attempt=1),
            record("hn:1", attempt=1),
            record("hn:2", failed=True, raw=""),
            record("hn:3"),
            record("hn:4", attempt=1, generated_at="2026-11-08T09:00:00-0500"),
            {**record("hn:5", attempt=1), "source": "qwen", "text_id": "hn:5|qwen"},
        ],
    )
    run_main(tmp_path, monkeypatch)
    patch = [
        json.loads(line) for line in (tmp_path / "patch.jsonl").read_text().splitlines()
    ]
    by_id = {p["text_id"]: p for p in patch}
    assert list(by_id) == ["hn:0|gemma", "hn:1|gemma", "hn:2|gemma"]
    restored = by_id["hn:0|gemma"]
    assert restored["d6"] == "first attempt restored"
    assert restored["raw"] == ENGLISH
    assert restored["attempt"] == 0
    assert restored["failed"] is False
    assert restored["truncated"] is True
    assert restored["seed"] == seed_for("hn:0", "gemma", 0)
    kept = by_id["hn:1|gemma"]
    assert kept["d6"] == "kept second attempt"
    assert (kept["raw"], kept["attempt"], kept["failed"]) == (SECOND, 1, False)
    still_failed = by_id["hn:2|gemma"]
    assert (still_failed["d6"], still_failed["failed"]) == ("kept second attempt", True)
    assert sorted(b["seed"] for b in bodies) == sorted(
        seed_for(f"hn:{i}", "gemma", 0) for i in range(3)
    )
    assert commands[0][:3] == [str(LMS), "load", "google/gemma-4-e4b"]
    assert commands[-1] == [str(LMS), "unload", "gemma"]
    assert "3 targets; 1 first attempts restored; 1 still failed" in (
        capsys.readouterr().out
    )


def test_nothing_to_fix_leaves_no_patch(
    setup: tuple[list[list[str]], list[dict[str, Any]]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    commands, bodies = setup
    write_records(tmp_path, [record("hn:3")])
    run_main(tmp_path, monkeypatch)
    assert capsys.readouterr().out.strip() == "nothing to fix"
    assert not (tmp_path / "patch.jsonl").exists()
    assert commands == []
    assert bodies == []
