import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from llm_tells import counters
from llm_tells import measure as measure_module
from llm_tells.measure import clean_human, texts_table
from llm_tells.text import Block, CleanText


def prompt_table(**extra: list[Any]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prompt_id": ["p1", "p2"],
            "register": ["hn", "se"],
            "stratum": ["a", "b"],
            "n_words": [40, 60],
            "human_raw": ["Human one.<p>Rather than this.", "**Human** two."],
            "unused": ["x", "y"],
            **extra,
        }
    )


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def record(
    text_id: str, prompt_id: str, raw: str | None, **extra: Any
) -> dict[str, Any]:
    return {
        "text_id": text_id,
        "prompt_id": prompt_id,
        "source": text_id.split("|")[1],
        "raw": raw,
        "failed": False,
        "finish_reason": "stop",
        **extra,
    }


def test_clean_human_hn_uses_html_rules():
    clean = clean_human("hn", "One &amp; two.<p>&gt; quoted bit<p>Three.")
    assert clean.blocks == (Block("para", "One & two."), Block("para", "Three."))
    assert clean.quoted_words == 2


def test_clean_human_se_unescapes_then_parses_markdown():
    clean = clean_human("se", "# Head\n\nUse &lt;b&gt; tags &amp; **bold**.")
    assert clean.blocks == (Block("heading", "Head"), Block("para", "Use tags & bold."))
    assert clean.formatting.bold_spans == 1


@pytest.mark.parametrize("register", ["se_cont", "other"])
def test_clean_human_continuation_splits_on_blank_lines_only(register: str):
    clean = clean_human(register, "First **para**\nline two.\n\n  \n\n# Not a heading")
    assert clean == CleanText(
        (Block("para", "First **para**\nline two."), Block("para", "# Not a heading"))
    )


def test_texts_table_without_generated_files_is_the_human_side(tmp_path: Path):
    table = texts_table(prompt_table(), [tmp_path / "missing.jsonl"])
    assert table.to_dicts() == [
        {
            "text_id": "p1|human",
            "prompt_id": "p1",
            "register": "hn",
            "stratum": "a",
            "n_words": 40,
            "source": "human",
            "raw": "Human one.<p>Rather than this.",
            "truncated": False,
            "failed": False,
        },
        {
            "text_id": "p2|human",
            "prompt_id": "p2",
            "register": "se",
            "stratum": "b",
            "n_words": 60,
            "source": "human",
            "raw": "**Human** two.",
            "truncated": False,
            "failed": False,
        },
    ]


def test_texts_table_stacks_model_records(tmp_path: Path):
    first = write_jsonl(
        tmp_path / "a.jsonl",
        [
            record("p1|qwen", "p1", "Old reply."),
            record("p2|qwen", "p2", None, withheld="quotes human text"),
            record("p9|qwen", "p9", "No such prompt."),
        ],
    )
    second = write_jsonl(
        tmp_path / "b.jsonl",
        [
            record("p1|qwen", "p1", "New reply.", failed=True),
            record("p2|llama", "p2", "Cut off", finish_reason="length"),
        ],
    )
    table = texts_table(prompt_table(split=["dev", "test"]), [first, second])
    model = table.filter(pl.col("source") != "human")
    assert model.select(
        "text_id", "register", "split", "raw", "truncated", "failed"
    ).rows() == [
        ("p1|qwen", "hn", "dev", "New reply.", False, True),
        ("p2|llama", "se", "test", "Cut off", True, False),
    ]
    assert table.columns == [
        "text_id",
        "prompt_id",
        "register",
        "stratum",
        "n_words",
        "split",
        "source",
        "raw",
        "truncated",
        "failed",
    ]
    assert table.height == 4


def test_texts_table_explicit_truncated_wins_over_finish_reason(tmp_path: Path):
    generated = write_jsonl(
        tmp_path / "g.jsonl",
        [
            record("p1|qwen", "p1", "Long.", truncated=False, finish_reason="length"),
            record("p2|qwen", "p2", "Short.", truncated=True),
            record("p2|llama", "p2", "Other.", finish_reason="length"),
            record("p1|llama", "p1", "Fine.", finish_reason=None),
        ],
    )
    table = texts_table(prompt_table(claude=[True, False]), [generated])
    model = table.filter(pl.col("source") != "human")
    assert model.select("text_id", "truncated", "claude").rows() == [
        ("p1|qwen", False, True),
        ("p2|qwen", True, False),
        ("p2|llama", True, False),
        ("p1|llama", False, True),
    ]


def test_main_writes_counts_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    prompts = tmp_path / "prompts.parquet"
    prompt_table().write_parquet(prompts)
    generated = write_jsonl(
        tmp_path / "g.jsonl",
        [
            record("p1|qwen", "p1", "# Plan\n\nWe walked rather than drove."),
            record("p2|qwen", "p2", None),
        ],
    )
    out_path = tmp_path / "out" / "counts.parquet"

    def measure_serially(texts: Sequence[CleanText]) -> list[dict[str, Any]]:
        return counters.measure(texts, n_process=1)

    monkeypatch.setattr(measure_module, "measure", measure_serially)
    monkeypatch.setattr(
        sys,
        "argv",
        ["measure", str(prompts), str(out_path), "--generated", str(generated)],
    )
    measure_module.main()
    table = pl.read_parquet(out_path)
    assert table.select(
        "text_id",
        "target_words",
        "words",
        "p2_rather_than",
        "s8b_heading",
        "clean_text",
    ).rows() == [
        ("p1|human", 40, 5, 1, 0, "Human one.\n\nRather than this."),
        ("p2|human", 60, 2, 0, 0, "Human two."),
        ("p1|qwen", 40, 6, 1, 1, "Plan\n\nWe walked rather than drove."),
    ]
    assert "register" in capsys.readouterr().out
