import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from llm_tells import release
from llm_tells.restore import hn_ids, se_ids

QUOTE = "one two three four five six seven eight nine ten eleven twelve"
BANNED = frozenset(release.ngrams(QUOTE))


def test_ngrams_need_the_full_run():
    assert list(release.ngrams("one two three")) == []
    assert len(list(release.ngrams(QUOTE + " thirteen"))) == 2


def test_quotes_human_matches_case_and_punctuation_insensitively():
    assert release.quotes_human(f"As they said: {QUOTE.upper()}!", BANNED)
    assert not release.quotes_human(
        "one two three four five six seven eight nine ten eleven", BANNED
    )
    assert not release.quotes_human(None, BANNED)


def test_strip_table_drops_text_columns(tmp_path: Path):
    source = tmp_path / "in.parquet"
    pl.DataFrame(
        {"text_id": ["a"], "raw": ["secret"], "clean_text": ["secret"], "p2": [1]}
    ).write_parquet(source)
    target = tmp_path / "out" / "in.parquet"
    release.strip_table(source, target)
    assert pl.read_parquet(target).columns == ["text_id", "p2"]


def test_strip_table_keeps_csv_values_as_written(tmp_path: Path):
    source = tmp_path / "in.csv"
    source.write_text("code_id,tricolon,note\n0450,01,a note\n")
    target = tmp_path / "in_out.csv"
    release.strip_table(source, target)
    assert target.read_text() == "code_id,tricolon\n0450,01\n"


def test_release_generated_withholds_quoting_replies(tmp_path: Path):
    source = tmp_path / "gen.jsonl"
    records = [
        {"text_id": "x|qwen", "raw": f"Quote: {QUOTE}"},
        {"text_id": "y|qwen", "raw": "An original reply."},
        {"text_id": "z|llama_base", "raw": "Clean.", "uncut": f"Clean. {QUOTE}"},
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    target = tmp_path / "out.jsonl"
    withheld = release.release_generated(source, target, BANNED)
    released = [json.loads(line) for line in target.read_text().splitlines()]
    assert withheld == ["x|qwen", "z|llama_base"]
    assert released[0]["raw"] is None
    assert "withheld" in released[0]
    assert released[1] == records[1]
    assert released[2]["raw"] is None
    assert released[2]["uncut"] is None


def test_leaks_finds_human_text_in_any_released_file(tmp_path: Path):
    (tmp_path / "ok.md").write_text("Nothing quoted here.")
    pl.DataFrame({"t": [f"x {QUOTE}"]}).write_parquet(tmp_path / "bad.parquet")
    assert release.leaks(tmp_path, BANNED) == ["bad.parquet"]


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        ([{"id": 3}, {"id": 1}], [3, 1]),
        ([{"answer_id": 9, "site": "cooking"}], [["cooking", 9]]),
    ],
)
def test_write_ids_keeps_file_order(
    tmp_path: Path, records: list[dict[str, Any]], expected: list[Any]
):
    source = tmp_path / "replies.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    release.write_ids(source, tmp_path / "out" / "replies.jsonl")
    assert json.loads((tmp_path / "out" / "replies.ids.json").read_text()) == expected


def test_id_helpers():
    assert hn_ids([{"id": 5}]) == [5]
    assert se_ids([{"site": "travel", "answer_id": 7}]) == [["travel", 7]]


def test_release_generated_can_withhold_every_reply(tmp_path: Path):
    source = tmp_path / "gen.jsonl"
    source.write_text(json.dumps({"text_id": "a|claude_opus", "raw": "Fine."}) + "\n")
    target = tmp_path / "out.jsonl"
    withheld = release.release_generated(source, target, BANNED, "provider terms")
    record = json.loads(target.read_text())
    assert withheld == []
    assert record["raw"] is None
    assert record["withheld"] == "provider terms"
