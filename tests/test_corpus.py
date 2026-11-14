import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import spacy

from llm_tells import corpus
from llm_tells.hn import Reply
from llm_tells.stackexchange import Answer
from llm_tells.text import count_words

SENTENCES = (
    "The team met on a cold morning to review the plan for the new bridge.",
    "Nobody expected the budget to hold, but it did for almost three months.",
    "When the rain came, the crew moved inside and worked on the drawings.",
    "A local baker brought bread to the site every Friday without being asked.",
    "By the end of the year, the bridge was open and the town was proud of it.",
)


def prose(n_sentences: int) -> str:
    return " ".join(SENTENCES[i % len(SENTENCES)] for i in range(n_sentences))


def hn_record(item_id: int) -> dict[str, Any]:
    reply = Reply(
        id=item_id,
        parent_id=item_id - 1,
        story_id=1,
        time=1_600_000_000 + item_id,
        story_title=f"Story {item_id}",
        parent_text=f"Parent text {item_id}",
        text=prose(4),
        n_words=count_words(prose(4)),
        raw_text=f"<p>raw {item_id}",
        parent_raw="<p>parent",
    )
    return asdict(reply)


def se_record(site: str, answer_id: int, text: str = "") -> dict[str, Any]:
    answer = Answer(
        answer_id=answer_id,
        question_id=answer_id + 1000,
        site=site,
        creation_date=1_600_000_000 + answer_id,
        last_edit_date=None,
        score=3,
        question_title=f"Title {answer_id}",
        question_text=f"Question body {answer_id}",
        text=text or prose(4),
        n_words=count_words(text or prose(4)),
        raw_markdown=f"raw {answer_id}",
        question_raw="question raw",
    )
    return asdict(answer)


@pytest.fixture(scope="module")
def nlp() -> spacy.language.Language:
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def test_rank_is_seeded_sha256():
    assert corpus.rank("hn:1") == corpus.rank("hn:1")
    assert corpus.rank("hn:1") != corpus.rank("hn:2")
    assert len(corpus.rank("hn:1")) == 64


def test_read_jsonl(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n')
    assert corpus.read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_hn_prompts_fill_the_template():
    row = corpus.hn_prompts([hn_record(7)]).row(0, named=True)
    assert row["prompt_id"] == "hn:7"
    assert (row["register"], row["stratum"], row["item_id"]) == ("hn", "hn", "7")
    assert row["created"] == 1_600_000_007
    assert row["human_raw"] == "<p>raw 7"
    assert row["prompt"] == (
        'Here is a comment from a Hacker News discussion of the story "Story 7".\n\n'
        f"Parent text 7\n\nWrite a reply to this comment, about {row['n_words']} "
        "words long. Give only the text of the reply."
    )


def test_se_prompts_fill_the_template_with_site_name():
    row = corpus.se_prompts([se_record("cooking", 5)]).row(0, named=True)
    assert row["prompt_id"] == "se:cooking:5"
    assert (row["register"], row["stratum"], row["item_id"]) == ("se", "cooking", "5")
    assert row["prompt"].startswith(
        "Here is a question posted on Seasoned Advice Stack Exchange.\n\n"
        "Title: Title 5\n\nQuestion body 5\n\n"
    )
    assert row["human_raw"] == "raw 5"


def prompt_table() -> pl.DataFrame:
    return pl.concat(
        [
            corpus.hn_prompts([hn_record(i) for i in range(1, 9)]),
            corpus.se_prompts([se_record("travel", i) for i in range(1, 5)]),
            corpus.se_prompts([se_record("cooking", i) for i in range(1, 13)]),
        ]
    )


def counts(table: pl.DataFrame, *filters: pl.Expr) -> dict[str, int]:
    rows = table.filter(*filters).group_by("stratum").len().iter_rows()
    return dict(rows)


def test_assign_splits_takes_a_quarter_per_stratum_by_rank():
    table = corpus.assign_splits(prompt_table())
    assert table["prompt_id"].to_list() == sorted(table["prompt_id"])
    explore = pl.col("split") == "exploratory"
    assert counts(table, explore) == {"hn": 2, "travel": 1, "cooking": 3}
    hn_ranked = sorted(f"hn:{i}" for i in range(1, 9))
    hn_ranked.sort(key=corpus.rank)
    assert set(table.filter(explore, pl.col("stratum") == "hn")["prompt_id"]) == set(
        hn_ranked[:2]
    )


def test_assign_splits_caps_claude_subset_per_register(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(corpus, "CLAUDE_PER_REGISTER", 4)
    table = corpus.assign_splits(prompt_table())
    claude = table.filter(pl.col("claude"))
    assert set(claude["split"]) == {"confirmatory"}
    assert counts(table, pl.col("claude")) == {"hn": 4, "travel": 2, "cooking": 2}


def test_keep_assignments_never_moves_existing_prompts():
    previous = corpus.assign_splits(prompt_table())
    flipped = previous.with_columns(
        pl.when(pl.col("prompt_id") == "se:travel:1")
        .then(pl.lit("exploratory"))
        .otherwise(pl.col("split"))
        .alias("split"),
        (pl.col("prompt_id") == "se:travel:2").alias("claude"),
    )
    current = pl.concat(
        [
            prompt_table().filter(pl.col("prompt_id") != "hn:1"),
            corpus.se_prompts([se_record("travel", i) for i in range(5, 9)]),
        ]
    )
    table = corpus.keep_assignments(current, flipped)
    assert table["prompt_id"].to_list() == sorted(current["prompt_id"])
    old = table.join(flipped, on="prompt_id", suffix="_before")
    assert (old["split"] == old["split_before"]).all()
    assert (old["claude"] == old["claude_before"]).all()
    new = table.filter(
        pl.col("prompt_id").is_in([f"se:travel:{i}" for i in range(5, 9)])
    )
    assert not new["claude"].any()
    have = flipped.filter(
        pl.col("stratum") == "travel", pl.col("split") == "exploratory"
    ).height
    exploratory_new = new.filter(pl.col("split") == "exploratory")["prompt_id"]
    assert exploratory_new.len() == max(2 - have, 0)
    assert (
        exploratory_new.to_list()
        == sorted(new["prompt_id"], key=corpus.rank)[: exploratory_new.len()]
    )


def test_keep_assignments_fills_quota_of_a_new_stratum():
    previous = corpus.assign_splits(prompt_table())
    current = pl.concat(
        [prompt_table(), corpus.se_prompts([se_record("diy", i) for i in range(1, 9)])]
    )
    table = corpus.keep_assignments(current, previous)
    diy = table.filter(pl.col("stratum") == "diy")
    assert diy.filter(pl.col("split") == "exploratory").height == 2
    assert not diy["claude"].any()


def test_split_halves_cuts_at_sentence_nearest_middle(nlp: spacy.language.Language):
    text = f"{prose(5)}\n\n{prose(21)}"
    halves = corpus.split_halves(text, nlp)
    assert halves is not None
    first, second = halves
    assert first.endswith(".")
    assert second[0].isupper()
    assert f"{first} {second}".replace("\n\n", " ") == text.replace("\n\n", " ")
    total = count_words(text)
    assert count_words(first) + count_words(second) == total
    sentence = count_words(SENTENCES[0])
    assert abs(count_words(first) - total / 2) <= sentence
    assert "\n\n" in first + second


def test_split_halves_keeps_paragraph_break_at_the_cut(nlp: spacy.language.Language):
    halves = corpus.split_halves(f"{prose(12)}\n\n{prose(12)}", nlp)
    assert halves == (prose(12), prose(12))


@pytest.mark.parametrize("text", ["", prose(20)])
def test_split_halves_rejects_halves_under_150_words(
    nlp: spacy.language.Language, text: str
):
    assert corpus.split_halves(text, nlp) is None


def test_continuation_items_build_both_prompt_forms(nlp: spacy.language.Language):
    records = [se_record("travel", 21, prose(26)), se_record("travel", 22, prose(20))]
    table = corpus.continuation_items(records, nlp)
    assert table.height == 1
    row = table.row(0, named=True)
    halves = corpus.split_halves(prose(26), nlp)
    assert halves is not None
    assert row["prompt_id"] == "cont:travel:21"
    assert (row["register"], row["stratum"]) == ("se_cont", "travel")
    assert row["base_prompt"] == f"Title 21\n\n{halves[0]}"
    assert row["human_raw"] == halves[1]
    assert row["n_words"] == count_words(halves[1])
    assert row["prompt"] == corpus.CONTINUE_TEMPLATE.format(
        n_words=row["n_words"], first_half=halves[0]
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def raw_folder(raw: Path, supplement: list[dict[str, Any]]) -> None:
    write_jsonl(
        raw / "hn_2019_2022" / "replies.jsonl", [hn_record(i) for i in range(1, 9)]
    )
    write_jsonl(
        raw / "hn_2025_2026" / "replies.jsonl", [hn_record(i) for i in (50, 51)]
    )
    write_jsonl(
        raw / "se_2019_2022" / "main.jsonl",
        [se_record("travel", i) for i in range(1, 5)]
        + [se_record("cooking", i) for i in range(11, 15)],
    )
    write_jsonl(
        raw / "se_2019_2022" / "long.jsonl",
        [
            se_record("travel", 21, f"{prose(12)}\n\n{prose(13)}"),
            se_record("cooking", 22, prose(20)),
            se_record("travel", 23, prose(30)),
        ],
    )
    write_jsonl(raw / "se_2019_2022_supplement" / "main.jsonl", supplement)
    kept: dict[str, Any] = {
        "main": [1, 2, 3, 11, 12, 13, 14],
        "long": [21, 22],
        "reasons": {},
    }
    (raw / "se_2019_2022_supplement" / "kept.json").write_text(json.dumps(kept))


def run_main(monkeypatch: pytest.MonkeyPatch, *argv: Path | str) -> None:
    monkeypatch.setattr(sys, "argv", ["corpus", *map(str, argv)])
    corpus.main()


def test_main_builds_tables_from_raw_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw_folder(raw, [se_record("travel", 5)])
    run_main(monkeypatch, raw, out)
    prompts = pl.read_parquet(out / "prompts.parquet")
    ids = set(prompts["prompt_id"])
    assert "se:travel:4" not in ids
    assert {"se:travel:5", "se:cooking:14", "hn:8"} <= ids
    assert prompts.height == 16
    explore = pl.col("split") == "exploratory"
    assert counts(prompts, explore) == {"hn": 2, "travel": 1, "cooking": 1}
    continuation = pl.read_parquet(out / "continuation.parquet")
    assert continuation["prompt_id"].to_list() == ["cont:travel:21"]
    assert continuation["n_words"][0] >= corpus.MIN_HALF_WORDS
    recent = pl.read_parquet(out / "recent_hn.parquet")
    assert recent["prompt_id"].to_list() == ["hn:50", "hn:51"]
    assert "continuation items: 1" in capsys.readouterr().out


def test_main_with_previous_keeps_earlier_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    raw, first, second = tmp_path / "raw", tmp_path / "first", tmp_path / "second"
    raw_folder(raw, [se_record("travel", 5)])
    run_main(monkeypatch, raw, first)
    raw_folder(raw, [se_record("travel", i) for i in range(5, 10)])
    previous = first / "prompts.parquet"
    run_main(monkeypatch, raw, second, "--previous", previous)
    before = pl.read_parquet(previous)
    after = pl.read_parquet(second / "prompts.parquet")
    joined = after.join(before, on="prompt_id", suffix="_before")
    assert joined.height == before.height
    assert (joined["split"] == joined["split_before"]).all()
    assert (joined["claude"] == joined["claude_before"]).all()
    new = after.filter(
        pl.col("prompt_id").is_in([f"se:travel:{i}" for i in range(6, 10)])
    )
    assert not new["claude"].any()
    travel = after.filter(pl.col("stratum") == "travel")
    assert travel.height == 8
    assert travel.filter(pl.col("split") == "exploratory").height == 2
