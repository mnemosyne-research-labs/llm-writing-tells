"""Build the prompt table, the split and the continuation items from the raw samples."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import spacy

from llm_tells.stackexchange import SITES
from llm_tells.text import count_words

SEED = "20261107"
HN_TEMPLATE = (
    'Here is a comment from a Hacker News discussion of the story "{story_title}".\n\n'
    "{parent_text}\n\n"
    "Write a reply to this comment, about {n_words} words long. Give only the text of the reply."
)
SE_TEMPLATE = (
    "Here is a question posted on {site_name} Stack Exchange.\n\n"
    "Title: {question_title}\n\n"
    "{question_text}\n\n"
    "Write an answer to this question, about {n_words} words long. Give only the text of the answer."
)
CONTINUE_TEMPLATE = (
    "Continue the following text. Write about {n_words} more words, in the same style. "
    "Give only the continuation.\n\n{first_half}"
)
EXPLORATORY_SHARE = 0.25
CLAUDE_PER_REGISTER = 50
MIN_HALF_WORDS = 150


def rank(key: str) -> str:
    """Seeded hash used for every random assignment.

    Args:
        key: Item key

    Returns:
        Hex digest that orders items reproducibly at random
    """
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file.

    Args:
        path: File path

    Returns:
        The records
    """
    with path.open() as lines:
        return [json.loads(line) for line in lines]


def hn_prompts(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Build HN prompt rows.

    Args:
        records: Raw HN reply records

    Returns:
        One row per reply
    """
    return pl.DataFrame(
        {
            "prompt_id": f"hn:{r['id']}",
            "register": "hn",
            "stratum": "hn",
            "item_id": str(r["id"]),
            "created": r["time"],
            "prompt": HN_TEMPLATE.format(
                story_title=r["story_title"],
                parent_text=r["parent_text"],
                n_words=r["n_words"],
            ),
            "n_words": r["n_words"],
            "human_raw": r["raw_text"],
        }
        for r in records
    )


def se_prompts(records: list[dict[str, Any]]) -> pl.DataFrame:
    """Build Stack Exchange prompt rows.

    Args:
        records: Raw answer records

    Returns:
        One row per answer
    """
    return pl.DataFrame(
        {
            "prompt_id": f"se:{r['site']}:{r['answer_id']}",
            "register": "se",
            "stratum": r["site"],
            "item_id": str(r["answer_id"]),
            "created": r["creation_date"],
            "prompt": SE_TEMPLATE.format(
                site_name=SITES[r["site"]],
                question_title=r["question_title"],
                question_text=r["question_text"],
                n_words=r["n_words"],
            ),
            "n_words": r["n_words"],
            "human_raw": r["raw_markdown"],
        }
        for r in records
    )


def assign_splits(prompts: pl.DataFrame) -> pl.DataFrame:
    """Assign the exploratory split (25% per stratum) and the Claude subset.

    Args:
        prompts: Prompt rows with `stratum`

    Returns:
        Prompts with `split` and `claude` columns
    """
    ranked = prompts.with_columns(
        pl.col("prompt_id").map_elements(rank, return_dtype=pl.String).alias("_rank")
    ).sort("_rank")
    order = pl.int_range(pl.len()).over("stratum")
    ranked = ranked.with_columns(
        pl.when(order < (pl.len().over("stratum") * EXPLORATORY_SHARE).round())
        .then(pl.lit("exploratory"))
        .otherwise(pl.lit("confirmatory"))
        .alias("split")
    )
    per_stratum = pl.col("stratum").n_unique().over("register")
    claude_rank = pl.int_range(pl.len()).over(["stratum", "split"])
    return (
        ranked.with_columns(
            (
                (pl.col("split") == "confirmatory")
                & (claude_rank < CLAUDE_PER_REGISTER // per_stratum)
            ).alias("claude")
        )
        .drop("_rank")
        .sort("prompt_id")
    )


def keep_assignments(prompts: pl.DataFrame, previous: pl.DataFrame) -> pl.DataFrame:
    """Assign splits to new prompts without moving any existing prompt between splits.

    Existing prompts keep their split and Claude flag. New prompts fill each
    stratum's exploratory quota (25%) in rank order; the rest are confirmatory
    and never in the Claude subset.

    Args:
        prompts: All current prompts
        previous: The earlier prompt table with `split` and `claude`

    Returns:
        Prompts with `split` and `claude`
    """
    old = prompts.join(
        previous.select("prompt_id", "split", "claude"), on="prompt_id", how="inner"
    )
    new = prompts.join(previous.select("prompt_id"), on="prompt_id", how="anti")
    have = old.group_by("stratum").agg(
        (pl.col("split") == "exploratory").sum().alias("have")
    )
    stratum_size = prompts.group_by("stratum").agg(
        (pl.len() * EXPLORATORY_SHARE).round().alias("quota")
    )
    ranked = (
        new.with_columns(
            pl.col("prompt_id")
            .map_elements(rank, return_dtype=pl.String)
            .alias("_rank")
        )
        .sort("_rank")
        .join(have, on="stratum", how="left")
        .join(stratum_size, on="stratum", how="left")
        .with_columns(pl.col("have").fill_null(0))
    )
    order = pl.int_range(pl.len()).over("stratum")
    assigned = ranked.with_columns(
        pl.when(order < pl.col("quota") - pl.col("have"))
        .then(pl.lit("exploratory"))
        .otherwise(pl.lit("confirmatory"))
        .alias("split"),
        pl.lit(False).alias("claude"),
    ).drop("_rank", "have", "quota")
    return pl.concat([old, assigned.select(old.columns)]).sort("prompt_id")


def split_halves(text: str, nlp: spacy.language.Language) -> tuple[str, str] | None:
    """Split a text at the sentence boundary nearest the middle by word count.

    Args:
        text: Cleaned text with paragraphs separated by blank lines
        nlp: spaCy pipeline used for sentence boundaries

    Returns:
        The first and second halves, or None if either is under 150 words
    """
    units: list[tuple[str, bool]] = []
    for paragraph in text.split("\n\n"):
        sentences = [
            sent.text.strip() for sent in nlp(paragraph).sents if sent.text.strip()
        ]
        units.extend(
            (sentence, index == len(sentences) - 1)
            for index, sentence in enumerate(sentences)
        )
    counts = [count_words(sentence) for sentence, _ in units]
    total = sum(counts)
    running = 0
    best_index, best_gap = 0, total
    for index, count in enumerate(counts[:-1]):
        running += count
        if abs(total / 2 - running) < best_gap:
            best_index, best_gap = index + 1, abs(total / 2 - running)
    halves = (units[:best_index], units[best_index:])
    joined = tuple(
        "".join(s + ("\n\n" if end else " ") for s, end in half).strip()
        for half in halves
    )
    if min(count_words(joined[0]), count_words(joined[1])) < MIN_HALF_WORDS:
        return None
    return joined[0], joined[1]


def continuation_items(
    records: list[dict[str, Any]], nlp: spacy.language.Language
) -> pl.DataFrame:
    """Build continuation-arm items from the long answers.

    Args:
        records: Raw long-answer records
        nlp: spaCy pipeline

    Returns:
        One row per eligible item with both prompt forms and the real continuation
    """
    rows: list[dict[str, Any]] = []
    for r in records:
        halves = split_halves(r["text"], nlp)
        if halves is None:
            continue
        n_words = count_words(halves[1])
        rows.append(
            {
                "prompt_id": f"cont:{r['site']}:{r['answer_id']}",
                "register": "se_cont",
                "stratum": r["site"],
                "item_id": str(r["answer_id"]),
                "created": r["creation_date"],
                "base_prompt": f"{r['question_title']}\n\n{halves[0]}",
                "prompt": CONTINUE_TEMPLATE.format(
                    n_words=n_words, first_half=halves[0]
                ),
                "n_words": n_words,
                "human_raw": halves[1],
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Write prompts.parquet, continuation.parquet and recent_hn.parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="earlier prompt table whose splits are kept",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    kept = json.loads(
        (args.raw_dir / "se_2019_2022_supplement" / "kept.json").read_text()
    )
    se_main = [
        r
        for r in read_jsonl(args.raw_dir / "se_2019_2022" / "main.jsonl")
        if r["answer_id"] in set(kept["main"])
    ]
    se_main += read_jsonl(args.raw_dir / "se_2019_2022_supplement" / "main.jsonl")
    se_long = [
        r
        for r in read_jsonl(args.raw_dir / "se_2019_2022" / "long.jsonl")
        if r["answer_id"] in set(kept["long"])
    ]
    prompts = pl.concat(
        [
            hn_prompts(read_jsonl(args.raw_dir / "hn_2019_2022" / "replies.jsonl")),
            se_prompts(se_main),
        ]
    )
    prompts = (
        assign_splits(prompts)
        if args.previous is None
        else keep_assignments(prompts, pl.read_parquet(args.previous))
    )
    prompts.write_parquet(args.out_dir / "prompts.parquet")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    continuation = continuation_items(se_long, nlp)
    continuation.write_parquet(args.out_dir / "continuation.parquet")
    hn_prompts(
        read_jsonl(args.raw_dir / "hn_2025_2026" / "replies.jsonl")
    ).write_parquet(args.out_dir / "recent_hn.parquet")
    summary = (
        prompts.group_by(["register", "split"])
        .agg(pl.len(), pl.col("claude").sum())
        .sort(["register", "split"])
    )
    print(summary, f"continuation items: {continuation.height}")


if __name__ == "__main__":
    main()
