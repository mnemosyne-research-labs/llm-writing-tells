"""Make the question the Stack Exchange cluster: one answer per question, then top each site up to quota."""

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from llm_tells.corpus import rank, read_jsonl
from llm_tells.stackexchange import (
    EXCLUDE,
    FIELDS,
    FRAME_END,
    FRAME_START,
    MAX_WORDS,
    WINDOW,
    StackExchangeClient,
    attach_questions,
    eligible,
    window_answers,
)

MAIN_PER_SITE = 40
MAX_WINDOWS = 12


def one_per_question(
    records: list[dict[str, Any]], banned: set[int]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Keep the lowest-ranked answer per question, dropping banned questions.

    Args:
        records: Answer records
        banned: Question IDs that may not appear

    Returns:
        The kept records and drop counts by reason
    """
    reasons: Counter[str] = Counter()
    kept: dict[int, dict[str, Any]] = {}
    for record in sorted(
        records, key=lambda r: rank(f"se:{r['site']}:{r['answer_id']}")
    ):
        question = record["question_id"]
        if question in banned:
            reasons["question_in_long"] += 1
        elif question in kept:
            reasons["duplicate_question"] += 1
        else:
            kept[question] = record
    return list(kept.values()), reasons


def top_up(
    api: StackExchangeClient,
    site: str,
    needed: int,
    used_questions: set[int],
    filter_id: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Draw replacement answers for one site under the pre-registered rules.

    Args:
        api: API client
        site: API site name
        needed: Number of answers wanted
        used_questions: Questions already in either sample
        filter_id: API filter
        rng: Seeded random generator

    Returns:
        Raw answer records, one per new question
    """
    reasons: Counter[str] = Counter()
    pool: dict[int, dict[str, Any]] = {}
    for _ in range(MAX_WINDOWS):
        start = rng.randrange(FRAME_START, FRAME_END - WINDOW)
        for answer in window_answers(api, site, start, filter_id):
            question = answer["question_id"]
            if question in used_questions or question in pool:
                continue
            if (n_words := eligible(answer, reasons)) and n_words <= MAX_WORDS:
                pool[question] = answer
        if len(pool) >= 3 * needed:
            break
    questions = sorted(pool)
    rng.shuffle(questions)
    return [pool[q] for q in questions[:needed]]


def main() -> None:
    """Write the supplement draw and the list of dropped answers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("se_dir", type=Path, help="original Stack Exchange download")
    parser.add_argument("out_dir", type=Path, help="new directory for the supplement")
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    long_kept, long_reasons = one_per_question(
        read_jsonl(args.se_dir / "long.jsonl"), set()
    )
    long_questions = {r["question_id"] for r in long_kept}
    main_kept, main_reasons = one_per_question(
        read_jsonl(args.se_dir / "main.jsonl"), long_questions
    )
    used = long_questions | {r["question_id"] for r in main_kept}
    rng = random.Random(args.seed)
    per_site = Counter(r["site"] for r in main_kept)
    supplement: list[dict[str, Any]] = []
    with httpx.Client(timeout=60) as client:
        api = StackExchangeClient(client)
        filter_id = api.get(
            "/filters/create",
            include=";".join(FIELDS),
            exclude=";".join(EXCLUDE),
            base="default",
            unsafe="false",
        )["items"][0]["filter"]
        for site in sorted(per_site):
            needed = MAIN_PER_SITE - per_site[site]
            if needed <= 0:
                continue
            raw = top_up(api, site, needed, used, filter_id, rng)
            records = [asdict(a) for a in attach_questions(api, site, raw, filter_id)]
            used |= {r["question_id"] for r in records}
            supplement.extend(records)
            print(
                f"{site}: needed {needed}, drew {len(records)}; quota left {api.quota_remaining}",
                flush=True,
            )
    with (args.out_dir / "main.jsonl").open("w") as out:
        out.writelines(json.dumps(r) + "\n" for r in supplement)
    kept_ids = {
        "main": sorted(r["answer_id"] for r in main_kept),
        "long": sorted(r["answer_id"] for r in long_kept),
        "reasons": {"main": dict(main_reasons), "long": dict(long_reasons)},
    }
    (args.out_dir / "kept.json").write_text(json.dumps(kept_ids, indent=2))


if __name__ == "__main__":
    main()
