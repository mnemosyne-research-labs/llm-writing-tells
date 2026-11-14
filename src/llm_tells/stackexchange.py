"""Random sample of Stack Exchange answers from the public API, stratified by site."""

import argparse
import html
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from llm_tells.hn import cap_words
from llm_tells.text import clean_markdown, is_english

API = "https://api.stackexchange.com/2.3"
SITES = {
    "english": "English Language & Usage",
    "travel": "Travel",
    "cooking": "Seasoned Advice",
    "money": "Personal Finance & Money",
    "parenting": "Parenting",
    "workplace": "The Workplace",
    "academia": "Academia",
    "history": "History",
    "diy": "Home Improvement",
    "bicycles": "Bicycles",
}
FRAME_START, FRAME_END = 1546300800, 1669766400  # 2019-01-01, 2022-11-30 (UTC)
WINDOW = 3 * 86400
MIN_WORDS, MAX_WORDS, LONG_MIN_WORDS, QUESTION_MAX_WORDS = 40, 600, 300, 300
FIELDS = [
    "answer.answer_id",
    "answer.question_id",
    "answer.creation_date",
    "answer.last_edit_date",
    "answer.score",
    "answer.body_markdown",
    "question.question_id",
    "question.title",
    "question.body_markdown",
    "question.creation_date",
]
EXCLUDE = ["answer.owner", "question.owner", "answer.body", "question.body"]


@dataclass(frozen=True, slots=True)
class Answer:
    """An eligible answer with its question.

    Attributes:
        answer_id: Stack Exchange answer ID
        question_id: Stack Exchange question ID
        site: API site name
        creation_date: Unix creation time
        last_edit_date: Unix time of the last edit, or None
        score: Answer score at download
        question_title: Question title, unescaped
        question_text: Cleaned question body, capped at 300 words
        text: Cleaned answer text
        n_words: Words in the cleaned answer
        raw_markdown: Answer body Markdown as downloaded
        question_raw: Question body Markdown as downloaded
    """

    answer_id: int
    question_id: int
    site: str
    creation_date: int
    last_edit_date: int | None
    score: int
    question_title: str
    question_text: str
    text: str
    n_words: int
    raw_markdown: str
    question_raw: str


class StackExchangeClient:
    """A small client that respects the API's backoff and quota."""

    def __init__(self, client: httpx.Client) -> None:
        """Wrap an HTTP client.

        Args:
            client: Open HTTP client
        """
        self._client = client
        self.quota_remaining = -1

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        """Call one API endpoint.

        Args:
            path: Endpoint path after the version
            **params: Query parameters

        Returns:
            The decoded response

        Raises:
            RuntimeError: If the API reports an error or the quota runs out
        """
        response = self._client.get(f"{API}{path}", params=params)
        body = response.json()
        if "error_id" in body:
            raise RuntimeError(f"Stack Exchange API error on {path}: {body}")
        self.quota_remaining = body.get("quota_remaining", -1)
        if self.quota_remaining == 0:
            raise RuntimeError(
                "Stack Exchange API quota exhausted; rerun after the daily reset"
            )
        time.sleep(body.get("backoff", 0) + 0.2)
        return body


def window_answers(
    api: StackExchangeClient, site: str, start: int, filter_id: str
) -> list[dict[str, Any]]:
    """Fetch up to 100 answers created in one window.

    Args:
        api: API client
        site: API site name
        start: Window start, Unix time
        filter_id: API filter that selects the fields

    Returns:
        Raw answer records
    """
    body = api.get(
        "/answers",
        site=site,
        fromdate=start,
        todate=min(start + WINDOW, FRAME_END) - 1,
        sort="creation",
        order="asc",
        pagesize=100,
        filter=filter_id,
    )
    return body["items"]


def eligible(answer: dict[str, Any], reasons: Counter[str]) -> int:
    """Check an answer and return its cleaned word count, or 0 if excluded.

    Args:
        answer: Raw answer record
        reasons: Counter of exclusion reasons, updated in place

    Returns:
        Cleaned word count if eligible, else 0
    """
    if answer.get("score", 0) < 1:
        reasons["score"] += 1
        return 0
    if answer.get("last_edit_date", 0) >= FRAME_END:
        reasons["edited_after_cutoff"] += 1
        return 0
    clean = clean_markdown(answer.get("body_markdown", ""), unescape=True)
    if clean.has_code:
        reasons["code"] += 1
        return 0
    if not is_english(clean.text):
        reasons["not_english"] += 1
        return 0
    if clean.n_words < MIN_WORDS:
        reasons["length"] += 1
        return 0
    return clean.n_words


def sample_site(
    api: StackExchangeClient,
    site: str,
    filter_id: str,
    quotas: tuple[int, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    """Sample long (continuation) answers first, then main answers, from one site.

    Args:
        api: API client
        site: API site name
        filter_id: API filter
        quotas: Number of long answers and main answers wanted
        rng: Seeded random generator

    Returns:
        The long answers, the main answers and the exclusion reasons
    """
    long_quota, main_quota = quotas
    reasons: Counter[str] = Counter()
    pool: dict[int, tuple[dict[str, Any], int]] = {}
    for _ in range(25):
        start = rng.randrange(FRAME_START, FRAME_END - WINDOW)
        for answer in window_answers(api, site, start, filter_id):
            if answer["answer_id"] not in pool and (
                n_words := eligible(answer, reasons)
            ):
                pool[answer["answer_id"]] = (answer, n_words)
        n_long = sum(LONG_MIN_WORDS <= n <= MAX_WORDS for _, n in pool.values())
        n_main = sum(n <= MAX_WORDS for _, n in pool.values())
        if n_long >= 2 * long_quota and n_main >= 2 * (main_quota + long_quota):
            break
    ids = sorted(pool)
    rng.shuffle(ids)
    long_ids = [i for i in ids if LONG_MIN_WORDS <= pool[i][1] <= MAX_WORDS][
        :long_quota
    ]
    main_ids = [i for i in ids if i not in set(long_ids) and pool[i][1] <= MAX_WORDS][
        :main_quota
    ]
    reasons["over_max_words"] = sum(pool[i][1] > MAX_WORDS for i in ids)
    return [pool[i][0] for i in long_ids], [pool[i][0] for i in main_ids], reasons


def attach_questions(
    api: StackExchangeClient, site: str, answers: list[dict[str, Any]], filter_id: str
) -> list[Answer]:
    """Fetch the questions and build Answer records.

    Args:
        api: API client
        site: API site name
        answers: Raw answer records
        filter_id: API filter

    Returns:
        Answers whose question was found
    """
    question_ids = sorted({answer["question_id"] for answer in answers})
    questions: dict[int, dict[str, Any]] = {}
    for start in range(0, len(question_ids), 100):
        chunk = ";".join(map(str, question_ids[start : start + 100]))
        questions |= {
            q["question_id"]: q
            for q in api.get(
                f"/questions/{chunk}", site=site, pagesize=100, filter=filter_id
            )["items"]
        }
    records: list[Answer] = []
    for answer in answers:
        question = questions.get(answer["question_id"])
        if question is None or not question.get("body_markdown", "").strip():
            continue
        clean = clean_markdown(answer["body_markdown"], unescape=True)
        question_clean = clean_markdown(
            question.get("body_markdown", ""), unescape=True
        )
        records.append(
            Answer(
                answer_id=answer["answer_id"],
                question_id=answer["question_id"],
                site=site,
                creation_date=answer["creation_date"],
                last_edit_date=answer.get("last_edit_date"),
                score=answer["score"],
                question_title=html.unescape(question["title"]),
                question_text=cap_words(question_clean.text, QUESTION_MAX_WORDS),
                text=clean.text,
                n_words=clean.n_words,
                raw_markdown=answer["body_markdown"],
                question_raw=question.get("body_markdown", ""),
            )
        )
    return records


def main() -> None:
    """Sample answers for every site and write them as JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--long-per-site", type=int, default=15)
    parser.add_argument("--main-per-site", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    all_reasons: dict[str, dict[str, int]] = {}
    with httpx.Client(timeout=60) as client:
        api = StackExchangeClient(client)
        filter_id = api.get(
            "/filters/create",
            include=";".join(FIELDS),
            exclude=";".join(EXCLUDE),
            base="default",
            unsafe="false",
        )["items"][0]["filter"]
        with (
            (args.out_dir / "long.jsonl").open("w") as long_out,
            (args.out_dir / "main.jsonl").open("w") as main_out,
        ):
            for site in SITES:
                long_raw, main_raw, reasons = sample_site(
                    api, site, filter_id, (args.long_per_site, args.main_per_site), rng
                )
                long_out.writelines(
                    json.dumps(asdict(a)) + "\n"
                    for a in attach_questions(api, site, long_raw, filter_id)
                )
                main_out.writelines(
                    json.dumps(asdict(a)) + "\n"
                    for a in attach_questions(api, site, main_raw, filter_id)
                )
                all_reasons[site] = dict(reasons)
                print(
                    f"{site}: {len(long_raw)} long, {len(main_raw)} main; quota left {api.quota_remaining}",
                    flush=True,
                )
    (args.out_dir / "reasons.json").write_text(json.dumps(all_reasons, indent=2))


if __name__ == "__main__":
    main()
