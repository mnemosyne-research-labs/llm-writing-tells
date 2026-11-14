"""Random sample of HN comment replies from the official Firebase API."""

import argparse
import asyncio
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from llm_tells.text import clean_hn, is_english

API = "https://hacker-news.firebaseio.com/v0/item/{}.json"
MIN_WORDS, MAX_WORDS, PARENT_MIN_WORDS, PARENT_MAX_WORDS = 40, 400, 10, 300
CONCURRENCY = 24
MAX_DEPTH = 200


@dataclass(frozen=True, slots=True)
class Reply:
    """An eligible HN reply with its prompt context.

    Attributes:
        id: HN item ID of the reply
        parent_id: HN item ID of the parent comment
        story_id: HN item ID of the story
        time: Unix creation time of the reply
        story_title: Title of the story
        parent_text: Cleaned parent text, capped at 300 words
        text: Cleaned reply text
        n_words: Words in the cleaned reply
        raw_text: The reply's HTML as downloaded
        parent_raw: The parent's HTML as downloaded
    """

    id: int
    parent_id: int
    story_id: int
    time: int
    story_title: str
    parent_text: str
    text: str
    n_words: int
    raw_text: str
    parent_raw: str


def cap_words(text: str, limit: int) -> str:
    """Cap text at a number of whitespace-separated words.

    Args:
        text: Text to cap
        limit: Maximum number of words

    Returns:
        The text, cut after `limit` words with an ellipsis if it was longer
    """
    words = text.split(" ")
    return text if len(words) <= limit else " ".join(words[:limit]) + " …"


async def fetch(client: httpx.AsyncClient, item_id: int) -> dict[str, Any] | None:
    """Fetch one HN item, retrying transient errors.

    Args:
        client: Shared HTTP client
        item_id: HN item ID

    Returns:
        The item, or None if it does not exist
    """
    for attempt in range(5):
        try:
            response = await client.get(API.format(item_id))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError, json.JSONDecodeError:
            await asyncio.sleep(2**attempt)
    return None


async def story_title(
    client: httpx.AsyncClient, item: dict[str, Any]
) -> tuple[int, str] | None:
    """Walk up the parent chain to the story.

    Args:
        client: Shared HTTP client
        item: A comment

    Returns:
        The story ID and title, or None if the chain is broken
    """
    current: dict[str, Any] | None = item
    for _ in range(MAX_DEPTH):
        if current is None:
            return None
        if current.get("type") == "story":
            return current["id"], current.get("title", "")
        if "parent" not in current:
            return None
        current = await fetch(client, current["parent"])
    return None


async def check(
    client: httpx.AsyncClient, item_id: int, end_time: int
) -> tuple[Reply | None, str]:
    """Check one random item ID against the eligibility rules.

    Args:
        client: Shared HTTP client
        item_id: Candidate HN item ID
        end_time: Exclusive upper bound on creation time

    Returns:
        The reply if eligible, and the reason code
    """
    item = await fetch(client, item_id)
    if item is None or item.get("type") != "comment":
        return None, "not_comment"
    if item.get("dead") or item.get("deleted") or "text" not in item:
        return None, "dead_or_deleted"
    if item["time"] >= end_time:
        return None, "after_cutoff"
    reply = clean_hn(item["text"])
    if reply.has_code:
        return None, "code"
    if not MIN_WORDS <= reply.n_words <= MAX_WORDS:
        return None, "length"
    if reply.quoted_words >= reply.n_words:
        return None, "mostly_quoted"
    if not is_english(reply.text):
        return None, "not_english"
    parent = await fetch(client, item["parent"])
    if parent is None or parent.get("type") != "comment":
        return None, "parent_not_comment"
    if parent.get("dead") or parent.get("deleted") or "text" not in parent:
        return None, "parent_dead"
    parent_clean = clean_hn(parent["text"])
    if parent_clean.n_words < PARENT_MIN_WORDS:
        return None, "parent_short"
    story = await story_title(client, parent)
    if story is None:
        return None, "no_story"
    return Reply(
        id=item["id"],
        parent_id=parent["id"],
        story_id=story[0],
        time=item["time"],
        story_title=story[1],
        parent_text=cap_words(parent_clean.text, PARENT_MAX_WORDS),
        text=reply.text,
        n_words=reply.n_words,
        raw_text=item["text"],
        parent_raw=parent["text"],
    ), "eligible"


async def sample(
    first_id: int, last_id: int, end_time: int, quota: int, seed: int
) -> tuple[list[Reply], Counter[str]]:
    """Draw random item IDs until the quota of eligible replies is met.

    Args:
        first_id: Lowest item ID in the frame
        last_id: Highest item ID in the frame
        end_time: Exclusive upper bound on creation time
        quota: Number of eligible replies wanted
        seed: Random seed

    Returns:
        The replies in draw order, and counts of each reason
    """
    rng = random.Random(seed)
    order = rng.sample(
        range(first_id, last_id + 1), k=min(quota * 40, last_id - first_id + 1)
    )
    replies: list[Reply] = []
    reasons: Counter[str] = Counter()
    async with httpx.AsyncClient(
        timeout=30, limits=httpx.Limits(max_connections=CONCURRENCY)
    ) as client:
        for start in range(0, len(order), CONCURRENCY * 4):
            batch = order[start : start + CONCURRENCY * 4]
            for reply, reason in await asyncio.gather(
                *(check(client, item_id, end_time) for item_id in batch)
            ):
                if reply is not None and len(replies) >= quota:
                    continue
                reasons[reason] += 1
                if reply is not None:
                    replies.append(reply)
            if len(replies) >= quota:
                return replies, reasons
    return replies, reasons


def main() -> None:
    """Sample HN replies and write them as JSON lines with a reasons file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--first-id", type=int, required=True)
    parser.add_argument("--last-id", type=int, required=True)
    parser.add_argument(
        "--end-time", type=int, required=True, help="exclusive Unix time"
    )
    parser.add_argument("--quota", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    replies, reasons = asyncio.run(
        sample(args.first_id, args.last_id, args.end_time, args.quota, args.seed)
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "replies.jsonl").open("w") as out:
        out.writelines(json.dumps(asdict(reply)) + "\n" for reply in replies)
    (args.out_dir / "reasons.json").write_text(json.dumps(dict(reasons), indent=2))
    print(f"{len(replies)} replies; reasons {dict(reasons)}")


if __name__ == "__main__":
    main()
