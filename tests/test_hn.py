import asyncio
import json
import random
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_tells import hn

REAL_ASYNC_CLIENT = httpx.AsyncClient
WORDS = (
    "the",
    "plan",
    "is",
    "good",
    "for",
    "the",
    "team",
    "and",
    "we",
    "can",
    "see",
    "it",
    "work",
    "well",
)
GERMAN = ("Der", "Hund", "lief", "schnell", "durch", "den", "Park", "zum", "Haus")
END_TIME = 1_000
STORY, PARENT, REPLY = 1, 2, 3

Items = dict[int, dict[str, Any] | None]


def prose(n_words: int, words: tuple[str, ...] = WORDS) -> str:
    tokens = [words[i % len(words)] for i in range(n_words)]
    sentences = [" ".join(tokens[i : i + 10]) for i in range(0, n_words, 10)]
    return " ".join(f"{s[0].upper()}{s[1:]}." for s in sentences)


def comment(item_id: int, parent: int, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "comment",
        "parent": parent,
        "time": 500,
        "text": text,
        **extra,
    }


def chain(reply_text: str, parent_text: str) -> Items:
    return {
        STORY: {"id": STORY, "type": "story", "title": "A story &amp; title"},
        PARENT: comment(PARENT, STORY, parent_text),
        REPLY: comment(REPLY, PARENT, reply_text),
    }


def handler(
    items: Items, requested: list[int], failures: dict[int, int] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    failures = {} if failures is None else failures

    def respond(request: httpx.Request) -> httpx.Response:
        item_id = int(request.url.path.rsplit("/", 1)[1].removesuffix(".json"))
        requested.append(item_id)
        if failures.get(item_id, 0) > 0:
            failures[item_id] -= 1
            return httpx.Response(500)
        return httpx.Response(200, content=json.dumps(items.get(item_id)).encode())

    return respond


def serve(
    monkeypatch: pytest.MonkeyPatch,
    items: Items,
    failures: dict[int, int] | None = None,
) -> list[int]:
    requested: list[int] = []
    transport = httpx.MockTransport(handler(items, requested, failures))

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return REAL_ASYNC_CLIENT(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return requested


def run[T](
    items: Items,
    call: Callable[[httpx.AsyncClient], Coroutine[Any, Any, T]],
    failures: dict[int, int] | None = None,
) -> T:
    async def go() -> T:
        transport = httpx.MockTransport(handler(items, [], failures))
        async with REAL_ASYNC_CLIENT(transport=transport) as client:
            return await call(client)

    return asyncio.run(go())


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def no_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    return delays


def check(items: Items, item_id: int = REPLY) -> tuple[hn.Reply | None, str]:
    return run(items, lambda client: hn.check(client, item_id, END_TIME))


def test_cap_words_keeps_short_text_and_cuts_long_text():
    assert hn.cap_words("one two three", 3) == "one two three"
    assert hn.cap_words("one two three four", 2) == "one two …"


def test_fetch_returns_item_or_none_for_missing():
    items: Items = {7: {"id": 7, "type": "story"}, 8: None}
    assert run(items, lambda client: hn.fetch(client, 7)) == items[7]
    assert run(items, lambda client: hn.fetch(client, 8)) is None


def test_fetch_retries_server_errors_with_backoff(sleeps: list[float]):
    items: Items = {7: {"id": 7}}
    assert run(items, lambda client: hn.fetch(client, 7), {7: 2}) == {"id": 7}
    assert sleeps == [1, 2]


def test_fetch_gives_up_after_five_attempts(sleeps: list[float]):
    assert run({7: {"id": 7}}, lambda client: hn.fetch(client, 7), {7: 5}) is None
    assert sleeps == [1, 2, 4, 8, 16]


def test_fetch_retries_invalid_json(sleeps: list[float]):
    calls: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, content=b"{" if len(calls) == 1 else b"{}")

    async def go() -> dict[str, Any] | None:
        async with REAL_ASYNC_CLIENT(transport=httpx.MockTransport(respond)) as client:
            return await hn.fetch(client, 1)

    assert asyncio.run(go()) == {}
    assert sleeps == [1]


def test_story_title_walks_up_to_the_story():
    items: Items = {
        1: {"id": 1, "type": "story", "title": "Root"},
        2: comment(2, 1, "a"),
        3: comment(3, 2, "b"),
    }
    assert run(items, lambda client: hn.story_title(client, comment(4, 3, "c"))) == (
        1,
        "Root",
    )


def test_story_title_defaults_to_empty_title():
    story = {"id": 1, "type": "story"}
    assert run({}, lambda client: hn.story_title(client, story)) == (1, "")


@pytest.mark.parametrize(
    "items",
    [
        pytest.param({2: None}, id="missing_ancestor"),
        pytest.param({2: {"id": 2, "type": "comment"}}, id="ancestor_without_parent"),
    ],
)
def test_story_title_reports_broken_chain(items: Items):
    assert run(items, lambda client: hn.story_title(client, comment(3, 2, "x"))) is None


def test_story_title_stops_at_max_depth(monkeypatch: pytest.MonkeyPatch):
    items: Items = {i: comment(i, i - 1, "x") for i in range(2, 10)}
    items[1] = {"id": 1, "type": "story", "title": "Deep"}
    start = comment(10, 9, "x")
    monkeypatch.setattr(hn, "MAX_DEPTH", 5)
    assert run(items, lambda client: hn.story_title(client, start)) is None
    monkeypatch.setattr(hn, "MAX_DEPTH", 10)
    assert run(items, lambda client: hn.story_title(client, start)) == (1, "Deep")


def test_check_builds_eligible_reply():
    parent_text = f"<p>{prose(320)}"
    reply_text = f"<p>{prose(50)}<p>&gt; {prose(20)}"
    items = chain(reply_text, parent_text)
    items[REPLY] = comment(REPLY, PARENT, reply_text)
    items[PARENT] = comment(PARENT, 9, parent_text)
    items[9] = comment(9, STORY, "grandparent")
    reply, reason = check(items)
    assert reason == "eligible"
    assert reply is not None
    assert (reply.id, reply.parent_id, reply.story_id) == (REPLY, PARENT, STORY)
    assert reply.story_title == "A story &amp; title"
    assert reply.time == 500
    assert reply.n_words == 50
    assert reply.text == prose(50)
    assert reply.parent_text == hn.cap_words(prose(320), hn.PARENT_MAX_WORDS)
    assert reply.parent_text.endswith(" …")
    assert (reply.raw_text, reply.parent_raw) == (reply_text, parent_text)


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        pytest.param({"id": 3, "type": "story"}, "not_comment", id="story"),
        pytest.param(None, "not_comment", id="missing"),
        pytest.param(comment(3, 2, prose(50), dead=True), "dead_or_deleted", id="dead"),
        pytest.param(
            comment(3, 2, prose(50), deleted=True), "dead_or_deleted", id="deleted"
        ),
        pytest.param(
            {"id": 3, "type": "comment", "parent": 2, "time": 5},
            "dead_or_deleted",
            id="no_text",
        ),
        pytest.param(
            comment(3, 2, prose(50), time=END_TIME), "after_cutoff", id="at_cutoff"
        ),
        pytest.param(
            comment(3, 2, f"{prose(50)}<pre><code>x = 1</code></pre>"),
            "code",
            id="code",
        ),
        pytest.param(comment(3, 2, prose(39)), "length", id="too_short"),
        pytest.param(comment(3, 2, prose(401)), "length", id="too_long"),
        pytest.param(
            comment(3, 2, f"&gt; {prose(45)}<p>{prose(45)}"),
            "mostly_quoted",
            id="half_quoted",
        ),
        pytest.param(comment(3, 2, prose(50, GERMAN)), "not_english", id="not_english"),
    ],
)
def test_check_excludes_reply(reply: dict[str, Any] | None, reason: str):
    items = chain(prose(50), prose(20))
    items[REPLY] = reply
    assert check(items) == (None, reason)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(prose(40), id="min_words"),
        pytest.param(prose(400), id="max_words"),
        pytest.param(f"&gt; {prose(44)}<p>{prose(45)}", id="quoted_under_half"),
    ],
)
def test_check_accepts_boundaries(text: str):
    assert check(chain(text, prose(20)))[1] == "eligible"


@pytest.mark.parametrize(
    ("parent", "reason"),
    [
        pytest.param({"id": 2, "type": "story"}, "parent_not_comment", id="story"),
        pytest.param(None, "parent_not_comment", id="missing"),
        pytest.param(comment(2, 1, prose(20), dead=True), "parent_dead", id="dead"),
        pytest.param(
            comment(2, 1, prose(20), deleted=True), "parent_dead", id="deleted"
        ),
        pytest.param(
            {"id": 2, "type": "comment", "parent": 1}, "parent_dead", id="no_text"
        ),
        pytest.param(comment(2, 1, prose(9)), "parent_short", id="short"),
        pytest.param(comment(2, 99, prose(20)), "no_story", id="broken_chain"),
    ],
)
def test_check_excludes_by_parent(parent: dict[str, Any] | None, reason: str):
    items = chain(prose(50), prose(20))
    items[PARENT] = parent
    assert check(items) == (None, reason)


def test_check_accepts_parent_of_ten_words():
    assert check(chain(prose(50), prose(10)))[1] == "eligible"


def frame(first: int, last: int) -> Items:
    items = chain(prose(50), prose(20))
    for item_id in range(first, last + 1):
        items[item_id] = (
            comment(item_id, PARENT, prose(50))
            if item_id % 2 == 0
            else {"id": item_id, "type": "story"}
        )
    return items


def test_sample_keeps_draw_order_and_stops_at_quota(monkeypatch: pytest.MonkeyPatch):
    serve(monkeypatch, frame(1000, 1039))
    replies, reasons = asyncio.run(hn.sample(1000, 1039, END_TIME, 3, seed=5))
    drawn = random.Random(5).sample(range(1000, 1040), k=40)
    assert [r.id for r in replies] == [i for i in drawn if i % 2 == 0][:3]
    assert reasons == {"eligible": 3, "not_comment": 20}


def test_sample_returns_after_the_batch_that_fills_quota(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(hn, "CONCURRENCY", 1)
    requested = serve(monkeypatch, frame(1000, 1039))
    replies, reasons = asyncio.run(hn.sample(1000, 1039, END_TIME, 1, seed=5))
    first_batch = random.Random(5).sample(range(1000, 1040), k=40)[:4]
    assert any(i % 2 == 0 for i in first_batch)
    assert {i for i in requested if i >= 1000} == set(first_batch)
    assert [r.id for r in replies] == [i for i in first_batch if i % 2 == 0][:1]
    assert reasons["eligible"] == 1
    assert reasons["not_comment"] == sum(i % 2 for i in first_batch)


def test_sample_returns_what_it_found_when_frame_runs_out(
    monkeypatch: pytest.MonkeyPatch,
):
    serve(monkeypatch, frame(1000, 1009))
    replies, reasons = asyncio.run(hn.sample(1000, 1009, END_TIME, 100, seed=1))
    assert sorted(r.id for r in replies) == [1000, 1002, 1004, 1006, 1008]
    assert reasons == {"eligible": 5, "not_comment": 5}


def test_main_writes_replies_and_reasons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    serve(monkeypatch, frame(1000, 1009))
    out = tmp_path / "out"
    argv = ["hn", str(out), "--first-id", "1000", "--last-id", "1009"]
    argv += ["--end-time", str(END_TIME), "--quota", "2", "--seed", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    hn.main()
    lines = (out / "replies.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert all(r["story_id"] == STORY and r["n_words"] == 50 for r in records)
    reasons = json.loads((out / "reasons.json").read_text())
    assert reasons["eligible"] == 2
    assert capsys.readouterr().out.startswith("2 replies; reasons ")
