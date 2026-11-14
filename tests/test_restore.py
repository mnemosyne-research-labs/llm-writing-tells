import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_tells import restore, stackexchange

REAL_ASYNC_CLIENT, REAL_CLIENT = httpx.AsyncClient, httpx.Client
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
OLD_END = restore.HN_END_TIMES["hn_2019_2022"]
Handler = Callable[[httpx.Request], httpx.Response]


def prose(n_words: int) -> str:
    tokens = [WORDS[i % len(WORDS)] for i in range(n_words)]
    sentences = [" ".join(tokens[i : i + 10]) for i in range(0, n_words, 10)]
    return " ".join(f"{s[0].upper()}{s[1:]}." for s in sentences)


def hn_items() -> dict[int, dict[str, Any]]:
    def reply(item_id: int, created: int, **extra: Any) -> dict[str, Any]:
        return {
            "id": item_id,
            "type": "comment",
            "parent": 2,
            "time": created,
            "text": prose(40 + item_id),
            **extra,
        }

    return {
        1: {"id": 1, "type": "story", "title": "Story"},
        2: {"id": 2, "type": "comment", "parent": 1, "time": 5, "text": prose(15)},
        10: reply(10, OLD_END - 1),
        11: reply(11, OLD_END - 1, deleted=True),
        12: reply(12, OLD_END - 1),
        13: reply(13, OLD_END + 1),
    }


def se_answers() -> dict[str, list[dict[str, Any]]]:
    def answer(answer_id: int, question_id: int) -> dict[str, Any]:
        return {
            "answer_id": answer_id,
            "question_id": question_id,
            "creation_date": 1_600_000_000,
            "score": 2,
            "body_markdown": prose(40 + answer_id),
        }

    return {
        "travel": [answer(2, 20), answer(3, 30), answer(4, 99), answer(5, 50)],
        "cooking": [answer(1, 10), answer(5, 51)],
    }


def se_questions() -> dict[str, list[dict[str, Any]]]:
    return {
        "travel": [
            {"question_id": q, "title": f"T{q}", "body_markdown": "Why is that?"}
            for q in (20, 30, 50)
        ],
        "cooking": [
            {"question_id": q, "title": f"C{q}", "body_markdown": "How so?"}
            for q in (10, 51)
        ],
    }


def hn_response(request: httpx.Request) -> httpx.Response:
    item_id = int(request.url.path.rsplit("/", 1)[1].removesuffix(".json"))
    return httpx.Response(200, content=json.dumps(hn_items().get(item_id)).encode())


def se_response(request: httpx.Request) -> httpx.Response:
    path = request.url.path.removeprefix("/2.3")
    if path == "/filters/create":
        return httpx.Response(200, json={"items": [{"filter": "F"}]})
    assert request.url.params["filter"] == "F"
    kind, _, ids = path.strip("/").partition("/")
    wanted = {int(i) for i in ids.split(";")}
    site = request.url.params["site"]
    source, key = (
        (se_answers(), "answer_id")
        if kind == "answers"
        else (se_questions(), "question_id")
    )
    return httpx.Response(
        200, json={"items": [r for r in source[site] if r[key] in wanted]}
    )


def no_sleep(delay: float) -> None:
    assert delay >= 0


@pytest.fixture
def api_requests(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def logged(handler: Handler) -> httpx.MockTransport:
        def respond(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        return httpx.MockTransport(respond)

    def async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return REAL_ASYNC_CLIENT(*args, transport=logged(hn_response), **kwargs)

    def client(*args: Any, **kwargs: Any) -> httpx.Client:
        return REAL_CLIENT(*args, transport=logged(se_response), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client)
    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(stackexchange, "time", SimpleNamespace(sleep=no_sleep))
    return seen


def test_ids_follow_file_order():
    assert restore.hn_ids([{"id": 3}, {"id": 1}]) == [3, 1]
    records = [{"site": "travel", "answer_id": 2}, {"site": "cooking", "answer_id": 1}]
    assert restore.se_ids(records) == [["travel", 2], ["cooking", 1]]


@pytest.mark.usefixtures("api_requests")
def test_fetch_hn_keeps_input_order_and_reports_missing():
    records, missing = asyncio.run(restore.fetch_hn([12, 11, 99, 13, 10], OLD_END))
    assert [r["id"] for r in records] == [12, 10]
    assert missing == [11, 99, 13]
    assert records[0]["text"] == prose(52)
    assert records[0]["story_title"] == "Story"


def test_fetch_se_pages_ids_per_site_and_reports_missing(
    monkeypatch: pytest.MonkeyPatch, api_requests: list[httpx.Request]
):
    monkeypatch.setattr(restore, "PAGE", 2)
    pairs: list[list[str | int]] = [
        ["travel", 3],
        ["cooking", 1],
        ["travel", 2],
        ["travel", 8],
        ["travel", 4],
    ]
    records, missing = restore.fetch_se(pairs)
    assert [(r["site"], r["answer_id"]) for r in records] == [
        ("travel", 3),
        ("cooking", 1),
        ("travel", 2),
    ]
    assert missing == [8, 4]
    assert records[0]["question_title"] == "T30"
    answer_paths = [
        r.url.path for r in api_requests if r.url.path.startswith("/2.3/answers/")
    ]
    assert answer_paths == ["/2.3/answers/3;2", "/2.3/answers/8;4", "/2.3/answers/1"]


@pytest.mark.usefixtures("api_requests")
def test_fetch_se_keeps_equal_answer_ids_on_different_sites():
    records, missing = restore.fetch_se([["travel", 5], ["cooking", 5]])
    assert missing == []
    assert [r["site"] for r in records] == ["travel", "cooking"]


@pytest.mark.usefixtures("api_requests")
def test_main_restores_every_file_with_its_own_cutoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    for folder, name, ids in [
        ("hn_2019_2022", "replies", [10, 11, 13]),
        ("hn_2025_2026", "replies", [13]),
        ("se_2019_2022", "main", [["travel", 2], ["travel", 4]]),
    ]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"{name}.ids.json").write_text(json.dumps(ids))
    monkeypatch.setattr(sys, "argv", ["restore", str(tmp_path)])
    restore.main()

    def written(path: str, key: str) -> list[int]:
        lines = (tmp_path / path).read_text().splitlines()
        return [json.loads(line)[key] for line in lines]

    assert written("hn_2019_2022/replies.jsonl", "id") == [10]
    assert written("hn_2025_2026/replies.jsonl", "id") == [13]
    assert written("se_2019_2022/main.jsonl", "answer_id") == [2]
    assert json.loads((tmp_path / "missing.json").read_text()) == {
        "hn_2019_2022/replies.jsonl": [11, 13],
        "hn_2025_2026/replies.jsonl": [],
        "se_2019_2022/main.jsonl": [4],
    }
    assert (
        "hn_2019_2022/replies.jsonl: 1 restored, 2 missing" in capsys.readouterr().out
    )
