import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_tells import se_repair, stackexchange
from llm_tells.corpus import rank
from llm_tells.stackexchange import FRAME_END, FRAME_START, WINDOW, StackExchangeClient

REAL_CLIENT = httpx.Client
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
FILTER = "F1"


def prose(n_words: int, words: tuple[str, ...] = WORDS) -> str:
    tokens = [words[i % len(words)] for i in range(n_words)]
    sentences = [" ".join(tokens[i : i + 10]) for i in range(0, n_words, 10)]
    return " ".join(f"{s[0].upper()}{s[1:]}." for s in sentences)


def answer(
    answer_id: int, question_id: int, n_words: int, **extra: Any
) -> dict[str, Any]:
    return {
        "answer_id": answer_id,
        "question_id": question_id,
        "creation_date": FRAME_START + answer_id,
        "score": 1,
        "body_markdown": prose(n_words),
        **extra,
    }


def question(question_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "title": f"Question {question_id} &amp; more",
        "body_markdown": prose(20),
        **extra,
    }


@dataclass
class FakeApi:
    answers: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    questions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=lambda: {"quota_remaining": 500})
    requests: list[httpx.Request] = field(default_factory=list)

    def calls(self, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.startswith(f"/2.3{path}")]

    def items(self, request: httpx.Request) -> list[dict[str, Any]]:
        path = request.url.path.removeprefix("/2.3")
        site = request.url.params.get("site", "")
        if path == "/filters/create":
            return [{"filter": FILTER}]
        if path == "/answers":
            return self.answers.get(site, [])
        kind, _, ids = path.strip("/").partition("/")
        wanted = {int(i) for i in ids.split(";")}
        key = "answer_id" if kind == "answers" else "question_id"
        source = self.answers if kind == "answers" else self.questions
        return [r for r in source.get(site, []) if r[key] in wanted]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"items": self.items(request), **self.extra})

    def client(self) -> httpx.Client:
        return REAL_CLIENT(transport=httpx.MockTransport(self.handle))

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = httpx.MockTransport(self.handle)

        def client(*args: Any, **kwargs: Any) -> httpx.Client:
            return REAL_CLIENT(*args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "Client", client)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(stackexchange, "time", SimpleNamespace(sleep=delays.append))
    return delays


def test_get_returns_body_tracks_quota_and_backs_off(sleeps: list[float]):
    fake = FakeApi(extra={"quota_remaining": 42, "backoff": 3})
    with fake.client() as client:
        api = StackExchangeClient(client)
        body = api.get("/filters/create", base="default")
    assert body["items"] == [{"filter": FILTER}]
    assert api.quota_remaining == 42
    assert sleeps == [3.2]
    assert (
        str(fake.requests[0].url) == f"{stackexchange.API}/filters/create?base=default"
    )


def test_get_without_quota_field_sleeps_minimum(sleeps: list[float]):
    with FakeApi(extra={}).client() as client:
        api = StackExchangeClient(client)
        api.get("/answers", site="travel")
    assert api.quota_remaining == -1
    assert sleeps == [0.2]


def test_get_raises_on_api_error(sleeps: list[float]):
    fake = FakeApi(extra={"error_id": 502, "error_name": "throttle_violation"})
    with (
        fake.client() as client,
        pytest.raises(RuntimeError, match="error on /answers"),
    ):
        StackExchangeClient(client).get("/answers")
    assert sleeps == []


def test_get_raises_when_quota_runs_out():
    with (
        FakeApi(extra={"quota_remaining": 0}).client() as client,
        pytest.raises(RuntimeError, match="quota exhausted"),
    ):
        StackExchangeClient(client).get("/answers")


@pytest.mark.parametrize(
    ("start", "todate"),
    [
        pytest.param(FRAME_START, FRAME_START + WINDOW - 1, id="inside_frame"),
        pytest.param(FRAME_END - 100, FRAME_END - 1, id="clamped_to_frame_end"),
    ],
)
def test_window_answers_requests_one_window(start: int, todate: int):
    fake = FakeApi(answers={"travel": [answer(1, 1, 50)]})
    with fake.client() as client:
        items = stackexchange.window_answers(
            StackExchangeClient(client), "travel", start, FILTER
        )
    assert [a["answer_id"] for a in items] == [1]
    params = fake.requests[0].url.params
    assert (int(params["fromdate"]), int(params["todate"])) == (start, todate)
    assert (params["sort"], params["order"], params["pagesize"]) == (
        "creation",
        "asc",
        "100",
    )
    assert params["filter"] == FILTER


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        pytest.param(answer(1, 1, 50, score=0), "score", id="score_zero"),
        pytest.param({"body_markdown": prose(50)}, "score", id="no_score"),
        pytest.param(
            answer(1, 1, 50, last_edit_date=FRAME_END),
            "edited_after_cutoff",
            id="edited",
        ),
        pytest.param(
            answer(1, 1, 50, body_markdown=f"{prose(50)}\n\n```\nx = 1\n```"),
            "code",
            id="fenced_code",
        ),
        pytest.param(
            answer(1, 1, 50, body_markdown=f"{prose(50)}\n\n    indented code"),
            "code",
            id="indented_code",
        ),
        pytest.param(
            answer(1, 1, 50, body_markdown=prose(50, GERMAN)),
            "not_english",
            id="german",
        ),
        pytest.param(answer(1, 1, 39), "length", id="too_short"),
    ],
)
def test_eligible_excludes_with_reason(record: dict[str, Any], reason: str):
    reasons: Counter[str] = Counter()
    assert stackexchange.eligible(record, reasons) == 0
    assert reasons == {reason: 1}


def test_eligible_returns_cleaned_word_count():
    reasons: Counter[str] = Counter()
    body = f"{prose(40)} See https://example.com &amp; <b>this</b>."
    record = answer(1, 1, 0, body_markdown=body, last_edit_date=FRAME_END - 1)
    assert stackexchange.eligible(record, reasons) == 42
    assert stackexchange.eligible(answer(2, 2, 900), reasons) == 900
    assert reasons == {}


def site_answers() -> list[dict[str, Any]]:
    return [
        *(answer(i, i, 350) for i in range(1, 5)),
        *(answer(i, i, 50) for i in range(5, 11)),
        answer(11, 11, 650),
        answer(12, 12, 50, score=0),
    ]


def test_sample_site_draws_disjoint_long_and_main():
    fake = FakeApi(answers={"travel": site_answers()})
    with fake.client() as client:
        long, main, reasons = stackexchange.sample_site(
            StackExchangeClient(client), "travel", FILTER, (2, 3), random.Random(1)
        )
    words = {a["answer_id"]: len(a["body_markdown"].split()) for a in site_answers()}
    long_ids = [a["answer_id"] for a in long]
    main_ids = [a["answer_id"] for a in main]
    assert len(fake.calls("/answers")) == 1
    assert (len(long_ids), len(main_ids)) == (2, 3)
    assert not set(long_ids) & set(main_ids)
    assert all(words[i] >= stackexchange.LONG_MIN_WORDS for i in long_ids)
    assert all(words[i] <= stackexchange.MAX_WORDS for i in main_ids)
    assert 12 not in long_ids + main_ids
    assert reasons == {"score": 1, "over_max_words": 1}


def test_sample_site_is_reproducible_by_seed():
    def draw(seed: int) -> list[int]:
        with FakeApi(answers={"travel": site_answers()}).client() as client:
            long, main, _ = stackexchange.sample_site(
                StackExchangeClient(client),
                "travel",
                FILTER,
                (2, 3),
                random.Random(seed),
            )
        return [a["answer_id"] for a in long + main]

    assert draw(7) == draw(7)


def test_sample_site_keeps_drawing_windows_until_pool_is_large_enough():
    fake = FakeApi(answers={"travel": [answer(1, 1, 50), answer(2, 2, 350)]})
    with fake.client() as client:
        long, main, reasons = stackexchange.sample_site(
            StackExchangeClient(client), "travel", FILTER, (1, 5), random.Random(1)
        )
    assert len(fake.calls("/answers")) == 25
    assert [a["answer_id"] for a in long] == [2]
    assert [a["answer_id"] for a in main] == [1]
    assert reasons == {"over_max_words": 0}


def test_sample_site_long_answers_respect_max_words():
    fake = FakeApi(answers={"travel": [answer(i, i, 650) for i in range(1, 5)]})
    with fake.client() as client:
        long, _, _ = stackexchange.sample_site(
            StackExchangeClient(client), "travel", FILTER, (2, 0), random.Random(1)
        )
    assert long == []


def test_attach_questions_chunks_ids_and_skips_missing_questions():
    answers = [answer(1000 + q, q, 50) for q in range(1, 102)]
    long_body = prose(350)
    questions = [question(q) for q in range(1, 101)]
    questions[0]["body_markdown"] = long_body
    fake = FakeApi(questions={"travel": questions})
    with fake.client() as client:
        records = stackexchange.attach_questions(
            StackExchangeClient(client), "travel", answers, FILTER
        )
    chunks = [r.url.path.rsplit("/", 1)[1].split(";") for r in fake.calls("/questions")]
    assert [len(c) for c in chunks] == [100, 1]
    assert [r.question_id for r in records] == list(range(1, 101))
    first = records[0]
    assert first.question_title == "Question 1 & more"
    assert first.question_text == stackexchange.cap_words(long_body, 300)
    assert first.question_raw == long_body
    assert (first.answer_id, first.site, first.n_words, first.score) == (
        1001,
        "travel",
        50,
        1,
    )
    assert first.text == prose(50)
    assert first.last_edit_date is None


def test_main_samples_every_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    sites = {"travel": "Travel", "cooking": "Seasoned Advice"}
    monkeypatch.setattr(stackexchange, "SITES", sites)
    offsets = {"travel": 0, "cooking": 100}
    fake = FakeApi(
        answers={
            site: [answer(o + i, o + i, 350 if i < 3 else 50) for i in range(1, 9)]
            for site, o in offsets.items()
        },
        questions={
            site: [question(o + i) for i in range(1, 9)] for site, o in offsets.items()
        },
    )
    fake.install(monkeypatch)
    out = tmp_path / "out"
    argv = ["se", str(out), "--long-per-site", "1", "--main-per-site", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    stackexchange.main()
    long = [json.loads(line) for line in (out / "long.jsonl").read_text().splitlines()]
    main = [json.loads(line) for line in (out / "main.jsonl").read_text().splitlines()]
    assert [r["site"] for r in long] == ["travel", "cooking"]
    assert Counter(r["site"] for r in main) == {"travel": 2, "cooking": 2}
    assert all(r["n_words"] >= 300 for r in long)
    assert json.loads((out / "reasons.json").read_text()) == {
        "travel": {"over_max_words": 0},
        "cooking": {"over_max_words": 0},
    }
    assert {r.url.params["filter"] for r in fake.requests[1:]} == {FILTER}
    assert "travel: 1 long, 2 main; quota left 500" in capsys.readouterr().out


def test_one_per_question_keeps_lowest_ranked_answer():
    records = [
        {"site": "travel", "answer_id": a, "question_id": q}
        for a, q in [(1, 10), (2, 10), (3, 10), (4, 20), (5, 30)]
    ]
    kept, reasons = se_repair.one_per_question(records, banned={30})
    best = min((1, 2, 3), key=lambda a: rank(f"se:travel:{a}"))
    assert sorted((r["question_id"], r["answer_id"]) for r in kept) == [
        (10, best),
        (20, 4),
    ]
    assert reasons == {"duplicate_question": 2, "question_in_long": 1}


def test_top_up_draws_new_questions_under_the_rules():
    answers = [
        answer(1, 1, 50),
        answer(2, 1, 50),
        answer(3, 2, 50),
        answer(4, 3, 650),
        answer(5, 4, 50, score=0),
        *(answer(i, i, 50) for i in range(10, 16)),
    ]
    fake = FakeApi(answers={"travel": answers})
    with fake.client() as client:
        drawn = se_repair.top_up(
            StackExchangeClient(client), "travel", 2, {2}, FILTER, random.Random(1)
        )
    questions = [a["question_id"] for a in drawn]
    assert len(questions) == len(set(questions)) == 2
    assert set(questions) <= {1, *range(10, 16)}
    assert len(fake.calls("/answers")) == 1


def test_top_up_stops_after_max_windows():
    fake = FakeApi(answers={"travel": [answer(1, 1, 50)]})
    with fake.client() as client:
        drawn = se_repair.top_up(
            StackExchangeClient(client), "travel", 5, set(), FILTER, random.Random(1)
        )
    assert [a["answer_id"] for a in drawn] == [1]
    assert len(fake.calls("/answers")) == se_repair.MAX_WINDOWS


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_se_repair_main_tops_up_short_sites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(se_repair, "MAIN_PER_SITE", 2)
    se_dir = tmp_path / "se"
    se_dir.mkdir()
    write_jsonl(
        se_dir / "long.jsonl",
        [{"site": "travel", "answer_id": 1, "question_id": 1}],
    )
    write_jsonl(
        se_dir / "main.jsonl",
        [
            {"site": "travel", "answer_id": 2, "question_id": 1},
            {"site": "travel", "answer_id": 3, "question_id": 3},
            {"site": "cooking", "answer_id": 4, "question_id": 4},
            {"site": "cooking", "answer_id": 5, "question_id": 5},
        ],
    )
    fake = FakeApi(
        answers={"travel": [answer(3, 3, 50), answer(6, 1, 50), answer(7, 7, 50)]},
        questions={"travel": [question(7)]},
    )
    fake.install(monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["se_repair", str(se_dir), str(out)])
    se_repair.main()
    supplement = [
        json.loads(line) for line in (out / "main.jsonl").read_text().splitlines()
    ]
    assert [(r["site"], r["answer_id"]) for r in supplement] == [("travel", 7)]
    kept = json.loads((out / "kept.json").read_text())
    assert kept == {
        "main": [3, 4, 5],
        "long": [1],
        "reasons": {"main": {"question_in_long": 1}, "long": {}},
    }
    assert {r.url.params.get("site") for r in fake.calls("/answers")} == {"travel"}
    assert "travel: needed 1, drew 1" in capsys.readouterr().out


def test_attach_questions_skips_questions_without_a_body():
    answers = [answer(1001, 1, 50), answer(1002, 2, 50)]
    questions = [question(1, body_markdown="  "), question(2)]
    fake = FakeApi(questions={"travel": questions})
    with fake.client() as client:
        records = stackexchange.attach_questions(
            StackExchangeClient(client), "travel", answers, FILTER
        )
    assert [r.question_id for r in records] == [2]
