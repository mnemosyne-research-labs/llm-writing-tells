import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anthropic
import polars as pl
import pytest
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from llm_tells import claude_batch
from llm_tells.generate import max_tokens

ENGLISH = "A plain English answer written for this test, with no tells at all."
SONNET = claude_batch.MODELS["claude_sonnet"]
OPUS = claude_batch.MODELS["claude_opus"]
Reply = Callable[[str], dict[str, Any]]


def succeeded(
    custom_id: str,
    text: str = ENGLISH,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    output_tokens: int = 200,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [
                    {"type": "thinking", "thinking": "hidden", "signature": "s"},
                    {"type": "text", "text": text[:10]},
                    {"type": "text", "text": text[10:]},
                ],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": stop_details,
                "usage": {"input_tokens": 1000, "output_tokens": output_tokens},
            },
        },
    }


def errored(custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "result": {
            "type": "errored",
            "error": {
                "type": "error",
                "error": {"type": "api_error", "message": "overloaded"},
            },
        },
    }


def response(data: dict[str, Any]) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate(data)


def batch(batch_id: str, status: str) -> MessageBatch:
    return MessageBatch.model_validate(
        {
            "id": batch_id,
            "created_at": "2026-11-14T00:00:00Z",
            "expires_at": "2026-11-15T00:00:00Z",
            "processing_status": status,
            "request_counts": {
                "canceled": 0,
                "errored": 0,
                "expired": 0,
                "processing": 0,
                "succeeded": 0,
            },
            "type": "message_batch",
        }
    )


@dataclass
class FakeBatches:
    reply: Reply
    polls_before_end: int = 0
    created: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict[str, list[dict[str, Any]]]
    )
    polls: list[str] = field(default_factory=list[str])

    def create(self, requests: list[dict[str, Any]]) -> MessageBatch:
        batch_id = f"msgbatch_{len(self.created)}"
        self.created[batch_id] = requests
        return batch(batch_id, "in_progress")

    def retrieve(self, batch_id: str) -> MessageBatch:
        self.polls.append(batch_id)
        ended = self.polls.count(batch_id) > self.polls_before_end
        return batch(batch_id, "ended" if ended else "in_progress")

    def results(self, batch_id: str) -> Iterator[MessageBatchIndividualResponse]:
        for request in self.created[batch_id]:
            yield response(self.reply(request["custom_id"]))


@dataclass
class FakeMessages:
    batches: FakeBatches


@dataclass
class FakeClient:
    messages: FakeMessages


def fake_client(reply: Reply, polls_before_end: int = 0) -> FakeClient:
    return FakeClient(FakeMessages(FakeBatches(reply, polls_before_end)))


def as_client(fake: FakeClient) -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, fake)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr(claude_batch.time, "sleep", waited.append)
    return waited


def rows(n: int) -> list[dict[str, Any]]:
    return [
        {"prompt_id": f"hn:{i}", "prompt": f"Prompt {i}", "n_words": 40}
        for i in range(n)
    ]


def test_params_sonnet_sets_thinking_only():
    params = SONNET.params(rows(1)[0])
    assert dict(params) == {
        "model": "claude-sonnet-5-5",
        "max_tokens": max_tokens(40),
        "messages": [{"role": "user", "content": "Prompt 0"}],
        "thinking": {"type": "between_tools"},
    }


def test_params_opus_sets_effort_and_reasoning_allowance():
    params = OPUS.params(rows(1)[0])
    assert params["max_tokens"] == max_tokens(40) + 2048
    assert params.get("output_config") == {"effort": "low"}
    assert "thinking" not in params


def test_cost_uses_batch_prices():
    assert OPUS.cost({"input_tokens": 1_000_000, "output_tokens": 500_000}) == 7.0
    assert SONNET.cost({"input_tokens": 2000, "output_tokens": 1000}) == pytest.approx(
        0.007
    )


def test_custom_id_keeps_only_allowed_characters():
    assert claude_batch.custom_id("hn:19012529", 0) == "hn_19012529__0"
    assert claude_batch.custom_id("se:math.12", 1) == "se_math-12__1"


def test_record_success_joins_text_blocks():
    out = claude_batch.record(OPUS, "hn:1", 0, response(succeeded("hn_1__0")))
    assert out["text_id"] == "hn:1|claude_opus"
    assert out["raw"] == ENGLISH
    assert out["model"] == "claude-opus-5-5"
    assert out["result_type"] == "succeeded"
    assert out["completion_tokens"] == 200
    assert out["cost_usd"] == pytest.approx((1000 * 2.0 + 200 * 10.0) / 1e6)
    assert out["stop_details"] is None
    assert out["truncated"] is False
    assert out["failed"] is False
    assert out["seed"] is None


def test_record_counts_refusal_stop_as_failure():
    data = succeeded(
        "hn_1__0",
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    out = claude_batch.record(SONNET, "hn:1", 0, response(data))
    assert out["failed"] is True
    assert out["finish_reason"] == "refusal"
    assert out["stop_details"]["category"] == "cyber"


def test_record_flags_max_tokens_as_truncated():
    data = succeeded("hn_1__0", stop_reason="max_tokens")
    out = claude_batch.record(SONNET, "hn:1", 0, response(data))
    assert out["truncated"] is True
    assert out["failed"] is False


def test_record_errored_result():
    out = claude_batch.record(SONNET, "hn:1", 1, response(errored("hn_1__1")))
    assert out["result_type"] == "errored"
    assert out["raw"] == ""
    assert out["failed"] is True
    assert out["attempt"] == 1
    assert "overloaded" in out["error"]
    assert "cost_usd" not in out


def test_run_batch_submits_saves_state_and_polls(
    tmp_path: Path, sleeps: list[float], capsys: pytest.CaptureFixture[str]
):
    fake = fake_client(succeeded, polls_before_end=2)
    state = tmp_path / "state.json"
    out = claude_batch.run_batch(as_client(fake), SONNET, rows(2), 0, state)
    assert json.loads(state.read_text()) == {"claude_sonnet|0|hn:0|2": "msgbatch_0"}
    requests = fake.messages.batches.created["msgbatch_0"]
    assert [r["custom_id"] for r in requests] == ["hn_0__0", "hn_1__0"]
    assert requests[0]["params"] == SONNET.params(rows(2)[0])
    assert sleeps == [claude_batch.POLL_SECONDS] * 2
    assert [r["prompt_id"] for r in out] == ["hn:0", "hn:1"]
    assert {r["batch_id"] for r in out} == {"msgbatch_0"}
    assert "batch msgbatch_0, 2 requests" in capsys.readouterr().out


def test_run_batch_resumes_a_saved_batch(tmp_path: Path, sleeps: list[float]):
    fake = fake_client(succeeded)
    fake.messages.batches.created["msgbatch_old"] = [
        {"custom_id": "hn_0__0"},
        {"custom_id": "hn_1__0"},
    ]
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"claude_sonnet|0|hn:0|2": "msgbatch_old"}))
    out = claude_batch.run_batch(as_client(fake), SONNET, rows(2), 0, state)
    assert list(fake.messages.batches.created) == ["msgbatch_old"]
    assert fake.messages.batches.polls == ["msgbatch_old"]
    assert sleeps == []
    assert [r["batch_id"] for r in out] == ["msgbatch_old"] * 2


def refusing_first(custom_id: str) -> dict[str, Any]:
    match custom_id:
        case "hn_0__0":
            return succeeded(custom_id, text="I'm sorry, I can't help with that.")
        case "hn_2__0":
            return errored(custom_id)
        case "hn_2__1":
            return succeeded(custom_id, stop_reason="refusal")
        case _:
            return succeeded(custom_id)


def prompt_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame(rows(n)).with_columns(
        split=pl.Series(["exploratory", "confirmatory"] * (n // 2))
    )


def test_generate_regenerates_failures_once_and_skips_done(
    tmp_path: Path, sleeps: list[float]
):
    fake = fake_client(refusing_first)
    out_path = tmp_path / "claude.jsonl"
    out_path.write_text(json.dumps({"text_id": "hn:3|claude_sonnet"}) + "\n")
    state = tmp_path / "state.json"
    final = claude_batch.generate(
        as_client(fake), SONNET, prompt_frame(4), out_path, state
    )
    created = fake.messages.batches.created
    assert [r["custom_id"] for r in created["msgbatch_0"]] == [
        "hn_0__0",
        "hn_1__0",
        "hn_2__0",
    ]
    assert [r["custom_id"] for r in created["msgbatch_1"]] == ["hn_0__1", "hn_2__1"]
    by_id = {r["prompt_id"]: r for r in final}
    assert [r["prompt_id"] for r in final] == ["hn:0", "hn:1", "hn:2"]
    assert (by_id["hn:0"]["attempt"], by_id["hn:0"]["failed"]) == (1, False)
    assert (by_id["hn:1"]["attempt"], by_id["hn:1"]["failed"]) == (0, False)
    assert (by_id["hn:2"]["attempt"], by_id["hn:2"]["failed"]) == (1, True)
    assert set(json.loads(state.read_text())) == {
        "claude_sonnet|0|hn:0|3",
        "claude_sonnet|1|hn:0|2",
    }
    written = [
        json.loads(line)["text_id"] for line in out_path.read_text().splitlines()
    ]
    assert written == [
        "hn:3|claude_sonnet",
        "hn:0|claude_sonnet",
        "hn:1|claude_sonnet",
        "hn:2|claude_sonnet",
    ]
    assert (
        claude_batch.generate(as_client(fake), SONNET, prompt_frame(4), out_path, state)
        == []
    )
    assert len(created) == 2


def test_generate_without_failures_sends_one_batch(tmp_path: Path, sleeps: list[float]):
    fake = fake_client(succeeded)
    final = claude_batch.generate(
        as_client(fake),
        OPUS,
        prompt_frame(2),
        tmp_path / "out.jsonl",
        tmp_path / "state.json",
    )
    assert len(fake.messages.batches.created) == 1
    assert [r["attempt"] for r in final] == [0, 0]


def test_summary_reports_counts_and_projection():
    records = [
        {
            "failed": False,
            "truncated": True,
            "attempt": 0,
            "finish_reason": "max_tokens",
            "completion_tokens": 100,
            "cost_usd": 0.5,
        },
        {
            "failed": True,
            "truncated": False,
            "attempt": 1,
            "finish_reason": "refusal",
            "completion_tokens": 300,
            "cost_usd": 1.5,
        },
        {
            "failed": True,
            "truncated": None,
            "attempt": 1,
            "finish_reason": None,
            "completion_tokens": None,
            "cost_usd": None,
        },
    ]
    assert claude_batch.summary(SONNET, records, 30) == (
        "claude_sonnet: 3 texts, 2 failed, 1 truncated, 2 regenerated, 1 refusals; "
        "output tokens median 200.0; US$2.00, projecting US$20.00 for 30"
    )


def test_main_runs_each_model_then_reports_nothing_to_do(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sleeps: list[float],
    capsys: pytest.CaptureFixture[str],
):
    key_file = tmp_path / "anthropic.json"
    key_file.write_text(json.dumps({"API_KEY_ANTHROPIC": "test-key"}))
    monkeypatch.setattr(claude_batch, "KEY_FILE", key_file)
    fake = fake_client(succeeded)
    made: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> FakeClient:
        made.append(kwargs)
        return fake

    monkeypatch.setattr(claude_batch.anthropic, "Anthropic", factory)
    prompts = tmp_path / "prompts.parquet"
    prompt_frame(6).write_parquet(prompts)
    out = tmp_path / "out" / "claude.jsonl"
    argv = [
        "claude_batch",
        str(prompts),
        str(out),
        "--models",
        "claude_sonnet",
        "--filter",
        "split = 'confirmatory'",
        "--limit",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    claude_batch.main()
    assert made == [{"api_key": "test-key", "max_retries": 6}]
    ids = [json.loads(line)["prompt_id"] for line in out.read_text().splitlines()]
    assert ids == ["hn:1", "hn:3"]
    assert out.with_suffix(".batches.json").exists()
    assert "claude_sonnet: 2 texts, 0 failed" in capsys.readouterr().out
    claude_batch.main()
    assert capsys.readouterr().out.strip() == "claude_sonnet: nothing to do"
