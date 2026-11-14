import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from llm_tells import generate

REAL_CLIENT = httpx.Client
PROMPT = {"prompt_id": "hn:1", "prompt": "Write a reply.", "n_words": 50}
ENGLISH = "A plain English reply about the weather, written for this test."
Handler = Callable[[httpx.Request], httpx.Response]


def chat_reply(content: str | None, **extra: Any) -> httpx.Response:
    message: dict[str, Any] = {"content": content, **extra}
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 7},
        },
    )


def body_of(req: httpx.Request) -> dict[str, Any]:
    return json.loads(req.content)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waited: list[float] = []
    monkeypatch.setattr(generate.time, "sleep", waited.append)
    return waited


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    path = tmp_path / "hosted.json"
    path.write_text(json.dumps({"NOTE": "x", "API_KEY_TEST": "secret-key"}))
    return path


def client_for(handler: Handler) -> httpx.Client:
    return REAL_CLIENT(transport=httpx.MockTransport(handler))


def patch_client(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    def factory(**kwargs: Any) -> httpx.Client:
        return REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_read_key_takes_the_api_key_field(key_file: Path):
    assert generate.read_key(key_file) == "secret-key"


def test_api_headers_are_empty_for_lm_studio(key_file: Path):
    assert generate.api_headers(None) == {}
    assert generate.api_headers(key_file) == {"Authorization": "Bearer secret-key"}


def test_post_retries_rate_limits_and_server_errors(sleeps: list[float]):
    statuses = iter([429, 500, 200])
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        status = next(statuses)
        headers = {"retry-after": "3"} if status == 429 else {}
        return httpx.Response(status, headers=headers, json={"ok": status})

    spec = generate.MODELS["qwen"]
    with client_for(handler) as client:
        response = generate.post(client, spec, "/chat/completions", {"a": 1})
    assert response.json() == {"ok": 200}
    assert sleeps == [3.0, 10]
    assert len(seen) == 3
    assert str(seen[0].url) == f"{generate.SERVER}/chat/completions"


def test_post_gives_up_after_the_retries(sleeps: list[float]):
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, headers={"retry-after": "soon"})

    with (
        client_for(handler) as client,
        pytest.raises(httpx.HTTPStatusError, match="503"),
    ):
        generate.post(client, generate.MODELS["qwen"], "/completions", {})
    assert len(calls) == generate.MAX_RETRIES + 1
    assert sleeps == [5, 10, 20, 40, 80, 120]


def test_post_does_not_retry_client_errors(sleeps: list[float]):
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400)

    with client_for(handler) as client, pytest.raises(httpx.HTTPStatusError):
        generate.post(client, generate.MODELS["qwen"], "/completions", {})
    assert calls == [1]
    assert sleeps == []


def test_seed_is_stable_and_differs_by_attempt():
    assert generate.seed_for("hn:1", "qwen", 0) == 799863228
    assert generate.seed_for("hn:1", "qwen", 1) == 1539515727
    assert generate.seed_for("hn:1", "gemma", 0) != 799863228
    assert all(0 <= generate.seed_for(f"se:{i}", "qwen", 0) < 2**31 for i in range(50))


def test_max_tokens_is_capped():
    assert generate.max_tokens(100) == 560
    assert generate.max_tokens(1000) == generate.MAX_TOKENS_CAP


def test_cut_base_output_stops_at_first_sentence_end_past_target():
    text = "One two three. Four five six. Seven eight."
    assert generate.cut_base_output(text, 4) == "One two three. Four five six."
    assert generate.cut_base_output(text, 1) == "One two three."


def test_cut_base_output_drops_a_new_document():
    text = "Short reply here.\n\nQuestion: what next? More words follow here."
    assert generate.cut_base_output(text, 20) == "Short reply here."
    assert generate.cut_base_output("Reply.\n12 Answers below", 50) == "Reply."


def test_cut_base_output_keeps_text_without_sentence_end():
    assert generate.cut_base_output("No sentence end at all", 2) == (
        "No sentence end at all"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("**  **", True),
        ("I'm sorry, but I can't write that.", True),
        ("As an AI, I have no opinion.", True),
        ("Это короткий русский текст о погоде.", True),  # noqa: RUF001
        (ENGLISH, False),
        ("Café naïve résumé is mostly fine here in English.", False),
        ("Dense nominalization utilization optimization implementation.", False),
    ],
)
def test_failed_triggers(text: str, expected: bool):
    assert generate.failed(text) is expected


def test_request_chat_body_and_result(sleeps: list[float]):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return chat_reply(ENGLISH, reasoning_content="abcd")

    spec = generate.MODELS["muse"]
    with client_for(handler) as client:
        result = generate.request(client, spec, PROMPT, 1)
    body = body_of(seen[0])
    assert seen[0].url.path.endswith("/chat/completions")
    assert "authorization" not in seen[0].headers
    assert body == {
        "model": "muse",
        "max_tokens": generate.max_tokens(50) + 2048,
        "seed": generate.seed_for("hn:1", "muse", 1),
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "messages": [
            {"role": "system", "content": "Reasoning strength: none."},
            {"role": "user", "content": "Write a reply."},
        ],
        "reasoning_effort": "none",
    }
    assert result == {
        "raw": ENGLISH,
        "finish_reason": "stop",
        "completion_tokens": 7,
        "reasoning_chars": 4,
        "usage": {"completion_tokens": 7},
        "seed": generate.seed_for("hn:1", "muse", 1),
    }


def test_request_hosted_sends_key_and_omits_seed(key_file: Path):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return chat_reply(None)

    spec = replace(generate.MODELS["gemini"], key_file=key_file)
    with client_for(handler) as client:
        result = generate.request(client, spec, PROMPT, 0)
    body = body_of(seen[0])
    assert str(seen[0].url) == f"{generate.GEMINI_SERVER}/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer secret-key"
    assert body["model"] == "gemini-3.8-flash"
    assert body["reasoning_effort"] == "low"
    assert "seed" not in body
    assert body["messages"] == [{"role": "user", "content": "Write a reply."}]
    assert result["raw"] == ""
    assert result["seed"] is None


def test_request_uses_max_completion_tokens_for_openai(key_file: Path):
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return chat_reply(ENGLISH)

    spec = replace(generate.MODELS["gpt_sol"], key_file=key_file)
    with client_for(handler) as client:
        generate.request(client, spec, PROMPT, 0)
    body = body_of(seen[0])
    assert body["max_completion_tokens"] == generate.max_tokens(50) + 2048
    assert "max_tokens" not in body
    assert body["seed"] == generate.seed_for("hn:1", "gpt_sol", 0)


def test_request_base_model_uses_completions():
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200, json={"choices": [{"text": "Base text.", "finish_reason": "length"}]}
        )

    spec = generate.MODELS["llama_base"]
    row = {**PROMPT, "base_prompt": "Question: q\nAnswer:"}
    with client_for(handler) as client:
        result = generate.request(client, spec, row, 0)
    body = body_of(seen[0])
    assert seen[0].url.path.endswith("/completions")
    assert not seen[0].url.path.endswith("/chat/completions")
    assert body["prompt"] == "Question: q\nAnswer:"
    assert "messages" not in body
    assert "reasoning_effort" not in body
    assert result["raw"] == "Base text."
    assert result["finish_reason"] == "length"
    assert result["completion_tokens"] is None


def test_generate_one_regenerates_once_after_refusal():
    replies = iter(["I'm sorry, I can't help with that.", ENGLISH])
    seeds: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seeds.append(body_of(req)["seed"])
        return chat_reply(next(replies))

    with client_for(handler) as client:
        out = generate.generate_one(client, generate.MODELS["qwen"], PROMPT)
    assert seeds == [generate.seed_for("hn:1", "qwen", a) for a in (0, 1)]
    assert out["text_id"] == "hn:1|qwen"
    assert out["source"] == "qwen"
    assert out["attempt"] == 1
    assert out["failed"] is False
    assert out["raw"] == ENGLISH
    assert out["truncated"] is False


def test_generate_one_keeps_second_failure_and_stops():
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return chat_reply("")

    with client_for(handler) as client:
        out = generate.generate_one(client, generate.MODELS["gemma"], PROMPT)
    assert len(calls) == 2
    assert out["attempt"] == 1
    assert out["failed"] is True


def test_generate_one_does_not_regenerate_a_good_first_attempt():
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(1)
        return chat_reply(ENGLISH)

    with client_for(handler) as client:
        out = generate.generate_one(client, generate.MODELS["gemma"], PROMPT)
    assert calls == [1]
    assert out["attempt"] == 0


def test_generate_one_cuts_base_output():
    raw = "First sentence is here. Second one ends. \nTitle: another post"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"text": raw, "finish_reason": "length"}]}
        )

    row = {**PROMPT, "n_words": 4, "base_prompt": "p"}
    with client_for(handler) as client:
        out = generate.generate_one(client, generate.MODELS["llama_base"], row)
    assert out["uncut"] == raw
    assert out["raw"] == "First sentence is here."
    assert out["truncated"] is False
    assert out["attempt"] == 0


def test_generate_one_marks_base_output_without_end_as_truncated():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"text": "Words with no end", "finish_reason": "stop"}]},
        )

    row = {**PROMPT, "n_words": 40, "base_prompt": "p"}
    with client_for(handler) as client:
        out = generate.generate_one(client, generate.MODELS["llama_base"], row)
    assert out["truncated"] is True


def fake_lms(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    commands: list[list[str]] = []

    def run(args: list[Any], **kwargs: Any) -> None:
        assert kwargs == {"check": True, "capture_output": True}
        commands.append([str(arg) for arg in args])

    monkeypatch.setattr(generate.subprocess, "run", run)
    return commands


def test_load_model_unloads_then_loads(monkeypatch: pytest.MonkeyPatch):
    commands = fake_lms(monkeypatch)
    generate.load_model(generate.MODELS["gemma"])
    lms = str(generate.LMS)
    assert commands == [
        [lms, "unload", "--all"],
        [
            lms,
            "load",
            "google/gemma-4-e4b",
            "-y",
            "-c",
            "8192",
            "--parallel",
            "4",
            "--identifier",
            "gemma",
        ],
    ]


def prompt_table(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "prompt_id": [f"hn:{i}" for i in range(n)],
            "prompt": [f"Prompt {i}" for i in range(n)],
            "n_words": [30] * n,
            "split": ["exploratory", "confirmatory"] * (n // 2),
        }
    )


def test_run_skips_finished_texts_and_appends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    commands = fake_lms(monkeypatch)
    prompts_seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        prompts_seen.append(body_of(req)["messages"][0]["content"])
        return chat_reply(ENGLISH)

    patch_client(monkeypatch, handler)
    out = tmp_path / "gen.jsonl"
    out.write_text(json.dumps({"text_id": "hn:1|gemma"}) + "\n")
    spec = generate.MODELS["gemma"]
    assert generate.run(spec, prompt_table(4), out) == 3
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [line["text_id"] for line in lines] == [
        "hn:1|gemma",
        "hn:0|gemma",
        "hn:2|gemma",
        "hn:3|gemma",
    ]
    assert sorted(prompts_seen) == ["Prompt 0", "Prompt 2", "Prompt 3"]
    assert commands[0][1:] == ["unload", "--all"]
    assert generate.run(spec, prompt_table(4), out) == 0
    assert len(commands) == 2


def test_run_hosted_does_not_touch_lm_studio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key_file: Path
):
    commands = fake_lms(monkeypatch)
    patch_client(monkeypatch, lambda req: chat_reply(ENGLISH))
    spec = replace(generate.MODELS["gemini"], key_file=key_file)
    out = tmp_path / "gen.jsonl"
    assert generate.run(spec, prompt_table(2), out) == 2
    assert commands == []
    assert len(out.read_text().splitlines()) == 2


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["generate", *args])
    generate.main()


def test_main_filters_limits_and_unloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    commands = fake_lms(monkeypatch)
    patch_client(monkeypatch, lambda req: chat_reply(ENGLISH))
    prompts = tmp_path / "prompts.parquet"
    prompt_table(6).write_parquet(prompts)
    out = tmp_path / "nested" / "gen.jsonl"
    run_main(
        monkeypatch,
        str(prompts),
        str(out),
        "--models",
        "qwen",
        "--filter",
        "split = 'confirmatory'",
        "--limit",
        "2",
    )
    ids = [json.loads(line)["text_id"] for line in out.read_text().splitlines()]
    assert ids == ["hn:1|qwen", "hn:3|qwen"]
    assert commands[-1][1:] == ["unload", "--all"]


def test_main_hosted_only_skips_final_unload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key_file: Path
):
    commands = fake_lms(monkeypatch)
    patch_client(monkeypatch, lambda req: chat_reply(ENGLISH))
    monkeypatch.setitem(
        generate.MODELS,
        "gpt_luna",
        replace(generate.MODELS["gpt_luna"], key_file=key_file),
    )
    prompts = tmp_path / "prompts.parquet"
    prompt_table(2).write_parquet(prompts)
    out = tmp_path / "gen.jsonl"
    run_main(monkeypatch, str(prompts), str(out), "--models", "gpt_luna")
    assert len(out.read_text().splitlines()) == 2
    assert commands == []
