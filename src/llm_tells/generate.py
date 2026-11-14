"""Generate model texts through the LM Studio server or a hosted API, resumably, one model at a time."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from llm_tells.text import clean_markdown, count_words

SERVER = "http://localhost:1234/v1"
GEMINI_SERVER = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENAI_SERVER = "https://api.openai.com/v1"
# Hosted-API key files; set LLM_TELLS_CREDS to use another folder.
CREDS = Path(
    os.environ.get("LLM_TELLS_CREDS", Path(__file__).resolve().parents[4] / "creds")
)
RETRY_STATUSES = {429, 500, 503}
MAX_RETRIES = 6
LMS = Path.home() / ".lmstudio" / "bin" / "lms"
PARALLEL = 4
CONTEXT = 8192
MAX_TOKENS_CAP = 2500
REFUSAL_RE = re.compile(
    r"^\s*(?:I'm sorry|I am sorry|I can't (?:help|assist)|I cannot (?:help|assist)|As an AI\b|I won't be able)",
    re.IGNORECASE,
)
NEW_DOCUMENT_RE = re.compile(
    r"\n\s*(?:Question:|Answer:|Title:|Q:|A:|#{1,3} |\d+ (?:Answers?|Votes?)\b)"
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model and its pre-registered sampling settings.

    Attributes:
        name: Short source name used in the data
        key: LM Studio model key
        sampling: Sampling parameters sent with every request
        chat: Whether to use the chat endpoint (False for base models)
        system: System message, used only to set a reasoning level the model cannot switch off
        reasoning_tokens: Extra max_tokens allowance for hidden reasoning
        reasoning_effort: Value sent as `reasoning_effort` on the chat endpoint
        server: OpenAI-compatible base URL; LM Studio unless hosted
        key_file: JSON file with the hosted API's key (see `api_headers`); None for LM Studio
        parallel: Concurrent requests
        sends_seed: Whether the server accepts `seed` (Gemini's endpoint rejects it)
        max_tokens_field: Name of the output-cap field (OpenAI's reasoning models need
            `max_completion_tokens`)
    """

    name: str
    key: str
    sampling: dict[str, float | int]
    chat: bool = True
    system: str | None = None
    reasoning_tokens: int = 0
    reasoning_effort: str = "none"
    server: str = SERVER
    key_file: Path | None = None
    parallel: int = PARALLEL
    sends_seed: bool = True
    max_tokens_field: str = "max_tokens"

    @property
    def hosted(self) -> bool:
        """Whether the model runs behind a hosted API rather than LM Studio."""
        return self.key_file is not None

    @property
    def model_id(self) -> str:
        """Model name sent in requests: the API's model key, or the LM Studio identifier."""
        return self.key if self.hosted else self.name


MODELS = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "qwen", "qwen/qwen3.8-27b", {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
        ),
        ModelSpec(
            "gemma",
            "google/gemma-4-e4b",
            {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        ),
        ModelSpec(
            "muse",
            "meta/muse-glimmer",
            {"temperature": 0.8, "top_p": 0.95, "top_k": 40},
            system="Reasoning strength: none.",
            reasoning_tokens=2048,
        ),
        ModelSpec(
            "llama_base",
            "meta-llama-3.1-8b",
            {"temperature": 0.6, "top_p": 0.9},
            chat=False,
        ),
        ModelSpec(
            "llama_instruct",
            "meta-llama-3.1-8b-instruct",
            {"temperature": 0.6, "top_p": 0.9},
        ),
        ModelSpec(
            "gemini",
            "gemini-3.8-flash",
            {},
            reasoning_tokens=2048,
            reasoning_effort="low",
            server=GEMINI_SERVER,
            key_file=CREDS / "gemini.json",
            parallel=2,
            sends_seed=False,
        ),
        ModelSpec(
            "gpt_sol",
            "gpt-6.1-sol",
            {},
            reasoning_tokens=2048,
            reasoning_effort="low",
            server=OPENAI_SERVER,
            key_file=CREDS / "openai.json",
            max_tokens_field="max_completion_tokens",
        ),
        ModelSpec(
            "gpt_luna",
            "gpt-6-luna",
            {},
            server=OPENAI_SERVER,
            key_file=CREDS / "openai.json",
            max_tokens_field="max_completion_tokens",
        ),
    )
}


def read_key(key_file: Path) -> str:
    """Read a hosted API's key from its JSON key file.

    Args:
        key_file: JSON file whose field starting with `API_KEY` holds the key

    Returns:
        The key
    """
    with key_file.open() as handle:
        return next(
            value
            for name, value in json.load(handle).items()
            if name.startswith("API_KEY")
        )


@cache
def api_headers(key_file: Path | None) -> dict[str, str]:
    """Authorization headers for a hosted API, read once from its key file.

    Args:
        key_file: JSON key file (see `read_key`), or None for LM Studio

    Returns:
        The headers (empty for LM Studio)
    """
    return {} if key_file is None else {"Authorization": f"Bearer {read_key(key_file)}"}


def post(
    client: httpx.Client, spec: ModelSpec, path: str, body: dict[str, Any]
) -> httpx.Response:
    """POST to the model's server, retrying rate-limit and server errors with backoff.

    Args:
        client: HTTP client
        spec: Model spec
        path: Endpoint path, e.g. "/chat/completions"
        body: JSON body

    Returns:
        The successful response

    Raises:
        httpx.HTTPStatusError: If the server still returns an error after the retries
            (for a hosted free tier, usually the daily quota: rerun the next day)
    """
    url, headers = f"{spec.server}{path}", api_headers(spec.key_file)
    response = client.post(url, json=body, headers=headers)
    for retry in range(MAX_RETRIES):
        if response.status_code not in RETRY_STATUSES:
            break
        wait = response.headers.get("retry-after")
        time.sleep(float(wait) if wait and wait.isdigit() else min(2**retry * 5, 120))
        response = client.post(url, json=body, headers=headers)
    response.raise_for_status()
    return response


def seed_for(prompt_id: str, model: str, attempt: int) -> int:
    """Derive the request seed from the prompt, the model and the attempt.

    Args:
        prompt_id: Prompt ID
        model: Source name
        attempt: 0 for the first try, 1 for the regeneration

    Returns:
        A 31-bit seed
    """
    return (
        int(
            hashlib.sha256(f"{prompt_id}:{model}:{attempt}".encode()).hexdigest()[:8],
            16,
        )
        & 0x7FFFFFFF
    )


def max_tokens(n_words: int) -> int:
    """Pre-registered output cap: four times the target words times 1.4, at most 2,500.

    Args:
        n_words: Target word count

    Returns:
        The max_tokens value
    """
    return min(int(4 * n_words * 1.4), MAX_TOKENS_CAP)


def cut_base_output(text: str, n_words: int) -> str:
    """Apply the base-model cut rule.

    Removes everything from a line that starts a new document, then cuts at the
    first sentence end after `n_words` words.

    Args:
        text: Raw base-model output
        n_words: Target word count

    Returns:
        The cut text
    """
    text = NEW_DOCUMENT_RE.split(text, maxsplit=1)[0]
    for match in re.finditer(r"[.!?](?=\s|$)", text):
        if count_words(text[: match.end()]) >= n_words:
            return text[: match.end()]
    return text


def failed(text: str) -> bool:
    """Check the regeneration triggers: empty, a refusal, or mostly non-ASCII letters.

    The function-word part of the English check is not applied to model output
    (deviation D6): it rejected dense but plainly English model prose.

    Args:
        text: Model output

    Returns:
        True if the output fails
    """
    clean = clean_markdown(text)
    letters = [char for char in clean.text if char.isalpha()]
    ascii_share = (
        sum(char.isascii() for char in letters) / len(letters) if letters else 0.0
    )
    return clean.n_words == 0 or bool(REFUSAL_RE.search(text)) or ascii_share < 0.9


def request(
    client: httpx.Client, spec: ModelSpec, row: dict[str, Any], attempt: int
) -> dict[str, Any]:
    """Send one generation request.

    Args:
        client: HTTP client
        spec: Model spec
        row: Prompt row with `prompt`, `n_words` and, for base models, `base_prompt`
        attempt: Attempt number

    Returns:
        The raw text, finish reason and usage

    Raises:
        httpx.HTTPStatusError: If the server returns an error status
    """
    seed = seed_for(row["prompt_id"], spec.name, attempt)
    limit = max_tokens(row["n_words"]) + spec.reasoning_tokens
    common = {
        "model": spec.model_id,
        spec.max_tokens_field: limit,
        **({"seed": seed} if spec.sends_seed else {}),
        **spec.sampling,
    }
    if spec.chat:
        system = [{"role": "system", "content": spec.system}] if spec.system else []
        messages = [*system, {"role": "user", "content": row["prompt"]}]
        body = {
            **common,
            "messages": messages,
            "reasoning_effort": spec.reasoning_effort,
        }
        response = post(client, spec, "/chat/completions", body)
    else:
        response = post(
            client, spec, "/completions", {**common, "prompt": row["base_prompt"]}
        )
    payload = response.json()
    choice = payload["choices"][0]
    raw = choice["message"]["content"] if spec.chat else choice["text"]
    reasoning = choice.get("message", {}).get("reasoning_content") or ""
    return {
        "raw": raw or "",
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "usage": payload.get("usage"),
        "seed": seed if spec.sends_seed else None,
    }


def generate_one(
    client: httpx.Client, spec: ModelSpec, row: dict[str, Any]
) -> dict[str, Any]:
    """Generate one text, regenerating once on failure.

    Args:
        client: HTTP client
        spec: Model spec
        row: Prompt row

    Returns:
        The output record
    """
    started = time.time()
    result: dict[str, Any] = {}
    for attempt in range(2):
        result = request(client, spec, row, attempt)
        result["truncated"] = result["finish_reason"] == "length"
        if not spec.chat:
            result["uncut"] = result["raw"]
            result["raw"] = cut_base_output(result["raw"], row["n_words"])
            result["truncated"] = not re.search(r"[.!?]\s*$", result["raw"])
        result["attempt"] = attempt
        result["failed"] = failed(result["raw"])
        if not result["failed"]:
            break
    return {
        "text_id": f"{row['prompt_id']}|{spec.name}",
        "prompt_id": row["prompt_id"],
        "source": spec.name,
        "seconds": round(time.time() - started, 2),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **result,
    }


def load_model(spec: ModelSpec) -> None:
    """Load a model under its source name, unloading everything else first.

    Args:
        spec: Model spec

    Raises:
        subprocess.CalledProcessError: If LM Studio fails to load the model
    """
    subprocess.run([LMS, "unload", "--all"], check=True, capture_output=True)
    subprocess.run(
        [
            LMS,
            "load",
            spec.key,
            "-y",
            "-c",
            str(CONTEXT),
            "--parallel",
            str(PARALLEL),
            "--identifier",
            spec.name,
        ],
        check=True,
        capture_output=True,
    )


def run(spec: ModelSpec, prompts: pl.DataFrame, out_path: Path) -> int:
    """Generate every missing text for one model and append to the output file.

    Args:
        spec: Model spec
        prompts: Prompt rows
        out_path: JSON-lines output, appended to (resumable)

    Returns:
        Number of texts generated
    """
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as lines:
            done = {json.loads(line)["text_id"] for line in lines}
    todo: list[dict[str, Any]] = [
        row
        for row in prompts.iter_rows(named=True)
        if f"{row['prompt_id']}|{spec.name}" not in done
    ]
    if not todo:
        return 0
    if not spec.hosted:
        load_model(spec)
    with (
        httpx.Client(timeout=900) as client,
        ThreadPoolExecutor(spec.parallel) as pool,
        out_path.open("a") as out,
    ):
        for record in pool.map(partial(generate_one, client, spec), todo):
            out.write(json.dumps(record) + "\n")
            out.flush()
    return len(todo)


def main() -> None:
    """Generate texts for the chosen models over a prompt table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path, help="parquet prompt table")
    parser.add_argument("out_path", type=Path, help="JSON-lines output")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--filter",
        default=None,
        help="polars SQL WHERE clause, e.g. \"split = 'exploratory'\"",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    prompts = pl.read_parquet(args.prompts)
    if args.filter:
        prompts = prompts.sql(f"SELECT * FROM self WHERE {args.filter}")
    if args.limit:
        prompts = prompts.head(args.limit)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        started = time.time()
        count = run(MODELS[name], prompts, args.out_path)
        print(f"{name}: {count} texts in {time.time() - started:.0f} s", flush=True)
    if not all(MODELS[name].hosted for name in args.models):
        subprocess.run([LMS, "unload", "--all"], check=True, capture_output=True)


if __name__ == "__main__":
    main()
