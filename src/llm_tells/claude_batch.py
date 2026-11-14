"""Generate the Claude API arm (D17) through the Message Batches API, resumably."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import polars as pl
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from llm_tells.generate import CREDS, failed, max_tokens, read_key

KEY_FILE = CREDS / "anthropic.json"
POLL_SECONDS = 60


@dataclass(frozen=True, slots=True)
class ClaudeSpec:
    """A Claude model and its registered request settings (D17).

    Attributes:
        name: Short source name used in the data
        model: API model ID
        thinking: `thinking` parameter, or None to omit it (the model's default)
        effort: `output_config.effort`, or None for the model's default
        reasoning_tokens: Extra max_tokens allowance for thinking
        batch_prices: Batch input and output prices, USD per 1M tokens
    """

    name: str
    model: str
    thinking: dict[str, str] | None
    effort: str | None
    reasoning_tokens: int
    batch_prices: tuple[float, float]

    def params(self, row: dict[str, Any]) -> MessageCreateParamsNonStreaming:
        """Request body for one prompt: the bare prompt, no system prompt, no sampling.

        Args:
            row: Prompt row with `prompt` and `n_words`

        Returns:
            The Messages API parameters
        """
        params = MessageCreateParamsNonStreaming(
            model=self.model,
            max_tokens=max_tokens(row["n_words"]) + self.reasoning_tokens,
            messages=[{"role": "user", "content": row["prompt"]}],
        )
        if self.thinking is not None:
            params["thinking"] = self.thinking  # pyright: ignore[reportGeneralTypeIssues]
        if self.effort is not None:
            params["output_config"] = {"effort": self.effort}  # pyright: ignore[reportGeneralTypeIssues]
        return params

    def cost(self, usage: dict[str, int]) -> float:
        """Batch cost of one request in USD.

        Args:
            usage: The response's usage

        Returns:
            The cost
        """
        price_in, price_out = self.batch_prices
        return (
            usage["input_tokens"] * price_in + usage["output_tokens"] * price_out
        ) / 1e6


MODELS = {
    spec.name: spec
    for spec in (
        ClaudeSpec(
            "claude_sonnet",
            "claude-sonnet-5-5",
            {"type": "between_tools"},
            None,
            0,
            (1.0, 5.0),
        ),
        ClaudeSpec("claude_opus", "claude-opus-5-5", None, "low", 2048, (2.0, 10.0)),
    )
}


def custom_id(prompt_id: str, attempt: int) -> str:
    """Batch custom ID for a prompt and attempt (IDs allow only [a-zA-Z0-9_-]).

    Args:
        prompt_id: Prompt ID, e.g. "hn:19012529"
        attempt: Attempt number

    Returns:
        The custom ID
    """
    return f"{prompt_id.replace(':', '_').replace('.', '-')}__{attempt}"


def record(
    spec: ClaudeSpec, prompt_id: str, attempt: int, result: Any
) -> dict[str, Any]:
    """Turn one batch result into an output record in the generator's format.

    Args:
        spec: Model spec
        prompt_id: Prompt ID
        attempt: Attempt number
        result: A `MessageBatchIndividualResponse`

    Returns:
        The output record
    """
    base = {
        "text_id": f"{prompt_id}|{spec.name}",
        "prompt_id": prompt_id,
        "source": spec.name,
        "attempt": attempt,
        "seed": None,
        "result_type": result.result.type,
    }
    if result.result.type != "succeeded":
        error = getattr(result.result, "error", None)
        return {**base, "raw": "", "failed": True, "error": str(error)}
    message = result.result.message
    raw = "".join(block.text for block in message.content if block.type == "text")
    usage = message.usage.to_dict()
    return {
        **base,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": message.model,
        "raw": raw,
        "finish_reason": message.stop_reason,
        "stop_details": message.stop_details.to_dict()
        if message.stop_details
        else None,
        "truncated": message.stop_reason == "max_tokens",
        "completion_tokens": usage["output_tokens"],
        "reasoning_chars": 0,
        "usage": usage,
        "cost_usd": spec.cost(usage),
        "failed": message.stop_reason == "refusal" or failed(raw),
    }


def run_batch(
    client: anthropic.Anthropic,
    spec: ClaudeSpec,
    rows: list[dict[str, Any]],
    attempt: int,
    state_path: Path,
) -> list[dict[str, Any]]:
    """Submit one batch (or resume the one already submitted), wait, and collect it.

    The batch ID is saved before polling, so an interrupted run collects the same
    batch instead of paying for a second one.

    Args:
        client: API client
        spec: Model spec
        rows: Prompt rows to generate
        attempt: Attempt number
        state_path: JSON file mapping "<model>|<attempt>|<first prompt>" to batch IDs

    Returns:
        One record per row
    """
    state: dict[str, str] = (
        json.loads(state_path.read_text()) if state_path.exists() else {}
    )
    key = f"{spec.name}|{attempt}|{rows[0]['prompt_id']}|{len(rows)}"
    if key not in state:
        batch = client.messages.batches.create(
            requests=[
                Request(
                    custom_id=custom_id(row["prompt_id"], attempt),
                    params=spec.params(row),
                )
                for row in rows
            ]
        )
        state[key] = batch.id
        state_path.write_text(json.dumps(state, indent=2))
        print(f"{spec.name} attempt {attempt}: batch {batch.id}, {len(rows)} requests")
    batch_id = state[key]
    while (
        batch := client.messages.batches.retrieve(batch_id)
    ).processing_status != "ended":
        time.sleep(POLL_SECONDS)
    ids = {custom_id(row["prompt_id"], attempt): row["prompt_id"] for row in rows}
    return [
        {**record(spec, ids[result.custom_id], attempt, result), "batch_id": batch.id}
        for result in client.messages.batches.results(batch_id)
    ]


def generate(
    client: anthropic.Anthropic,
    spec: ClaudeSpec,
    prompts: pl.DataFrame,
    out_path: Path,
    state_path: Path,
) -> list[dict[str, Any]]:
    """Generate every missing text for one model, regenerating failures once.

    Args:
        client: API client
        spec: Model spec
        prompts: Prompt rows
        out_path: JSON-lines output, appended to (resumable)
        state_path: Batch ID file (see `run_batch`)

    Returns:
        The final record per text written in this run
    """
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as lines:
            done = {json.loads(line)["text_id"] for line in lines}
    todo = [
        row
        for row in prompts.iter_rows(named=True)
        if f"{row['prompt_id']}|{spec.name}" not in done
    ]
    if not todo:
        return []
    first = {
        item["prompt_id"]: item for item in run_batch(client, spec, todo, 0, state_path)
    }
    retry = [row for row in todo if first[row["prompt_id"]]["failed"]]
    second = (
        {
            item["prompt_id"]: item
            for item in run_batch(client, spec, retry, 1, state_path)
        }
        if retry
        else {}
    )
    final = [second.get(row["prompt_id"], first[row["prompt_id"]]) for row in todo]
    with out_path.open("a") as out:
        out.writelines(json.dumps(item) + "\n" for item in final)
    return final


def summary(spec: ClaudeSpec, records: list[dict[str, Any]], total: int) -> str:
    """One-line report of a model's run: counts, tokens and cost.

    Args:
        spec: Model spec
        records: Final records written in this run
        total: Prompts in the full arm, for the cost projection

    Returns:
        The report line
    """
    frame = pl.DataFrame(records, infer_schema_length=None)
    cost = frame["cost_usd"].fill_null(0).sum()
    return (
        f"{spec.name}: {frame.height} texts, {frame['failed'].sum()} failed, "
        f"{frame['truncated'].fill_null(False).sum()} truncated, "
        f"{(frame['attempt'] > 0).sum()} regenerated, "
        f"{frame.filter(pl.col('finish_reason') == 'refusal').height} refusals; "
        f"output tokens median {frame['completion_tokens'].median()}; "
        f"US${cost:.2f}, projecting US${cost / frame.height * total:.2f} for {total}"
    )


def main() -> None:
    """Generate the Claude arm for the chosen models over a prompt table."""
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
    total = prompts.height
    if args.filter:
        prompts = prompts.sql(f"SELECT * FROM self WHERE {args.filter}")
    if args.limit:
        prompts = prompts.head(args.limit)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.out_path.with_suffix(".batches.json")
    client = anthropic.Anthropic(api_key=read_key(KEY_FILE), max_retries=6)
    for name in args.models:
        records = generate(client, MODELS[name], prompts, args.out_path, state_path)
        print(
            summary(MODELS[name], records, total)
            if records
            else f"{name}: nothing to do"
        )


if __name__ == "__main__":
    main()
