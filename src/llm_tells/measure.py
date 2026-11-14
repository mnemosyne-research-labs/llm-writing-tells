"""Clean every human and model text and count the tells, one row per text."""

import argparse
import json
from pathlib import Path

import polars as pl

from llm_tells.counters import measure
from llm_tells.text import Block, CleanText, clean_hn, clean_markdown


def clean_human(register: str, raw: str) -> CleanText:
    """Clean a human text with the rule for its register.

    Args:
        register: "hn", "se" or "se_cont"
        raw: Text as stored in the prompt table

    Returns:
        The cleaned text
    """
    match register:
        case "hn":
            return clean_hn(raw)
        case "se":
            return clean_markdown(raw, unescape=True)
        case _:
            return CleanText(
                tuple(Block("para", p) for p in raw.split("\n\n") if p.strip())
            )


def texts_table(prompts: pl.DataFrame, generated: list[Path]) -> pl.DataFrame:
    """Stack human and model texts with their prompt metadata.

    Args:
        prompts: Prompt table (main, continuation or recent)
        generated: JSON-lines files of model outputs

    Returns:
        One row per text with `raw`, `source` and `truncated`
    """
    keep = ["prompt_id", "register", "stratum", "n_words"] + [
        c for c in ("split", "claude") if c in prompts.columns
    ]
    human = prompts.select(
        pl.concat_str(pl.col("prompt_id"), pl.lit("|human")).alias("text_id"),
        *keep,
        pl.lit("human").alias("source"),
        pl.col("human_raw").alias("raw"),
        pl.lit(False).alias("truncated"),
        pl.lit(False).alias("failed"),
    )
    records = [
        json.loads(line)
        for path in generated
        if path.exists()
        for line in path.read_text().splitlines()
    ]
    if not records:
        return human
    # A released reply that quotes human text has `raw` withheld (llm_tells.release);
    # its counts are in the released counts table, so it is skipped here.
    frame = (
        pl.DataFrame(records, infer_schema_length=None)
        .unique("text_id", keep="last", maintain_order=True)
        .filter(pl.col("raw").is_not_null())
    )
    if "truncated" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Boolean).alias("truncated"))
    model = frame.select(
        "text_id",
        "prompt_id",
        "source",
        "raw",
        pl.coalesce(pl.col("truncated"), pl.col("finish_reason") == "length")
        .fill_null(False)
        .alias("truncated"),
        pl.col("failed"),
    ).join(prompts.select(keep), on="prompt_id", how="inner")
    return pl.concat([human, model.select(human.columns)])


def main() -> None:
    """Write the counts table for one prompt table and its generated texts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompts", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--generated", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    texts = texts_table(pl.read_parquet(args.prompts), args.generated)
    cleaned = [
        clean_human(register, raw) if source == "human" else clean_markdown(raw)
        for register, source, raw in texts.select(
            "register", "source", "raw"
        ).iter_rows()
    ]
    counts = pl.DataFrame(measure(cleaned), infer_schema_length=None).rename(
        {"n_words": "words"}
    )
    table = pl.concat(
        [texts.rename({"n_words": "target_words"}), counts], how="horizontal"
    ).with_columns(pl.Series("clean_text", [clean.text for clean in cleaned]))
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out_path)
    print(
        table.group_by(["register", "source"])
        .agg(pl.len(), pl.col("words").median())
        .sort(["register", "source"])
    )


if __name__ == "__main__":
    main()
