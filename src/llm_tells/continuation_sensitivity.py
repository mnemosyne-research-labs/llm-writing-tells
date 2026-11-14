"""Rerun the continuation comparisons without listed items (deviation D19)."""

import argparse
import json
from pathlib import Path

import polars as pl

from llm_tells.analysis import PRIMARY, finish, test_family, write
from llm_tells.stats import Measure

COMPARISONS = (
    ("instruct_vs_base", ["llama_base", "llama_instruct"]),
    ("instruct_vs_human", ["human", "llama_instruct"]),
    ("base_vs_human", ["human", "llama_base"]),
)


def rerun(
    counts: pl.DataFrame,
    excluded: list[str],
    measures: dict[str, Measure],
    seed: int,
    out_dir: Path,
) -> None:
    """Write the three continuation tables for the items not excluded.

    Uses the same estimator, seed offset and correction as `llm_tells.report`.

    Args:
        counts: Continuation counts
        excluded: Prompt IDs to leave out
        measures: Switch-rule measure per tell
        seed: The report's seed
        out_dir: Output folder
    """
    kept = counts.filter(~pl.col("prompt_id").is_in(excluded))
    cont_measures: dict[str, Measure] = {t.id: measures[t.id] for t in PRIMARY}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sources in COMPARISONS:
        write(
            finish(test_family(kept, PRIMARY, cont_measures, sources, seed + 6), "bh"),
            out_dir / f"continuation_{name}.csv",
        )


def main() -> None:
    """Rerun the continuation arm without the items in a JSON list of prompt IDs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path, help="continuation counts parquet")
    parser.add_argument("measures", type=Path, help="switch-rule JSON")
    parser.add_argument("excluded", type=Path, help="JSON list of prompt IDs")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    rerun(
        pl.read_parquet(args.counts),
        json.loads(args.excluded.read_text()),
        json.loads(args.measures.read_text()),
        args.seed,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
