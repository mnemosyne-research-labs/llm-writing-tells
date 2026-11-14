"""Frontier arms: each hosted model against the human texts (D15), and the pooled test over every generator (D18)."""

import argparse
import json
from pathlib import Path

import polars as pl

from llm_tells.analysis import (
    MAIN_SOURCES,
    PRIMARY,
    SECONDARY,
    finish,
    per_cell,
    test_family,
    write,
)

REPLICATION_TELLS = ("P2", "S1", "S3", "P3")
REPLICATION_RATIO = 2.0
REPLICATION_Q = 0.05


def model_table(
    data: pl.DataFrame, model: str, measures: dict, seed: int
) -> pl.DataFrame:
    """Score one model's 21 counted tells against the human texts (D15).

    Pooled over both registers with the main arm's estimator; the model's tells are
    one Benjamini-Hochberg family. A tell in `REPLICATION_TELLS` replicates if its
    ratio is at least 2 with q < 0.05.

    Args:
        data: Confirmatory-split counts with human and model rows
        model: Source name
        measures: Switch-rule measure per tell
        seed: Random seed

    Returns:
        One row per tell, with `model` and `replicates` (null outside the four tells)
    """
    results = test_family(data, PRIMARY + SECONDARY, measures, ["human", model], seed)
    return finish(results, "bh").with_columns(
        pl.lit(model).alias("model"),
        pl.when(pl.col("id").is_in(REPLICATION_TELLS))
        .then(
            (pl.col("effect") >= REPLICATION_RATIO) & (pl.col("p_adj") < REPLICATION_Q)
        )
        .alias("replicates"),
    )


def pooled(
    data: pl.DataFrame, models: list[str], measures: dict, seed: int
) -> pl.DataFrame:
    """Recalculate the registered pooled test over a longer model list (D18).

    Same estimator, Holm across the primary tells, BH across the counted secondary
    tells and the registered verdict rules; equal weight per model and register.

    Args:
        data: Confirmatory-split counts with human rows once and every model's rows
        models: Model sources to pool
        measures: Switch-rule measure per tell
        seed: Random seed

    Returns:
        One row per tell with `family` ("primary" or "secondary")
    """
    sources = ["human", *models]
    return pl.concat(
        [
            finish(test_family(data, PRIMARY, measures, sources, seed), "holm"),
            finish(test_family(data, SECONDARY, measures, sources, seed + 1), "bh"),
        ],
        how="diagonal",
    ).with_columns(
        pl.Series("family", ["primary"] * len(PRIMARY) + ["secondary"] * len(SECONDARY))
    )


def main() -> None:
    """Write each frontier model's tell table, per-register estimates and the pooled test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path, help="frontier-arm counts parquet")
    parser.add_argument("measures", type=Path, help="switch-rule JSON")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--main", type=Path, required=True, help="main-arm counts, pooled in (D18)"
    )
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    measures = json.loads(args.measures.read_text())
    data = pl.read_parquet(args.counts).filter(pl.col("split") == "confirmatory")
    tables = [
        model_table(data, model, measures, args.seed + offset)
        for offset, model in enumerate(args.models)
    ]
    write(pl.concat(tables, how="diagonal"), args.out_dir / "tells.csv")
    write(
        per_cell(
            data, PRIMARY + SECONDARY, measures, ["human", *args.models], args.seed + 99
        ),
        args.out_dir / "per_register.csv",
    )
    main_rows = pl.read_parquet(args.main).filter(
        (pl.col("split") == "confirmatory") & pl.col("source").is_in(MAIN_SOURCES)
    )
    combined = pl.concat(
        [main_rows, data.filter(pl.col("source").is_in(args.models))], how="diagonal"
    )
    models = [*MAIN_SOURCES[1:], *args.models]
    write(
        pooled(combined, models, measures, args.seed + 200),
        args.out_dir / f"pooled_{len(models)}.csv",
    )


if __name__ == "__main__":
    main()
