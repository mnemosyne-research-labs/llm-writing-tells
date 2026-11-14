"""Run the full confirmatory analysis once and write every result table."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import polars as pl

from llm_tells import extras
from llm_tells.analysis import (
    MAIN_SOURCES,
    PRIMARY,
    SECONDARY,
    Tell,
    describe,
    finish,
    frame_for,
    per_cell,
    test_family,
    write,
)


def main() -> None:
    """Write the confirmatory, secondary and exploratory result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "counts", type=Path, help="main-arm counts parquet (all sources)"
    )
    parser.add_argument(
        "measures", type=Path, help="switch-rule JSON from the exploratory split"
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--coding", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("polars", "numpy", "scipy", "statsmodels", "scikit-learn", "spacy")
    }
    (args.out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "run_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "seed": args.seed,
                "versions": versions,
            },
            indent=2,
        )
    )
    measures = json.loads(args.measures.read_text())
    table = pl.read_parquet(args.counts)
    data = table.filter(pl.col("split") == "confirmatory")
    main_data = data.filter(pl.col("source").is_in(MAIN_SOURCES))

    primary = finish(
        test_family(main_data, PRIMARY, measures, MAIN_SOURCES, args.seed), "holm"
    )
    write(primary, args.out_dir / "primary.csv")

    secondary_rows = test_family(
        main_data, SECONDARY, measures, MAIN_SOURCES, args.seed + 1
    )
    content_path = args.coding / "glm_content.parquet"
    if content_path.exists():
        content = pl.read_parquet(args.coding / "content_sample.parquet").join(
            pl.read_parquet(content_path), on="code_id", how="inner"
        )
        s9 = (
            Tell(
                "S9a",
                "generic",
                "Content: generic, no specifics (coder A)",
                fixed="presence",
            ),
            Tell("S9b", "hype", "Content: hype (coder A)", fixed="presence"),
        )
        content = content.filter(
            pl.col("generic").is_not_null() & pl.col("hype").is_not_null()
        )
        secondary_rows += test_family(
            content,
            s9,
            {**measures, "S9a": "presence", "S9b": "presence"},
            MAIN_SOURCES,
            args.seed + 9,
        )
    write(finish(secondary_rows, "bh"), args.out_dir / "secondary.csv")

    write(
        per_cell(main_data, PRIMARY + SECONDARY, measures, MAIN_SOURCES, args.seed + 2),
        args.out_dir / "per_model_register.csv",
    )
    descriptive = pl.concat(
        [
            describe(
                frame_for(data, t, [*MAIN_SOURCES, "claude"]), t.column
            ).with_columns(pl.lit(t.id).alias("id"))
            for t in PRIMARY + SECONDARY
        ]
    )
    write(descriptive, args.out_dir / "descriptive.csv")
    write(
        extras.robustness(main_data, measures, args.seed + 3),
        args.out_dir / "robustness.csv",
    )
    write(
        extras.presence_table(main_data, measures),
        args.out_dir / "presence_at_median.csv",
    )
    write(extras.p10_nulls(table), args.out_dir / "p10_too_short.csv")
    write(
        extras.usefulness(main_data, measures, args.seed + 7),
        args.out_dir / "lr_plus.csv",
    )
    write(
        extras.discrimination(main_data, args.seed + 4),
        args.out_dir / "discrimination.csv",
    )
    write(
        extras.claude_arm(data, measures, args.seed + 5),
        args.out_dir / "claude_arm.csv",
    )

    recent = pl.read_parquet(args.recent).filter(pl.col("source") == "human")
    baseline = table.filter(
        (pl.col("source") == "human") & (pl.col("register") == "hn")
    )
    write(extras.recent_hn(recent, baseline), args.out_dir / "recent_hn.csv")

    cont = pl.read_parquet(args.continuation)
    cont_measures = {t.id: measures[t.id] for t in PRIMARY}
    for name, sources in (
        ("instruct_vs_base", ["llama_base", "llama_instruct"]),
        ("instruct_vs_human", ["human", "llama_instruct"]),
        ("base_vs_human", ["human", "llama_base"]),
    ):
        write(
            finish(
                test_family(cont, PRIMARY, cont_measures, sources, args.seed + 6), "bh"
            ),
            args.out_dir / f"continuation_{name}.csv",
        )
    print(
        primary.select(
            "id",
            "measure",
            "effect",
            "ci95_lo",
            "ci95_hi",
            "p_adj",
            "fpr",
            "hit",
            "lr",
            "verdict",
        )
    )


if __name__ == "__main__":
    main()
