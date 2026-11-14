"""The pre-registered analysis: switch rule, primary and secondary tests, LR+, verdicts and arms."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from llm_tells.stats import (
    Contrast,
    Measure,
    bh,
    estimate_tell,
    holm,
    human_total,
    lr_plus,
    verdict,
)

MAIN_SOURCES = ["human", "qwen", "gemma", "muse"]
CONTINUATION_SOURCES = ["human", "llama_base", "llama_instruct"]
END_POSITION = {"p6_closer", "p6_lastpara_only", "s7_hedge_closer"}
SWITCH_PREVALENCE, SWITCH_TOTAL = 0.05, 20


@dataclass(frozen=True, slots=True)
class Tell:
    """A pre-registered tell.

    Attributes:
        id: P1-P10 or S1-S9
        column: Counts-table column
        label: Short label
        fixed: Measure fixed by the pre-registration, or None if the switch rule decides
        se_only: Tested in the Stack Exchange register only
    """

    id: str
    column: str
    label: str
    fixed: Measure | None = None
    se_only: bool = False


PRIMARY = (
    Tell("P1", "p1_contrast", "Contrast frame (it's not X, it's Y)"),
    Tell("P2", "p2_rather_than", "Y rather than X"),
    Tell("P3", "p3_tricolon", "Rule of three"),
    Tell("P4", "p4_staccato", "Staccato fragments"),
    Tell("P5", "p5_signpost", "Signposting"),
    Tell("P6", "p6_closer", "Summary closer", fixed="presence"),
    Tell("P7", "p7_worth_noting", "It's worth noting"),
    Tell("P8", "p8_participial", "Present participial clauses", fixed="rate"),
    Tell("P9", "p9_nominal", "Nominalizations", fixed="rate"),
    Tell("P10", "p10_cv", "Sentence-length variation (CV)", fixed="cv"),
)
SECONDARY = (
    Tell("S1", "s1_dash", "Em dash family"),
    Tell("S2", "s2_transition", "Stacked transitions"),
    Tell("S3", "s3_vocab", "Flagged vocabulary (delve, crucial...)"),
    Tell("S4", "s4_new_vocab", "2026 vocabulary (quietly, load-bearing...)"),
    Tell("S5", "s5_self_answer", "Self-answered question"),
    Tell("S6", "s6_opener", "Sycophantic or chatbot opener", fixed="presence"),
    Tell("S7", "s7_hedge_closer", "Hedging closer", fixed="presence"),
    Tell("S8a", "s8a_list", "Any list (SE)", fixed="presence", se_only=True),
    Tell("S8b", "s8b_heading", "Any heading (SE)", fixed="presence", se_only=True),
    Tell("S8c", "s8c_bold", "Bold spans (SE)", fixed="rate", se_only=True),
    Tell(
        "S8d",
        "s8d_bold_lead",
        "Bold lead-ins on list items (SE)",
        fixed="presence",
        se_only=True,
    ),
)


def switch_rule(
    exploratory: pl.DataFrame, tells: tuple[Tell, ...]
) -> dict[str, Measure]:
    """Choose rate or presence per tell from the exploratory human texts.

    Args:
        exploratory: Exploratory-split counts (human rows are used)
        tells: Tells to decide

    Returns:
        Measure per tell ID
    """
    human = exploratory.filter(pl.col("source") == "human")
    chosen: dict[str, Measure] = {}
    for tell in tells:
        if tell.fixed is not None:
            chosen[tell.id] = tell.fixed
            continue
        prevalence = float(human.select((pl.col(tell.column) > 0).mean()).item() or 0)
        total = float(human[tell.column].sum())
        chosen[tell.id] = (
            "presence"
            if prevalence < SWITCH_PREVALENCE or total < SWITCH_TOTAL
            else "rate"
        )
    return chosen


def frame_for(table: pl.DataFrame, tell: Tell, sources: list[str]) -> pl.DataFrame:
    """Rows used for one tell: the sources, the register scope and the truncation rule.

    Args:
        table: Counts table
        tell: The tell
        sources: Sources included

    Returns:
        The filtered rows
    """
    rows = table.filter(pl.col("source").is_in(sources) & ~pl.col("failed"))
    if tell.se_only:
        rows = rows.filter(pl.col("register") == "se")
    if tell.column in END_POSITION:
        rows = rows.filter(~pl.col("truncated").fill_null(False))
    return rows


def describe(rows: pl.DataFrame, column: str) -> pl.DataFrame:
    """Rates per 1,000 words and presence shares by register and source.

    Args:
        rows: Rows for one tell
        column: Tell column

    Returns:
        One row per register and source
    """
    return (
        rows.group_by(["register", "source"])
        .agg(
            pl.len().alias("n"),
            (pl.col(column).sum() / pl.col("words").sum() * 1000).alias("per_1000"),
            (pl.col(column) > 0).mean().alias("share"),
            pl.col(column).mean().alias("mean"),
        )
        .sort(["register", "source"])
    )


def test_family(
    table: pl.DataFrame,
    tells: tuple[Tell, ...],
    measures: dict[str, Measure],
    sources: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    """Estimate, test and describe a family of tells.

    Args:
        table: Counts table (confirmatory split)
        tells: The family
        measures: Measure per tell
        sources: Source order; the first is human
        seed: Random seed

    Returns:
        One result row per tell (unadjusted)
    """
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    for tell in tells:
        rows = frame_for(table, tell, sources)
        measure = measures[tell.id]
        contrast = estimate_tell(rows, tell.column, measure, sources, rng)
        if measure == "cv":
            cv_rows = rows.filter(pl.col(tell.column).is_not_null())
            cutoff = (
                cv_rows.filter(pl.col("source") == sources[0])
                .group_by("register")
                .agg(pl.col(tell.column).quantile(0.25).alias("cut"))
            )
            flag_rows = cv_rows.join(cutoff, on="register").with_columns(
                (pl.col(tell.column) < pl.col("cut")).cast(pl.Int64).alias("_low_cv")
            )
            usefulness = lr_plus(flag_rows, "_low_cv", sources, rng)
        else:
            usefulness = lr_plus(rows, tell.column, sources, rng)
        results.append(
            {
                "id": tell.id,
                "label": tell.label,
                "measure": measure,
                "human_total": float("nan")
                if measure == "cv"
                else human_total(rows, tell.column, measure, sources[0]),
                **{f"c_{k}": v for k, v in asdict(contrast).items()},
                **usefulness,
                "_contrast": contrast,
            }
        )
    return results


def finish(results: list[dict[str, Any]], adjust: str) -> pl.DataFrame:
    """Adjust p-values, assign verdicts and format ratios.

    Args:
        results: Rows from test_family
        adjust: "holm" or "bh"

    Returns:
        The family's result table
    """
    raw = [row["c_p"] for row in results]
    adjusted = holm(raw) if adjust == "holm" else bh(raw)
    rows: list[dict[str, Any]] = []
    for row, p_adj in zip(results, adjusted, strict=True):
        contrast: Contrast = row.pop("_contrast")
        scale = (lambda x: x) if row["measure"] == "cv" else math.exp
        rows.append(
            {
                **{k: v for k, v in row.items() if not k.startswith("c_ci")},
                "effect": scale(contrast.estimate),
                "ci95_lo": scale(contrast.ci95[0]),
                "ci95_hi": scale(contrast.ci95[1]),
                "ci90_lo": scale(contrast.ci90[0]),
                "ci90_hi": scale(contrast.ci90[1]),
                "p_adj": p_adj,
                "verdict": verdict(contrast, p_adj, row["lr"], row["measure"]),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def per_cell(
    table: pl.DataFrame,
    tells: tuple[Tell, ...],
    measures: dict[str, Measure],
    sources: list[str],
    seed: int,
) -> pl.DataFrame:
    """Per-model and per-register estimates (their own BH family).

    Args:
        table: Counts table
        tells: Tells
        measures: Measure per tell
        sources: Source order; the first is human
        seed: Random seed

    Returns:
        One row per tell, model and register
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for tell in tells:
        for register in ("hn", "se"):
            if tell.se_only and register == "hn":
                continue
            for model in sources[1:]:
                sub = frame_for(table, tell, ["human", model]).filter(
                    pl.col("register") == register
                )
                try:
                    contrast = estimate_tell(
                        sub, tell.column, measures[tell.id], ["human", model], rng
                    )
                except RuntimeError:
                    continue
                scale = (lambda x: x) if measures[tell.id] == "cv" else math.exp
                rows.append(
                    {
                        "id": tell.id,
                        "register": register,
                        "model": model,
                        "effect": scale(contrast.estimate),
                        "ci95_lo": scale(contrast.ci95[0]),
                        "ci95_hi": scale(contrast.ci95[1]),
                        "p": contrast.p,
                        "method": contrast.method,
                    }
                )
    frame = pl.DataFrame(rows)
    return frame.with_columns(pl.Series("q", bh(frame["p"].to_list())))


def write(frame: pl.DataFrame, path: Path) -> None:
    """Write a result table as CSV.

    Args:
        frame: Table
        path: Output path
    """
    frame.write_csv(path, float_precision=4)


def main() -> None:
    """Apply the switch rule on the exploratory split and write exploratory tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path, help="main-arm counts parquet")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pl.read_parquet(args.counts).filter(pl.col("split") == "exploratory")
    measures = switch_rule(data, PRIMARY + SECONDARY)
    (args.out_dir / "switch_rule.json").write_text(json.dumps(measures, indent=2))
    human = data.filter(pl.col("source") == "human")
    write(
        pl.DataFrame(
            [
                {
                    "id": t.id,
                    "column": t.column,
                    "measure": measures[t.id],
                    "human_share": float(
                        human.select((pl.col(t.column) > 0).mean()).item() or 0
                    ),
                    "human_total": float(human[t.column].sum()),
                }
                for t in PRIMARY + SECONDARY
                if t.column != "p10_cv"
            ]
        ),
        args.out_dir / "switch_rule_inputs.csv",
    )
    descriptive = pl.concat(
        [
            describe(frame_for(data, t, MAIN_SOURCES), t.column).with_columns(
                pl.lit(t.id).alias("id")
            )
            for t in PRIMARY + SECONDARY
        ]
    )
    write(descriptive, args.out_dir / "exploratory_descriptive.csv")
    print(json.dumps(measures))


if __name__ == "__main__":
    main()
