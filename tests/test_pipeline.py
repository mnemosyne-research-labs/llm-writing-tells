import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from llm_tells import analysis, extras, frontier, power, report, stats, summarize
from llm_tells.analysis import MAIN_SOURCES, PRIMARY, SECONDARY

SEED = 20261114
ALL_TELLS = PRIMARY + SECONDARY
NULL_RATE = 1.5
RATES: dict[str, tuple[float, float]] = {
    "p1_contrast": (10.0, 1.6),
    "p1_within_only": (8.0, 1.6),
    "p2_rather_than": (0.8, 5.0),
    "p3_tricolon": (3.0, 1.0),
    "p7_worth_noting": (0.0, 1.0),
    "p8_participial": (15.0, 0.5),
    "p9_nominal": (20.0, 1.0),
    "s1_dash": (2.0, 3.0),
    "s3_vocab": (0.5, 6.0),
}
SPARSE_MODEL_RATE = 1.5
COUNT_COLUMNS = (
    "p1_contrast",
    "p1_within_only",
    "p2_rather_than",
    "p3_tricolon",
    "p4_staccato",
    "p5_signpost",
    "p6_closer",
    "p6_lastpara_only",
    "p7_worth_noting",
    "p8_participial",
    "p9_nominal",
    "s1_dash",
    "s2_transition",
    "s3_vocab",
    "s4_new_vocab",
    "s5_self_answer",
    "s6_opener",
    "s7_hedge_closer",
    "s8a_list",
    "s8b_heading",
    "s8c_bold",
    "s8d_bold_lead",
    "symbols_emoji",
    "not_only_but_also",
)
VOCABULARY = np.array(
    [
        "the",
        "a",
        "of",
        "to",
        "and",
        "in",
        "that",
        "is",
        "was",
        "for",
        "on",
        "with",
        "as",
        "it",
        "at",
        "by",
        "from",
        "this",
        "be",
        "or",
    ]
)
MODEL_WORDS = np.array(["moreover", "ultimately", "robust", "nuanced", "landscape"])
BASELINE_SOURCES = ("human", "llama_base")


@dataclass(frozen=True, slots=True)
class Inputs:
    main: Path
    frontier: Path
    continuation: Path
    recent: Path
    coding: Path
    measures: Path


def text(rng: np.random.Generator, words: int, model: bool) -> str:
    pool = np.concatenate([VOCABULARY, MODEL_WORDS]) if model else VOCABULARY
    return " ".join(rng.choice(pool, size=min(words, 60)).tolist())


def counts_table(
    rng: np.random.Generator,
    sources: tuple[str, ...],
    prompts: dict[str, int],
    registers: tuple[str, ...] = ("hn", "se"),
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for split, n_prompts in prompts.items():
        for register in registers:
            for prompt in range(n_prompts):
                prompt_id = f"{register}-{split[0]}{prompt:03d}"
                target = int(rng.choice([60, 150, 250]))
                frailty = float(rng.gamma(10.0, 0.1))
                for source in sources:
                    model = source not in BASELINE_SOURCES
                    if source == "claude" and prompt % 4:
                        continue
                    words = max(30, int(target * rng.uniform(0.7, 1.3)))
                    row: dict[str, object] = {
                        "text_id": f"{prompt_id}|{source}",
                        "prompt_id": prompt_id,
                        "register": register,
                        "stratum": "synthetic",
                        "target_words": target,
                        "split": split,
                        "claude": prompt % 4 == 0,
                        "source": source,
                        "truncated": model and rng.random() < 0.03,
                        "failed": model and rng.random() < 0.01,
                        "words": words,
                        "n_sentences": max(1, words // 18),
                    }
                    for column in COUNT_COLUMNS:
                        human_rate, ratio = RATES.get(column, (NULL_RATE, 1.0))
                        if column == "p7_worth_noting" and model:
                            human_rate = SPARSE_MODEL_RATE
                        rate = human_rate * (ratio if model else 1.0)
                        row[column] = int(rng.poisson(rate * words / 1000 * frailty))
                    sd = 0.15
                    cv = rng.normal(0.55 - (1.5 * sd if model else 0.0), sd)
                    row["p10_cv"] = None if words < 45 else max(0.01, float(cv))
                    row["raw"] = row["clean_text"] = text(rng, words, model)
                    rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def coding_folder(rng: np.random.Generator, main: pl.DataFrame, path: Path) -> None:
    path.mkdir()
    sample = (
        main.filter(
            (pl.col("split") == "confirmatory") & pl.col("source").is_in(MAIN_SOURCES)
        )
        .sort("text_id")
        .with_row_index("code_id")
        .with_columns(pl.format("{}", pl.col("code_id").cast(pl.String).str.zfill(4)))
    )
    sample.write_parquet(path / "content_sample.parquet")
    model = (sample["source"] != "human").to_numpy()
    generic = (rng.random(sample.height) < np.where(model, 0.6, 0.15)).astype(float)
    generic[:3] = np.nan
    pl.DataFrame(
        {
            "code_id": sample["code_id"],
            "generic": pl.Series(generic).fill_nan(None).cast(pl.Int64),
            "hype": (rng.random(sample.height) < 0.2).astype(np.int64),
        }
    ).write_parquet(path / "glm_content.parquet")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> Inputs:
    root = tmp_path_factory.mktemp("inputs")
    rng = np.random.default_rng(SEED)
    main = counts_table(
        rng, (*MAIN_SOURCES, "claude"), {"confirmatory": 40, "exploratory": 16}
    )
    human = main.filter(pl.col("source") == "human")
    extra = counts_table(rng, ("human", "gpt_a", "gpt_b"), {"confirmatory": 40})
    frontier_table = pl.concat(
        [
            human.filter(pl.col("split") == "confirmatory"),
            extra.filter(pl.col("source") != "human"),
        ]
    )
    continuation = counts_table(
        rng, ("human", "llama_base", "llama_instruct"), {"confirmatory": 30}
    )
    recent = counts_table(rng, ("human",), {"recent": 60}, registers=("hn",))
    paths = Inputs(
        main=root / "main.parquet",
        frontier=root / "frontier.parquet",
        continuation=root / "continuation.parquet",
        recent=root / "recent.parquet",
        coding=root / "coding",
        measures=root / "switch_rule.json",
    )
    main.write_parquet(paths.main)
    frontier_table.write_parquet(paths.frontier)
    continuation.write_parquet(paths.continuation)
    recent.write_parquet(paths.recent)
    coding_folder(rng, main, paths.coding)
    exploratory = main.filter(pl.col("split") == "exploratory")
    paths.measures.write_text(json.dumps(analysis.switch_rule(exploratory, ALL_TELLS)))
    return paths


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bootstrap and permutation counts are function defaults and module constants;
    # shrink them so the full pipeline runs in seconds. The permutation count stays
    # high enough for a planted effect to survive Holm over ten tells.
    monkeypatch.setattr(stats.sparse_contrast, "__defaults__", (100, 999))
    monkeypatch.setattr(stats.lr_plus, "__defaults__", (1, 100))
    monkeypatch.setattr(extras, "N_BOOT", 50)


def run(monkeypatch: pytest.MonkeyPatch, module: str, *args: object) -> None:
    monkeypatch.setattr(sys, "argv", [module, *map(str, args)])


def by_id(frame: pl.DataFrame, column: str) -> dict[str, object]:
    return dict(zip(frame["id"].to_list(), frame[column].to_list(), strict=True))


def test_exploratory_step_applies_the_switch_rule(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run(monkeypatch, "analysis", inputs.main, tmp_path)
    analysis.main()
    measures = json.loads((tmp_path / "switch_rule.json").read_text())
    assert measures == json.loads(inputs.measures.read_text())
    assert set(measures) == {t.id for t in ALL_TELLS}
    assert measures["P1"] == "rate"
    assert measures["P7"] == "presence"
    assert {measures[t.id] for t in ALL_TELLS if t.fixed} <= {"rate", "presence", "cv"}
    assert all(measures[t.id] == t.fixed for t in ALL_TELLS if t.fixed)
    switch_inputs = pl.read_csv(tmp_path / "switch_rule_inputs.csv")
    assert switch_inputs.columns == [
        "id",
        "column",
        "measure",
        "human_share",
        "human_total",
    ]
    assert "P10" not in switch_inputs["id"].to_list()
    assert by_id(switch_inputs, "human_total")["P7"] == 0
    descriptive = pl.read_csv(tmp_path / "exploratory_descriptive.csv")
    assert set(descriptive["source"].unique()) == set(MAIN_SOURCES)
    assert set(descriptive["id"].unique()) == {t.id for t in ALL_TELLS}


def test_report_and_summary_give_the_planted_verdicts(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast: None
):
    out = tmp_path / "results"
    run(
        monkeypatch,
        "report",
        inputs.main,
        inputs.measures,
        out,
        "--continuation",
        inputs.continuation,
        "--recent",
        inputs.recent,
        "--coding",
        inputs.coding,
        "--seed",
        SEED,
    )
    report.main()
    primary = pl.read_csv(out / "primary.csv")
    assert primary["id"].to_list() == [t.id for t in PRIMARY]
    verdicts = by_id(primary, "verdict")
    assert verdicts["P1"] == "real but weak"
    assert verdicts["P2"] == "holds up"
    assert verdicts["P7"] == "holds up"
    assert verdicts["P8"] == "reversed"
    assert verdicts["P9"] == "does not hold up"
    assert verdicts["P10"] == "holds up"
    methods = by_id(primary, "c_method")
    assert methods["P7"] == "haldane-permutation"
    assert methods["P10"] == "gee-gaussian"
    assert methods["P9"] == "gee"
    assert by_id(primary, "effect")["P10"] == pytest.approx(-1.5, abs=0.4)
    assert by_id(primary, "effect")["P8"] == pytest.approx(0.5, abs=0.1)

    secondary = pl.read_csv(out / "secondary.csv")
    assert secondary["id"].to_list() == [t.id for t in SECONDARY] + ["S9a", "S9b"]
    assert by_id(secondary, "verdict")["S3"] == "holds up"
    assert secondary.filter(pl.col("id") == "S9a")["effect"].item() > 2

    per_model = pl.read_csv(out / "per_model_register.csv")
    assert set(per_model["model"].unique()) == set(MAIN_SOURCES[1:])
    assert per_model.filter(pl.col("id").str.starts_with("S8"))[
        "register"
    ].unique().to_list() == ["se"]
    assert (per_model["q"] >= per_model["p"]).all()

    for name, columns in {
        "descriptive.csv": {"register", "source", "n", "per_1000", "share", "id"},
        "robustness.csv": {"id", "check", "effect", "ci95_lo", "ci95_hi", "p"},
        "presence_at_median.csv": {"id", "human:hn", "median_words:se"},
        "p10_too_short.csv": {"register", "source", "share_too_short"},
        "lr_plus.csv": {"id", "scope", "threshold", "lr", "lr_share_infinite"},
        "discrimination.csv": {"score", "auc", "auc_lo", "balanced_accuracy"},
        "claude_arm.csv": {"id", "measure", "ratio_haldane", "claude", "human"},
        "recent_hn.csv": {"id", "recent_share", "baseline_lo", "baseline_n"},
        "continuation_instruct_vs_base.csv": {"id", "verdict", "p_adj"},
        "continuation_instruct_vs_human.csv": {"id", "verdict", "p_adj"},
        "continuation_base_vs_human.csv": {"id", "verdict", "p_adj"},
    }.items():
        assert columns <= set(pl.read_csv(out / name).columns), name
    assert json.loads((out / "run_info.json").read_text())["seed"] == SEED

    descriptive = pl.read_csv(out / "descriptive.csv")
    assert "claude" in descriptive["source"].to_list()
    robustness = pl.read_csv(out / "robustness.csv")
    checks = set(zip(robustness["id"], robustness["check"], strict=True))
    assert {("P1", "variant_p1_within_only"), ("P6", "variant_p6_lastpara_only")} <= (
        checks
    )
    assert ("P9", "nb_gee") in checks
    assert ("P10", "cluster_bootstrap_haldane") not in checks
    auc = pl.read_csv(out / "discrimination.csv").select("score", "auc").rows()
    discrimination: dict[str, float] = dict(auc)
    assert discrimination["character 2-4-gram ceiling"] > 0.9
    assert discrimination["all tells, logistic"] > 0.9
    assert discrimination["P9"] == pytest.approx(0.5, abs=0.1)
    base = pl.read_csv(out / "continuation_base_vs_human.csv")
    assert by_id(base, "verdict")["P2"] != "holds up"
    instruct = pl.read_csv(out / "continuation_instruct_vs_human.csv")
    assert by_id(instruct, "effect")["P8"] == pytest.approx(0.5, abs=0.15)

    run(monkeypatch, "summarize", out)
    summarize.main()
    tables = (out / "results_tables.md").read_text()
    assert "## Primary (Holm)" in tables
    p9_effect = summarize.fmt(primary.filter(pl.col("id") == "P9")["effect"].item())
    assert f"| P9 Nominalizations | rate | ratio {p9_effect} (" in tables
    assert "| P10 Sentence-length variation (CV) | cv | d -" in tables
    assert "| S9a Content: generic, no specifics (coder A) | presence |" in tables
    assert tables.count("| holds up |") >= 4
    assert "| qwen HN |" in tables


def test_frontier_scores_each_model_and_the_pooled_test(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast: None
):
    run(
        monkeypatch,
        "frontier",
        inputs.frontier,
        inputs.measures,
        tmp_path,
        "--models",
        "gpt_a",
        "gpt_b",
        "--main",
        inputs.main,
    )
    frontier.main()
    tells = pl.read_csv(tmp_path / "tells.csv")
    assert tells.height == 2 * len(ALL_TELLS)
    assert set(tells["model"].unique()) == {"gpt_a", "gpt_b"}
    replicating = tells.filter(pl.col("replicates").is_not_null())
    assert set(replicating["id"].unique()) == set(frontier.REPLICATION_TELLS)
    replicated = {
        (row["model"], row["id"])
        for row in replicating.filter(pl.col("replicates")).iter_rows(named=True)
    }
    assert {("gpt_a", "S3"), ("gpt_b", "S3"), ("gpt_a", "P2"), ("gpt_b", "P2")} <= (
        replicated
    )
    assert ("gpt_a", "P3") not in replicated
    per_register = pl.read_csv(tmp_path / "per_register.csv")
    assert set(per_register["model"].unique()) == {"gpt_a", "gpt_b"}
    pooled = pl.read_csv(tmp_path / "pooled_5.csv")
    assert pooled["family"].to_list() == ["primary"] * len(PRIMARY) + [
        "secondary"
    ] * len(SECONDARY)
    assert by_id(pooled, "verdict")["P8"] == "reversed"
    assert by_id(pooled, "verdict")["P2"] == "holds up"


def test_test_family_is_reproducible_with_the_same_seed(inputs: Inputs, fast: None):
    table = pl.read_parquet(inputs.main).filter(pl.col("split") == "confirmatory")
    measures = json.loads(inputs.measures.read_text())
    sparse = tuple(t for t in ALL_TELLS if t.id in {"P7", "S2", "S6"})

    def family(seed: int) -> pl.DataFrame:
        rows = analysis.test_family(table, sparse, measures, MAIN_SOURCES, seed)
        return analysis.finish(rows, "bh")

    first = family(1)
    assert first.equals(family(1))
    assert not first.equals(family(2))


def test_power_simulation_on_tiny_settings(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(power, "N_PROMPTS", 25)
    # Threads stand in for processes so the replicates run inside the test process.
    monkeypatch.setattr(power, "ProcessPoolExecutor", ThreadPoolExecutor)
    out = tmp_path / "power.json"
    run(monkeypatch, "power", inputs.main, inputs.measures, out, "--replicates", 4)
    power.main()
    results = json.loads(out.read_text())
    measures = json.loads(inputs.measures.read_text())
    expected = {t.column for t in PRIMARY if measures[t.id] != "cv"}
    assert set(results) == expected
    assert results["p9_nominal"]["power_ratio_2"] == 1.0
    assert results["p9_nominal"]["estimate_only"] is False
    assert results["p7_worth_noting"]["power_ratio_2"] == 0.0
    assert results["p7_worth_noting"]["base"] == {"hn": 0.005, "se": 0.005}
    assert all(0 <= r["power_ratio_2"] <= 1 for r in results.values())

    exploratory = pl.read_parquet(inputs.main).filter(pl.col("split") == "exploratory")
    setting = next(
        s
        for s in power.build_settings(exploratory, measures)
        if s.column == "p3_tricolon"
    )
    assert power.power(setting, 3, 7) == power.power(setting, 3, 7)
    assert set(setting.words["hn"]) == set(MAIN_SOURCES)


def test_overdispersion_is_zero_for_poisson_like_or_empty_counts():
    exposure = np.ones(4)
    assert power.overdispersion(np.zeros(4), exposure) == 0.0
    assert power.overdispersion(np.array([1.0, 1.0, 1.0, 1.0]), exposure) == 0.0
    spread = power.overdispersion(np.array([0.0, 0.0, 0.0, 8.0]), exposure)
    assert spread == pytest.approx((4 + 4 + 4 + 36 - 8) / 16)
    assert math.isfinite(spread)
