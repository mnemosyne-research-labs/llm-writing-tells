"""Secondary and exploratory analyses: robustness, interaction, discrimination, Claude arm, recent HN."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from llm_tells.analysis import MAIN_SOURCES, PRIMARY, SECONDARY, Tell, frame_for
from llm_tells.stats import (
    Measure,
    bh,
    cells,
    estimate_tell,
    gee_contrast,
    haldane_contrast,
    identity,
    interaction_contrast,
    lr_plus,
    presence_at_median,
    resample,
)

FEATURE_TELLS = [t for t in PRIMARY + SECONDARY if not t.se_only]
N_BOOT = 1000
DASH_QUOTE = str.maketrans(
    {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a proportion.

    Args:
        successes: Count of successes
        n: Trials

    Returns:
        Lower and upper bounds
    """
    if n == 0:
        return float("nan"), float("nan")
    z = float(scipy_stats.norm.ppf(0.975))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def ratio_row(contrast: Any, measure: Measure) -> dict[str, float]:
    """Express a contrast on its reporting scale.

    Args:
        contrast: A Contrast or None
        measure: Its measure

    Returns:
        Effect, 95% interval and p-value
    """
    if contrast is None:
        return {
            "effect": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "p": float("nan"),
        }
    scale: Callable[[float], float] = identity if measure == "cv" else math.exp
    return {
        "effect": scale(contrast.estimate),
        "ci95_lo": scale(contrast.ci95[0]),
        "ci95_hi": scale(contrast.ci95[1]),
        "p": contrast.p,
    }


def safe_estimate(
    frame: pl.DataFrame,
    column: str,
    measure: Measure,
    sources: list[str],
    rng: np.random.Generator,
) -> Any:
    """estimate_tell for exploratory callers: a failed fit gives None instead of stopping the run.

    Args:
        frame: Rows
        column: Tell column
        measure: Measure
        sources: Source order
        rng: Random generator

    Returns:
        The contrast or None
    """
    try:
        return estimate_tell(frame, column, measure, sources, rng)
    except RuntimeError:
        return None


def usefulness(
    data: pl.DataFrame, measures: dict[str, Measure], seed: int
) -> pl.DataFrame:
    """LR+ per register and pooled, at "at least one", "at least two" and "rate above the human 90th percentile".

    Args:
        data: Confirmatory counts
        measures: Measure per tell
        seed: Random seed

    Returns:
        One row per tell, scope and threshold
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for tell in PRIMARY + SECONDARY:
        if measures[tell.id] == "cv":
            continue
        frame = frame_for(data, tell, MAIN_SOURCES).with_columns(
            (pl.col(tell.column) / pl.col("words") * 1000).alias("_rate")
        )
        p90 = (
            frame.filter(pl.col("source") == "human")
            .group_by("register")
            .agg(pl.col("_rate").quantile(0.9).alias("_p90"))
        )
        frame = frame.join(p90, on="register").with_columns(
            ((pl.col("_rate") > pl.col("_p90")) & (pl.col(tell.column) > 0))
            .cast(pl.Int64)
            .alias("_above_p90")
        )
        scopes = {
            "pooled": frame,
            **{
                r: frame.filter(pl.col("register") == r)
                for r in frame["register"].unique().sort()
            },
        }
        for scope, sub in scopes.items():
            for threshold_name, column, threshold in (
                ("at least 1", tell.column, 1),
                ("at least 2", tell.column, 2),
                ("above human P90 rate", "_above_p90", 1),
            ):
                rows.append(
                    {
                        "id": tell.id,
                        "scope": scope,
                        "threshold": threshold_name,
                        **lr_plus(sub, column, MAIN_SOURCES, rng, threshold, N_BOOT),
                    }
                )
    return pl.DataFrame(rows)


def robustness(
    data: pl.DataFrame, measures: dict[str, Measure], seed: int
) -> pl.DataFrame:
    """Pre-registered robustness checks for each primary tell.

    NB-family GEE (alpha by method of moments on human texts), cluster bootstrap of
    the Haldane ratio, P1 without the cross-sentence rule, P6 on the last paragraph
    only, and the HN length floor at 80 words.

    Args:
        data: Confirmatory counts
        measures: Measure per tell
        seed: Random seed

    Returns:
        One row per tell and check
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    long_hn = data.filter((pl.col("register") != "hn") | (pl.col("target_words") >= 80))
    variants = {"P1": "p1_within_only", "P6": "p6_lastpara_only"}
    for tell in PRIMARY:
        measure = measures[tell.id]
        frame = frame_for(data, tell, MAIN_SOURCES)
        checks: dict[str, Any] = {}
        if measure == "rate":
            human = frame.filter(pl.col("source") == "human")
            counts: NDArray[np.float64] = human[tell.column].to_numpy().astype(float)
            words: NDArray[np.float64] = human["words"].to_numpy().astype(float)
            expected = counts.sum() / words.sum() * words
            alpha = max(
                1e-3,
                float(
                    np.sum((counts - expected) ** 2 - expected)
                    / max(np.sum(expected**2), 1e-9)
                ),
            )
            checks["nb_gee"] = gee_contrast(
                frame, tell.column, measure, MAIN_SOURCES, nb_alpha=alpha
            )
        if measure != "cv":
            arrays = cells(frame, tell.column, measure, MAIN_SOURCES)
            replicates = np.array(
                [haldane_contrast(resample(arrays, rng)) for _ in range(N_BOOT)]
            )
            rows.append(
                {
                    "id": tell.id,
                    "check": "cluster_bootstrap_haldane",
                    "effect": math.exp(haldane_contrast(arrays)),
                    "ci95_lo": math.exp(float(np.quantile(replicates, 0.025))),
                    "ci95_hi": math.exp(float(np.quantile(replicates, 0.975))),
                    "p": float("nan"),
                }
            )
        if tell.id in variants:
            variant = Tell(tell.id, variants[tell.id], tell.label, tell.fixed)
            checks[f"variant_{variants[tell.id]}"] = safe_estimate(
                frame_for(data, variant, MAIN_SOURCES),
                variant.column,
                measure,
                MAIN_SOURCES,
                rng,
            )
        checks["hn_floor_80"] = safe_estimate(
            frame_for(long_hn, tell, MAIN_SOURCES),
            tell.column,
            measure,
            MAIN_SOURCES,
            rng,
        )
        checks["interaction_hn_minus_se"] = interaction_contrast(
            frame, tell.column, measure, MAIN_SOURCES
        )
        rows.extend(
            {"id": tell.id, "check": name, **ratio_row(c, measure)}
            for name, c in checks.items()
        )
    return pl.DataFrame(rows)


def presence_table(data: pl.DataFrame, measures: dict[str, Measure]) -> pl.DataFrame:
    """Presence shares at the human median length for the presence-measured tells.

    Args:
        data: Confirmatory counts
        measures: Measure per tell

    Returns:
        One row per tell with a column per cell
    """
    rows: list[dict[str, Any]] = []
    for tell in PRIMARY + SECONDARY:
        if measures[tell.id] != "presence":
            continue
        rows.append(
            {
                "id": tell.id,
                **presence_at_median(
                    frame_for(data, tell, MAIN_SOURCES), tell.column, MAIN_SOURCES
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def p10_nulls(data: pl.DataFrame) -> pl.DataFrame:
    """Share of texts with too few sentences for P10, by source and register.

    Args:
        data: Counts table

    Returns:
        Null shares
    """
    return (
        data.filter(~pl.col("failed"))
        .group_by(["register", "source"])
        .agg(
            pl.len().alias("n"),
            pl.col("p10_cv").is_null().mean().alias("share_too_short"),
        )
        .sort(["register", "source"])
    )


def features(data: pl.DataFrame) -> np.ndarray:
    """Tell features per text: rates per 1,000 words, CV with a missing flag.

    Args:
        data: Counts table

    Returns:
        Feature matrix
    """
    columns = [t.column for t in FEATURE_TELLS if t.column != "p10_cv"]
    end_position = {"p6_closer", "s7_hedge_closer"}
    rates = data.select(
        [
            pl.when(pl.col("truncated") & pl.lit(c in end_position))
            .then(0)
            .otherwise(pl.col(c) / pl.col("words") * 1000)
            .alias(c)
            for c in columns
        ]
    ).to_numpy()
    cv = data["p10_cv"].to_numpy().astype(float)
    return np.column_stack([rates, cv, np.isnan(cv).astype(float)])


def grouped_scores(
    model_factory: Any, x: Any, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Out-of-fold probabilities with folds grouped by prompt.

    Args:
        model_factory: Callable returning a fresh classifier
        x: Features
        y: Labels (1 = model)
        groups: Prompt IDs

    Returns:
        Out-of-fold scores
    """
    scores = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        model = model_factory()
        model.fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return scores


def group_bootstrap_auc(
    y: NDArray[np.int_],
    scores: NDArray[np.float64],
    groups: NDArray[Any],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% group-bootstrap interval for AUC.

    Args:
        y: Labels
        scores: Scores
        groups: Prompt IDs
        rng: Random generator

    Returns:
        Lower and upper bounds
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    members: list[NDArray[np.intp]] = [
        np.flatnonzero(inverse == g) for g in range(len(unique))
    ]
    values: list[float] = []
    for _ in range(N_BOOT):
        picked: NDArray[np.intp] = np.concatenate(
            [members[int(g)] for g in rng.integers(0, len(unique), len(unique))]
        )
        if len(set(y[picked])) == 2:
            values.append(roc_auc_score(y[picked], scores[picked]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def discrimination(data: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Per-tell AUC, the multi-tell classifier and the character n-gram ceiling.

    Args:
        data: Confirmatory counts for the main sources
        seed: Random seed

    Returns:
        One row per score
    """
    rng = np.random.default_rng(seed)
    rows = data.filter(~pl.col("failed")).sort("prompt_id")
    y = (rows["source"] != "human").to_numpy().astype(int)
    groups = rows["prompt_id"].to_numpy()
    results: list[dict[str, Any]] = []
    for tell in FEATURE_TELLS:
        if tell.column == "p10_cv":
            sub = rows.filter(pl.col("p10_cv").is_not_null())
            score, label, group = (
                -sub["p10_cv"].to_numpy(),
                (sub["source"] != "human").to_numpy().astype(int),
                sub["prompt_id"].to_numpy(),
            )
        else:
            score = (rows[tell.column] / rows["words"]).to_numpy()
            label, group = y, groups
        lo, hi = group_bootstrap_auc(label, score, group, rng)
        results.append(
            {
                "score": tell.id,
                "auc": roc_auc_score(label, score),
                "auc_lo": lo,
                "auc_hi": hi,
                "balanced_accuracy": float("nan"),
            }
        )
    x = features(rows)
    logistic = grouped_scores(
        lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        x,
        y,
        groups,
    )
    classifiers = [("all tells, logistic", logistic)]
    # The released counts carry no text (llm_tells.release); the ceiling then needs
    # the human texts restored and re-measured first.
    if "clean_text" in rows.columns:
        texts = np.asarray(
            [t.translate(DASH_QUOTE) for t in rows["clean_text"].to_list()],
            dtype=object,
        )
        ceiling = grouped_scores(
            lambda: make_pipeline(
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True
                ),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
            ),
            texts,
            y,
            groups,
        )
        classifiers.append(("character 2-4-gram ceiling", ceiling))
    for name, scores in classifiers:
        lo, hi = group_bootstrap_auc(y, scores, groups, rng)
        results.append(
            {
                "score": name,
                "auc": roc_auc_score(y, scores),
                "auc_lo": lo,
                "auc_hi": hi,
                "balanced_accuracy": balanced_accuracy_score(y, scores >= 0.5),
            }
        )
    return pl.DataFrame(results)


def claude_arm(
    data: pl.DataFrame, measures: dict[str, Measure], seed: int
) -> pl.DataFrame:
    """Claude texts against the human texts on the same prompts; descriptive, untested.

    Args:
        data: Confirmatory counts including the Claude source
        measures: Measure per tell
        seed: Random seed

    Returns:
        One row per tell
    """
    rng = np.random.default_rng(seed)
    prompts = data.filter(pl.col("source") == "claude")["prompt_id"].unique()
    sub = data.filter(
        pl.col("prompt_id").is_in(prompts.implode())
        & pl.col("source").is_in(["human", "claude"])
    )
    rows: list[dict[str, Any]] = []
    for tell in PRIMARY + SECONDARY:
        measure = measures[tell.id]
        frame = frame_for(sub, tell, ["human", "claude"])
        if measure == "cv":
            values = (
                frame.filter(pl.col("p10_cv").is_not_null())
                .group_by("source")
                .agg(pl.col("p10_cv").mean())
                .sort("source")
            )
            rows.append({"id": tell.id, "measure": measure, **dict(values.iter_rows())})
            continue
        arrays = cells(frame, tell.column, measure, ["human", "claude"])
        replicates = np.array(
            [haldane_contrast(resample(arrays, rng)) for _ in range(N_BOOT)]
        )
        usefulness = lr_plus(
            frame, tell.column, ["human", "claude"], rng, n_boot=N_BOOT
        )
        rows.append(
            {
                "id": tell.id,
                "measure": measure,
                "n_prompts": frame["prompt_id"].n_unique(),
                "ratio_haldane": math.exp(haldane_contrast(arrays)),
                "ci95_lo": math.exp(float(np.quantile(replicates, 0.025))),
                "ci95_hi": math.exp(float(np.quantile(replicates, 0.975))),
                "claude_share": usefulness["hit"],
                "human_share": usefulness["fpr"],
                "lr": usefulness["lr"],
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def recent_hn(recent: pl.DataFrame, baseline: pl.DataFrame) -> pl.DataFrame:
    """False-positive rate of each tell in 2025-2026 HN replies against 2019-2022.

    Args:
        recent: Counts for the recent HN sample (human only)
        baseline: Counts for the 2019-2022 HN human texts (both splits)

    Returns:
        One row per tell with shares and Wilson intervals
    """
    rows: list[dict[str, Any]] = []
    for tell in PRIMARY + SECONDARY:
        if tell.se_only or tell.column == "p10_cv":
            continue
        row: dict[str, Any] = {"id": tell.id}
        for name, frame in (("recent", recent), ("baseline", baseline)):
            hits = int((frame[tell.column] > 0).sum())
            lo, hi = wilson(hits, frame.height)
            row |= {
                f"{name}_share": hits / frame.height,
                f"{name}_lo": lo,
                f"{name}_hi": hi,
                f"{name}_n": frame.height,
            }
        rows.append(row)
    return pl.DataFrame(rows)


def adjust(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """BH-adjust a list of result rows with a `p` column.

    Args:
        rows: Rows

    Returns:
        The rows with `q`
    """
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.with_columns(pl.Series("q", bh(frame["p"].fill_nan(1.0).to_list())))
