"""Estimators for the pre-registered analysis: GEE contrasts, sparse fallbacks, LR+ and verdicts."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import stats as scipy_stats
from statsmodels.genmod.cov_struct import Exchangeable

Measure = Literal["rate", "presence", "cv"]
Z95, Z90 = float(scipy_stats.norm.ppf(0.975)), float(scipy_stats.norm.ppf(0.95))
SESOI_RATIO, SESOI_LR, SESOI_D = 2.0, 2.0, 0.5
BAND_RATIO, BAND_D = (0.67, 1.5), (-0.2, 0.2)
MIN_HUMAN_COUNT = 5


@dataclass(frozen=True, slots=True)
class Contrast:
    """An averaged model-versus-human contrast.

    Attributes:
        estimate: Average log ratio (or standardized difference for CV)
        se: Standard error, or NaN for the bootstrap fallback
        ci95: 95% interval on the same scale
        ci90: 90% interval on the same scale
        p: Two-sided p-value
        method: "gee", "haldane-permutation" or "gee-gaussian"
    """

    estimate: float
    se: float
    ci95: tuple[float, float]
    ci90: tuple[float, float]
    p: float
    method: str


@dataclass(frozen=True, slots=True)
class Cells:
    """Prompt-by-source arrays for one register.

    Attributes:
        counts: Counts (or 0/1 presence) per prompt and source, NaN where missing
        exposure: Words / 1,000 (rate) or 1 (presence), NaN where missing
    """

    counts: NDArray[np.float64]
    exposure: NDArray[np.float64]


def identity(x: float) -> float:
    """Return the value unchanged (the scale for a standardized difference).

    Args:
        x: Value

    Returns:
        The same value
    """
    return x


def cells(
    frame: pl.DataFrame, column: str, measure: Measure, sources: list[str]
) -> dict[str, Cells]:
    """Pivot one tell into prompt-by-source arrays per register.

    Args:
        frame: Counts table (one row per text)
        column: Tell column
        measure: "rate" or "presence"
        sources: Source order; the first is the reference (human)

    Returns:
        Arrays per register
    """
    value = (
        pl.col(column) if measure == "rate" else (pl.col(column) > 0).cast(pl.Float64)
    )
    exposure = pl.col("words") / 1000 if measure == "rate" else pl.lit(1.0)
    long = frame.select(
        "register",
        "prompt_id",
        "source",
        value.alias("value"),
        exposure.alias("exposure"),
    )
    result: dict[str, Cells] = {}
    for register in sorted(frame["register"].unique()):
        sub = long.filter(pl.col("register") == register)
        wide_value = sub.pivot(on="source", index="prompt_id", values="value").sort(
            "prompt_id"
        )
        wide_exposure = sub.pivot(
            on="source", index="prompt_id", values="exposure"
        ).sort("prompt_id")
        result[register] = Cells(
            counts=wide_value.select(
                [
                    pl.col(s) if s in wide_value.columns else pl.lit(None).alias(s)
                    for s in sources
                ]
            )
            .to_numpy()
            .astype(float),
            exposure=wide_exposure.select(
                [
                    pl.col(s) if s in wide_exposure.columns else pl.lit(None).alias(s)
                    for s in sources
                ]
            )
            .to_numpy()
            .astype(float),
        )
    return result


def haldane_contrast(arrays: dict[str, Cells]) -> float:
    """Average log ratio of pooled rates, with 0.5 added to each cell's count.

    Args:
        arrays: Prompt-by-source arrays per register (column 0 is human)

    Returns:
        The average over models and registers of log(model rate / human rate)
    """
    terms: list[float] = []
    for cell in arrays.values():
        totals = np.nansum(cell.counts, axis=0) + 0.5
        exposure = np.nansum(cell.exposure, axis=0)
        log_rates = np.log(totals / exposure)
        terms.extend(log_rates[1:] - log_rates[0])
    return float(np.mean(terms))


def resample(arrays: dict[str, Cells], rng: np.random.Generator) -> dict[str, Cells]:
    """Resample prompts with replacement within each register.

    Args:
        arrays: Prompt-by-source arrays per register
        rng: Random generator

    Returns:
        A bootstrap replicate
    """
    replicate: dict[str, Cells] = {}
    for register, cell in arrays.items():
        rows = rng.integers(0, cell.counts.shape[0], cell.counts.shape[0])
        replicate[register] = Cells(cell.counts[rows], cell.exposure[rows])
    return replicate


def permute_within_rows(cell: Cells, rng: np.random.Generator) -> Cells:
    """Shuffle source labels within each prompt among the texts that exist.

    Missing texts stay missing in their own column, so every prompt is used and the
    permuted statistic is computed on the same texts as the reported estimate.

    Args:
        cell: Prompt-by-source arrays for one register
        rng: Random generator

    Returns:
        The permuted arrays
    """
    missing = np.isnan(cell.counts) | np.isnan(cell.exposure)
    keys = np.where(missing, np.inf, rng.random(cell.counts.shape))
    source_order = np.argsort(keys, axis=1, kind="stable")
    target_order = np.argsort(missing, axis=1, kind="stable")
    rows = np.arange(int(cell.counts.shape[0])).reshape(-1, 1)
    counts = np.empty_like(cell.counts)
    exposure = np.empty_like(cell.exposure)
    counts[rows, target_order] = cell.counts[rows, source_order]
    exposure[rows, target_order] = cell.exposure[rows, source_order]
    return Cells(counts, exposure)


def permutation_p(
    arrays: dict[str, Cells], n_permutations: int, rng: np.random.Generator
) -> float:
    """Within-prompt permutation test of the Haldane contrast, on every prompt.

    Args:
        arrays: Prompt-by-source arrays per register
        n_permutations: Number of permutations
        rng: Random generator

    Returns:
        Two-sided permutation p-value (with the +1 correction)
    """
    observed = abs(haldane_contrast(arrays))
    exceed = 0
    for _ in range(n_permutations):
        permuted = {
            register: permute_within_rows(cell, rng)
            for register, cell in arrays.items()
        }
        exceed += abs(haldane_contrast(permuted)) >= observed - 1e-12
    return (exceed + 1) / (n_permutations + 1)


def sparse_contrast(
    arrays: dict[str, Cells],
    rng: np.random.Generator,
    n_boot: int = 2000,
    n_perm: int = 5000,
) -> Contrast:
    """The pre-registered fallback for sparse tells or failed GEE fits.

    Args:
        arrays: Prompt-by-source arrays per register
        rng: Random generator
        n_boot: Bootstrap replicates
        n_perm: Permutations

    Returns:
        The Haldane contrast with a cluster-bootstrap interval and a permutation p-value
    """
    estimate = haldane_contrast(arrays)
    replicates = np.array(
        [haldane_contrast(resample(arrays, rng)) for _ in range(n_boot)]
    )
    return Contrast(
        estimate=estimate,
        se=float("nan"),
        ci95=(
            float(np.quantile(replicates, 0.025)),
            float(np.quantile(replicates, 0.975)),
        ),
        ci90=(
            float(np.quantile(replicates, 0.05)),
            float(np.quantile(replicates, 0.95)),
        ),
        p=permutation_p(arrays, n_perm, rng),
        method="haldane-permutation",
    )


def design(
    frame: pl.DataFrame, sources: list[str], registers: list[str]
) -> tuple[NDArray[np.float64], list[str]]:
    """Cell-means design matrix (one column per source and register).

    Args:
        frame: Rows to model
        sources: Source order
        registers: Register order

    Returns:
        The matrix and its column names ("source:register")
    """
    names = [f"{s}:{r}" for r in registers for s in sources]
    cell = (frame["source"] + ":" + frame["register"]).to_numpy()
    return np.column_stack([(cell == name).astype(float) for name in names]), names


def contrast_vector(names: list[str], sources: list[str]) -> NDArray[np.float64]:
    """Weights for the average of (model - human) over models and registers.

    Args:
        names: Design column names
        sources: Source order; the first is human

    Returns:
        The contrast weights over the cell columns
    """
    models = sources[1:]
    registers = sorted({name.split(":")[1] for name in names})
    weight = 1 / (len(models) * len(registers))
    vector = np.zeros(len(names))
    for index, name in enumerate(names):
        source = name.split(":")[0]
        vector[index] = weight if source in models else -weight * len(models)
    return vector


@dataclass(frozen=True, slots=True)
class GeeFit:
    """A fitted cell-means GEE.

    Attributes:
        params: Coefficients (cells first, then covariates)
        cov: Robust covariance
        names: Cell names ("source:register")
        data: Rows used, sorted by prompt
        log_words_mean: Centring constant of the log-words covariate (presence only)
    """

    params: NDArray[np.float64]
    cov: NDArray[np.float64]
    names: list[str]
    data: pl.DataFrame
    log_words_mean: float


def fit_gee(
    frame: pl.DataFrame,
    column: str,
    measure: Measure,
    sources: list[str],
    nb_alpha: float | None = None,
) -> GeeFit | None:
    """Fit the pre-registered GEE for one tell.

    Poisson (or, as a sensitivity, NB with fixed alpha) for rates with a log
    word-count offset; modified Poisson with centred log words for presence;
    Gaussian for CV. Cell means by source and register, exchangeable within prompt.

    Args:
        frame: Counts table restricted to `sources`
        column: Tell column
        measure: "rate", "presence" or "cv"
        sources: Source order; the first is human
        nb_alpha: Use NB-family GEE with this alpha (rates only)

    Returns:
        The fit, or None if it fails or a cell has no occurrences
    """
    data = frame.filter(pl.col(column).is_not_null()) if measure == "cv" else frame
    data = data.sort("prompt_id")
    registers = sorted(data["register"].unique().to_list())
    exog, names = design(data, sources, registers)
    groups = data["prompt_id"].to_numpy()
    log_words_mean = 0.0
    if measure == "cv":
        endog = data[column].to_numpy().astype(float)
        family: sm.families.Family = sm.families.Gaussian()
        offset = None
    else:
        raw = data[column].to_numpy().astype(float)
        endog = raw if measure == "rate" else (raw > 0).astype(float)
        cell_totals = np.array(
            [endog[exog[:, j] == 1].sum() for j in range(exog.shape[1])]
        )
        if (cell_totals == 0).any():
            return None
        family = (
            sm.families.NegativeBinomial(alpha=nb_alpha)
            if nb_alpha is not None and measure == "rate"
            else sm.families.Poisson()
        )
        words = data["words"].to_numpy().astype(float)
        if measure == "rate":
            offset = np.log(words / 1000)
        else:
            offset = None
            log_words = np.log(words)
            log_words_mean = float(log_words.mean())
            exog = np.column_stack([exog, log_words - log_words_mean])
    try:
        fit = sm.GEE(
            endog,
            exog,
            groups=groups,
            family=family,
            cov_struct=Exchangeable(),
            offset=offset,
        ).fit(maxiter=100)
    except ValueError, np.linalg.LinAlgError:
        return None
    if not getattr(fit, "converged", True):
        return None
    return GeeFit(
        np.asarray(fit.params),
        np.asarray(fit.cov_params()),
        names,
        data,
        log_words_mean,
    )


def wald(fit: GeeFit, vector: NDArray[np.float64], method: str) -> Contrast | None:
    """Wald estimate, intervals and p-value for a linear contrast.

    Args:
        fit: The GEE fit
        vector: Contrast weights over the cell columns (covariates get 0)
        method: Label for the result

    Returns:
        The contrast, or None if its standard error is degenerate
    """
    full = np.concatenate([vector, np.zeros(len(fit.params) - len(vector))])
    estimate = float(full @ fit.params)
    variance = float(full @ fit.cov @ full)
    if not math.isfinite(variance) or variance <= 0:
        return None
    se = math.sqrt(variance)
    return Contrast(
        estimate=estimate,
        se=se,
        ci95=(estimate - Z95 * se, estimate + Z95 * se),
        ci90=(estimate - Z90 * se, estimate + Z90 * se),
        p=float(2 * scipy_stats.norm.sf(abs(estimate / se))),
        method=method,
    )


def cv_scaled(
    fit: GeeFit, column: str, sources: list[str], vector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Divide each register's weights by that register's human SD (standardized differences).

    Args:
        fit: The Gaussian GEE fit
        column: CV column
        sources: Source order; the first is human
        vector: Contrast weights over the cells

    Returns:
        The scaled weights
    """
    human = fit.data.filter(pl.col("source") == sources[0])
    sd = dict(human.group_by("register").agg(pl.col(column).std()).iter_rows())
    return vector / np.array([sd[name.split(":")[1]] for name in fit.names])


def gee_contrast(
    frame: pl.DataFrame,
    column: str,
    measure: Measure,
    sources: list[str],
    nb_alpha: float | None = None,
) -> Contrast | None:
    """Fit the pre-registered GEE and return the averaged model-versus-human contrast.

    Args:
        frame: Counts table restricted to `sources`
        column: Tell column
        measure: "rate", "presence" or "cv"
        sources: Source order; the first is human
        nb_alpha: NB-family alpha for the sensitivity analysis

    Returns:
        The contrast, or None if the fit fails or a cell has no occurrences
    """
    fit = fit_gee(frame, column, measure, sources, nb_alpha)
    if fit is None:
        return None
    vector = contrast_vector(fit.names, sources)
    if measure == "cv":
        vector = cv_scaled(fit, column, sources, vector)
    return wald(
        fit,
        vector,
        "gee-gaussian" if measure == "cv" else "gee-nb" if nb_alpha else "gee",
    )


def interaction_contrast(
    frame: pl.DataFrame, column: str, measure: Measure, sources: list[str]
) -> Contrast | None:
    """Register by source interaction: the HN model-human log ratio minus the SE one.

    Args:
        frame: Counts table restricted to `sources`
        column: Tell column
        measure: "rate", "presence" or "cv"
        sources: Source order; the first is human

    Returns:
        The interaction contrast, or None if the fit fails
    """
    fit = fit_gee(frame, column, measure, sources)
    if fit is None or len({n.split(":")[1] for n in fit.names}) < 2:
        return None
    models = sources[1:]
    vector = np.array(
        [
            (1 if name.split(":")[1] == "hn" else -1)
            * (1 / len(models) if name.split(":")[0] in models else -1.0)
            for name in fit.names
        ]
    )
    if measure == "cv":
        vector = cv_scaled(fit, column, sources, vector)
    return wald(fit, vector, "gee-interaction")


def presence_at_median(
    frame: pl.DataFrame, column: str, sources: list[str]
) -> dict[str, float]:
    """Predicted presence share per source and register at the human median length.

    Args:
        frame: Counts table restricted to `sources`
        column: Tell column
        sources: Source order; the first is human

    Returns:
        Share per "source:register" key, plus the human median words per register
    """
    fit = fit_gee(frame, column, "presence", sources)
    if fit is None:
        return {}
    medians = dict(
        fit.data.filter(pl.col("source") == sources[0])
        .group_by("register")
        .agg(pl.col("words").median())
        .sort("register")
        .iter_rows()
    )
    slope = fit.params[-1]
    shares = {
        name: float(
            min(
                1.0,
                math.exp(
                    fit.params[i]
                    + slope
                    * (math.log(medians[name.split(":")[1]]) - fit.log_words_mean)
                ),
            )
        )
        for i, name in enumerate(fit.names)
    }
    return shares | {f"median_words:{r}": float(m) for r, m in medians.items()}


def human_total(
    frame: pl.DataFrame, column: str, measure: Measure, reference: str = "human"
) -> float:
    """Total occurrences in the reference source (or texts with the tell, for presence).

    Args:
        frame: Counts table
        column: Tell column
        measure: "rate" or "presence"
        reference: Reference source

    Returns:
        The reference total
    """
    human = frame.filter(pl.col("source") == reference)[column]
    return float(human.sum() if measure == "rate" else (human > 0).sum())


def estimate_tell(
    frame: pl.DataFrame,
    column: str,
    measure: Measure,
    sources: list[str],
    rng: np.random.Generator,
) -> Contrast:
    """Primary estimator with the pre-registered fallback.

    Args:
        frame: Counts table restricted to `sources`
        column: Tell column
        measure: "rate", "presence" or "cv"
        sources: Source order; the first is human
        rng: Random generator for the fallback

    Returns:
        The contrast
    """
    if measure == "cv":
        result = gee_contrast(frame, column, measure, sources)
        if result is None:
            raise RuntimeError(f"Gaussian GEE failed for {column}")
        return result
    if human_total(frame, column, measure, sources[0]) >= MIN_HUMAN_COUNT and (
        result := gee_contrast(frame, column, measure, sources)
    ):
        return result
    return sparse_contrast(cells(frame, column, measure, sources), rng)


def lr_plus(
    frame: pl.DataFrame,
    column: str,
    sources: list[str],
    rng: np.random.Generator,
    threshold: float = 1,
    n_boot: int = 2000,
) -> dict[str, float]:
    """Hit rate, false-positive rate and LR+ with cluster-bootstrap intervals.

    Pooled with equal weights: hit rate averaged over models and registers,
    false-positive rate averaged over registers.

    Args:
        frame: Counts table restricted to `sources`
        column: Tell column (or a 0/1 indicator column)
        sources: Source order; the first is human
        rng: Random generator
        threshold: A text is flagged when the column is at least this value
        n_boot: Bootstrap replicates

    Returns:
        Point estimates and 95% intervals for hit rate, FPR and LR+
    """
    flagged = frame.with_columns(
        (pl.col(column) >= threshold).cast(pl.Float64).alias("_flag")
    )
    arrays = cells(flagged, "_flag", "rate", sources)
    arrays = {
        r: Cells(c.counts, np.where(np.isnan(c.counts), np.nan, 1.0))
        for r, c in arrays.items()
    }

    def point(sample: dict[str, Cells]) -> tuple[float, float, float]:
        shares = np.array([np.nanmean(cell.counts, axis=0) for cell in sample.values()])
        hit = float(shares[:, 1:].mean())
        fpr = float(shares[:, 0].mean())
        return hit, fpr, hit / fpr if fpr > 0 else float("inf")

    hit, fpr, ratio = point(arrays)
    replicates = np.array([point(resample(arrays, rng)) for _ in range(n_boot)])
    finite = replicates[np.isfinite(replicates[:, 2]), 2]
    return {
        "hit": hit,
        "hit_lo": float(np.quantile(replicates[:, 0], 0.025)),
        "hit_hi": float(np.quantile(replicates[:, 0], 0.975)),
        "fpr": fpr,
        "fpr_lo": float(np.quantile(replicates[:, 1], 0.025)),
        "fpr_hi": float(np.quantile(replicates[:, 1], 0.975)),
        "lr": ratio,
        "lr_lo": bound(
            float(
                np.quantile(np.nan_to_num(replicates[:, 2], nan=0.0, posinf=1e9), 0.025)
            )
        ),
        "lr_hi": bound(
            float(
                np.quantile(np.nan_to_num(replicates[:, 2], nan=0.0, posinf=1e9), 0.975)
            )
        ),
        "lr_share_infinite": float(1 - finite.size / len(replicates)),
    }


def bound(value: float) -> float:
    """Report bootstrap bounds at the 1e9 stand-in as infinite.

    Args:
        value: A bound

    Returns:
        The bound, or infinity
    """
    return float("inf") if value >= 1e9 else value


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values.

    Args:
        p_values: Raw p-values

    Returns:
        Adjusted p-values in the input order
    """
    order: NDArray[np.intp] = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[int(index)]))
        adjusted[index] = running
    return adjusted.tolist()


def bh(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values.

    Args:
        p_values: Raw p-values

    Returns:
        q-values in the input order
    """
    m = len(p_values)
    order: NDArray[np.intp] = np.argsort(p_values)[::-1]
    q = np.empty(m)
    running = 1.0
    for position, index in enumerate(order):
        rank = m - position
        running = min(running, p_values[int(index)] * m / rank)
        q[index] = running
    return q.tolist()


def verdict(contrast: Contrast, adjusted_p: float, lr: float, measure: Measure) -> str:
    """Apply the pre-registered verdict rules in order.

    Args:
        contrast: The averaged contrast
        adjusted_p: Holm (primary) or BH (secondary) adjusted p-value
        lr: Point LR+
        measure: "rate", "presence" or "cv"

    Returns:
        One of: holds up, does not hold up, real but weak, reversed, inconclusive
    """
    hypothesized = contrast.estimate < 0 if measure == "cv" else contrast.estimate > 0
    significant = adjusted_p < 0.05
    if measure == "cv":
        big = contrast.estimate <= -SESOI_D
        in_band = BAND_D[0] < contrast.ci90[0] and contrast.ci90[1] < BAND_D[1]
    else:
        big = math.exp(contrast.estimate) >= SESOI_RATIO
        in_band = math.log(BAND_RATIO[0]) < contrast.ci90[0] and contrast.ci90[
            1
        ] < math.log(BAND_RATIO[1])
    if significant and hypothesized and big and lr >= SESOI_LR:
        return "holds up"
    if in_band:
        return "does not hold up"
    if significant and hypothesized:
        return "real but weak"
    if significant:
        return "reversed"
    return "inconclusive"
