import math

import numpy as np
import polars as pl
import pytest

from llm_tells import analysis, extras, stats, summarize
from llm_tells.analysis import Tell
from llm_tells.stats import Cells, Contrast, GeeFit

LOG2 = math.log(2)
DASH = "\N{EN DASH}"


def contrast(
    estimate: float, ci90: tuple[float, float] = (-5.0, 5.0), p: float = 0.01
) -> Contrast:
    return Contrast(estimate, 0.1, ci90, ci90, p, "gee")


def counts_frame(
    rng: np.random.Generator,
    n_prompts: int,
    human_rate: float,
    model_rate: float,
    models: tuple[str, ...] = ("qwen", "gemma"),
    registers: tuple[str, ...] = ("hn", "se"),
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for register in registers:
        for prompt in range(n_prompts):
            for source in ("human", *models):
                words = int(rng.integers(80, 300))
                rate = human_rate if source == "human" else model_rate
                rows.append(
                    {
                        "prompt_id": f"{register}{prompt:03d}",
                        "register": register,
                        "source": source,
                        "words": words,
                        "tell": int(rng.poisson(rate * words / 1000)),
                    }
                )
    return pl.DataFrame(rows)


def test_holm_matches_hand_worked_values():
    adjusted = stats.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_stays_monotone():
    assert stats.holm([0.4, 0.9, 0.5]) == pytest.approx([1.0, 1.0, 1.0])


def test_bh_matches_hand_worked_values():
    assert stats.bh([0.01, 0.04, 0.03, 0.005]) == pytest.approx(
        [0.02, 0.04, 0.04, 0.02]
    )


def test_bh_takes_the_running_minimum_from_the_top():
    assert stats.bh([0.04, 0.05]) == pytest.approx([0.05, 0.05])
    assert stats.bh([1.0, 0.5, 0.01]) == pytest.approx([1.0, 0.75, 0.03])


def test_contrast_vector_averages_models_against_human():
    names = ["human:hn", "qwen:hn", "gemma:hn", "human:se", "qwen:se", "gemma:se"]
    vector = stats.contrast_vector(names, ["human", "qwen", "gemma"])
    assert vector == pytest.approx([-0.5, 0.25, 0.25, -0.5, 0.25, 0.25])
    assert vector.sum() == pytest.approx(0)


def test_design_has_one_indicator_per_cell():
    frame = pl.DataFrame(
        {"source": ["human", "qwen", "qwen"], "register": ["hn", "hn", "se"]}
    )
    matrix, names = stats.design(frame, ["human", "qwen"], ["hn", "se"])
    assert names == ["human:hn", "qwen:hn", "human:se", "qwen:se"]
    assert matrix.tolist() == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


def gee_fit(params: list[float], cov: list[list[float]]) -> GeeFit:
    return GeeFit(
        np.array(params),
        np.array(cov),
        ["human:hn", "qwen:hn"],
        pl.DataFrame(),
        0.0,
    )


def test_wald_pads_covariates_and_builds_intervals():
    fit = gee_fit([0.5, 1.5, 9.0], [[0.04, 0, 0], [0, 0.05, 0], [0, 0, 100.0]])
    result = stats.wald(fit, np.array([-1.0, 1.0]), "gee")
    assert result is not None
    assert result.estimate == pytest.approx(1.0)
    assert result.se == pytest.approx(0.3)
    assert result.ci95 == pytest.approx((1 - stats.Z95 * 0.3, 1 + stats.Z95 * 0.3))
    assert result.ci90 == pytest.approx((1 - stats.Z90 * 0.3, 1 + stats.Z90 * 0.3))
    assert result.p == pytest.approx(2 * (1 - 0.999570940), abs=1e-6)
    assert result.method == "gee"


@pytest.mark.parametrize("variance", [0.0, -1.0, float("nan")])
def test_wald_rejects_degenerate_variance(variance: float):
    fit = gee_fit([0.0, 1.0], [[variance, 0], [0, 0]])
    assert stats.wald(fit, np.array([1.0, 0.0]), "gee") is None


def test_verdict_holds_up_exactly_at_the_thresholds():
    assert stats.verdict(contrast(LOG2), 0.0499, 2.0, "rate") == "holds up"


@pytest.mark.parametrize(
    ("estimate", "adjusted_p", "lr", "expected"),
    [
        (LOG2 - 1e-9, 0.01, 5.0, "real but weak"),
        (LOG2, 0.01, 1.99, "real but weak"),
        (LOG2, 0.05, 5.0, "inconclusive"),
        (-LOG2, 0.01, 5.0, "reversed"),
        (-LOG2, 0.2, 5.0, "inconclusive"),
    ],
)
def test_verdict_rules_below_the_thresholds(
    estimate: float, adjusted_p: float, lr: float, expected: str
):
    assert stats.verdict(contrast(estimate), adjusted_p, lr, "presence") == expected


def test_verdict_band_is_open_at_its_edges():
    inside = (math.log(0.67) + 1e-6, math.log(1.5) - 1e-6)
    edge = (math.log(0.67), math.log(1.5) - 1e-6)
    assert stats.verdict(contrast(0.1, inside), 0.01, 1.0, "rate") == (
        "does not hold up"
    )
    assert stats.verdict(contrast(0.1, edge), 0.01, 1.0, "rate") == "real but weak"


def test_verdict_cv_is_hypothesized_negative():
    assert stats.verdict(contrast(-0.5), 0.01, 2.0, "cv") == "holds up"
    assert stats.verdict(contrast(-0.49), 0.01, 2.0, "cv") == "real but weak"
    assert stats.verdict(contrast(0.6), 0.01, 2.0, "cv") == "reversed"
    assert stats.verdict(contrast(0.0, (-0.19, 0.19)), 0.5, 1.0, "cv") == (
        "does not hold up"
    )
    assert stats.verdict(contrast(0.0, (-0.2, 0.19)), 0.5, 1.0, "cv") == (
        "inconclusive"
    )


def test_haldane_contrast_adds_half_to_each_pooled_count():
    arrays = {
        "hn": Cells(
            counts=np.array([[0.0, 3.0, 1.0], [0.0, 1.0, np.nan]]),
            exposure=np.array([[1.0, 1.0, 2.0], [1.0, 1.0, np.nan]]),
        )
    }
    human, qwen, gemma = 0.5 / 2, 4.5 / 2, 1.5 / 2
    expected = (math.log(qwen / human) + math.log(gemma / human)) / 2
    assert stats.haldane_contrast(arrays) == pytest.approx(expected)


def test_haldane_contrast_averages_over_registers():
    one = Cells(np.array([[1.0, 3.0]]), np.array([[1.0, 1.0]]))
    two = Cells(np.array([[3.0, 1.0]]), np.array([[1.0, 1.0]]))
    assert stats.haldane_contrast({"hn": one, "se": two}) == pytest.approx(0.0)


def sparse_cells(rng: np.random.Generator) -> Cells:
    counts = rng.poisson(2.0, (30, 4)).astype(float)
    exposure = rng.uniform(0.5, 2.0, (30, 4))
    counts[[1, 5, 9], [2, 3, 1]] = np.nan
    exposure[[1, 5, 9], [2, 3, 1]] = np.nan
    return Cells(counts, exposure)


def test_permute_within_rows_keeps_each_prompts_texts_and_gaps():
    cell = sparse_cells(np.random.default_rng(1))
    permuted = stats.permute_within_rows(cell, np.random.default_rng(2))
    assert np.array_equal(np.isnan(permuted.counts), np.isnan(cell.counts))
    for row in range(cell.counts.shape[0]):
        present = ~np.isnan(cell.counts[row])
        before = sorted(
            zip(cell.counts[row, present], cell.exposure[row, present], strict=True)
        )
        after = sorted(
            zip(
                permuted.counts[row, present],
                permuted.exposure[row, present],
                strict=True,
            )
        )
        assert before == after
    assert not np.array_equal(permuted.counts, cell.counts, equal_nan=True)


def test_permute_within_rows_is_reproducible():
    cell = sparse_cells(np.random.default_rng(1))
    first = stats.permute_within_rows(cell, np.random.default_rng(3))
    second = stats.permute_within_rows(cell, np.random.default_rng(3))
    assert np.array_equal(first.counts, second.counts, equal_nan=True)


def test_permutation_p_is_one_without_a_difference():
    cell = Cells(np.ones((10, 3)), np.ones((10, 3)))
    assert stats.permutation_p({"hn": cell}, 99, np.random.default_rng(0)) == 1.0


def test_permutation_p_reaches_its_floor_for_a_planted_effect():
    rng = np.random.default_rng(4)
    counts = np.column_stack([np.zeros(40), rng.poisson(3.0, (40, 2))]).astype(float)
    cell = Cells(counts, np.ones((40, 3)))
    assert stats.permutation_p({"hn": cell}, 199, rng) == pytest.approx(1 / 200)


def test_resample_draws_whole_prompts_within_each_register():
    cell = sparse_cells(np.random.default_rng(5))
    replicate = stats.resample({"hn": cell}, np.random.default_rng(6))["hn"]
    assert replicate.counts.shape == cell.counts.shape
    originals = {tuple(np.nan_to_num(row, nan=-1)) for row in cell.counts}
    assert all(
        tuple(np.nan_to_num(row, nan=-1)) in originals for row in replicate.counts
    )


def test_sparse_contrast_reports_a_bootstrap_interval_and_permutation_p():
    rng = np.random.default_rng(7)
    counts = np.column_stack([np.zeros(30), rng.poisson(1.0, (30, 2))]).astype(float)
    arrays = {"hn": Cells(counts, np.ones((30, 3)))}
    result = stats.sparse_contrast(arrays, rng, n_boot=200, n_perm=199)
    assert result.method == "haldane-permutation"
    assert math.isnan(result.se)
    assert result.estimate == pytest.approx(stats.haldane_contrast(arrays))
    assert result.ci95[0] <= result.ci90[0] <= result.ci90[1] <= result.ci95[1]
    assert result.ci95[0] > 0
    assert result.p == pytest.approx(1 / 200)


def test_cells_pivots_by_prompt_and_fills_missing_sources():
    frame = pl.DataFrame(
        {
            "register": ["hn", "hn", "hn", "se"],
            "prompt_id": ["b", "a", "a", "c"],
            "source": ["human", "human", "qwen", "human"],
            "words": [500, 1000, 2000, 250],
            "tell": [0, 2, 3, 1],
        }
    )
    rate = stats.cells(frame, "tell", "rate", ["human", "qwen"])
    assert np.array_equal(
        rate["hn"].counts, np.array([[2.0, 3.0], [0.0, np.nan]]), equal_nan=True
    )
    assert np.array_equal(
        rate["hn"].exposure, np.array([[1.0, 2.0], [0.5, np.nan]]), equal_nan=True
    )
    assert np.array_equal(rate["se"].counts, np.array([[1.0, np.nan]]), equal_nan=True)
    presence = stats.cells(frame, "tell", "presence", ["human", "qwen"])
    assert np.array_equal(
        presence["hn"].counts, np.array([[1.0, 1.0], [0.0, np.nan]]), equal_nan=True
    )
    assert np.array_equal(
        presence["hn"].exposure, np.array([[1.0, 1.0], [1.0, np.nan]]), equal_nan=True
    )


def test_human_total_counts_occurrences_or_texts():
    frame = pl.DataFrame(
        {"source": ["human", "human", "human", "qwen"], "tell": [0, 3, 2, 9]}
    )
    assert stats.human_total(frame, "tell", "rate") == 5
    assert stats.human_total(frame, "tell", "presence") == 2
    assert stats.human_total(frame, "tell", "rate", "qwen") == 9


def test_gee_contrast_recovers_a_planted_ratio():
    frame = counts_frame(np.random.default_rng(8), 60, 10.0, 30.0)
    result = stats.gee_contrast(frame, "tell", "rate", ["human", "qwen", "gemma"])
    assert result is not None
    assert result.method == "gee"
    assert result.ci95[0] < math.log(3) < result.ci95[1]
    assert result.p < 1e-6


def test_gee_contrast_nb_sensitivity_is_labelled():
    frame = counts_frame(np.random.default_rng(8), 60, 10.0, 30.0)
    result = stats.gee_contrast(
        frame, "tell", "rate", ["human", "qwen", "gemma"], nb_alpha=0.2
    )
    assert result is not None
    assert result.method == "gee-nb"
    assert result.estimate == pytest.approx(math.log(3), abs=0.2)


def test_fit_gee_refuses_an_empty_cell():
    frame = counts_frame(np.random.default_rng(9), 20, 0.0, 10.0)
    assert stats.fit_gee(frame, "tell", "rate", ["human", "qwen", "gemma"]) is None


def test_estimate_tell_switches_to_the_sparse_fallback():
    rng = np.random.default_rng(10)
    frame = counts_frame(rng, 30, 0.0, 10.0)
    sparse = stats.estimate_tell(frame, "tell", "rate", ["human", "qwen"], rng)
    assert sparse.method == "haldane-permutation"
    dense = counts_frame(rng, 30, 10.0, 10.0)
    assert stats.human_total(dense, "tell", "rate") >= stats.MIN_HUMAN_COUNT
    assert stats.estimate_tell(
        dense, "tell", "rate", ["human", "qwen"], rng
    ).method == ("gee")


def test_estimate_tell_stops_when_the_cv_fit_fails(monkeypatch: pytest.MonkeyPatch):
    def no_fit(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(stats, "gee_contrast", no_fit)
    frame = counts_frame(np.random.default_rng(11), 5, 1.0, 1.0)
    with pytest.raises(RuntimeError, match="Gaussian GEE failed for tell"):
        stats.estimate_tell(
            frame, "tell", "cv", ["human", "qwen"], np.random.default_rng(0)
        )


def cv_frame(rng: np.random.Generator, shift: dict[str, float]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for register in ("hn", "se"):
        for prompt in range(50):
            for source in ("human", "qwen"):
                offset = 0.0 if source == "human" else shift[register]
                rows.append(
                    {
                        "prompt_id": f"{register}{prompt}",
                        "register": register,
                        "source": source,
                        "words": 200,
                        "cv": float(rng.normal(0.5 + offset, 0.1)),
                    }
                )
    return pl.DataFrame(rows)


def test_cv_contrast_is_a_standardized_difference():
    frame = cv_frame(np.random.default_rng(12), {"hn": -0.1, "se": -0.1})
    result = stats.gee_contrast(frame, "cv", "cv", ["human", "qwen"])
    assert result is not None
    assert result.method == "gee-gaussian"
    assert result.estimate == pytest.approx(-1.0, abs=0.3)


def test_interaction_contrast_is_hn_minus_se():
    frame = cv_frame(np.random.default_rng(13), {"hn": -0.1, "se": 0.1})
    result = stats.interaction_contrast(frame, "cv", "cv", ["human", "qwen"])
    assert result is not None
    assert result.method == "gee-interaction"
    assert result.estimate == pytest.approx(-2.0, abs=0.5)
    rates = counts_frame(np.random.default_rng(14), 40, 10.0, 20.0)
    rate_result = stats.interaction_contrast(rates, "tell", "rate", ["human", "qwen"])
    assert rate_result is not None
    assert abs(rate_result.estimate) < 3 * rate_result.se


def test_interaction_contrast_needs_two_registers():
    frame = counts_frame(np.random.default_rng(15), 30, 10.0, 20.0, registers=("hn",))
    assert stats.interaction_contrast(frame, "tell", "rate", ["human", "qwen"]) is None


def test_presence_at_median_reports_shares_and_lengths():
    frame = counts_frame(np.random.default_rng(16), 60, 2.0, 6.0)
    shares = stats.presence_at_median(frame, "tell", ["human", "qwen", "gemma"])
    human = frame.filter(pl.col("source") == "human")
    for register in ("hn", "se"):
        median = human.filter(pl.col("register") == register)["words"].median()
        assert shares[f"median_words:{register}"] == median
        assert 0 < shares[f"human:{register}"] < shares[f"qwen:{register}"] <= 1


def test_presence_at_median_is_empty_when_the_fit_fails():
    frame = counts_frame(np.random.default_rng(17), 20, 0.0, 6.0)
    assert stats.presence_at_median(frame, "tell", ["human", "qwen"]) == {}


def test_lr_plus_matches_hand_counted_shares():
    frame = pl.DataFrame(
        {
            "register": ["hn"] * 8,
            "prompt_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "source": ["human", "qwen"] * 4,
            "words": [100] * 8,
            "tell": [1, 2, 0, 1, 0, 1, 0, 0],
        }
    )
    result = stats.lr_plus(
        frame, "tell", ["human", "qwen"], np.random.default_rng(0), n_boot=50
    )
    assert result["hit"] == pytest.approx(0.75)
    assert result["fpr"] == pytest.approx(0.25)
    assert result["lr"] == pytest.approx(3.0)
    assert result["hit_lo"] <= result["hit"] <= result["hit_hi"]
    assert result["lr_hi"] == math.inf
    assert 0 < result["lr_share_infinite"] < 1
    strict = stats.lr_plus(
        frame,
        "tell",
        ["human", "qwen"],
        np.random.default_rng(0),
        threshold=2,
        n_boot=50,
    )
    assert strict["hit"] == pytest.approx(0.25)
    assert strict["lr"] == math.inf


def test_bound_maps_the_stand_in_to_infinity():
    assert stats.bound(1e9) == math.inf
    assert stats.bound(3.5) == 3.5


def test_identity_returns_its_argument():
    assert stats.identity(-0.25) == -0.25


def human_rows(values: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"source": ["human"] * len(values) + ["qwen"], "tell": [*values, 100]}
    )


def test_switch_rule_keeps_rate_at_both_thresholds():
    values = [20] + [0] * 19
    tell = (Tell("X", "tell", "x"),)
    assert analysis.switch_rule(human_rows(values), tell) == {"X": "rate"}


@pytest.mark.parametrize(
    "values",
    [[19] + [0] * 19, [20] + [0] * 20],
    ids=["total below 20", "prevalence below 5%"],
)
def test_switch_rule_chooses_presence_below_a_threshold(values: list[int]):
    tell = (Tell("X", "tell", "x"),)
    assert analysis.switch_rule(human_rows(values), tell) == {"X": "presence"}


def test_switch_rule_respects_fixed_measures():
    tells = (Tell("X", "tell", "x", fixed="rate"), Tell("Y", "tell", "y", fixed="cv"))
    assert analysis.switch_rule(human_rows([0] * 5), tells) == {"X": "rate", "Y": "cv"}


def test_frame_for_applies_scope_and_truncation():
    table = pl.DataFrame(
        {
            "source": ["human", "qwen", "qwen", "gemma", "human"],
            "register": ["hn", "se", "se", "se", "se"],
            "failed": [False, False, True, False, False],
            "truncated": [None, True, False, False, False],
        }
    )
    sources = ["human", "qwen"]
    assert analysis.frame_for(table, Tell("X", "x", "x"), sources).height == 3
    se_only = Tell("X", "x", "x", se_only=True)
    assert analysis.frame_for(table, se_only, sources).height == 2
    closer = Tell("P6", "p6_closer", "x")
    kept = analysis.frame_for(table, closer, sources)
    assert kept["register"].to_list() == ["hn", "se"]


def test_finish_adjusts_and_scales_each_family():
    results = [
        {
            "id": tell_id,
            "measure": "rate",
            "lr": 3.0,
            "c_p": p,
            "c_ci95": None,
            "_contrast": Contrast(LOG2 * 2, 0.1, (1.0, 1.8), (1.1, 1.7), p, "gee"),
        }
        for tell_id, p in (("A", 0.001), ("B", 0.03))
    ]
    holm = analysis.finish([dict(r) for r in results], "holm")
    assert holm["p_adj"].to_list() == pytest.approx([0.002, 0.03])
    assert holm["effect"].to_list() == pytest.approx([4.0, 4.0])
    assert holm["ci90_lo"].to_list() == pytest.approx([math.exp(1.1)] * 2)
    assert "c_ci95" not in holm.columns
    assert holm["verdict"].to_list() == ["holds up", "holds up"]
    bh = analysis.finish([dict(r) for r in results], "bh")
    assert bh["p_adj"].to_list() == pytest.approx([0.002, 0.03])


def test_wilson_interval():
    assert all(math.isnan(v) for v in extras.wilson(0, 0))
    lo, hi = extras.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = extras.wilson(5, 10)
    assert lo + hi == pytest.approx(1.0)


def test_ratio_row_handles_missing_contrasts():
    missing = extras.ratio_row(None, "rate")
    assert all(math.isnan(v) for v in missing.values())
    cv = extras.ratio_row(contrast(-0.4), "cv")
    assert cv["effect"] == -0.4
    rate = extras.ratio_row(contrast(LOG2), "rate")
    assert rate["effect"] == pytest.approx(2.0)


def test_adjust_treats_missing_p_values_as_one():
    frame = extras.adjust([{"p": 0.01}, {"p": float("nan")}, {"p": 0.02}])
    assert frame["q"].to_list() == pytest.approx([0.03, 1.0, 0.03])


def test_summarize_formatters():
    assert summarize.fmt(None) == DASH
    assert summarize.fmt(float("nan")) == DASH
    assert summarize.fmt(math.inf) == "\N{INFINITY}"
    assert summarize.fmt(2.345, 1) == "2.3"
    assert summarize.pct(None) == DASH
    assert summarize.pct(0.1234) == "12.3%"
    assert summarize.p_value(0.0004) == "< 0.001"
    assert summarize.p_value(0.0123) == "0.012"
