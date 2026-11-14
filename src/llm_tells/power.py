"""Power simulation for the primary tells, from exploratory-split rates (pre-registration 6.6)."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from llm_tells.analysis import MAIN_SOURCES, PRIMARY
from llm_tells.stats import gee_contrast

N_PROMPTS = 300
RATIO = 2.0
ALPHA = 0.005
PROMPT_SHARE = 0.5


@dataclass(frozen=True, slots=True)
class Setting:
    """Inputs for one tell's simulation, per register.

    Attributes:
        column: Tell column
        measure: "rate" or "presence"
        base: Human rate per 1,000 words (rate) or prevalence (presence), per register
        inverse_k: Extra-Poisson variance of the human counts (1 / NB k), per register
        words: Observed word counts per register and source, resampled for exposure
    """

    column: str
    measure: str
    base: dict[str, float]
    inverse_k: dict[str, float]
    words: dict[str, dict[str, list[int]]]


def overdispersion(counts: NDArray[np.float64], exposure: NDArray[np.float64]) -> float:
    """Method-of-moments 1/k for counts with exposure.

    Args:
        counts: Counts per text
        exposure: Words / 1,000 per text

    Returns:
        1/k, at least 0
    """
    rate = counts.sum() / exposure.sum()
    expected = rate * exposure
    if expected.sum() == 0:
        return 0.0
    return max(
        0.0, float(np.sum((counts - expected) ** 2 - expected) / np.sum(expected**2))
    )


def simulate_once(setting: Setting, seed: int) -> bool:
    """Simulate one confirmatory dataset at ratio 2 and test it.

    Args:
        setting: Simulation inputs
        seed: Replicate seed

    Returns:
        True if the averaged contrast is significant at alpha 0.005
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for register, base in setting.base.items():
        inverse_k = setting.inverse_k[register]
        prompt_var, text_var = PROMPT_SHARE * inverse_k, (1 - PROMPT_SHARE) * inverse_k
        for prompt in range(N_PROMPTS):
            prompt_frailty = (
                rng.gamma(1 / prompt_var, prompt_var) if prompt_var > 0 else 1.0
            )
            for source in MAIN_SOURCES:
                words = int(rng.choice(setting.words[register][source]))
                multiplier = 1.0 if source == "human" else RATIO
                text_frailty = (
                    rng.gamma(1 / text_var, text_var) if text_var > 0 else 1.0
                )
                if setting.measure == "rate":
                    value = rng.poisson(
                        base * words / 1000 * multiplier * prompt_frailty * text_frailty
                    )
                else:
                    value = int(
                        rng.random() < min(1.0, base * multiplier * prompt_frailty)
                    )
                rows.append(
                    {
                        "prompt_id": f"{register}{prompt}",
                        "register": register,
                        "source": source,
                        "words": words,
                        "value": value,
                    }
                )
    frame = pl.DataFrame(rows)
    contrast = gee_contrast(frame, "value", setting.measure, MAIN_SOURCES)  # type: ignore[arg-type]
    return contrast is not None and contrast.p < ALPHA and contrast.estimate > 0


def build_settings(
    exploratory: pl.DataFrame, measures: dict[str, str]
) -> list[Setting]:
    """Estimate simulation inputs from the exploratory split.

    Args:
        exploratory: Exploratory counts
        measures: Switch-rule measures

    Returns:
        One setting per count or presence tell
    """
    settings: list[Setting] = []
    for tell in PRIMARY:
        measure = measures[tell.id]
        if measure == "cv":
            continue
        base: dict[str, float] = {}
        inverse_k: dict[str, float] = {}
        words: dict[str, dict[str, list[int]]] = {}
        for register in ("hn", "se"):
            sub = exploratory.filter(pl.col("register") == register)
            human = sub.filter(pl.col("source") == "human")
            counts = human[tell.column].to_numpy().astype(float)
            exposure = human["words"].to_numpy() / 1000
            if measure == "rate":
                base[register] = max(float(counts.sum() / exposure.sum()), 1e-3)
            else:
                base[register] = max(float((counts > 0).mean()), 0.005)
            inverse_k[register] = overdispersion(counts, exposure)
            words[register] = {
                source: sub.filter(pl.col("source") == source)["words"].to_list()
                or human["words"].to_list()
                for source in MAIN_SOURCES
            }
        settings.append(Setting(tell.column, measure, base, inverse_k, words))
    return settings


def power(setting: Setting, replicates: int, seed: int) -> float:
    """Share of significant replicates.

    Args:
        setting: Simulation inputs
        replicates: Number of replicates
        seed: Base seed

    Returns:
        Estimated power
    """
    with ProcessPoolExecutor(8) as pool:
        hits = list(
            pool.map(
                simulate_once,
                [setting] * replicates,
                range(seed, seed + replicates),
                chunksize=8,
            )
        )
    return float(np.mean(hits))


def main() -> None:
    """Run the simulation and write power.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts", type=Path)
    parser.add_argument("measures", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    exploratory = pl.read_parquet(args.counts).filter(pl.col("split") == "exploratory")
    measures = json.loads(args.measures.read_text())
    results = {}
    for setting in build_settings(exploratory, measures):
        estimate = power(setting, args.replicates, 1000)
        results[setting.column] = {
            "measure": setting.measure,
            "base": setting.base,
            "inverse_k": setting.inverse_k,
            "power_ratio_2": estimate,
            "estimate_only": estimate < 0.8,
        }
        print(setting.column, setting.measure, round(estimate, 3), flush=True)
    args.out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
