import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import pytest

from llm_tells import coding
from llm_tells.text import CleanText, clean_markdown

REAL_CLIENT = httpx.Client
TEXT_ID_RE = re.compile(r"<<<TEXT\nSynthetic text (\S+) here\.")
SOURCES = ("human", "gemma", "qwen", "muse")
REGISTERS = ("hn", "se")
Handler = Callable[[httpx.Request], httpx.Response]


def counts_table(
    per_cell: int,
    sources: Sequence[str] = SOURCES,
    registers: Sequence[str] = REGISTERS,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for register in registers:
            for i in range(per_cell):
                text_id = f"{register}:{i}|{source}"
                rows.append(
                    {
                        "text_id": text_id,
                        "source": source,
                        "register": register,
                        "split": "confirmatory",
                        "failed": False,
                        "raw": f"Synthetic text {text_id} here.",
                        "clean_text": f"Synthetic text {text_id} here.",
                        **{
                            column: int((i + k) % 3 == 0)
                            for k, column in enumerate(coding.CODES.values())
                        },
                    }
                )
    return pl.DataFrame(rows)


def test_opaque_id_is_a_stable_short_hash():
    assert coding.opaque_id("hn:1|qwen") == coding.opaque_id("hn:1|qwen")
    assert coding.opaque_id("hn:1|qwen") != coding.opaque_id("hn:1|gemma")
    assert re.fullmatch(r"[0-9a-f]{10}", coding.opaque_id("hn:1|qwen"))


def test_cell_rng_depends_only_on_its_cell():
    first = coding.cell_rng(1, "qwen", "hn").integers(0, 10**9, 5)
    again = coding.cell_rng(1, "qwen", "hn").integers(0, 10**9, 5)
    other = coding.cell_rng(1, "qwen", "se").integers(0, 10**9, 5)
    assert first.tolist() == again.tolist()
    assert first.tolist() != other.tolist()


def test_validation_sample_covers_each_side_of_each_tell_per_cell():
    counts = counts_table(20)
    sample = coding.validation_sample(counts, seed=5)
    picked = set(sample["text_id"])
    for (_, _), cell in counts.group_by(["source", "register"]):
        for column in coding.CODES.values():
            for positive in (True, False):
                side = set(cell.filter((pl.col(column) > 0) == positive)["text_id"])
                assert len(side & picked) >= min(coding.PER_CELL, len(side))
    assert sample["code_id"].to_list() == sorted(
        coding.opaque_id(t) for t in sample["text_id"]
    )
    assert sample.height < counts.height


def test_validation_sample_cells_are_independent():
    small = coding.validation_sample(counts_table(20, ("human", "gemma")), seed=5)
    large = coding.validation_sample(counts_table(20), seed=5)
    kept = large.filter(pl.col("source").is_in(["human", "gemma"]))
    assert set(kept["text_id"]) == set(small["text_id"])


def test_validation_sample_takes_every_positive_of_a_rare_tell():
    counts = counts_table(20, ("human",), ("hn",)).with_columns(
        p7_worth_noting=(pl.col("text_id") == "hn:3|human").cast(pl.Int64)
    )
    sample = coding.validation_sample(counts, seed=1)
    assert "hn:3|human" in set(sample["text_id"])


def test_content_sample_draws_sixty_per_cell_and_a_claude_subset():
    counts = pl.concat(
        [
            counts_table(70, ("human",), ("hn",)),
            counts_table(10, ("gemma",), ("hn",)),
        ]
    )
    sample = coding.content_sample(counts, seed=3)
    per_source = sample.group_by("source").agg(
        pl.len().alias("n"), pl.col("claude_subset").sum().alias("claude")
    )
    assert dict(per_source.select("source", "n").iter_rows()) == {
        "human": 60,
        "gemma": 10,
    }
    assert dict(per_source.select("source", "claude").iter_rows()) == {
        "human": 19,
        "gemma": 10,
    }
    assert sample["code_id"].is_sorted()
    assert coding.content_sample(counts, seed=3).equals(sample)
    assert not coding.content_sample(counts, seed=4).equals(sample)


def check_pool() -> tuple[pl.DataFrame, set[str]]:
    human_tri = [f"h_t{i:02d}" for i in range(30)]
    human_neg = [f"h_n{i:02d}" for i in range(20)]
    model_both = [f"m_b{i:02d}" for i in range(25)]
    model_part = [f"m_p{i:02d}" for i in range(10)]
    model_neg = [f"m_n{i:02d}" for i in range(5)]
    rows = (
        [(t, "human", 1, 0) for t in human_tri]
        + [(t, "human", 0, 0) for t in human_neg]
        + [(t, "qwen", 1, 1) for t in model_both]
        + [(t, "qwen", 0, 1) for t in model_part]
        + [(t, "qwen", 0, 0) for t in model_neg]
    )
    counts = pl.DataFrame(
        rows,
        schema=["text_id", "source", "p3_tricolon", "p8_participial"],
        orient="row",
    )
    return counts, set(human_tri[5:])


def test_human_check_sample_prefers_haiku_coded_and_reuses_picks():
    counts, haiku = check_pool()
    sheet = coding.human_check_sample(counts, haiku, seed=9)
    ids = set(sheet["text_id"])
    assert {t for t in ids if t.startswith("h_t")} == haiku
    assert len({t for t in ids if t.startswith("h_n")}) == 15
    assert {t for t in ids if t.startswith("m_")} == {
        f"m_b{i:02d}" for i in range(25)
    } | {f"m_n{i:02d}" for i in range(5)}
    assert sheet.height == 70
    assert sheet["human_side"].sum() == 40
    assert set(sheet.filter(pl.col("haiku_coded"))["text_id"]) == haiku
    assert sheet["code_id"].to_list() == sorted(sheet["code_id"].to_list())


def check_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    code_ids = [f"s{i:02d}" for i in range(12)]
    sample = pl.DataFrame(
        {
            "code_id": code_ids,
            "source": ["human"] * 6 + ["gemma"] * 6,
            "human_side": [True] * 6 + [False] * 6,
            "p3_tricolon": [1, 1, 1, 1, 0, 0] * 2,
            "p8_participial": [2, 1, 1, 0, 0, 0] * 2,
        }
    )
    coded = pl.DataFrame(
        {
            "code_id": code_ids,
            "tricolon": [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
            "participial": [1, 1, 1, 0, 0, 0] * 2,
        }
    )

    def side(source: str, n: int, tri: int, part: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "source": [source] * n,
                "p3_tricolon": [int(i < tri) for i in range(n)],
                "p8_participial": [int(i < part) for i in range(n)],
            }
        )

    pool = pl.concat([side("human", 30, 10, 5), side("gemma", 20, 10, 5)])
    return sample, coded, pool


def test_human_check_scores_against_hand_worked_values():
    table = coding.human_check(*check_tables()).sort("code", "group")
    rows = {(r["code"], r["group"]): r for r in table.iter_rows(named=True)}
    human_tri = rows["tricolon", "human"]
    assert human_tri["n_counter_pos"] == 4
    assert human_tri["n_counter_neg"] == 2
    assert human_tri["precision"] == pytest.approx(0.75)
    assert human_tri["recall"] == pytest.approx(7.5 / 17.5)
    assert human_tri["precision_gap"] == pytest.approx(0.25)
    assert human_tri["rate_ratio_factor"] == pytest.approx(4 / 7)
    assert (
        rows["tricolon", "model"]["precision"],
        rows["tricolon", "model"]["recall"],
    ) == (
        1.0,
        1.0,
    )
    assert human_tri["validated"] is False
    assert rows["tricolon", "model"]["validated"] is False
    for group in ("human", "model"):
        part = rows["participial", group]
        assert (part["precision"], part["recall"]) == (1.0, 1.0)
        assert part["rate_ratio_factor"] == pytest.approx(1.0)
        assert part["validated"] is True


def test_correction_factor_matches_hand_worked_value():
    empty = np.array([], dtype=np.bool_)
    cells: dict[bool, coding.Cell] = {
        True: (10, 20, empty, empty),
        False: (10, 10, empty, empty),
    }
    factor = coding.correction_factor(
        cells, {True: 0.75, False: 1.0}, {True: 0.5, False: 0.0}
    )
    assert factor == pytest.approx(4 / 7)


def test_human_check_sensitivity_values():
    sample, coded, pool = check_tables()
    table = coding.human_check_sensitivity(
        sample, coded, pool, {"tricolon": 3.0, "participial": 2.0}, n_boot=300
    )
    rows = {r["code"]: r for r in table.iter_rows(named=True)}
    tri = rows["tricolon"]
    assert tri["factor"] == pytest.approx(4 / 7)
    assert tri["corrected_ratio"] == pytest.approx(12 / 7)
    assert tri["precision_only_ratio"] == pytest.approx(4.0)
    assert tri["model_miss_bound_ratio"] == pytest.approx(30 / 7)
    assert 1 / 3 <= tri["factor_lo"] <= 4 / 7 <= tri["factor_hi"] <= 4
    assert tri["corrected_lo"] == pytest.approx(3 * tri["factor_lo"])
    assert 0 < tri["share_corrected_at_least_2"] < 1
    part = rows["participial"]
    assert (part["factor"], part["factor_lo"], part["factor_hi"]) == (1.0, 1.0, 1.0)
    assert part["share_corrected_at_least_2"] == 1.0
    assert part["model_miss_bound_ratio"] == pytest.approx(8.0)


def adjudication_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    code_ids = [f"c{i:02d}" for i in range(12)]
    sample = pl.DataFrame(
        {
            "code_id": code_ids,
            "human_side": [i % 2 == 0 for i in range(12)],
            "p3_tricolon": [1, 0, 2] + [1] * 9,
            "p8_participial": [0] * 12,
        }
    )
    coded = pl.DataFrame(
        {
            "code_id": code_ids,
            "tricolon": [0, 0, 1] + [1] * 9,
            "participial": [0] * 5 + [1] + [0] * 6,
        }
    )
    fable = pl.DataFrame(
        {
            "code_id": code_ids,
            "tricolon": [0, 1, 1] + [1] * 9,
            "participial": [0] * 12,
        }
    )
    return sample, coded, fable


def test_adjudication_items_takes_all_disagreements_and_an_agree_sample():
    items = coding.adjudication_items(*adjudication_tables(), seed=4)
    disagree = items.filter(pl.col("stratum") == "disagree")
    agree = items.filter(pl.col("stratum") == "agree")
    assert sorted(disagree.filter(pl.col("code") == "tricolon")["code_id"]) == [
        "c00",
        "c01",
        "c02",
    ]
    assert disagree.filter(pl.col("code") == "participial")["code_id"].to_list() == [
        "c05"
    ]
    for code in coding.HUMAN_CHECK_CODES:
        picks = agree.filter(pl.col("code") == code)
        assert picks.height == coding.AGREE_SAMPLE
        assert picks["code_id"].n_unique() == coding.AGREE_SAMPLE
        assert (picks["founder"] == picks["counter"]).all()
        assert (picks["founder"] == picks["fable"]).all()
    row = items.filter((pl.col("code_id") == "c02") & (pl.col("code") == "tricolon"))
    assert row.select("founder", "counter", "fable").row(0) == (1, 2, 1)
    assert row["item_id"].item() == coding.opaque_id("c02:tricolon")
    assert items["item_id"].is_sorted()
    assert coding.adjudication_items(*adjudication_tables(), seed=4).equals(items)


def test_adjudication_sheet_hides_sources_and_shows_phrases(
    monkeypatch: pytest.MonkeyPatch,
):
    received: list[CleanText] = []

    def fake_phrases(texts: Sequence[CleanText]) -> list[dict[str, list[str]]]:
        received.extend(texts)
        return [
            {"tricolon": [f"red, green and blue {i}"], "participial": ["running", "x"]}
            for i, _ in enumerate(texts)
        ]

    monkeypatch.setattr(coding, "matched_phrases", fake_phrases)
    items = pl.DataFrame(
        {
            "item_id": ["i1", "i2"],
            "code_id": ["a", "b"],
            "code": ["tricolon", "participial"],
            "founder": [0, 1],
            "counter": [10, 1],
            "fable": [2, 1],
        }
    )
    sheet = pl.DataFrame(
        {
            "code_id": ["a", "b"],
            "register": ["se_cont", "hn"],
            "source": ["human", "qwen"],
            "raw": ["Para one.\n\nPara two.", "**Bold** reply."],
            "clean_text": ["Para one.\n\nPara two.", "Bold reply."],
        }
    )
    out = coding.adjudication_sheet(items, sheet)
    assert out.columns == [
        "item_id",
        "pattern",
        "text",
        "options",
        "flagged_phrases",
        "final",
        "instances",
        "note",
    ]
    assert out["options"].to_list() == ["0 or 2 or 10", "1"]
    assert out["flagged_phrases"].to_list() == ["red, green and blue 0", "running | x"]
    assert out["pattern"].to_list() == ["tricolon", "participial"]
    assert set(out["final"]) == {""}
    assert [block.text for block in received[0].blocks] == ["Para one.", "Para two."]
    assert received[1] == clean_markdown("**Bold** reply.")


def test_adjudicate_applies_final_counts_and_summarizes():
    coded = pl.DataFrame(
        {"code_id": ["a", "b", "c"], "tricolon": [0, 1, 2], "participial": [1, 0, 0]}
    )
    items = pl.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "code_id": ["a", "b", "c"],
            "code": ["tricolon", "participial", "tricolon"],
            "founder": [0, 0, 2],
            "counter": [1, 0, 2],
            "fable": [0, 0, 1],
            "stratum": ["disagree", "agree", "disagree"],
        }
    )
    reviewed = pl.DataFrame({"item_id": ["i1", "i2", "i3"], "final": [1, 0, 3]})
    adjudicated, summary = coding.adjudicate(coded, items, reviewed)
    assert adjudicated.columns == coded.columns
    assert adjudicated.sort("code_id").rows() == [("a", 1, 1), ("b", 1, 0), ("c", 3, 0)]
    assert summary.rows(named=True) == [
        {
            "code": "participial",
            "stratum": "agree",
            "n_items": 1,
            "n_changed": 0,
            "n_final_is_counter": 1,
            "n_final_is_fable": 1,
            "n_final_is_neither": 0,
        },
        {
            "code": "tricolon",
            "stratum": "disagree",
            "n_items": 2,
            "n_changed": 2,
            "n_final_is_counter": 1,
            "n_final_is_fable": 0,
            "n_final_is_neither": 1,
        },
    ]


def test_write_human_check_writes_all_tables(tmp_path: Path):
    sample, coded, pool = check_tables()
    primary = tmp_path / "primary.csv"
    primary.write_text("id,effect\nP3,3.0\nP8,2.0\nS1,9.0\n")
    coded = coded.with_columns(
        tricolon=pl.when(pl.col("code_id") == "s03").then(1).otherwise("tricolon")
    )
    haiku = coded.with_columns(tricolon=pl.lit(1))
    coding.write_human_check(
        sample, coded, pool, primary, tmp_path, {"haiku": haiku}, "_x"
    )
    check = pl.read_csv(tmp_path / "human_check_x.csv")
    assert check.height == 4
    sensitivity = pl.read_csv(tmp_path / "human_check_sensitivity_x.csv")
    tri = sensitivity.filter(pl.col("code") == "tricolon")
    assert tri["registered_ratio"].item() == 3.0
    assert tri["corrected_ratio"].item() == pytest.approx(1.5)
    versus = pl.read_csv(tmp_path / "human_vs_haiku_x.csv")
    tri_all = versus.filter((pl.col("code") == "tricolon") & (pl.col("group") == "all"))
    assert tri_all["precision"].item() == pytest.approx(9 / 12, abs=1e-3)
    assert tri_all["recall"].item() == 1.0


def test_coder_prompt_fences_the_text():
    prompt = coding.coder_prompt("BOOK", "Some text.", ("contrast", "closer"))
    assert prompt.startswith("BOOK\n\n---")
    assert "<<<TEXT\nSome text.\nTEXT>>>" in prompt
    assert prompt.endswith('{"contrast": <integer>, "closer": <integer>}')
    quoted = coding.coder_prompt("BOOK", "Some text.", ("contrast",), quotes=True)
    assert quoted.endswith(
        '{"contrast": {"quotes": [<exact quoted instances>], "count": <integer>}}'
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"a": 2, "b": 0}', {"a": 2, "b": 0}),
        ('Sure:\n```json\n{"a": "3", "b": -1, "c": 9}\n```', {"a": 3, "b": 0}),
        ('{"a": {"quotes": ["x"], "count": 1}, "b": 2.9}', {"a": 1, "b": 2}),
        ('{"a": 1}', None),
        ('{"a": "many", "b": 1}', None),
        ('{"a": null, "b": 1}', None),
        ('{"a": 1, "b": 1', None),
        ("no json here", None),
    ],
)
def test_parse_codes(reply: str, expected: dict[str, int] | None):
    assert coding.parse_codes(reply, ("a", "b")) == expected


def glm_handler(
    bodies: list[dict[str, Any]], values: dict[str, dict[str, Any]]
) -> Handler:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        bodies.append(body)
        match = TEXT_ID_RE.search(body["messages"][0]["content"])
        text_id = match.group(1) if match else ""
        reply = json.dumps(values[text_id]) if text_id in values else "not json"
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    return handler


def patch_client(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    def factory(**kwargs: Any) -> httpx.Client:
        return REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_code_with_glm_retries_once_then_gives_nulls(monkeypatch: pytest.MonkeyPatch):
    bodies: list[dict[str, Any]] = []
    values = {"t1": {"contrast": 2, "closer": 0}}
    patch_client(monkeypatch, glm_handler(bodies, values))
    sample = pl.DataFrame(
        {
            "code_id": ["x1", "x2"],
            "clean_text": ["Synthetic text t1 here.", "Synthetic text t2 here."],
        }
    )
    out = coding.code_with_glm(sample, "BOOK", ("contrast", "closer"))
    assert out.rows() == [("x1", 2, 0), ("x2", None, None)]
    assert out.schema["contrast"] == pl.Int64
    assert len(bodies) == 3
    assert all(b["temperature"] == 0 for b in bodies)
    assert all(b["max_tokens"] == 300 for b in bodies)
    assert all(b["reasoning_effort"] == "none" for b in bodies)
    assert all(b["model"] == "glm" for b in bodies)


def test_code_with_glm_reasoning_asks_for_quotes(monkeypatch: pytest.MonkeyPatch):
    bodies: list[dict[str, Any]] = []
    values = {"t1": {"contrast": {"quotes": ["a", "b"], "count": 2}}}
    patch_client(monkeypatch, glm_handler(bodies, values))
    sample = pl.DataFrame(
        {"code_id": ["x1"], "clean_text": ["Synthetic text t1 here."]}
    )
    out = coding.code_with_glm(sample, "BOOK", ("contrast",), reasoning=True)
    assert out.rows() == [("x1", 2)]
    assert bodies[0]["max_tokens"] == 6000
    assert "reasoning_effort" not in bodies[0]
    assert '"quotes"' in bodies[0]["messages"][0]["content"]


def test_write_batches_skips_texts_already_batched(tmp_path: Path):
    def sample(n: int) -> pl.DataFrame:
        return pl.DataFrame(
            {"code_id": [f"k{i:02d}" for i in range(n)], "clean_text": ["t"] * n}
        )

    assert coding.write_batches(sample(30), tmp_path) == 2
    assert coding.write_batches(sample(30), tmp_path) == 0
    assert coding.write_batches(sample(35), tmp_path) == 1
    batches = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("batch_*.json"))]
    assert [len(b) for b in batches] == [25, 5, 5]
    ids = [item["id"] for b in batches for item in b]
    assert sorted(ids) == [f"k{i:02d}" for i in range(35)]
    assert {item["id"] for item in batches[2]} == {f"k{i:02d}" for i in range(30, 35)}
    assert batches[0][0]["text"] == "t"


def test_read_claude_drops_rows_failing_the_schema(tmp_path: Path):
    (tmp_path / "batch_00.out.json").write_text(
        json.dumps(
            [
                {"id": "a", "contrast": 1, "closer": 0},
                {"id": "b", "contrast": 1},
                {"contrast": 1, "closer": 1},
                {"id": "a", "contrast": 1, "closer": 0},
            ]
        )
    )
    (tmp_path / "batch_01.out.json").write_text(
        json.dumps([{"id": "c", "contrast": {"count": 3}, "closer": 2}])
    )
    (tmp_path / "batch_02.json").write_text(json.dumps([{"id": "z"}]))
    out = coding.read_claude(tmp_path, ("contrast", "closer")).sort("code_id")
    assert out.rows() == [("a", 1, 0), ("c", 3, 2)]
    empty = coding.read_claude(tmp_path / "missing", ("contrast",))
    assert empty.height == 0
    assert empty.columns == ["code_id", "contrast"]


def test_alpha_nominal_hand_worked():
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 0, 0])
    assert coding.alpha_nominal(a, b) == pytest.approx(16 / 30)
    assert coding.alpha_nominal(a, a) == 1.0
    assert math.isnan(coding.alpha_nominal(np.zeros(3, int), np.zeros(3, int)))


def agreement_tables() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ids = ["a", "b", "c", "d", "e"]
    sample = pl.DataFrame(
        {
            "code_id": ids,
            "source": ["human", "human", "qwen", "qwen", "qwen"],
            "p1_contrast": [1, 1, 1, 0, 0],
        }
    )
    glm = pl.DataFrame({"code_id": ids, "contrast": [1, 1, 0, 0, None]})
    claude = pl.DataFrame({"code_id": ids, "contrast": [2, 0, 0, 0, 1]})
    return sample, glm, claude


def test_agreement_hand_worked():
    codes = {"contrast": "p1_contrast", "closer": "p6_closer"}
    table = coding.agreement(*agreement_tables(), codes)
    rows = {r["group"]: r for r in table.iter_rows(named=True)}
    assert set(table["code"]) == {"contrast"}
    whole = rows["all"]
    assert whole["n"] == 4
    assert whole["kappa"] == pytest.approx(0.5)
    assert whole["alpha"] == pytest.approx(16 / 30)
    assert whole["positive_agreement"] == pytest.approx(2 / 3)
    assert whole["n_consensus"] == 3
    assert whole["precision"] == pytest.approx(0.5)
    assert whole["recall"] == 1.0
    assert (whole["n_counter_pos"], whole["n_consensus_pos"]) == (2, 1)
    human = rows["human"]
    assert human["kappa"] == pytest.approx(0.0)
    assert human["precision"] == 1.0
    model = rows["model"]
    assert math.isnan(model["kappa"])
    assert math.isnan(model["alpha"])
    assert math.isnan(model["positive_agreement"])
    assert model["precision"] == 0.0
    assert math.isnan(model["recall"])


def test_agreement_when_counter_column_has_the_code_name():
    sample, glm, claude = agreement_tables()
    sample = sample.rename({"p1_contrast": "contrast"})
    table = coding.agreement(sample, glm, claude, {"contrast": "contrast"})
    whole = table.filter(pl.col("group") == "all")
    assert whole["kappa"].item() == pytest.approx(0.5)
    assert whole["n"].item() == 4


def test_counter_vs_coder_hand_worked():
    sample = pl.DataFrame(
        {
            "code_id": ["a", "b", "c", "d"],
            "source": ["human", "qwen", "qwen", "qwen"],
            "p3_tricolon": [1, 0, 0, 0],
        }
    )
    coder = pl.DataFrame({"code_id": ["a", "b", "c", "d"], "tricolon": [1, 1, 3, 0]})
    table = coding.counter_vs_coder(sample, coder, {"tricolon": "p3_tricolon"})
    rows = {r["group"]: r for r in table.iter_rows(named=True)}
    assert rows["all"]["kappa"] == pytest.approx(0.2)
    assert rows["all"]["precision"] == 1.0
    assert rows["all"]["recall"] == pytest.approx(1 / 3)
    assert (rows["all"]["n_counter_pos"], rows["all"]["n_coder_pos"]) == (1, 3)
    assert math.isnan(rows["human"]["kappa"])
    assert math.isnan(rows["model"]["precision"])
    assert rows["model"]["recall"] == 0.0
    renamed = coding.counter_vs_coder(
        sample.rename({"p3_tricolon": "tricolon"}), coder, {"tricolon": "tricolon"}
    )
    assert renamed.filter(pl.col("group") == "all")["kappa"].item() == pytest.approx(
        0.2
    )


CODEBOOK = "# Codebook\n\nTell codes.\n\n## Content codes\n\nGeneric and hype.\n"


def coder_values(counts: pl.DataFrame) -> dict[str, dict[str, int]]:
    return {
        row["text_id"]: {
            **{code: row[column] for code, column in coding.CODES.items()},
            "generic": int(row["text_id"].startswith("hn")),
            "hype": 0,
        }
        for row in counts.iter_rows(named=True)
    }


def write_outputs(path: Path, frame: pl.DataFrame, codes: Sequence[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "batch_00.out.json").write_text(
        json.dumps(
            [
                {"id": row["code_id"], **{code: row[code] for code in codes}}
                for row in frame.iter_rows(named=True)
            ]
        )
    )


def run_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, step: str, *extra: str
) -> None:
    argv = [
        "coding",
        step,
        "--counts",
        str(tmp_path / "counts.parquet"),
        "--dir",
        str(tmp_path / "coding"),
        "--primary",
        str(tmp_path / "primary.csv"),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    coding.main()


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pl.DataFrame:
    counts = counts_table(12)
    excluded = counts_table(1, ("gemma",), ("hn",)).with_columns(
        text_id=pl.lit("x:0|gemma"), split=pl.lit("exploratory")
    )
    pl.concat([counts, excluded]).write_parquet(tmp_path / "counts.parquet")
    (tmp_path / "coding").mkdir()
    (tmp_path / "coding" / "codebook.md").write_text(CODEBOOK)
    (tmp_path / "primary.csv").write_text("id,effect\nP3,3.0\nP8,2.5\n")
    monkeypatch.setattr(coding, "HUMAN_SHEET", 10)
    run_main(monkeypatch, tmp_path, "sample", "--final")
    return counts


def test_main_sample_glm_and_agreement(
    pipeline: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "coding"
    validation = pl.read_parquet(root / "validation_sample.parquet")
    content = pl.read_parquet(root / "content_sample.parquet")
    assert "x:0|gemma" not in set(validation["text_id"])
    assert content.height == pipeline.height
    assert pl.read_csv(root / "human_sheet.csv").height == 10
    assert (root / "claude_tells" / "batch_00.json").exists()

    bodies: list[dict[str, Any]] = []
    patch_client(monkeypatch, glm_handler(bodies, coder_values(pipeline)))
    run_main(monkeypatch, tmp_path, "glm")
    assert len(bodies) == validation.height + content.height
    run_main(monkeypatch, tmp_path, "glm")
    assert len(bodies) == validation.height + content.height
    glm = pl.read_parquet(root / "glm_tells.parquet")
    assert glm.height == validation.height
    assert glm["contrast"].null_count() == 0
    run_main(monkeypatch, tmp_path, "glm-reason")
    run_main(monkeypatch, tmp_path, "glm-reason")
    reason = pl.read_parquet(root / "glm_reason_tells.parquet")
    assert reason.columns == ["code_id", *coding.REASONED_CODES]
    assert reason.height == validation.height
    assert bodies[-1]["max_tokens"] == 6000

    claude = validation.select(
        "code_id", *(pl.col(c).alias(code) for code, c in coding.CODES.items())
    )
    write_outputs(root / "claude_tells", claude, list(coding.CODES))
    write_outputs(root / "claude_tells_rerun", claude.head(20), list(coding.CODES))
    claude_content = (
        content.filter(pl.col("claude_subset"))
        .select(
            "code_id",
            generic=pl.col("register") == "hn",
            hype=pl.lit(0),
        )
        .with_columns(pl.col("generic").cast(pl.Int64))
    )
    write_outputs(root / "claude_content", claude_content, coding.CONTENT_CODES)
    run_main(monkeypatch, tmp_path, "agreement")
    tells = pl.read_csv(root / "agreement_tells.csv")
    assert tells.height == 3 * len(coding.CODES)
    assert (tells.filter(pl.col("group") == "all")["kappa"] == 1.0).all()
    stability = pl.read_csv(root / "claude_stability.csv")
    assert set(stability["group"]) == {"all"}
    assert (stability["n"] == 20).all()
    content_agreement = pl.read_csv(root / "agreement_content.csv")
    generic = content_agreement.filter(
        (pl.col("code") == "generic") & (pl.col("group") == "all")
    )
    assert generic["kappa"].item() == 1.0
    assert (root / "counter_vs_glm.csv").exists()
    assert (root / "counter_vs_claude.csv").exists()


def founder_codes(sheet: pl.DataFrame) -> pl.DataFrame:
    flip = sheet["code_id"].sort().head(3).to_list()
    return (
        sheet.filter(pl.int_range(pl.len()) % 5 != 4)
        .select(
            "code_id",
            tricolon=pl.when(pl.col("code_id").is_in(flip))
            .then(1 - pl.col("p3_tricolon"))
            .otherwise(pl.col("p3_tricolon")),
            participial=pl.col("p8_participial"),
        )
        .sort("code_id")
    )


def test_main_human_check_and_adjudication(
    pipeline: pl.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    root = tmp_path / "coding"
    run_main(monkeypatch, tmp_path, "human-sheet")
    sheet = pl.read_parquet(root / "human_sheet_v2.parquet")
    haiku_ids = set(pl.read_parquet(root / "validation_sample.parquet")["text_id"])
    assert set(sheet.filter(pl.col("haiku_coded"))["text_id"]) <= haiku_ids
    assert pl.read_csv(root / "human_sheet_v2.csv").columns == [
        "code_id",
        "text",
        *coding.HUMAN_CHECK_CODES,
    ]

    coded = founder_codes(sheet)
    coded_path = root / "human_sheet_v2_coded.csv"
    coded.with_columns(tricolon=pl.lit(None, pl.Int64)).write_csv(coded_path)
    with pytest.raises(SystemExit, match="every row needs both codes"):
        run_main(monkeypatch, tmp_path, "human-check")
    coded.write_csv(coded_path)
    flipped = coded.with_columns(participial=1 - pl.col("participial"))
    write_outputs(
        root / "fable_tells_v2",
        flipped.head(4).vstack(coded[4:]),
        ["tricolon", "participial"],
    )
    write_outputs(root / "claude_tells", coded, ["tricolon", "participial"])
    capsys.readouterr()
    run_main(monkeypatch, tmp_path, "human-check")
    assert (
        f"{coded.height} of {sheet.height} sheet texts coded" in capsys.readouterr().out
    )
    check = pl.read_csv(root / "human_check.csv")
    assert check.height == 4
    participial = check.filter(pl.col("code") == "participial")
    assert (participial["precision"] == 1.0).all()
    assert participial["validated"].all()
    for name in ("human_vs_haiku", "human_vs_fable", "counter_vs_fable"):
        assert (root / f"{name}.csv").exists()

    def fake_phrases(texts: Sequence[CleanText]) -> list[dict[str, list[str]]]:
        return [{"tricolon": [], "participial": []} for _ in texts]

    monkeypatch.setattr(coding, "matched_phrases", fake_phrases)
    run_main(monkeypatch, tmp_path, "adjudication-sheet")
    items = pl.read_parquet(root / "adjudication_sheet.parquet")
    assert set(items.filter(pl.col("stratum") == "disagree")["code"]) == {
        "tricolon",
        "participial",
    }
    review = pl.read_csv(root / "adjudication_sheet.csv")
    assert review["item_id"].sort().equals(items["item_id"].sort())

    reviewed_path = root / "adjudication_sheet_reviewed.csv"
    reviewed = items.select("item_id", final=pl.col("counter"))
    reviewed.head(len(items) - 1).write_csv(reviewed_path)
    with pytest.raises(SystemExit, match="whole-number"):
        run_main(monkeypatch, tmp_path, "adjudication-check")
    reviewed.write_csv(reviewed_path)
    run_main(monkeypatch, tmp_path, "adjudication-check")
    summary = pl.read_csv(root / "adjudication_summary.csv")
    assert (summary["n_final_is_counter"] == summary["n_items"]).all()
    adjudicated = pl.read_csv(root / "human_sheet_v2_adjudicated.csv")
    assert adjudicated.columns == coded.columns
    expected = sheet.select(
        "code_id", tricolon="p3_tricolon", participial="p8_participial"
    )
    assert (
        adjudicated.sort("code_id")
        .join(expected, on="code_id")
        .filter(pl.col("tricolon") != pl.col("tricolon_right"))
        .height
        == 0
    )
    assert (root / "human_check_adjudicated.csv").exists()
