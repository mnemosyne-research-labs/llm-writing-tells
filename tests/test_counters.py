from statistics import mean, stdev
from typing import Any

import pytest

from llm_tells.counters import end_sentences, matched_phrases, measure, parse_texts
from llm_tells.counters import sentences_of as to_sentences
from llm_tells.text import Block, CleanText, Formatting, clean_markdown


def counts(markdown: str) -> dict[str, Any]:
    return measure([clean_markdown(markdown)], n_process=1)[0]


def phrases(markdown: str) -> dict[str, list[str]]:
    return matched_phrases([clean_markdown(markdown)], n_process=1)[0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It's not a bug, it's a feature.", 1),
        ("She is not tired; she is bored.", 1),
        ("Which isn't a bug, it's a feature.", 0),
        ("This is not just a tool but a platform.", 1),
        ("This is not just a tool but also a platform.", 0),
        ("The answer is not money, but time.", 1),
        ("He was not happy, but he left.", 1),
        ("I did not like it, but I stayed.", 0),
        ("We will not stop, but we will slow down.", 0),
        ("I'm not sure, but I think so.", 0),
        ("It wasn't the cost. It was the delay.", 1),
        ("They are not wrong. They're early.", 1),
        ("It wasn't the cost. It's worth asking why.", 0),
        ("It wasn't the cost. This is because nobody asked.", 0),
        ("It isn't a bug, it's a feature. It's fine.", 1),
        ("The plan is good. It is cheap.", 0),
    ],
)
def test_contrast_frames(text: str, expected: int):
    assert counts(text)["p1_contrast"] == expected


def test_cross_sentence_frame_is_not_a_within_sentence_frame():
    row = counts("It wasn't the cost. It was the delay.")
    assert (row["p1_contrast"], row["p1_within_only"]) == (1, 0)


def test_not_only_but_also_is_counted_apart_from_contrast():
    row = counts("Not only fast but also cheap.")
    assert (row["not_only_but_also"], row["p1_contrast"]) == (1, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We bought apples, pears and plums.", ["apples, pears and plums"]),
        ("We bought apples, pears, and plums.", ["apples, pears, and plums"]),
        ("The car was small, cheap and slow.", ["small, cheap and slow"]),
        ("Is it red, blue or green?", ["red, blue or green"]),
        ("We met Anna, Ben and Carl.", ["Anna, Ben and Carl"]),
        ("We bought apples and pears.", []),
        ("We bought apples, pears, plums and figs.", []),
        ("We bought apples and pears and plums.", []),
        ("The car was small, cheap, but slow.", []),
        ("The room was warm, quiet and a library.", []),
    ],
)
def test_tricolon(text: str, expected: list[str]):
    assert phrases(text)["tricolon"] == expected
    assert counts(text)["p3_tricolon"] == len(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It rained. We stayed.", 1),
        ("It rained. We stayed. The long afternoon went slowly by for us.", 1),
        ("It rained. We stayed.\n\nThe long afternoon went slowly by for us.", 1),
        ("It rained.\n\nWe stayed.", 0),
        (
            "It rained. We stayed. Nobody came. The long afternoon went slowly by "
            "for everyone. Then it stopped. We left.",
            2,
        ),
        ("# It rained\n\n- It rained. We stayed.", 0),
        ("It rained. The long afternoon went slowly by for us. We left.", 0),
    ],
)
def test_staccato_runs(text: str, expected: int):
    assert counts(text)["p4_staccato"] == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Walking home, she saw the fire.", ["Walking home"]),
        ("She left early, hoping to avoid traffic.", ["hoping to avoid traffic"]),
        ("While walking home, she saw the fire.", []),
        ("She is walking home.", []),
    ],
)
def test_participial_clauses(text: str, expected: list[str]):
    assert phrases(text)["participial"] == expected
    assert counts(text)["p8_participial"] == len(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The information and the management improved.", 2),
        ("His happiness and their creativity grew.", 2),
        ("The city station had a comment.", 0),
        ("The motion passed.", 0),
    ],
)
def test_nominalizations(text: str, expected: int):
    assert counts(text)["p9_nominal"] == expected


def test_sentence_length_cv_over_prose_sentences():
    row = counts("One. One two three four five six seven. One two. One two three.")
    lengths = [1, 7, 2, 3]
    assert row["p10_cv"] == pytest.approx(stdev(lengths) / mean(lengths))


@pytest.mark.parametrize(
    "text",
    [
        "One two three. One two three. One two.",
        "# A long heading\n\n- One. Two. Three.",
    ],
)
def test_sentence_length_cv_needs_four_prose_sentences(text: str):
    assert counts(text)["p10_cv"] is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The plan \u2014 if it works \u2014 is simple.", 2),
        ("The plan\u2014simple.", 1),
        ("It works \u2013 mostly.", 1),
        ("It works -- mostly.", 1),
        ("It works--mostly.", 1),
        ("Pages 10\u201420 were missing.", 0),
        ("Read pages 10 \u2013 20.", 0),
        ("Years 2001\u20132005 matter.", 0),
        ("Use 10--20 items.", 0),
        ("The arrow ---> points.", 0),
        ("A well-known fact.", 0),
    ],
)
def test_dashes(text: str, expected: int):
    assert counts(text)["s1_dash"] == expected


@pytest.mark.parametrize(
    ("key", "text", "expected"),
    [
        ("p2_rather_than", "We walked rather than drove. Rather, we ran.", 1),
        ("p5_signpost", "Here's the thing: it was close.", 1),
        ("p5_signpost", "Spoiler: it ends well. The truth is simple.", 2),
        ("p5_signpost", "Let\u2019s dive in.", 1),
        ("p7_worth_noting", "It's worth noting that rain fell.", 1),
        (
            "p7_worth_noting",
            "It's worth noting that rain fell. Note that snow fell too. "
            "It is important to remember this.",
            3,
        ),
        ("p7_worth_noting", "We note that snow fell.", 0),
        (
            "s2_transition",
            "Moreover, it rained. Furthermore it snowed. In addition, hail.",
            3,
        ),
        ("s2_transition", "It rained, moreover.", 0),
        (
            "s3_vocab",
            "We delved into the tapestry of robust, seamless landscapes. Delve deeper.",
            6,
        ),
        ("s3_vocab", "The robot fostered nothing.", 1),
        ("s4_new_vocab", "The quiet room was genuinely legible and load-bearing.", 4),
        ("s4_new_vocab", "The quieter room.", 0),
        ("s5_self_answer", "Is it safe? Yes, mostly.", 1),
        ("s5_self_answer", "Is it safe? No, it is not safe at all today.", 0),
        ("s5_self_answer", "It is safe. Yes, mostly.", 0),
        ("s6_opener", "Great question! The answer is yes.", 1),
        ("s6_opener", "Certainly! Here it is.", 1),
        ("s6_opener", "The question is great. Certainly.", 0),
        ("s7_hedge_closer", "It works well. However, it is slow.", 1),
        ("s7_hedge_closer", "It works. Still, it is slow.", 1),
        ("s7_hedge_closer", "It works. Still it is slow.", 1),
        ("s7_hedge_closer", "It works. Stillness helps.", 0),
        ("s7_hedge_closer", "However, it is slow.", 0),
        ("s7_hedge_closer", "However, it is slow. It works.", 0),
    ],
)
def test_phrase_tells(key: str, text: str, expected: int):
    assert counts(text)[key] == expected


@pytest.mark.parametrize(
    ("text", "closer", "last_para_only"),
    [
        ("We tested it. Overall it works.", 1, 0),
        ("We tested it. Ultimately, it works. Then more.", 1, 0),
        ("We tested it. TL;DR it works.", 1, 0),
        ("We tested it. It ran.\n\nIn conclusion, it works. It ran. It ended.", 1, 1),
        ("In conclusion, it works. It ran.\n\nIt ended. It stopped. It slept.", 0, 0),
        ("In conclusion, it works. It ran. It ended.", 1, 1),
        ("We tested it. Overalls are blue.", 0, 0),
        ("Overall it works.", 0, 0),
    ],
)
def test_summary_closer(text: str, closer: int, last_para_only: int):
    row = counts(text)
    assert (row["p6_closer"], row["p6_lastpara_only"]) == (closer, last_para_only)


def test_formatting_tells_come_from_markdown():
    row = counts(
        "# Plan\n\n- **One** first step.\n- **Two** second step.\n\nDone \U0001f600."
    )
    assert {
        key: row[key]
        for key in (
            "s8a_list",
            "s8b_heading",
            "s8c_bold",
            "s8d_bold_lead",
            "symbols_emoji",
        )
    } == {
        "s8a_list": 1,
        "s8b_heading": 1,
        "s8c_bold": 2,
        "s8d_bold_lead": 1,
        "symbols_emoji": 1,
    }


def test_one_bold_lead_item_is_not_a_bold_lead_in():
    assert counts("- **One** first step.\n- second step.")["s8d_bold_lead"] == 0


def test_word_and_sentence_counts():
    row = counts("It's late. We left at 10.")
    assert (row["n_words"], row["n_sentences"]) == (6, 2)


def test_empty_text_counts_zero():
    row = measure([CleanText(())], n_process=1)[0]
    assert row["n_sentences"] == 0
    assert row["p10_cv"] is None
    assert row["s6_opener"] == row["s7_hedge_closer"] == row["p6_closer"] == 0


def test_measure_keeps_text_order():
    texts = [clean_markdown("We walked rather than drove."), clean_markdown("Fine.")]
    assert [row["p2_rather_than"] for row in measure(texts, n_process=1)] == [1, 0]


def test_parse_texts_groups_blocks_by_text():
    texts = [
        CleanText((Block("heading", "Title"), Block("para", "Body text."))),
        CleanText((Block("item", "An item."),), Formatting(list_items=1)),
    ]
    parsed = parse_texts(texts, n_process=1)
    assert [[meta for meta, _ in text] for text in parsed] == [
        [(0, "heading"), (1, "para")],
        [(0, "item")],
    ]
    assert parsed[0][1][1].text == "Body text."


def test_sentences_carry_block_kind_and_clean_lowercase():
    clean = CleanText((Block("para", "\u201cHello there,\u201d she said. Bye."),))
    parsed = parse_texts([clean], n_process=1)[0]
    sentences = to_sentences([meta for meta, _ in parsed], [doc for _, doc in parsed])
    assert [(s.block, s.kind, s.lower, s.n_words) for s in sentences] == [
        (0, "para", 'hello there," she said.', 4),
        (0, "para", "bye.", 1),
    ]


def test_end_sentences_are_first_of_last_block_and_last_two():
    clean = clean_markdown("One here. Two here.\n\nThree here. Four here. Five here.")
    parsed = parse_texts([clean], n_process=1)[0]
    sentences = to_sentences([meta for meta, _ in parsed], [doc for _, doc in parsed])
    assert [s.lower for s in end_sentences(sentences)] == [
        "three here.",
        "four here.",
        "five here.",
    ]
    assert end_sentences(sentences[:1]) == []
