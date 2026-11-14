"""Deterministic counters for the pre-registered tells (P1-P10, S1-S8)."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from statistics import mean, stdev
from typing import Any

import spacy
from spacy.tokens import Doc, Span

from llm_tells.text import QUOTE_TRANSLATION, CleanText, count_words

PRON = r"(?:it|this|that|they|he|she|we|you|there|these|those)"
NEGCOP = r"(?:'s not|'re not| is not| isn't| are not| aren't| was not| wasn't| were not| weren't)"
COP = r"(?:'s|'re| is| are| was| were)"
CONTRAST_RES = (
    re.compile(
        rf"\b{PRON}{NEGCOP}\b[^.?!;]{{1,80}}?[,;:\u2014\u2013-]\s*{PRON}{COP}\b"
    ),
    re.compile(r"\bnot (?:just|only|merely|simply)\b[^.?!]{1,80}?\bbut\b(?! also)"),
)
NOT_BUT_RE = re.compile(r"^[^.?!,;]{1,60}, but\b")
NOT_BUT_AUX = frozenset(
    [
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "may",
        "might",
        "must",
        "'ll",
        "'d",
        "ca",
        "wo",
    ]
)
NEGATED_PRONOUN_RE = re.compile(rf"\b{PRON}{NEGCOP}\b")
CONTRAST_FOLLOW_RE = re.compile(
    r"^(?:it's|it is|it was|this is|that's|they're)\b(?! (?:worth|because)\b)"
)
NOT_ONLY_RE = re.compile(r"\bnot only\b[^.?!]{1,80}?\bbut also\b")
RATHER_THAN_RE = re.compile(r"\brather than\b")
SIGNPOST_RE = re.compile(
    r"\b(?:here's the thing|here is the thing|here's why|here is why|here's the kicker|the key insight"
    r"|the real question is|the bottom line|let's dive in|let's dive into|let's delve|let that sink in"
    r"|the truth is|make no mistake)\b|\bspoiler:"
)
WORTH_NOTING_RE = re.compile(
    r"\b(?:it's |it is )?(?:worth (?:noting|mentioning|pointing out)"
    r"|important to (?:note|remember|understand|keep in mind))\b"
)
NOTE_THAT_RE = re.compile(r"^note that\b")
CLOSER_RE = re.compile(
    r"^(?:in conclusion|in summary|to summarize|to sum up|overall|ultimately|in short|all in all|in the end"
    r"|the takeaway|key takeaway|the key takeaway|tl;dr|tldr)(?![a-z])"
)
TRANSITION_RE = re.compile(
    r"^(?:moreover|furthermore|additionally|in addition|consequently|notably|importantly)\b"
)
FLAGGED_VOCAB_RE = re.compile(
    r"\b(?:delv(?:e|es|ed|ing)|tapestr(?:y|ies)|testaments?|realms?|crucial(?:ly)?|pivotal|leverag(?:e|es|ed|ing)"
    r"|seamless(?:ly)?|robust(?:ly|ness)?|landscapes?|underscor(?:e|es|ed|ing)|showcas(?:e|es|ed|ing)|intricate(?:ly)?"
    r"|meticulous(?:ly)?|commendable|camaraderie|nuanced?|foster(?:s|ed|ing)?|multifaceted|navigat(?:e|es|ed|ing))\b"
)
NEW_VOCAB_RE = re.compile(r"\b(?:quiet|quietly|load-bearing|genuinely|legible)\b")
ANSWER_START_RE = re.compile(
    r"^(?:yes|no|absolutely|not really|of course|sure|exactly|nope|definitely|probably|maybe|the answer)\b"
)
OPENER_RE = re.compile(
    r"^(?:great question|good question|you're right|you're absolutely right|absolutely|certainly|of course"
    r"|i understand|that's a great point|that's a fair point|totally|sure)\b"
)
HEDGE_CLOSER_RE = re.compile(
    r"^(?:however|that said|that being said|of course|still)(?![a-z])"
)
DASH_RE = re.compile(
    r"(?<!\d)\u2014(?!\d)|(?<!\d)\s\u2013\s(?!\d)|(?<=[^\W\d]) ?--(?!-) ?(?=[^\W\d])"
)
LEADING_JUNK_RE = re.compile(r"^[\W_]+")
NOMINAL_SUFFIXES = ("tion", "tions", "ment", "ments", "ness", "nesses", "ity", "ities")
NOMINAL_EXCLUDE = frozenset(
    [
        "city",
        "cities",
        "pity",
        "nation",
        "nations",
        "station",
        "stations",
        "ration",
        "rations",
        "cement",
        "comment",
        "comments",
        "moment",
        "moments",
        "element",
        "elements",
        "apartment",
        "apartments",
        "department",
        "departments",
        "segment",
        "segments",
        "garment",
        "garments",
        "document",
        "documents",
    ]
)
CC_WORDS = frozenset({"and", "or"})
POS_GROUPS = {
    "NOUN": "N",
    "PROPN": "N",
    "PRON": "N",
    "ADJ": "A",
    "VERB": "V",
    "AUX": "V",
    "ADV": "R",
}
STACCATO_MAX_WORDS = 5
CV_MIN_SENTENCES = 4


@dataclass(frozen=True, slots=True)
class Sentence:
    """A sentence with its position.

    Attributes:
        span: The spaCy span
        block: Index of its block in the text
        kind: Kind of its block
        lower: Lowercased text with straight quotes and leading punctuation removed
        n_words: Word count
    """

    span: Span
    block: int
    kind: str
    lower: str
    n_words: int


@cache
def nlp() -> spacy.language.Language:
    """Load the pre-registered spaCy pipeline once.

    Returns:
        en_core_web_sm without NER
    """
    return spacy.load("en_core_web_sm", disable=["ner"])


def sentences_of(
    blocks: Sequence[tuple[int, str]], docs: Sequence[Doc]
) -> list[Sentence]:
    """Turn parsed blocks into sentences.

    Args:
        blocks: (block index, kind) per parsed block
        docs: The parsed blocks, in the same order

    Returns:
        The text's sentences in order
    """
    return [
        Sentence(
            span=sent,
            block=index,
            kind=kind,
            lower=LEADING_JUNK_RE.sub(
                "", sent.text.translate(QUOTE_TRANSLATION).lower()
            ).strip(),
            n_words=count_words(sent.text),
        )
        for (index, kind), doc in zip(blocks, docs, strict=True)
        for sent in doc.sents
        if sent.text.strip()
    ]


def contrast_frames(sentences: Sequence[Sentence]) -> int:
    """P1: sentences with a contrast frame, plus cross-sentence frames.

    Args:
        sentences: The text's sentences

    Returns:
        Count of frames
    """
    within = [
        any(pattern.search(s.lower) for pattern in CONTRAST_RES) or not_but(s.span)
        for s in sentences
    ]
    across = sum(
        not within[i]
        and bool(NEGATED_PRONOUN_RE.search(first.lower))
        and bool(CONTRAST_FOLLOW_RE.match(second.lower))
        for i, (first, second) in enumerate(pairwise(sentences))
    )
    return sum(within) + across


def not_but(span: Span) -> bool:
    """P1 rule (c): "not X, but Y" where "not" negates a non-verb and is not do-support or a modal.

    Args:
        span: A sentence

    Returns:
        True if the sentence has the frame
    """
    for token in span:
        if token.lower_ != "not" or token.i + 1 >= span.end:
            continue
        before = token.doc[token.i - 1].lower_ if token.i > span.start else ""
        after = token.doc[token.i + 1]
        if (
            before in NOT_BUT_AUX
            or before in {"i'm", "'m", "am"}
            or after.tag_.startswith("VB")
            or after.lower_ == "sure"
        ):
            continue
        if NOT_BUT_RE.match(
            token.doc[token.i + 1 : span.end].text.translate(QUOTE_TRANSLATION).lower()
        ):
            return True
    return False


def tricolon_spans(docs: Iterable[Doc]) -> list[Span]:
    """P3: coordination groups of exactly three conjuncts joined by "and" or "or".

    Args:
        docs: Parsed blocks

    Returns:
        One span per group, from its first conjunct to its last
    """
    spans: list[Span] = []
    for doc in docs:
        for token in doc:
            if token.dep_ == "conj":
                continue
            group = [token]
            frontier = [token]
            while frontier:
                children = [
                    child
                    for node in frontier
                    for child in node.children
                    if child.dep_ == "conj"
                ]
                group.extend(children)
                frontier = children
            ccs = [
                child.lower_
                for node in group
                for child in node.children
                if child.dep_ == "cc"
            ]
            same_kind = (
                len({POS_GROUPS.get(node.pos_, node.pos_) for node in group}) == 1
            )
            if len(group) == 3 and len(ccs) == 1 and ccs[0] in CC_WORDS and same_kind:
                indexes = [node.i for node in group]
                spans.append(doc[min(indexes) : max(indexes) + 1])
    return spans


def staccato_runs(sentences: Sequence[Sentence]) -> int:
    """P4: runs of two or more consecutive short prose sentences in one paragraph.

    Args:
        sentences: The text's sentences

    Returns:
        Count of runs
    """
    runs = 0
    length = 0
    previous_block = -1
    for sentence in sentences:
        if sentence.block != previous_block:
            runs += length >= 2
            length = 0
        previous_block = sentence.block
        if sentence.kind == "para" and 0 < sentence.n_words <= STACCATO_MAX_WORDS:
            length += 1
            continue
        runs += length >= 2
        length = 0
    return runs + (length >= 2)


def participial_spans(docs: Iterable[Doc]) -> list[Span]:
    """P8: comma-attached or sentence-initial present participial adverbial clauses.

    Args:
        docs: Parsed blocks

    Returns:
        One span per clause, covering the participle's subtree
    """
    return [
        doc[token.left_edge.i : token.right_edge.i + 1]
        for doc in docs
        for token in doc
        if token.tag_ == "VBG"
        and token.dep_ == "advcl"
        and not any(child.dep_ == "mark" for child in token.children)
        and (
            token.i == token.sent.start
            or (token.i > 0 and doc[token.i - 1].text == ",")
        )
    ]


def nominalizations(docs: Iterable[Doc]) -> int:
    """P9: Biber nominalizations, minus non-derived nouns.

    Args:
        docs: Parsed blocks

    Returns:
        Count of nominalizations
    """
    return sum(
        token.pos_ == "NOUN"
        and len(token.lower_) >= 7
        and token.lower_.endswith(NOMINAL_SUFFIXES)
        and token.lower_ not in NOMINAL_EXCLUDE
        for doc in docs
        for token in doc
    )


def sentence_length_cv(sentences: Sequence[Sentence]) -> float | None:
    """P10: coefficient of variation of prose sentence lengths.

    Args:
        sentences: The text's sentences

    Returns:
        Sample SD over mean, or None with fewer than four prose sentences
    """
    lengths = [s.n_words for s in sentences if s.kind == "para" and s.n_words > 0]
    if len(lengths) < CV_MIN_SENTENCES:
        return None
    return stdev(lengths) / mean(lengths)


def end_sentences(sentences: Sequence[Sentence]) -> list[Sentence]:
    """Sentences checked for end-position tells: first of the last block and the last two.

    Args:
        sentences: The text's sentences

    Returns:
        The candidate sentences (empty for one-sentence texts)
    """
    body = list(sentences)
    if len(body) < 2:
        return []
    last_block = body[-1].block
    first_of_last = next(s for s in body if s.block == last_block)
    return [first_of_last, *body[-2:]]


def measure_text(
    clean: CleanText, sentences: Sequence[Sentence], docs: Sequence[Doc]
) -> dict[str, Any]:
    """Count every tell in one text.

    Args:
        clean: The cleaned text
        sentences: Its sentences
        docs: Its parsed blocks

    Returns:
        One row of counts
    """
    lower = clean.text.translate(QUOTE_TRANSLATION).lower()
    body = list(sentences)
    ends = end_sentences(sentences)
    formatting = clean.formatting
    return {
        "n_words": clean.n_words,
        "n_sentences": len(body),
        "p1_contrast": contrast_frames(body),
        "p1_within_only": sum(
            any(pattern.search(s.lower) for pattern in CONTRAST_RES) or not_but(s.span)
            for s in body
        ),
        "p2_rather_than": len(RATHER_THAN_RE.findall(lower)),
        "p3_tricolon": len(tricolon_spans(docs)),
        "p4_staccato": staccato_runs(sentences),
        "p5_signpost": len(SIGNPOST_RE.findall(lower)),
        "p6_closer": int(any(CLOSER_RE.match(s.lower) for s in ends)),
        "p6_lastpara_only": int(bool(ends) and bool(CLOSER_RE.match(ends[0].lower))),
        "p7_worth_noting": len(WORTH_NOTING_RE.findall(lower))
        + sum(bool(NOTE_THAT_RE.match(s.lower)) for s in body),
        "p8_participial": len(participial_spans(docs)),
        "p9_nominal": nominalizations(docs),
        "p10_cv": sentence_length_cv(sentences),
        "s1_dash": len(DASH_RE.findall(clean.text)),
        "s2_transition": sum(bool(TRANSITION_RE.match(s.lower)) for s in body),
        "s3_vocab": len(FLAGGED_VOCAB_RE.findall(lower)),
        "s4_new_vocab": len(NEW_VOCAB_RE.findall(lower)),
        "s5_self_answer": sum(
            first.span.text.rstrip().endswith("?")
            and second.n_words <= 4
            and bool(ANSWER_START_RE.match(second.lower))
            for first, second in pairwise(body)
        ),
        "s6_opener": int(bool(body) and bool(OPENER_RE.match(body[0].lower))),
        "s7_hedge_closer": int(
            len(body) >= 2 and bool(HEDGE_CLOSER_RE.match(body[-1].lower))
        ),
        "s8a_list": int(formatting.list_items > 0),
        "s8b_heading": int(formatting.headings > 0),
        "s8c_bold": formatting.bold_spans,
        "s8d_bold_lead": int(formatting.bold_lead_items >= 2),
        "symbols_emoji": formatting.emoji,
        "not_only_but_also": len(NOT_ONLY_RE.findall(lower)),
    }


def parse_texts(
    texts: Sequence[CleanText], n_process: int = 4
) -> list[list[tuple[tuple[int, str], Doc]]]:
    """Parse every block of every text in one batch.

    Args:
        texts: Cleaned texts
        n_process: spaCy worker processes

    Returns:
        Per text, each block's (index, kind) and its parse, in order
    """
    index = [
        (t, b, block.kind)
        for t, clean in enumerate(texts)
        for b, block in enumerate(clean.blocks)
    ]
    docs = nlp().pipe(
        (texts[t].blocks[b].text for t, b, _ in index),
        batch_size=256,
        n_process=n_process,
    )
    per_text: list[list[tuple[tuple[int, str], Doc]]] = [[] for _ in texts]
    for (t, b, kind), doc in zip(index, docs, strict=True):
        per_text[t].append(((b, kind), doc))
    return per_text


def measure(texts: Sequence[CleanText], n_process: int = 4) -> list[dict[str, Any]]:
    """Count every tell in every text, parsing all blocks in one batch.

    Args:
        texts: Cleaned texts
        n_process: spaCy worker processes

    Returns:
        One row of counts per text, in order
    """
    rows: list[dict[str, Any]] = []
    for clean, parsed in zip(texts, parse_texts(texts, n_process), strict=True):
        blocks = [meta for meta, _ in parsed]
        text_docs = [doc for _, doc in parsed]
        rows.append(measure_text(clean, sentences_of(blocks, text_docs), text_docs))
    return rows


def matched_phrases(
    texts: Sequence[CleanText], n_process: int = 4
) -> list[dict[str, list[str]]]:
    """The phrases the P3 and P8 counters matched, for review by a human coder.

    Args:
        texts: Cleaned texts
        n_process: spaCy worker processes

    Returns:
        Per text, the matched phrases under "tricolon" and "participial"
    """
    return [
        {
            "tricolon": [span.text for span in tricolon_spans(docs)],
            "participial": [span.text for span in participial_spans(docs)],
        }
        for docs in (
            [doc for _, doc in parsed] for parsed in parse_texts(texts, n_process)
        )
    ]
