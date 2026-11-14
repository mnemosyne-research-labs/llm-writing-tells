"""Cleaning of HN HTML and Markdown into blocks of prose, identical for every source."""

import html
import re
from dataclasses import dataclass, field
from functools import cache
from typing import Literal

from markdown_it import MarkdownIt
from markdown_it.token import Token

BlockKind = Literal["para", "heading", "item"]

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)*")
# Kept as run for this study: it also strips prose such as "x < y and y > z" (in
# this corpus only "<->" in two texts; see results.md, Limitations). A tag-only
# pattern would change one model text's word count.
TAG_RE = re.compile(r"<[^>]+>")
PRE_RE = re.compile(r"<pre>.*?</pre>", re.DOTALL | re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t\u00a0]+")
QUOTE_TRANSLATION = str.maketrans(
    {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}
)
FUNCTION_WORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "not",
        "no",
        "as",
        "at",
        "by",
        "from",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "will",
        "would",
        "should",
        "could",
        "if",
        "so",
        "what",
        "which",
        "who",
        "there",
        "their",
        "my",
        "your",
    ]
)


@dataclass(frozen=True, slots=True)
class Block:
    """One paragraph, heading or list item of cleaned prose.

    Attributes:
        kind: "para", "heading" or "item"
        text: The cleaned text of the block
    """

    kind: BlockKind
    text: str


@dataclass(frozen=True, slots=True)
class Formatting:
    """Formatting counts taken from a Markdown rendering.

    Attributes:
        list_items: Number of list items
        headings: Number of headings
        bold_spans: Number of strong-emphasis spans
        bold_lead_items: Number of list items whose text starts with bold
        emoji: Number of emoji and pictographic symbol characters
    """

    list_items: int = 0
    headings: int = 0
    bold_spans: int = 0
    bold_lead_items: int = 0
    emoji: int = 0


@dataclass(frozen=True, slots=True)
class CleanText:
    """A text after cleaning, ready for counting.

    Attributes:
        blocks: The prose blocks in order
        formatting: Formatting counts (zero for sources without Markdown)
        quoted_words: Words removed as quoted lines or block quotes
        has_code: Whether a code block was present
    """

    blocks: tuple[Block, ...]
    formatting: Formatting = field(default_factory=Formatting)
    quoted_words: int = 0
    has_code: bool = False

    @property
    def text(self) -> str:
        """The blocks joined with blank lines."""
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def n_words(self) -> int:
        """Number of words in the cleaned text."""
        return count_words(self.text)


SYMBOL_EMOJI_RE = re.compile("[\U0001f300-\U0001faff☀-➿\U0001f1e6-\U0001f1ff]")


def normalize(text: str) -> str:
    """Normalize quotes, URLs and spacing in a piece of prose.

    Args:
        text: Raw text

    Returns:
        Text with straight quotes, no URLs and single spaces
    """
    text = URL_RE.sub("", TAG_RE.sub("", text).translate(QUOTE_TRANSLATION))
    return SPACE_RE.sub(" ", text).strip()


def count_words(text: str) -> int:
    """Count words with the pre-registered word pattern.

    Args:
        text: Cleaned text

    Returns:
        Number of words
    """
    return len(WORD_RE.findall(text.translate(QUOTE_TRANSLATION)))


def is_english(text: str) -> bool:
    """Apply the pre-registered English check.

    Args:
        text: Cleaned text

    Returns:
        True if at least 90% of letters are ASCII and 20% of words are function words
    """
    text = text.translate(QUOTE_TRANSLATION)
    letters = [char for char in text if char.isalpha()]
    words = [word.lower() for word in WORD_RE.findall(text)]
    if not letters or not words:
        return False
    ascii_share = sum(char.isascii() for char in letters) / len(letters)
    function_share = sum(word in FUNCTION_WORDS for word in words) / len(words)
    return ascii_share >= 0.9 and function_share >= 0.2


def clean_hn(raw_html: str) -> CleanText:
    """Clean an HN comment's HTML.

    Args:
        raw_html: The `text` field of an HN item

    Returns:
        The cleaned text; quoted lines (starting with ">") are removed and counted
    """
    has_code = bool(PRE_RE.search(raw_html))
    body = PRE_RE.sub("", raw_html)
    paragraphs = (
        html.unescape(TAG_RE.sub("", part)).strip()
        for part in re.split(r"<p>", body, flags=re.IGNORECASE)
    )
    blocks: list[Block] = []
    quoted_words = 0
    for paragraph in paragraphs:
        if paragraph.startswith(">"):
            quoted_words += count_words(paragraph)
            continue
        text = normalize(paragraph)
        if text:
            blocks.append(Block("para", text))
    return CleanText(tuple(blocks), quoted_words=quoted_words, has_code=has_code)


@cache
def _markdown() -> MarkdownIt:
    return MarkdownIt("commonmark", {"html": False}).enable("table")


def _inline_text(token: Token) -> tuple[str, int, bool]:
    """Extract prose from an inline token.

    Returns:
        The text, the number of bold spans and whether the text starts with bold
    """
    parts: list[str] = []
    bold = 0
    starts_bold = False
    for child in token.children or []:
        match child.type:
            case "text":
                parts.append(child.content)
            case "softbreak" | "hardbreak":
                parts.append(" ")
            case "strong_open":
                bold += 1
                starts_bold = starts_bold or not "".join(parts).strip()
            case _:
                pass
    return "".join(parts), bold, starts_bold


def clean_markdown(raw_markdown: str, *, unescape: bool = False) -> CleanText:
    """Clean Markdown (Stack Exchange bodies and all model output).

    Code blocks, inline code, images, HTML and block quotes are removed; link
    text is kept; headings and list items become their own blocks.

    Args:
        raw_markdown: The Markdown source
        unescape: Unescape HTML entities first (the Stack Exchange API escapes them)

    Returns:
        The cleaned text with formatting counts from the same rendering
    """
    source = html.unescape(raw_markdown) if unescape else raw_markdown
    tokens = _markdown().parse(source)
    blocks: list[Block] = []
    counts = {"list_items": 0, "headings": 0, "bold_spans": 0, "bold_lead_items": 0}
    quote_depth = 0
    item_depth = 0
    table_depth = 0
    heading = False
    quoted_words = 0
    has_code = False
    for token in tokens:
        match token.type:
            case "blockquote_open":
                quote_depth += 1
            case "blockquote_close":
                quote_depth -= 1
            case "list_item_open":
                item_depth += 1
                counts["list_items"] += quote_depth == 0
            case "list_item_close":
                item_depth -= 1
            case "table_open":
                table_depth += 1
            case "table_close":
                table_depth -= 1
            case "heading_open":
                heading = True
                counts["headings"] += quote_depth == 0
            case "heading_close":
                heading = False
            case "fence" | "code_block":
                has_code = True
            case "inline":
                text, bold, starts_bold = _inline_text(token)
                if quote_depth:
                    quoted_words += count_words(text)
                    continue
                counts["bold_spans"] += bold
                kind: BlockKind = (
                    "heading"
                    if heading
                    else "item"
                    if item_depth or table_depth
                    else "para"
                )
                counts["bold_lead_items"] += kind == "item" and starts_bold
                if cleaned := normalize(text):
                    blocks.append(Block(kind, cleaned))
            case _:
                pass
    formatting = Formatting(**counts, emoji=len(SYMBOL_EMOJI_RE.findall(source)))
    return CleanText(tuple(blocks), formatting, quoted_words, has_code)
