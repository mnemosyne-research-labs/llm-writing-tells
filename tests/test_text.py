import pytest

from llm_tells.text import (
    Block,
    CleanText,
    Formatting,
    clean_hn,
    clean_markdown,
    count_words,
    is_english,
    normalize,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("It's late, isn't it?", 4),
        ("It\u2019s late.", 2),
        ("We left at 10:30.", 5),
        ("well-known rock'n'roll", 3),
        ("", 0),
        ("--- ... !!!", 0),
    ],
)
def test_count_words(text: str, expected: int):
    assert count_words(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\u201cHi\u201d  she\tsaid\u00a0twice ", '"Hi" she said twice'),
        ("See https://example.com/a?b=1 or www.example.org now.", "See or now."),
        ("Some <b>bold</b> text", "Some bold text"),
    ],
)
def test_normalize(text: str, expected: str):
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The cat is on the mat.", True),
        ("the cat dog fish bird", True),
        ("the cat dog fish bird owl", False),
        ("the cats is \u00e9", True),
        ("the cat \u00e9\u00e9", False),
        ("Cats dogs birds fish", False),
        ("\u042d\u0442\u043e \u043d\u0435 \u0442\u0430\u043a, it is", False),
        ("", False),
        ("123 456", False),
    ],
)
def test_is_english(text: str, expected: bool):
    assert is_english(text) is expected


def test_clean_text_joins_blocks_and_counts_words():
    clean = CleanText((Block("heading", "A title"), Block("para", "It's here.")))
    assert clean.text == "A title\n\nIt's here."
    assert clean.n_words == 4
    assert clean.formatting == Formatting()


def test_clean_hn_splits_paragraphs_and_unescapes():
    clean = clean_hn("First &amp; best.<p>It&#x27;s <i>fine</i>.<P>&quot;Done&quot;")
    assert clean.blocks == (
        Block("para", "First & best."),
        Block("para", "It's fine."),
        Block("para", '"Done"'),
    )
    assert (clean.quoted_words, clean.has_code) == (0, False)


def test_clean_hn_removes_quoted_lines_and_counts_their_words():
    clean = clean_hn("&gt; four words quoted here<p>My reply.<p>&gt;two more")
    assert clean.blocks == (Block("para", "My reply."),)
    assert clean.quoted_words == 6


def test_clean_hn_removes_code_and_keeps_link_text_without_urls():
    clean = clean_hn(
        'See <a href="https://x.org">the docs</a> or https://x.org now.'
        "<p><pre><code>x = 1\ny = 2</code></pre><p>After."
    )
    assert clean.blocks == (
        Block("para", "See the docs or now."),
        Block("para", "After."),
    )
    assert clean.has_code is True


def test_clean_hn_drops_empty_paragraphs():
    assert clean_hn("<p><p> <p>").blocks == ()


def test_clean_markdown_blocks_by_kind():
    clean = clean_markdown("# Title\n\nSome prose here.\n\n- one\n- two\n\n1. three\n")
    assert clean.blocks == (
        Block("heading", "Title"),
        Block("para", "Some prose here."),
        Block("item", "one"),
        Block("item", "two"),
        Block("item", "three"),
    )
    assert (clean.formatting.list_items, clean.formatting.headings) == (3, 1)


def test_clean_markdown_drops_inline_code_images_and_urls():
    clean = clean_markdown(
        "Run `make` and see [the guide](https://e.com) or ![a chart](c.png) "
        "at www.example.com/a today."
    )
    assert clean.text == "Run and see the guide or at today."


def test_clean_markdown_removes_code_blocks():
    clean = clean_markdown(
        "Before.\n\n```python\nx = 1\n```\n\n    indented = 2\n\nAfter."
    )
    assert clean.text == "Before.\n\nAfter."
    assert clean.has_code is True


def test_clean_markdown_without_code():
    assert clean_markdown("Plain.").has_code is False


def test_clean_markdown_removes_block_quotes_and_their_formatting():
    clean = clean_markdown(
        "> quoted line one two\n>\n> # quoted heading\n> - quoted item\n\nMine."
    )
    assert clean.blocks == (Block("para", "Mine."),)
    assert clean.quoted_words == 8
    assert (clean.formatting.headings, clean.formatting.list_items) == (0, 0)


def test_clean_markdown_drops_blocks_left_empty():
    clean = clean_markdown("![a chart](c.png)\n\n- `code`\n\nText.")
    assert clean.blocks == (Block("para", "Text."),)
    assert clean.formatting.list_items == 1


def test_clean_markdown_joins_line_breaks():
    clean = clean_markdown("Line one\nline two  \nhard break.")
    assert clean.blocks == (Block("para", "Line one line two hard break."),)


def test_clean_markdown_shows_raw_html_as_text_and_strips_tags():
    clean = clean_markdown("<div>raw html</div>\n\nAn <em>inline</em> tag.")
    assert clean.blocks == (Block("para", "raw html"), Block("para", "An inline tag."))


def test_clean_markdown_table_cells_are_items():
    clean = clean_markdown("| a | b |\n| --- | --- |\n| cell one | cell two |\n")
    assert clean.blocks == tuple(
        Block("item", text) for text in ("a", "b", "cell one", "cell two")
    )
    assert clean.formatting.list_items == 0


def test_clean_markdown_bold_counts():
    clean = clean_markdown(
        "# Title **bold**\n\n- **Lead** one\n- **Lead** two\n- plain **late** bold\n"
    )
    assert clean.formatting == Formatting(
        list_items=3, headings=1, bold_spans=4, bold_lead_items=2
    )
    assert clean.text == "Title bold\n\nLead one\n\nLead two\n\nplain late bold"


def test_clean_markdown_bold_paragraph_is_not_a_bold_lead_item():
    assert clean_markdown("**Lead** in prose.").formatting.bold_lead_items == 0


def test_clean_markdown_counts_emoji():
    clean = clean_markdown("Done \U0001f600 and \u2600 and \U0001f1e8\U0001f1e6.")
    assert clean.formatting.emoji == 4


@pytest.mark.parametrize(
    ("unescape", "expected"), [(True, "x & y"), (False, "x &amp; y")]
)
def test_clean_markdown_unescape(unescape: bool, expected: str):
    assert clean_markdown("x &amp;amp; y", unescape=unescape).text == expected
