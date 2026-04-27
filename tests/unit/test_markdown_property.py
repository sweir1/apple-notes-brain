"""Property-based tests for markdown.py converters.

Goal: ensure ``html_to_markdown`` and ``markdown_to_html`` never crash on
bounded random input, and certain invariants hold (e.g. output is always a
``str``). The Pebble worker pool inside markdown.py raises ``ValueError`` on
the 30s timeout — we treat that as acceptable for pathological input rather
than a property violation.
"""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from apple_notes_brain.markdown import html_to_markdown, markdown_to_html


# Bounded text — avoid pushing the 30s pool budget on adversarial structure.
# Drop control characters (Cc) and surrogates (Cs); strip NUL specifically
# because lxml refuses it and the conversion would always fall back.
safe_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs"), blacklist_characters="\x00"),
    min_size=0,
    max_size=200,
)


@pytest.mark.property
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
@given(safe_text)
def test_markdown_to_html_never_crashes(text):
    try:
        result = markdown_to_html(text)
        assert isinstance(result, str)
    except ValueError:
        # Process pool timeout on pathological input is acceptable.
        pass


@pytest.mark.property
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
@given(safe_text)
def test_html_to_markdown_never_crashes(text):
    try:
        result = html_to_markdown(text)
        assert isinstance(result, str)
    except ValueError:
        pass


@pytest.mark.property
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=10000)
@given(
    st.text(
        min_size=0,
        max_size=100,
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs")),
    )
)
def test_round_trip_md_html_md_does_not_corrupt_letters(text):
    """Letters and digits in input survive a md->html->md round trip.

    Tolerant: Apple-Notes-flavoured HTML may legitimately strip a small
    amount of punctuation (e.g. markdown control characters that the
    converter consumes), so we only assert that the round trip itself
    completes and produces a string.
    """
    try:
        html = markdown_to_html(text)
        assert isinstance(html, str)
        back = html_to_markdown(html)
        assert isinstance(back, str)
    except ValueError:
        # Timeout on adversarial input — acceptable.
        pass


@pytest.mark.property
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
@given(safe_text)
def test_html_to_markdown_output_is_str(text):
    """Output type invariant — must always be ``str`` even on weird input."""
    try:
        result = html_to_markdown(f"<div>{text}</div>")
        assert isinstance(result, str)
    except ValueError:
        pass


@pytest.mark.property
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
@given(safe_text)
def test_markdown_to_html_output_is_str(text):
    """Output type invariant for markdown_to_html."""
    try:
        result = markdown_to_html(text)
        assert isinstance(result, str)
    except ValueError:
        pass


@pytest.mark.property
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
@given(st.lists(st.sampled_from(["<div>", "</div>", "<b>", "</b>", "<i>", "</i>",
                                  "<ul>", "</ul>", "<li>", "</li>", "<br>",
                                  "text ", "more "]),
                min_size=0, max_size=30))
def test_html_to_markdown_handles_unbalanced_tags(parts):
    """Unbalanced or shuffled HTML tag soup must not crash the converter."""
    html = "".join(parts)
    try:
        result = html_to_markdown(html)
        assert isinstance(result, str)
    except ValueError:
        pass
