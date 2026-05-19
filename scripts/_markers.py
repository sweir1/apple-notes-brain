"""Shared helpers for marker-bounded markdown regions.

Every auto-generated section in this repo is wrapped in HTML comments of the
form:

    <!-- GENERATED:<name> — DO NOT EDIT. ...instructions... -->
    ...auto-content...
    <!-- /GENERATED:<name> -->

The generators replace ONLY what's between the markers. Everything outside is
preserved byte-for-byte. The `--check` mode of each generator compares the
file on disk against what would be written, and exits non-zero on drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _build_pattern(marker: str) -> re.Pattern[str]:
    open_re = re.escape(f"<!-- GENERATED:{marker}")
    close_re = re.escape(f"<!-- /GENERATED:{marker} -->")
    # Match the entire bounded region including the opening + closing
    # comments. The opening comment may carry extra commentary after the
    # marker name and before the closing `-->`.
    return re.compile(
        rf"({open_re}[^>]*-->)(.*?)({close_re})",
        re.DOTALL,
    )


def render_into(path: Path, marker: str, new_block: str) -> str:
    """Return what the file would look like with the marker section replaced.

    `new_block` is the content between the opening and closing markers. A
    single leading + trailing newline are inserted automatically so the
    surrounding HTML comments sit on their own lines.
    """
    text = path.read_text(encoding="utf-8")
    pat = _build_pattern(marker)
    m = pat.search(text)
    if m is None:
        raise SystemExit(
            f"error: marker `{marker}` not found in {path}. "
            f"Expected `<!-- GENERATED:{marker} ... -->` and "
            f"`<!-- /GENERATED:{marker} -->` on their own lines."
        )
    return pat.sub(
        lambda mm: f"{mm.group(1)}\n{new_block.rstrip()}\n{mm.group(3)}",
        text,
    )


def apply_or_check(path: Path, marker: str, new_block: str, check: bool) -> int:
    """Write the regenerated file, or in check mode diff-print and exit non-zero on drift.

    Returns the suggested process exit code.
    """
    rendered = render_into(path, marker, new_block)
    current = path.read_text(encoding="utf-8")
    if rendered == current:
        if not check:
            # No-op write avoids touching mtime when nothing changed.
            return 0
        return 0
    if check:
        sys.stderr.write(
            f"\nDrift detected in {path} (marker `{marker}`).\n"
            f"Run the generator without --check to regenerate.\n\n"
        )
        # Tiny inline diff so the failure is actionable in CI logs.
        import difflib

        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"{path} (on disk)",
            tofile=f"{path} (regenerated)",
            n=3,
        )
        sys.stderr.writelines(diff)
        return 1
    path.write_text(rendered, encoding="utf-8")
    print(f"wrote {path} (marker `{marker}`)")
    return 0
