"""Render math with typst

## Usage

1. Install the markdown extensions pymdownx.arithmatex.
2. Add `math: typst` to pages' metadata.

## Requirements

- typst

"""

from __future__ import annotations

import html
import re
from functools import cache
from subprocess import CalledProcessError, run
from typing import TYPE_CHECKING
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page


def should_render(page: Page) -> bool:
    return page.meta.get("math") == "typst"


def on_page_markdown(
    markdown: str, page: Page, config: MkDocsConfig, files: Files
) -> str | None:
    if should_render(page):
        assert "pymdownx.arithmatex" in config.markdown_extensions, (
            "Missing pymdownx.arithmatex in config.markdown_extensions. "
            "Setting `math: typst` requires it to parse markdown."
        )


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str | None:
    if should_render(page):
        # Read configuration for typst integration from mkdocs config.extra
        extra = getattr(config, "extra", {}) or {}
        global _TYPST_CMD, _TYPST_CWD, _TYPST_CACHE_DIR
        _TYPST_CMD = extra.get("typst_cmd", _TYPST_CMD)
        _TYPST_CWD = extra.get("typst_cwd", _TYPST_CWD)
        _TYPST_CACHE_DIR = extra.get("typst_cache_dir", _TYPST_CACHE_DIR)
        # Read user prelude (imports) to inject before every compile
        global _TYPST_USER_PRELUDE
        _TYPST_USER_PRELUDE = extra.get("typst_prelude", _TYPST_USER_PRELUDE)
        _ensure_typst_available()

        output = re.sub(
            r'<span class="arithmatex">(.+?)</span>', render_inline_math, output
        )

        output = re.sub(
            r'<div class="arithmatex">(.+?)</div>',
            render_block_math,
            output,
            flags=re.MULTILINE | re.DOTALL,
        )
        return output


def render_inline_math(match: re.Match[str]) -> str:
    src = html.unescape(match.group(1)).removeprefix("$(").removesuffix(")$").strip()
    typ = f"${src}$"
    return (
        '<span class="typst-math">'
        + fix_svg(typst_compile(typ))
        + for_screen_reader(typ)
        + "</span>"
    )


def render_block_math(match: re.Match[str]) -> str:
    src = html.unescape(match.group(1)).removeprefix("$[").removesuffix("]$").strip()
    typ = f"$\n{src}\n$"
    return (
        '<div class="typst-math">'
        + fix_svg(typst_compile(typ))
        + for_screen_reader(typ)
        + "</div>"
    )


def for_screen_reader(typ: str) -> str:
    return f'<span class="sr-only">{html.escape(typ)}</span>'


def fix_svg(svg: bytes) -> str:
    """Fix the compiled SVG to be embedded in HTML

    - Strip trailing spaces
    - Support dark theme
    """
    return re.sub(
        r' (fill|stroke)="#000000"',
        r' \1="var(--md-typeset-color)"',
        svg.decode().strip(),
    )


@cache
def typst_compile(
    typ: str,
    *,
    prelude="#set page(width: auto, height: auto, margin: 0pt, fill: none)\n",
    format="svg",
) -> bytes:
    """Compile a Typst document

    https://github.com/marimo-team/marimo/discussions/2441
    """
    # Use a disk cache to avoid recompiling identical fragments across builds
    cache_dir = Path(_TYPST_CACHE_DIR or Path.cwd() / ".typst_cache")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_dir = Path(tempfile.gettempdir())

    # Merge user prelude (if any) so cache key and input include it
    user_prelude = (_TYPST_USER_PRELUDE or "")
    final_prelude = user_prelude + prelude
    key = hashlib.sha256((final_prelude + typ + format).encode()).hexdigest()
    cache_file = cache_dir / f"{key}.svg"
    if cache_file.exists():
        return cache_file.read_bytes()

    cmd = [_TYPST_CMD, "compile", "-", "-", "--format", format]
    cwd = _TYPST_CWD or None
    try:
        proc = run(
            cmd,
            input=(final_prelude + typ).encode(),
            check=True,
            capture_output=True,
            cwd=cwd,
        )
        out = proc.stdout
        # Validate that output looks like SVG
        if not out.lstrip().startswith(b"<svg"):
            # Fallback: wrap the original typ text into a simple SVG so page still renders
            fallback = (
                '<svg xmlns="http://www.w3.org/2000/svg"><desc>typst fallback</desc><text x="0" y="14' +
                '">' + html.escape(typ) + '</text></svg>'
            ).encode()
            out = fallback

        # Atomically write cache
        try:
            tmp_path = cache_file.with_suffix(".tmp")
            tmp_path.write_bytes(out)
            os.replace(tmp_path, cache_file)
        except Exception:
            pass

        return out
    except FileNotFoundError:
        # typst not installed
        return (
            '<svg xmlns="http://www.w3.org/2000/svg"><desc>typst not found</desc><text x="0" y="14">'
            + html.escape(typ)
            + '</text></svg>'
        ).encode()
    except CalledProcessError as err:
        # On compilation failure, don't abort the build; return a readable SVG fallback
        err_msg = err.stderr.decode(errors="ignore")
        fallback = (
            '<svg xmlns="http://www.w3.org/2000/svg"><desc>typst error</desc><text x="0" y="14">'
            + html.escape(typ)
            + '</text><text x="0" y="32">'
            + html.escape(err_msg[:200])
            + '</text></svg>'
        ).encode()
        return fallback


# Module-level configuration defaults
_TYPST_CMD = "typst"
_TYPST_CWD: str | None = None
_TYPST_CACHE_DIR: str | None = None
_TYPST_AVAILABLE = None
_TYPST_USER_PRELUDE: str | None = None


def _ensure_typst_available() -> None:
    global _TYPST_AVAILABLE
    if _TYPST_AVAILABLE is not None:
        return
    cmd = shutil.which(_TYPST_CMD)
    if not cmd:
        print(f"[typst_math] typst executable not found: '{_TYPST_CMD}' (will use fallback SVG)")
        _TYPST_AVAILABLE = False
        return
    try:
        proc = run([_TYPST_CMD, "--version"], capture_output=True, check=True)
        ver = proc.stdout.decode().strip() or proc.stderr.decode().strip()
        print(f"[typst_math] typst found: {ver}")
        _TYPST_AVAILABLE = True
    except Exception:
        _TYPST_AVAILABLE = False