"""Integration tests for the FastMCP server boot path.

These tests exercise tool/prompt/resource registration without spawning a
real MCP client. They mock the underlying I/O layer (cache.prewarm,
cache.start_background_refresh, db.recent_notes) so importing the server
module doesn't block on real AppleScript or SQLite.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def server_module(mocker):
    """Import apple_notes_brain.server with all I/O subsystems mocked.

    Forces a fresh import so prior tests can't leak state. Returns the
    imported module.
    """
    # Patch BEFORE the module imports so _startup_log() sees the mocks.
    mocker.patch("apple_notes_brain.cache.prewarm", return_value=True)
    mocker.patch("apple_notes_brain.cache.start_background_refresh", return_value=False)
    mocker.patch("apple_notes_brain.cache.sync_after_write", return_value=None)
    mocker.patch("apple_notes_brain.sqlite_reader.recent_notes", return_value=[])

    import importlib
    import sys

    # Drop any prior import so _startup_log re-runs with the mocks.
    for mod_name in list(sys.modules):
        if mod_name.startswith("apple_notes_brain.server"):
            del sys.modules[mod_name]

    import apple_notes_brain.server as server_mod  # noqa: WPS433
    importlib.reload(server_mod)
    return server_mod


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_mcp_instance_has_correct_name(server_module):
    """FastMCP server identifies as 'apple-notes-brain'."""
    assert server_module.mcp.name == "apple-notes-brain"


def test_list_tools_returns_all_registered_tools(server_module):
    """All 16 documented MCP tools are registered (12 lexical + 4 semantic).

    v1.0.x had 12 tools (the lexical/CRUD set). v1.1 added four
    embedding-backed tools that require the [semantic] install extra;
    those return a structured `missing-extras` error when the extra
    isn't installed but are ALWAYS registered so MCP clients can
    discover them.
    """
    tools = asyncio.run(server_module.mcp.list_tools())
    names = sorted(t.name for t in tools)
    expected = sorted([
        # v1.0 lexical / CRUD surface
        "list_folders",
        "list_notes",
        "search_notes",
        "get_note",
        "create_note",
        "update_note",
        "rename_note",
        "move_note",
        "create_folder",
        "rename_folder",
        "delete_folder",
        "delete_note",
        # v1.1 semantic + hybrid additions
        "semantic_search",
        "hybrid_search",
        "reindex_semantic",
        "semantic_index_status",
    ])
    assert names == expected, f"tool list drift; got {names}"


def test_read_only_tools_carry_read_only_annotation(server_module):
    """Tools that don't mutate are annotated readOnlyHint=True."""
    tools = asyncio.run(server_module.mcp.list_tools())
    by_name = {t.name: t for t in tools}
    for read_only in ("list_folders", "list_notes", "search_notes", "get_note"):
        ann = by_name[read_only].annotations
        assert ann is not None, f"{read_only} missing annotations"
        assert ann.readOnlyHint is True, f"{read_only} should be readOnly"
        assert ann.idempotentHint is True
        assert ann.openWorldHint is False


def test_write_tools_carry_write_annotation(server_module):
    """Mutating-but-not-destructive tools are annotated readOnlyHint=False, destructiveHint=False."""
    tools = asyncio.run(server_module.mcp.list_tools())
    by_name = {t.name: t for t in tools}
    for write_tool in ("create_note", "update_note", "rename_note", "move_note", "create_folder", "rename_folder"):
        ann = by_name[write_tool].annotations
        assert ann is not None, f"{write_tool} missing annotations"
        assert ann.readOnlyHint is False, f"{write_tool} should not be readOnly"
        assert ann.destructiveHint is False, f"{write_tool} should not be destructiveHint=True"


def test_destructive_tools_carry_destructive_annotation(server_module):
    """Tools that delete data carry destructiveHint=True."""
    tools = asyncio.run(server_module.mcp.list_tools())
    by_name = {t.name: t for t in tools}
    for destructive in ("delete_note", "delete_folder"):
        ann = by_name[destructive].annotations
        assert ann is not None, f"{destructive} missing annotations"
        assert ann.readOnlyHint is False
        assert ann.destructiveHint is True, f"{destructive} should be destructiveHint=True"


def test_every_tool_has_non_empty_description(server_module):
    """All tools have descriptions (LLMs use them to decide tool selection)."""
    tools = asyncio.run(server_module.mcp.list_tools())
    for tool in tools:
        assert tool.description, f"{tool.name} has empty description"
        assert len(tool.description) > 50, f"{tool.name} description suspiciously short ({len(tool.description)} chars)"


def test_every_tool_has_input_schema(server_module):
    """All tools have a JSON Schema for inputs."""
    tools = asyncio.run(server_module.mcp.list_tools())
    for tool in tools:
        assert tool.inputSchema is not None, f"{tool.name} missing inputSchema"
        assert tool.inputSchema.get("type") == "object", f"{tool.name} inputSchema not object-shaped"


# ---------------------------------------------------------------------------
# Prompt registration
# ---------------------------------------------------------------------------


def test_list_prompts_returns_overview_prompt(server_module):
    """The notes_server_overview prompt is registered."""
    prompts = asyncio.run(server_module.mcp.list_prompts())
    names = [p.name for p in prompts]
    assert "notes_server_overview" in names


def test_overview_prompt_has_no_required_arguments(server_module):
    """The architecture overview prompt is parameter-free."""
    prompts = asyncio.run(server_module.mcp.list_prompts())
    overview = next(p for p in prompts if p.name == "notes_server_overview")
    assert overview.arguments is None or len(overview.arguments) == 0


# ---------------------------------------------------------------------------
# Resource registration
# ---------------------------------------------------------------------------


def test_list_resources_returns_list(server_module):
    """list_resources returns a (possibly empty) list without raising."""
    resources = asyncio.run(server_module.mcp.list_resources())
    assert isinstance(resources, list)


def test_populate_recent_resources_handles_empty_db(server_module, mocker):
    """_populate_recent_resources gracefully handles an empty NoteStore."""
    mocker.patch("apple_notes_brain.sqlite_reader.recent_notes", return_value=[])
    count = server_module._populate_recent_resources(count=10)
    assert count == 0


def test_populate_recent_resources_skips_locked_notes(server_module, mocker):
    """Locked notes are excluded from the @-mention autocomplete list."""
    mocker.patch(
        "apple_notes_brain.sqlite_reader.recent_notes",
        return_value=[
            {"id": "p1", "title": "Open", "locked": False},
            {"id": "p2", "title": "Locked", "locked": True},
            {"id": "p3", "title": "Open 2", "locked": False},
        ],
    )
    count = server_module._populate_recent_resources(count=10)
    assert count == 2  # locked one skipped
