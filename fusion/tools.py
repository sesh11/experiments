"""Tool schemas and a dispatcher over a Workspace.

Tools are grouped so policies can hand an agent exactly the surface it should
have: a read-only Scout gets navigation tools only; a write-capable agent also
gets edit + run_tests. The `scout` and `finish` tools are handled by the
orchestrator, not here.
"""

from __future__ import annotations

from .workspace import Workspace

# --- tool schemas -----------------------------------------------------------
READ_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file's contents (line-numbered). Optionally pass start/end line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories at a path (default: repo root).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    },
    {
        "name": "search",
        "description": "Regex search across all .py files. Returns file:line: text hits.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]

WRITE_TOOLS = [
    {
        "name": "str_replace",
        "description": "Replace an exact unique substring in a file with new text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "create_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run the repository test suite. Returns pass/fail and output. "
            "On large repos, pass `target` (a test file/dir path, or a pytest "
            "expression like `-k name`) to scope the run — whole-suite runs may "
            "time out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Optional test path or pytest -k expression to scope the run.",
                },
            },
        },
    },
]

FINISH_TOOL = {
    "name": "finish",
    "description": "Declare the task complete. Provide a short summary of the change.",
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
}

SCOUT_TOOL = {
    "name": "scout",
    "description": (
        "Ask the read-only Scout sidekick to explore the repository and return a "
        "compact, cited map (file:line references + why they matter). Use this "
        "instead of reading files yourself: describe what you need to locate or "
        "understand."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}


class WorkspaceTools:
    """Dispatch model tool calls onto a Workspace."""

    def __init__(self, ws: Workspace):
        self.ws = ws

    def execute(self, name: str, inp: dict) -> tuple[str, bool]:
        """Returns (result_text, is_error)."""
        try:
            if name == "read_file":
                return self.ws.read_file(inp["path"], inp.get("start"), inp.get("end")), False
            if name == "list_dir":
                return self.ws.list_dir(inp.get("path", ".")), False
            if name == "search":
                return self.ws.search(inp["pattern"]), False
            if name == "str_replace":
                res = self.ws.str_replace(inp["path"], inp["old_str"], inp["new_str"])
                return res, res.startswith("ERROR")
            if name == "create_file":
                return self.ws.create_file(inp["path"], inp["content"]), False
            if name == "run_tests":
                ok, out = self.ws.run_tests(inp.get("target"))
                header = "TESTS PASSED\n" if ok else "TESTS FAILED\n"
                return header + out, False
            return f"ERROR: unknown tool {name}", True
        except Exception as exc:  # never let a tool crash the loop
            return f"ERROR: {type(exc).__name__}: {exc}", True
