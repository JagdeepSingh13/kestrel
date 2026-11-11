"""Every built-in tool, by name. Agent files pick from this list."""

from kestrel.tools.files import list_dir, read_file, write_file
from kestrel.tools.interact import ask_user
from kestrel.tools.process import python_exec, run_shell
from kestrel.tools.registry import Tool

BUILTIN_TOOLS: dict[str, Tool] = {
    t.name: t for t in [list_dir, read_file, write_file, run_shell, python_exec, ask_user]
}
