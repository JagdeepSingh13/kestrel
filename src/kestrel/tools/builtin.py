"""Every built-in tool, by name. Agent files pick from this list."""

from kestrel.tools.files import (
    append_file,
    edit_file,
    list_dir,
    read_file,
    search_files,
    write_file,
)
from kestrel.tools.interact import ask_user
from kestrel.tools.process import python_exec, run_shell
from kestrel.tools.registry import Tool
from kestrel.tools.research import research

BUILTIN_TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        list_dir,
        read_file,
        search_files,
        write_file,
        append_file,
        edit_file,
        run_shell,
        python_exec,
        ask_user,
        research,
    ]
}
