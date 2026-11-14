"""Every built-in tool, by name. Agent files pick from this list."""

from kestrel.tools.checks import run_tests
from kestrel.tools.code import repo_map
from kestrel.tools.files import (
    append_file,
    edit_file,
    list_dir,
    read_file,
    search_files,
    write_file,
)
from kestrel.tools.git import git_diff, git_status
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
        run_tests,
        repo_map,
        git_status,
        git_diff,
        ask_user,
        research,
    ]
}
