# Interactive shell (M1.5, reworked in M3.5)

`kestrel` with no arguments (or `kestrel chat`) opens Kestrel's own prompt. Every message is one agent turn: the agent gets the conversation so far plus its tools and decides itself whether to use them. `/` commands control the session. `kestrel doctor`, `kestrel ask` and `kestrel run` keep working from the normal terminal.

## Structure

![What happens to each line you type](img/shell.svg)

- In a terminal, `Tui` (`cli/tui.py`) runs the shell: one prompt_toolkit app for the whole session holds the input box at the bottom, a spinner line above it and the permission menu. A worker thread takes typed lines from a queue and runs `Shell.handle`. All printing goes through `patch_stdout`, which draws it above the box, so the box never moves. Lines typed during a turn wait in the queue.
- Without a terminal (tests), `Shell.run` reads lines from a `read_line` function instead; tests pass scripted lines and a `FakeProvider`. `tests/test_cli_tui.py` drives the real box with piped key presses.
- `Shell.send` calls `start_run` (`cli/agent_run.py`) with `Session.conversation`, a `ContextManager` that lives as long as the shell. The run adds to it; a stopped or failed turn is cut back out.
- `CliFrontend` (`cli/frontend.py`) draws the run. With the box (`BottomBox`), its spinner, menu and `ask_user` questions go through the box, Esc cancels the turn from the box's thread, and the answer is printed once it is complete. `kestrel run` has no box: there `watch_stop_keys` (`cli/keys.py`) handles Esc, and the answer streams live.
- No terminal attached (pipes, tests): plain typed y/n/a answers and plain answer text.

## How one message is handled

1. The message is printed above the box as `> message` (the box stays where it is).
2. `start_run` sets the conversation's system prompt, window and tool list from the current agent (so `/agent` can switch between messages).
3. The agent loop runs: the context manager trims old turns to fit, the model streams, tools run through the permission gate.
4. The answer is printed as markdown after a `⏺` when it is complete; a summary line gives steps, tools, tokens, seconds and the run id.
5. The turn stays in the conversation if it ended normally or hit a limit; it is rolled back on Esc, Ctrl-C or a backend error.

## Data, before and after

The screen for one message:

```
> what is the total revenue per region in data/sales.csv?

⏺ read_file(data/sales.csv)
  ⎿  data/sales.csv: 7 lines total
     1| month,region,units,price
     2| jan,north,120,9.5
     … +5 lines
⏺ python_exec(import csv…)
╭─ python_exec wants to run ───────────────╮
│ import csv                               │
│ ...                                      │
╰──────────────────────────────────────────╯
  ⎿  allowed
  ⎿  exit code 0
     north 3760.0
     south 2720.0
⏺ North made 3755.0 and south 2720.0.
  3 steps · 2 tools · 4.1K tokens · 41s · run 20261112-091502-8f3a

╭──────────────────────────────────────────╮
│ >                                        │
╰──────────────────────────────────────────╯
  qwen3:8b · agent default · context 18% of 16K · /help
```

The conversation after it (what the next message builds on):

```python
[
    Message(role=user, content="what is the total revenue per region in data/sales.csv?"),
    Message(role=assistant, tool_calls=[read_file(path="data/sales.csv")]),
    Message(role=tool, content='<untrusted source="read_file:data/sales.csv">...'),
    Message(role=assistant, tool_calls=[python_exec(code="import csv...")]),
    Message(role=tool, content='<untrusted source="python_exec">exit code 0 ...'),
    Message(role=assistant, content="North made 3755.0 and south 2720.0."),
]
```

When it grows past the window, old tool results are emptied first, then the oldest turns are dropped. The newest message from you is never dropped. Reasoning is never kept.
