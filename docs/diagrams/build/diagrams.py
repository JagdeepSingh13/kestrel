"""Every Kestrel diagram, defined in one place.

Run `python3 docs/diagrams/build/diagrams.py` to write `src/<name>.json` (Excalidraw
skeletons), then open build.html (see server.py) to render them with Excalidraw into
`docs/diagrams/img/<name>.svg` (published) and `.excalidraw` (local only, not in git).

Style rules: one picture per idea, big enough to read on GitHub (about 920 wide, text
18 to 20), boxes that explain themselves in a line or two, short straight arrows, labels
beside arrows (never on them), plain words. Arrows connect boxes along the middle of where they overlap, so line boxes
up (same row or same column) to keep arrows straight.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

SRC = Path(__file__).parent / "src"

YELLOW, BLUE, VIOLET, RED = "#ffec99", "#a5d8ff", "#d0bfff", "#ffc9c9"
GREEN, GRAY, ORANGE, TEAL = "#b2f2bb", "#e9ecef", "#ffd8a8", "#96f2d7"
NOTE = "#495057"


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Diagram:
    name: str
    elements: list[dict[str, object]] = field(default_factory=list)
    boxes: dict[str, Box] = field(default_factory=dict)

    def box(
        self,
        id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        color: str,
        text: str,
        *,
        shape: str = "rectangle",
        size: int = 20,
    ) -> None:
        self.boxes[id] = Box(x, y, w, h)
        self.elements.append(
            {
                "type": shape,
                "id": id,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "backgroundColor": color,
                "fillStyle": "solid",
                "roundness": {"type": 3},
                "label": {"text": text, "fontSize": size},
            }
        )

    def title(self, x: float, y: float, text: str) -> None:
        """A heading over a part of the picture."""
        self.text(x, y, text, size=22, color="#1e1e1e")

    def text(self, x: float, y: float, text: str, size: int = 18, color: str = NOTE) -> None:
        self.elements.append(
            {"type": "text", "x": x, "y": y, "text": text, "fontSize": size, "strokeColor": color}
        )

    def arrow(
        self, a: str, b: str, label: str | None = None, *, shift: float = 0, dashed: bool = False
    ) -> None:
        """Straight arrow from box a to box b, through the middle of their overlap."""
        p, q = self.boxes[a], self.boxes[b]
        gap = 3
        if q.y >= p.y + p.h or q.y + q.h <= p.y:  # one above the other
            x = (max(p.x, q.x) + min(p.x + p.w, q.x + q.w)) / 2 + shift
            down = q.y >= p.y + p.h
            y0 = p.y + p.h + gap if down else p.y - gap
            y1 = q.y - gap if down else q.y + q.h + gap
            start, end = (x, y0), (x, y1)
            label_at = (x + 12, (y0 + y1) / 2 - 12)
        else:  # side by side
            y = (max(p.y, q.y) + min(p.y + p.h, q.y + q.h)) / 2 + shift
            right = q.x >= p.x + p.w
            x0 = p.x + p.w + gap if right else p.x - gap
            x1 = q.x - gap if right else q.x + q.w + gap
            start, end = (x0, y), (x1, y)
            label_at = ((x0 + x1) / 2 - 5 * len(label or ""), y + (10 if shift > 0 else -32))
        dx, dy = end[0] - start[0], end[1] - start[1]
        self.elements.append(
            {
                "type": "arrow",
                "x": start[0],
                "y": start[1],
                "width": dx or 1,
                "height": dy or 1,
                "start": {"id": a},
                "end": {"id": b},
                "strokeStyle": "dashed" if dashed else "solid",
            }
        )
        if label:
            self.text(*label_at, label)

    def save(self) -> None:
        (SRC / f"{self.name}.json").write_text(json.dumps(self.elements, indent=1))


def overview() -> Diagram:
    """README: the whole of Kestrel on one page."""
    d = Diagram("overview")
    d.box(
        "you",
        0,
        0,
        420,
        110,
        BLUE,
        "Where you type\nthe Kestrel shell: chat with a model,\n/run an agent, /trace a run",
    )
    d.box(
        "log",
        500,
        0,
        420,
        110,
        YELLOW,
        "Run log\nevery step saved, secrets removed;\n/trace shows it as a timeline",
    )
    d.box(
        "loop",
        0,
        170,
        920,
        110,
        VIOLET,
        "Agent loop + memory\nask the model, run the tools it asks for, show it the results, repeat;\nkeeps the conversation inside the model's small window",
    )
    d.box(
        "rel",
        0,
        340,
        420,
        110,
        ORANGE,
        "Reliability (M4, next)\nfixes small models' broken tool calls,\nstops loops",
    )
    d.box(
        "tools",
        500,
        340,
        420,
        110,
        RED,
        "Tools + safety\nfiles, shell, Python, ask you;\nasks before risky actions",
    )
    d.box(
        "prov",
        0,
        510,
        420,
        110,
        GREEN,
        "Providers\none HTTP client for every backend;\nstreams replies, keeps thinking apart",
    )
    d.box("ws", 500, 510, 420, 110, YELLOW, "workspace/\nthe only folder agents can touch")
    d.box(
        "back",
        0,
        680,
        420,
        110,
        GRAY,
        "Backend on your machine\nOllama, LM Studio, llama.cpp, mlx-lm, vLLM\nruns the model",
        size=18,
    )
    d.arrow("you", "loop")
    d.arrow("loop", "log", "every step", dashed=True)
    d.arrow("loop", "rel")
    d.arrow("loop", "tools")
    d.arrow("rel", "prov")
    d.arrow("tools", "ws")
    d.arrow("prov", "back")
    return d


def providers() -> Diagram:
    """How one message reaches the model, and how /doctor finds backends."""
    d = Diagram("providers")
    d.title(0, -50, "Answering one message")
    steps = [
        ("1. You type a message", YELLOW),
        ("2. Kestrel turns the conversation\ninto the backend's JSON format", BLUE),
        ("3. Sent to the backend\nPOST /v1/chat/completions", BLUE),
        ("4. The reply streams back in pieces;\nthinking is kept apart from the answer", GREEN),
        ("5. Shown live; the last piece\nholds the complete reply", GREEN),
    ]
    for i, (text, color) in enumerate(steps):
        d.box(f"s{i}", 0, i * 150, 420, 100, color, text, size=18)
        if i:
            d.arrow(f"s{i - 1}", f"s{i}")
    d.title(500, -50, "Finding backends: /doctor")
    d.box(
        "doc",
        500,
        0,
        420,
        100,
        BLUE,
        "/doctor asks every known port\nat the same time (1.5 s limit)",
        size=18,
    )
    d.box(
        "ports",
        500,
        160,
        420,
        190,
        GRAY,
        "Ollama :11434\nLM Studio :1234\nllama.cpp or mlx-lm :8080\n(asks /props to tell them apart)\nvLLM :8000",
        size=18,
    )
    d.box(
        "table",
        500,
        410,
        420,
        140,
        GREEN,
        "A table: running or not,\nwhich models each one has,\nwarnings (Ollama's 4K default,\nREMOTE = data leaves your machine)",
        size=18,
    )
    d.arrow("doc", "ports")
    d.arrow("ports", "table")
    d.text(
        500,
        600,
        "If a backend is not up yet, Kestrel retries the\nconnection twice. A reply that already started\nis never silently repeated.",
        18,
    )
    return d


def shell() -> Diagram:
    """How the shell handles each line you type."""
    d = Diagram("shell")
    d.box(
        "line", 0, 0, 400, 90, YELLOW, "You type a line\n(Tab completes, up arrow recalls)", size=18
    )
    d.box("q", 60, 150, 280, 150, VIOLET, "Starts\nwith / ?", shape="diamond")
    d.box(
        "cmd",
        500,
        120,
        420,
        210,
        ORANGE,
        "Run the command\n/run  /trace  /agent  /agents\n/tools  /workspace  /model\n/provider  /think  /doctor\n/clear  /help  /exit",
        size=18,
    )
    d.box(
        "send",
        0,
        370,
        400,
        100,
        BLUE,
        "Send it to the model with\nthe conversation so far",
        size=18,
    )
    d.box(
        "reply",
        0,
        530,
        400,
        100,
        BLUE,
        "The reply streams in;\nCtrl-C stops it and drops that turn",
        size=18,
    )
    d.box(
        "keep",
        0,
        690,
        400,
        100,
        GREEN,
        "Both are added to the conversation;\nold turns are left out when it\noutgrows the model's window",
        size=18,
    )
    d.arrow("line", "q")
    d.arrow("q", "cmd", "yes")
    d.arrow("q", "send", "no")
    d.arrow("send", "reply")
    d.arrow("reply", "keep")
    d.text(500, 700, "Then back to the prompt.\nCtrl-D or /exit leaves.", 18)
    return d


def agent_run() -> Diagram:
    """One agent run: the loop, the window, and the checks before every tool."""
    d = Diagram("agent-run")
    d.box(
        "task", 0, 0, 420, 90, YELLOW, "Your task\n/run count the lines in every .py file", size=18
    )
    d.box(
        "ctx",
        0,
        150,
        420,
        130,
        ORANGE,
        "Fit the conversation into the window\nalways keeps the task and the latest step;\nempties old tool results first,\nthen drops the oldest steps",
        size=18,
    )
    d.box("ask", 0, 340, 420, 90, BLUE, "Ask the model\n(the reply streams in live)", size=18)
    d.box("q", 70, 490, 280, 150, VIOLET, "Did it ask\nfor a tool?", shape="diamond")
    d.box("answer", 0, 710, 420, 90, GREEN, "No: its reply is the answer", size=18)
    d.box(
        "result",
        540,
        150,
        380,
        130,
        GREEN,
        "Result added to the conversation,\nmarked untrusted (data, not orders)",
        size=18,
    )
    d.box(
        "run",
        540,
        440,
        380,
        250,
        RED,
        "Yes: run it, after checks\n\n1. known tool, right argument types\n2. permission: allow, ask you, or deny\n3. path stays inside workspace/\n4. time limit and output cap\n\nany failure comes back as ERROR",
        size=18,
    )
    d.arrow("task", "ctx")
    d.arrow("ctx", "ask")
    d.arrow("ask", "q")
    d.arrow("q", "answer")
    d.arrow("q", "run")
    d.arrow("run", "result")
    d.arrow("result", "ctx", "next step")
    d.text(
        540,
        730,
        "Also stops when out of steps, tokens\nor time, or on Ctrl-C. Every box\nwrites a line to the run log.",
        18,
    )
    return d


def run_log() -> Diagram:
    """How every step gets recorded, and what the record is used for."""
    d = Diagram("run-log")
    d.box("loop", 0, 60, 260, 120, VIOLET, "Agent loop\nannounces\neach step", size=18)
    d.box(
        "bus", 320, 40, 220, 160, YELLOW, "Event bus\nhands every step\nto each listener", size=18
    )
    d.box("screen", 660, 0, 260, 100, BLUE, "Screen\nshows it live", size=18)
    d.box("writer", 660, 150, 260, 110, RED, "Log writer\nremoves secrets first", size=18)
    d.box(
        "file",
        660,
        330,
        260,
        130,
        YELLOW,
        "Run log file\none line per step,\nwritten instantly",
        size=18,
    )
    d.box(
        "replay",
        280,
        330,
        260,
        130,
        GREEN,
        "Replay\nthe same run without\nthe model (tests)",
        size=18,
    )
    d.box("trace", 660, 530, 260, 100, GREEN, "/trace\na readable timeline", size=18)
    d.arrow("loop", "bus")
    d.arrow("bus", "screen", "listener")
    d.arrow("bus", "writer", "listener", shift=10)
    d.arrow("writer", "file")
    d.arrow("file", "replay")
    d.arrow("file", "trace")
    return d


def tools_and_safety() -> Diagram:
    """Agents are files; tools are guarded."""
    d = Diagram("tools-and-safety")
    d.box(
        "file",
        0,
        0,
        420,
        170,
        YELLOW,
        "An agent file (YAML)\nname: counter\ntools: [list_dir, read_file, write_file]\npermissions:\n  write_file: confirm",
        size=18,
    )
    d.box(
        "load",
        0,
        230,
        420,
        130,
        VIOLET,
        "Checked when loaded\nunknown settings or tools refused;\nlooser permissions need --trust-agent",
        size=18,
    )
    d.box(
        "run",
        0,
        420,
        420,
        110,
        BLUE,
        "/run uses the agent's model,\ninstructions, tools and limits",
        size=18,
    )
    d.box(
        "tools",
        540,
        0,
        380,
        200,
        GREEN,
        "Tools\nlist_dir, read_file\nwrite_file (asks before overwriting)\nrun_shell, python_exec (always ask)\nask_user",
        size=18,
    )
    d.box(
        "safety",
        540,
        260,
        380,
        270,
        RED,
        "Safety on every call\nonly workspace/ (paths outside,\nincluding via shortcuts, refused)\ncommands: time limit, output cap,\nno secret environment variables\noutput marked untrusted\n(not a sandbox, so they ask first)",
        size=18,
    )
    d.arrow("file", "load")
    d.arrow("load", "run")
    d.arrow("run", "safety", "every call")
    d.arrow("safety", "tools", "guards")
    return d


def core_types() -> Diagram:
    d = Diagram("core-types")
    d.box("call", 0, 0, 300, 100, RED, "ToolCall\nid, name, arguments")
    d.box(
        "msg", 0, 170, 300, 120, BLUE, "Message\nrole + content\n+ tool calls + reasoning", size=18
    )
    d.box(
        "spec",
        0,
        360,
        300,
        120,
        ORANGE,
        "ToolSpec\na tool's name +\nJSON schema of arguments",
        size=18,
    )
    d.box(
        "prov",
        390,
        170,
        240,
        310,
        VIOLET,
        "Provider\n(any backend)\n\nchat()\nstream()\nlist_models()",
    )
    d.box(
        "resp", 720, 170, 300, 120, GREEN, "ModelResponse\nthe full reply\n+ token usage", size=18
    )
    d.box(
        "chunk",
        720,
        360,
        300,
        120,
        GREEN,
        "StreamChunk\na piece of the reply;\nthe last holds it all",
        size=18,
    )
    d.arrow("call", "msg", "inside")
    d.arrow("msg", "prov", "in")
    d.arrow("spec", "prov", "in")
    d.arrow("prov", "resp", "out")
    d.arrow("prov", "chunk", "out")
    return d


def all_diagrams() -> list[Diagram]:
    return [
        overview(),
        providers(),
        shell(),
        agent_run(),
        run_log(),
        tools_and_safety(),
        core_types(),
    ]


if __name__ == "__main__":
    SRC.mkdir(exist_ok=True)
    for old in SRC.glob("*.json"):
        old.unlink()  # drop skeletons of diagrams that no longer exist
    diagrams = all_diagrams()
    for diagram in diagrams:
        diagram.save()
    (SRC / "index.json").write_text(json.dumps([d.name for d in diagrams]))
    print(f"wrote {len(diagrams)} skeletons to {SRC}")
