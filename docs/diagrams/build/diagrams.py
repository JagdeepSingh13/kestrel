"""Every Kestrel diagram, defined in one place.

Run `python3 docs/diagrams/build/diagrams.py` to write `src/<name>.json` (Excalidraw
skeletons), then open build.html (see server.py) to render them with Excalidraw into
`docs/diagrams/img/<name>.svg` and `.excalidraw`.

Style rules: few boxes, short straight arrows, labels beside arrows (never on them),
plain words. Arrows connect boxes along the middle of where they overlap, so line boxes
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
        size: int = 18,
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

    def text(self, x: float, y: float, text: str, size: int = 16, color: str = NOTE) -> None:
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
            label_at = (x + 10, (y0 + y1) / 2 - 10)
        else:  # side by side
            y = (max(p.y, q.y) + min(p.y + p.h, q.y + q.h)) / 2 + shift
            right = q.x >= p.x + p.w
            x0 = p.x + p.w + gap if right else p.x - gap
            x1 = q.x - gap if right else q.x + q.w + gap
            start, end = (x0, y), (x1, y)
            label_at = ((x0 + x1) / 2 - 4.5 * len(label or ""), y + (10 if shift > 0 else -30))
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


def layers(name: str, code: bool) -> Diagram:
    """The layers of Kestrel. `code=True` labels them with folders (for docs/diagrams)."""
    d = Diagram(name)
    t = (
        {
            "fe": "cli/\nshell, commands",
            "loop": "core/loop.py + core/context.py",
            "rel": "reliability/\n(M4)",
            "tools": "tools/ + safety/\n(M3)",
            "prov": "providers/\nHTTP client",
            "back": "Backend\nOllama, LM Studio ...",
            "log": "core/events.py\nrun log",
        }
        if code
        else {
            "fe": "Where you type\nthe Kestrel shell",
            "loop": "Agent loop + memory: ask, act, repeat",
            "rel": "Reliability\nfixes bad calls (M4)",
            "tools": "Tools + safety\nfiles, shell (M3)",
            "prov": "Providers\ntalk to the model",
            "back": "Backend\nOllama runs the model",
            "log": "Run log\nevery step saved",
        }
    )
    d.box("fe", 0, 0, 500, 70, BLUE, t["fe"])
    d.box("loop", 0, 130, 500, 70, VIOLET, t["loop"])
    d.box("rel", 0, 260, 240, 80, ORANGE, t["rel"])
    d.box("tools", 260, 260, 240, 80, RED, t["tools"])
    d.box("prov", 0, 400, 240, 80, GREEN, t["prov"])
    d.box("back", 0, 540, 240, 80, GRAY, t["back"])
    d.box("log", 600, 130, 220, 70, YELLOW, t["log"])
    d.arrow("fe", "loop")
    d.arrow("loop", "rel")
    d.arrow("loop", "tools")
    d.arrow("rel", "prov")
    d.arrow("prov", "back")
    d.arrow("loop", "log", dashed=True)
    return d


def core_types() -> Diagram:
    d = Diagram("core-types")
    d.box("call", 0, 0, 280, 80, RED, "ToolCall\nid, name, arguments")
    d.box("msg", 0, 140, 280, 100, BLUE, "Message\nrole + content\n+ tool calls + reasoning")
    d.box("spec", 0, 300, 280, 90, ORANGE, "ToolSpec\na tool's name +\nJSON schema of arguments")
    d.box(
        "prov",
        380,
        140,
        240,
        250,
        VIOLET,
        "Provider\n(any backend)\n\nchat()\nstream()\nlist_models()",
    )
    d.box("resp", 720, 140, 280, 100, GREEN, "ModelResponse\nthe full reply\n+ token usage")
    d.box(
        "chunk",
        720,
        300,
        280,
        90,
        GREEN,
        "StreamChunk\na piece of the reply;\nthe last holds it all",
    )
    d.arrow("call", "msg", "inside")
    d.arrow("msg", "prov", "in")
    d.arrow("spec", "prov", "in")
    d.arrow("prov", "resp", "out")
    d.arrow("prov", "chunk", "out")
    return d


def providers() -> Diagram:
    d = Diagram("providers")
    d.box("wire", 250, -140, 210, 80, ORANGE, "wire.py\nJSON in and out")
    d.box("sse", 480, -140, 230, 80, ORANGE, "sse.py + think.py\nstitch the pieces")
    d.box("ask", 0, 0, 220, 70, BLUE, "chat, ask, /run")
    d.box("oai", 300, 0, 380, 70, VIOLET, "OpenAICompatProvider")
    d.box("doc", 0, 200, 220, 70, BLUE, "/doctor")
    d.box("disc", 300, 200, 380, 70, VIOLET, "discovery.py\nasks every known port")
    d.box(
        "back",
        900,
        0,
        240,
        270,
        GRAY,
        "Backend\n\nOllama :11434\nLM Studio :1234\nllama.cpp / mlx-lm :8080\nvLLM :8000",
        size=16,
    )
    d.arrow("ask", "oai")
    d.arrow("oai", "wire")
    d.arrow("oai", "sse")
    d.arrow("oai", "back", "HTTP")
    d.arrow("doc", "disc")
    d.arrow("disc", "back", "GET /models")
    return d


def steps(name: str, items: list[tuple[str, str]], notes: list[str] | None = None) -> Diagram:
    """A numbered column of steps, optionally with what gets logged beside each."""
    d = Diagram(name)
    for i, (text, color) in enumerate(items):
        d.box(f"s{i}", 0, i * 120, 380, 76, color, f"{i + 1}. {text}")
        if i:
            d.arrow(f"s{i - 1}", f"s{i}")
        if notes and notes[i]:
            d.box(f"n{i}", 480, i * 120 + 8, 260, 60, GRAY, notes[i], size=16)
            d.arrow(f"s{i}", f"n{i}", dashed=True)
    if notes:
        d.text(480, -36, "written to the run log", 16)
    return d


def doctor() -> Diagram:
    d = Diagram("doctor")
    d.box("doc", 0, 0, 780, 70, BLUE, "/doctor asks every backend at once (1.5 s limit)")
    ports = ["Ollama\n:11434", "LM Studio\n:1234", "llama.cpp or\nmlx-lm :8080", "vLLM\n:8000"]
    for i, port in enumerate(ports):
        d.box(f"p{i}", i * 200, 140, 180, 80, GRAY, port)
        d.arrow("doc", f"p{i}")
    d.box(
        "table",
        0,
        300,
        780,
        80,
        GREEN,
        "Table: running or not, which models,\nwarnings (4K context, REMOTE)",
    )
    for i in range(4):
        d.arrow(f"p{i}", "table")
    return d


def talking() -> Diagram:
    d = Diagram("talking-to-model")
    d.box("you", 0, 40, 160, 80, YELLOW, "You")
    d.box("k", 260, 20, 260, 120, VIOLET, "Kestrel\nturns your chat\ninto JSON")
    d.box("back", 780, 20, 260, 120, GRAY, "Backend\nOllama runs\nthe model")
    d.box("screen", 260, 240, 260, 90, GREEN, "Your screen\nanswer appears live")
    d.arrow("you", "k")
    d.arrow("k", "back", "your chat", shift=-28)
    d.arrow("back", "k", "reply, piece by piece", shift=28)
    d.arrow("k", "screen")
    return d


def shell_structure() -> Diagram:
    d = Diagram("shell-structure")
    d.box("app", 0, 0, 760, 70, BLUE, "app.py: kestrel, kestrel chat")
    d.box("shell", 0, 140, 760, 70, VIOLET, "shell.py: the session + read, handle, repeat")
    d.box("prompt", 0, 280, 240, 80, GRAY, "prompt.py\ntyping, history, Tab")
    d.box("cmds", 260, 280, 240, 80, ORANGE, "commands.py\n/model /run /trace ...")
    d.box("out", 520, 280, 240, 80, GREEN, "output.py\nstreams answers")
    d.box("run", 260, 430, 240, 80, RED, "agent_run.py\none agent run")
    d.arrow("app", "shell")
    d.arrow("shell", "prompt")
    d.arrow("shell", "cmds")
    d.arrow("shell", "out")
    d.arrow("cmds", "run")
    return d


def shell_line() -> Diagram:
    d = Diagram("shell-line")
    d.box("line", 0, 0, 280, 70, YELLOW, "You type a line")
    d.box("q", 30, 130, 220, 130, VIOLET, "Starts\nwith / ?", shape="diamond")
    d.box("cmd", 380, 160, 280, 70, ORANGE, "Run the command\n/model /run /trace ...")
    d.box("send", 0, 320, 280, 70, BLUE, "Send it with the\nconversation so far")
    d.box("reply", 0, 450, 280, 70, BLUE, "The reply streams in")
    d.box("keep", 0, 580, 280, 70, GREEN, "Both are added to\nthe conversation")
    d.arrow("line", "q")
    d.arrow("q", "cmd", "yes")
    d.arrow("q", "send", "no")
    d.arrow("send", "reply")
    d.arrow("reply", "keep")
    d.text(320, 600, "then back to the prompt", 16)
    return d


def agent_loop() -> Diagram:
    d = Diagram("agent-loop")
    d.box("task", 0, 0, 240, 70, YELLOW, 'Your task\n"what is 17 + 25?"')
    d.box("ask", 0, 150, 240, 70, BLUE, "Ask the model", size=20)
    d.box("q", 10, 290, 220, 120, VIOLET, "Wants a\ntool?", shape="diamond", size=20)
    d.box("run", 360, 315, 220, 70, RED, "Run the tool\nadd(17, 25) = 42")
    d.box("show", 360, 150, 220, 70, GREEN, "Give the model\nthe result")
    d.box("answer", 0, 490, 240, 70, GREEN, 'Answer: "It is 42."', size=20)
    d.arrow("task", "ask")
    d.arrow("ask", "q")
    d.arrow("q", "run", "yes")
    d.arrow("run", "show")
    d.arrow("show", "ask", "repeat")
    d.arrow("q", "answer", "no")
    d.text(300, 500, "also stops when out of steps,\ntokens or time, or on Ctrl-C", 16)
    return d


def loop_structure() -> Diagram:
    d = Diagram("loop-structure")
    d.box(
        "cmd", 0, 0, 1100, 70, BLUE, "/run (shell) or kestrel run: agent_run.py sets everything up"
    )
    d.box("loop", 0, 140, 1100, 70, VIOLET, "loop.py: run_agent() asks, runs tools, repeats")
    d.box("ctx", 0, 290, 200, 80, ORANGE, "context.py\nfits the window")
    d.box("prov", 220, 290, 200, 80, GRAY, "Provider\nthe model")
    d.box("tools", 440, 290, 200, 80, RED, "Toolbox\nadd, echo (demo)")
    d.box("bus", 660, 290, 440, 80, YELLOW, "events.py: EventBus\nannounces every step")
    d.box("screen", 660, 450, 210, 80, GREEN, "frontend.py\nprints it live")
    d.box("writer", 890, 450, 210, 80, GREEN, "JsonlWriter\n+ redact.py")
    d.box("file", 890, 600, 210, 70, GRAY, "runs/<id>.jsonl")
    d.arrow("cmd", "loop")
    for target in ["ctx", "prov", "tools", "bus"]:
        d.arrow("loop", target)
    d.arrow("bus", "screen", "listener")
    d.arrow("bus", "writer", "listener")
    d.arrow("writer", "file")
    return d


def context_budget() -> Diagram:
    d = Diagram("context-budget")
    d.text(60, 0, "always kept", 18)
    d.text(420, 0, "can be trimmed", 18)
    d.text(790, 0, "always kept", 18)
    d.box("sys", 0, 40, 130, 80, GREEN, "system\nprompt")
    d.box("task", 145, 40, 130, 80, GREEN, "the task")
    d.box("o1", 320, 40, 130, 80, ORANGE, "old step")
    d.box("o2", 465, 40, 130, 80, ORANGE, "old step")
    d.box("o3", 610, 40, 130, 80, ORANGE, "old step")
    d.box("last", 785, 40, 160, 80, GREEN, "latest\nstep")
    d.box("r1", 320, 200, 420, 70, RED, "1. empty old tool results first")
    d.box("r2", 320, 340, 420, 70, RED, "2. still too big? drop the oldest steps")
    d.arrow("o2", "r1")
    d.arrow("r1", "r2")
    d.text(0, 450, "Budget = context window (16K) minus 1K kept free for the reply.", 16)
    return d


def run_log() -> Diagram:
    d = Diagram("run-log")
    d.box("run", 0, 60, 180, 80, VIOLET, "A run\nevery step")
    d.box("scrub", 250, 60, 200, 80, RED, "Remove\nsecrets")
    d.box("file", 520, 20, 220, 160, YELLOW, "Run log\none line\nper step")
    d.box("trace", 820, 0, 220, 80, GREEN, "/trace\nreadable timeline")
    d.box("replay", 820, 120, 220, 80, BLUE, "Replay\nsame run, no model")
    d.arrow("run", "scrub")
    d.arrow("scrub", "file")
    d.arrow("file", "trace")
    d.arrow("file", "replay")
    return d


def all_diagrams() -> list[Diagram]:
    return [
        layers("layers", code=False),
        layers("layers-code", code=True),
        core_types(),
        providers(),
        steps(
            "ask-steps",
            [
                ("You type a question", YELLOW),
                ("Kestrel turns it into JSON", BLUE),
                ("Sends it to the backend\nPOST /v1/chat/completions", BLUE),
                ("The reply streams back\nin small pieces", GREEN),
                ("Pieces shown live;\nthe last holds the full reply", GREEN),
            ],
        ),
        doctor(),
        talking(),
        shell_structure(),
        shell_line(),
        agent_loop(),
        loop_structure(),
        steps(
            "run-steps",
            [
                ("The task goes into memory", YELLOW),
                ("Fit the conversation\ninto the window", ORANGE),
                ("Ask the model\n(reply streams live)", BLUE),
                ("Tool calls? Run each one,\nadd the results, go to 2", RED),
                ("No tool calls:\nthe reply is the answer", GREEN),
                ("End: reason + totals", GRAY),
            ],
            notes=[
                "run_start",
                "context_trim\n(only if needed)",
                "model_request\nmodel_response",
                "tool_call\ntool_result",
                "final",
                "run_end",
            ],
        ),
        context_budget(),
        run_log(),
    ]


if __name__ == "__main__":
    SRC.mkdir(exist_ok=True)
    diagrams = all_diagrams()
    for diagram in diagrams:
        diagram.save()
    (SRC / "index.json").write_text(json.dumps([d.name for d in diagrams]))
    print(f"wrote {len(diagrams)} skeletons to {SRC}")
