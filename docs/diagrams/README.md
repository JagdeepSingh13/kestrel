# Diagrams

Every picture here is an [Excalidraw](https://excalidraw.com) drawing. New ones use the clean style (`Diagram(..., clean=True)` with `card`, `cell` and `circle`): straight lines, plain font, cards with a title, a dotted rule and a few lines of words, dashed borders for planned parts. The older pictures are hand-drawn and will move to the clean style when their part changes.

- `img/<name>.svg`: the image shown in the README and these pages (the font is built in).
- `img/<name>.excalidraw`: the editable drawing, written next to it when you redraw. It stays on your machine (ignored by git); drag it onto excalidraw.com to look at or tweak it.

## Redrawing

All diagrams are defined in one file, [build/diagrams.py](build/diagrams.py): boxes, arrows, colors, text.

1. Edit `build/diagrams.py`, then run `python3 docs/diagrams/build/diagrams.py`.
2. Start `python3 docs/diagrams/build/server.py 8766` and open http://localhost:8766/build.html
   (in Claude's desktop app: the "diagrams" preview). The page draws every diagram with
   Excalidraw's own code and saves both files into `img/`.

Style: one picture per idea, big enough to read on GitHub (about 920 wide, text 18 to 20), boxes that explain themselves, short straight arrows, labels beside arrows, plain words. No Mermaid.

The build page loads Excalidraw from esm.sh and its font from unpkg while drawing. Nothing
here is part of Kestrel itself.
