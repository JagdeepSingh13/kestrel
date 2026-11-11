# Diagrams

Every picture here is a hand-drawn [Excalidraw](https://excalidraw.com) drawing:

- `img/<name>.svg`: the image shown in the README and these pages (the font is built in).
- `img/<name>.excalidraw`: the editable drawing. Drag it onto excalidraw.com to change it.

## Redrawing

All diagrams are defined in one file, [build/diagrams.py](build/diagrams.py): boxes, arrows, colors, text.

1. Edit `build/diagrams.py`, then run `python3 docs/diagrams/build/diagrams.py`.
2. Start `python3 docs/diagrams/build/server.py 8766` and open http://localhost:8766/build.html
   (in Claude's desktop app: the "diagrams" preview). The page draws every diagram with
   Excalidraw's own code and saves both files into `img/`.

Style: few boxes, short straight arrows, labels beside arrows, plain words. No Mermaid.

The build page loads Excalidraw from esm.sh and its font from unpkg while drawing. Nothing
here is part of Kestrel itself.
