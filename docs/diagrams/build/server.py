"""Local helper for redrawing diagrams: serves build.html and saves what it exports.

Run through the Claude desktop preview (`.claude/launch.json`, "diagrams") or by hand:
    python3 docs/diagrams/build/server.py 8765   then open http://localhost:8765/build.html
Saved files land in docs/diagrams/img/. Stdlib only; not part of Kestrel itself.
"""

import http.server
import pathlib
import sys
import urllib.parse

HERE = pathlib.Path(__file__).parent
IMG = HERE.parent / "img"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_POST(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = pathlib.Path(query["name"][0]).name  # no paths, only a file name
        body = self.rfile.read(int(self.headers["Content-Length"]))
        (IMG / name).write_bytes(body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"saved")


if __name__ == "__main__":
    IMG.mkdir(exist_ok=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
