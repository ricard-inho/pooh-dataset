"""Browser page for clicking object keypoints: works locally and over SSH, no GUI toolkit.

The page is served on 127.0.0.1 only. Over SSH, forward the port:
``ssh -L 8765:localhost:8765 <host>`` and open http://localhost:8765 locally.
"""

from __future__ import annotations

import io
import json
import webbrowser
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>pooh calibrate</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 html,body{margin:0;height:100%;background:#111;color:#ddd;font:14px system-ui,sans-serif}
 body{display:flex;flex-direction:column}
 #bar{padding:8px 12px;background:#1c1c1c;display:flex;gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333}
 #c{flex:1;min-height:0;width:100%;display:block;cursor:crosshair}
 button{background:#2d2d2d;color:#eee;border:1px solid #555;padding:5px 12px;border-radius:4px;cursor:pointer;font:inherit}
 button:hover{background:#3a3a3a}
 .k{color:#8cc8ff} #msg{color:#ffb347;min-width:22em} #prog{color:#9f9}
</style></head><body>
<div id="bar"><b id="title">loading…</b><span id="prog"></span>
 <span><span class="k">click</span> corner · <span class="k">right-click / u</span> undo · <span class="k">wheel</span> zoom ·
 <span class="k">drag</span> pan · <span class="k">f</span> fit · <span class="k">n / Enter</span> next · <span class="k">s</span> skip · <span class="k">q</span> finish</span>
 <button id="bn">next (n)</button><button id="bs">skip (s)</button><button id="bq">finish (q)</button><span id="msg"></span></div>
<canvas id="c"></canvas>
<script>
const c = document.getElementById('c'), ctx = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
let st = null, img = new Image(), pts = [], view = {s: 1, x: 0, y: 0}, down = null, busy = false;
const $ = id => document.getElementById(id);
function msg(t) { $('msg').textContent = t || ''; }
function fit() {
  const r = c.getBoundingClientRect(); c.width = r.width * dpr; c.height = r.height * dpr;
  if (img.width) { const s = Math.min(c.width / img.width, c.height / img.height);
    view = {s, x: (c.width - img.width * s) / 2, y: (c.height - img.height * s) / 2}; }
  draw();
}
function toImg(e) {  // canvas event -> image pixel coords (pixel centres at integers)
  // use the canvas's *current* on-screen size: the top bar can change height at any time
  const r = c.getBoundingClientRect();
  const cx = (e.clientX - r.left) * c.width / r.width, cy = (e.clientY - r.top) * c.height / r.height;
  return [(cx - view.x) / view.s - 0.5, (cy - view.y) / view.s - 0.5, cx, cy];
}
function P(u, v) { return [(u + 0.5) * view.s + view.x, (v + 0.5) * view.s + view.y]; }
function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#111'; ctx.fillRect(0, 0, c.width, c.height);
  if (!img.width || !st) return;
  ctx.setTransform(view.s, 0, 0, view.s, view.x, view.y); ctx.imageSmoothingEnabled = view.s < 3;
  ctx.drawImage(img, 0, 0); ctx.setTransform(1, 0, 0, 1, 0, 0);
  const a = 9 * dpr; ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = '#ffd200';
  for (const [u, v] of (st.hint || [])) { const [x, y] = P(u, v);
    ctx.beginPath(); ctx.moveTo(x - a, y - a); ctx.lineTo(x + a, y + a); ctx.moveTo(x + a, y - a); ctx.lineTo(x - a, y + a); ctx.stroke(); }
  ctx.strokeStyle = '#ff3b3b'; ctx.fillStyle = '#ff3b3b';
  for (const [u, v] of pts) { const [x, y] = P(u, v);
    ctx.beginPath(); ctx.arc(x, y, a, 0, 2 * Math.PI); ctx.stroke(); ctx.fillRect(x - dpr, y - dpr, 2 * dpr, 2 * dpr); }
}
async function call(path, body) {
  busy = true; if (path !== '/state') msg(path === '/accept' ? 'saving… (updating prediction)' : '');
  try {
    const r = await fetch(path, body === undefined ? {} : {method: 'POST', body: JSON.stringify(body)});
    show(await r.json());
  } catch (e) { msg('connection lost — is `pooh calibrate` still running?'); }
  busy = false;
}
function show(s) {
  st = s; pts = [];
  if (s.finished) { $('title').textContent = 'done — you can close this tab'; $('prog').textContent =
      s.done + ' frames saved'; img = new Image(); draw(); msg(''); return; }
  $('title').textContent = s.title; $('prog').textContent = `[${s.done}/${s.target} frames done]`;
  msg(s.hint && s.hint.length ? 'yellow = predicted corners from the frames so far' : '');
  const im = new Image(); im.onload = () => { img = im; fit(); }; im.src = '/image/' + s.frame_id;
}
function next() { if (busy || !st || st.finished) return;
  if (pts.length < st.min_points) { msg(`click at least ${st.min_points} corners (or press s to skip)`); return; }
  call('/accept', {frame_id: st.frame_id, points: pts}); }
function skip() { if (!busy && st && !st.finished) call('/skip', {frame_id: st.frame_id}); }
function finish() { if (!busy && st && !st.finished) call('/finish', {}); }
function undo() { pts.pop(); draw(); }
c.addEventListener('mousedown', e => { if (e.button === 0) down = {x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false}; });
window.addEventListener('mousemove', e => { if (!down) return;
  const r = c.getBoundingClientRect();
  const dx = (e.clientX - down.x) * c.width / r.width, dy = (e.clientY - down.y) * c.height / r.height;
  if (Math.abs(dx) + Math.abs(dy) > 4 * dpr) down.moved = true;
  if (down.moved) { view.x = down.vx + dx; view.y = down.vy + dy; draw(); } });
window.addEventListener('mouseup', e => { if (down && !down.moved && e.button === 0 && e.target === c && st && !st.finished) {
    const [u, v] = toImg(e); pts.push([u, v]); draw(); } down = null; });
c.addEventListener('contextmenu', e => { e.preventDefault(); undo(); });
c.addEventListener('wheel', e => { e.preventDefault(); const [, , cx, cy] = toImg(e), f = Math.exp(-e.deltaY * 0.0015);
  view.x = cx - (cx - view.x) * f; view.y = cy - (cy - view.y) * f; view.s *= f; draw(); }, {passive: false});
window.addEventListener('keydown', e => { const k = e.key;
  if (k === 'n' || k === 'Enter') next(); else if (k === 's') skip(); else if (k === 'q') finish();
  else if (k === 'u' || k === 'Backspace') { e.preventDefault(); undo(); } else if (k === 'f') fit(); });
$('bn').onclick = next; $('bs').onclick = skip; $('bq').onclick = finish;
// keep the backing store equal to the on-screen size without moving the view (no refit)
new ResizeObserver(() => { const r = c.getBoundingClientRect();
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (w !== c.width || h !== c.height) { c.width = w; c.height = h; draw(); } }).observe(c);
call('/state');
</script></body></html>
"""


class WebClickSession:
    """Shows frames one by one in a browser page; the user clicks the visible keypoints.

    ``frames`` yields ``(key, title, image)``; ``hint(key)`` may return predicted keypoints
    (K, 2), drawn as yellow crosses; ``on_accept(key, points)`` is called for every frame.
    """

    def __init__(self, frames: Iterator, n_target: int, hint: Callable | None = None,
                 on_accept: Callable | None = None, min_points: int = 3, port: int = 8765,
                 open_browser: bool = True):
        self.frames = frames
        self.n_target = n_target
        self.hint = hint
        self.on_accept = on_accept
        self.min_points = min_points
        self.port = port
        self.open_browser = open_browser
        self.accepted: list[tuple] = []
        self.finished = False
        self.frame_id = 0
        self.current = None
        self._jpeg = b""
        self._hint: list = []

    # ---------------------------------------------------------- state machine
    def _advance(self) -> None:
        if len(self.accepted) >= self.n_target:
            self.finished = True
            return
        try:
            self.current = next(self.frames)
        except StopIteration:
            self.finished = True
            return
        self.frame_id += 1
        key, _, image = self.current
        buf = io.BytesIO()
        Image.fromarray(np.asarray(image)).convert("RGB").save(buf, format="JPEG", quality=95)
        self._jpeg = buf.getvalue()
        h = self.hint(key) if (self.hint and self.accepted) else None
        self._hint = [] if h is None else np.asarray(h, float).reshape(-1, 2).tolist()

    def state(self) -> dict:
        if self.finished:
            return {"finished": True, "done": len(self.accepted)}
        return {"finished": False, "done": len(self.accepted), "target": self.n_target,
                "title": self.current[1], "frame_id": self.frame_id, "hint": self._hint,
                "min_points": self.min_points}

    def accept(self, frame_id: int, points) -> dict:
        pts = np.asarray(points, float).reshape(-1, 2)
        if frame_id == self.frame_id and not self.finished and len(pts) >= self.min_points:
            self.accepted.append((self.current[0], pts))
            if self.on_accept:
                self.on_accept(self.current[0], pts)
            self._advance()
        return self.state()

    def skip(self, frame_id: int) -> dict:
        if frame_id == self.frame_id and not self.finished:
            self._advance()
        return self.state()

    def finish(self) -> dict:
        self.finished = True
        return self.state()

    # ----------------------------------------------------------------- server
    def _make_server(self) -> HTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal quiet
                pass

            def _send(self, body: bytes, ctype: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    self._send(json.dumps(app.state()).encode(), "application/json")
                elif self.path.startswith("/image/"):
                    self._send(app._jpeg, "image/jpeg")
                else:
                    self.send_error(404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/accept":
                    out = app.accept(int(body.get("frame_id", -1)), body.get("points", []))
                elif self.path == "/skip":
                    out = app.skip(int(body.get("frame_id", -1)))
                elif self.path == "/finish":
                    out = app.finish()
                else:
                    self.send_error(404)
                    return
                self._send(json.dumps(out).encode(), "application/json")

        last_err = None
        for port in range(self.port, self.port + 20):
            try:
                srv = HTTPServer(("127.0.0.1", port), Handler)
                self.port = port
                return srv
            except OSError as e:  # port in use
                last_err = e
        raise OSError(f"no free port in {self.port}-{self.port + 19}") from last_err

    def run(self) -> list[tuple]:
        self._advance()
        if self.finished:
            print("no candidate frames to click")
            return self.accepted
        srv = self._make_server()
        srv.timeout = 0.5
        url = f"http://localhost:{self.port}"
        print(f"\nopen {url} in a browser to click the corners")
        print(f"  (remote machine? on your laptop run:  ssh -L {self.port}:localhost:{self.port} <this host>"
              f"  then open {url})")
        print("  Ctrl-C here also finishes; clicks are saved after every frame\n")
        if self.open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            while not self.finished:
                srv.handle_request()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            srv.server_close()
        return self.accepted
