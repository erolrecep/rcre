#!/usr/bin/env python3
"""
RCRE-SR web demo — Python 3 stdlib only (no pip dependencies).

Serves an interactive before/after page:
  * drop / choose an image (PNG or JPG-converted-PNG; 8-bit PNG supported:
    grayscale, RGB, RGBA)
  * pick scale (1-4), style (auto/sharp/clean/binarize/upscale), sharpen
  * POSTs to the local sr_cli binary (subprocess) and streams back the
    enhanced PNG + the pipeline report

A minimal 8-bit PNG encoder/decoder (zlib + struct + binascii, all stdlib)
translates between the browser's PNG and the CLI's PPM I/O.

Run:  cd web && python3 server.py [port]
"""

import base64
import binascii
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLI = os.path.join(ROOT, "bin", "sr_cli")

MAX_INPUT_PX = 8_000_000  # keep the demo responsive


# =========================================================================
# Minimal 8-bit PNG codec (color types 0/2/6, no interlace)
# =========================================================================

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def png_decode(data: bytes):
    """Return (width, height, channels, rows-as-bytes) or raise ValueError."""
    if data[:8] != _PNG_SIG:
        raise ValueError("not a PNG")
    pos = 8
    idat = bytearray()
    width = height = bitdepth = colortype = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bitdepth, colortype, comp, filt, interlace = \
                struct.unpack(">IIBBBBB", chunk)
            if bitdepth != 8 or interlace != 0:
                raise ValueError("only 8-bit non-interlaced PNG supported")
            if colortype not in (0, 2, 6):
                raise ValueError("grayscale/RGB/RGBA only")
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
    if width is None:
        raise ValueError("truncated PNG")
    ch = {0: 1, 2: 3, 6: 4}[colortype]
    # PNG IDAT is a zlib (RFC 1950) stream; fall back to raw deflate for
    # tolerance of non-conforming writers.
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error:
        d = zlib.decompressobj(wbits=-15)
        raw = d.decompress(bytes(idat)) + d.flush()
    stride = width * ch
    out = bytearray(height * stride)
    prev = bytearray(stride)
    rp = 0
    for y in range(height):
        f = raw[rp]
        rp += 1
        line = bytearray(raw[rp:rp + stride])
        rp += stride
        if f == 1:      # Sub
            for i in range(ch, stride):
                line[i] = (line[i] + line[i - ch]) & 0xFF
        elif f == 2:    # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif f == 3:    # Average
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif f == 4:    # Paeth
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                b = prev[i]
                c = prev[i - ch] if i >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, ch, bytes(out)


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)


def png_encode(width, height, ch, pixels: bytes) -> bytes:
    stride = width * ch
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter: None
        raw += pixels[y * stride:(y + 1) * stride]
    colortype = {1: 0, 3: 2, 4: 6}[ch]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colortype, 0, 0, 0)

    def chunk(tag, body):
        c = struct.pack(">I", len(body)) + tag + body
        return c + struct.pack(">I", binascii.crc32(tag + body) & 0xFFFFFFFF)

    return (_PNG_SIG + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b""))


# =========================================================================
# CLI bridge
# =========================================================================

def run_sr_cli(pgm_in: str, pgm_out: str, scale: int, style: str,
               sharpen: float, deblur: bool):
    import time
    cmd = [CLI, "enhance", pgm_in, pgm_out,
           "--scale", str(scale), "--style", style]
    if not deblur:
        cmd.append("--no-deblur")
    if sharpen >= 0:
        cmd += ["--sharpen", f"{sharpen:.2f}"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ms = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        raise RuntimeError(f"sr_cli failed: {p.stderr.strip()[:400]}")
    return p.stdout, ms


def write_pgm(path, w, h, gray: bytes):
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(gray)


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    # P5\n<w> <h>\n255\n<binary>
    m = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", data)
    if not m:
        raise ValueError("bad PGM")
    w, h, maxv = int(m.group(1)), int(m.group(2)), int(m.group(3))
    off = m.end()
    return w, h, data[off:off + w * h]


def gray_from_pixels(w, h, ch, pixels: bytes) -> bytes:
    if ch == 1:
        return pixels
    out = bytearray(w * h)
    if ch == 3:
        for i in range(w * h):
            r, g, b = pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]
            out[i] = (r * 77 + g * 150 + b * 29) >> 8
    else:  # 4 -> drop alpha
        for i in range(w * h):
            r, g, b = pixels[4 * i], pixels[4 * i + 1], pixels[4 * i + 2]
            out[i] = (r * 77 + g * 150 + b * 29) >> 8
    return bytes(out)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RCRE-SR — Document Super-Resolution</title>
<style>
  :root { --bg:#0e1116; --panel:#161b23; --line:#2a3140; --txt:#dbe2ec;
          --dim:#8b96a8; --acc:#5aa9ff; --ok:#3ecf8e; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { padding:18px 24px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0; letter-spacing:.3px; }
  header .sub { color:var(--dim); font-size:12.5px; }
  main { display:grid; grid-template-columns:300px 1fr; gap:0;
         min-height:calc(100vh - 61px); }
  aside { border-right:1px solid var(--line); padding:18px;
          background:var(--panel); }
  .card { border:1px solid var(--line); border-radius:10px; padding:14px;
          margin-bottom:14px; background:#121722; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.8px;
             color:var(--dim); margin:0 0 10px; }
  label { display:block; margin:8px 0 3px; color:var(--dim); font-size:12.5px; }
  select,input[type=range] { width:100%; accent-color:var(--acc); }
  select { background:#0e131b; color:var(--txt); border:1px solid var(--line);
           border-radius:6px; padding:6px; }
  .row { display:flex; gap:8px; align-items:center; }
  .row span { min-width:26px; text-align:right; color:var(--txt);
              font-variant-numeric:tabular-nums; }
  button { width:100%; margin-top:12px; padding:10px; border:0; border-radius:8px;
           background:var(--acc); color:#08101c; font-weight:600; cursor:pointer;
           font-size:14px; }
  button:disabled { opacity:.5; cursor:wait; }
  .check { display:flex; gap:8px; align-items:center; margin-top:10px;
           color:var(--txt); font-size:13px; }
  .stage { display:flex; align-items:center; justify-content:center;
           min-height:400px; position:relative; overflow:hidden; }
  .drop { border:2px dashed var(--line); border-radius:12px; padding:60px 30px;
          text-align:center; color:var(--dim); cursor:pointer; width:100%;
          max-width:420px; margin:80px auto; }
  .drop.hot { border-color:var(--acc); color:var(--txt); }
  #viewer { position:relative; width:100%; height:100%; }
  #viewer img { position:absolute; inset:0; width:100%; height:100%;
                object-fit:contain; }
  #after { clip-path:inset(0 0 0 var(--split,50%)); }
  #handle { position:absolute; top:0; bottom:0; width:2px; background:var(--acc);
            left:var(--split,50%); pointer-events:none; }
  #handle::after { content:"\\2194"; position:absolute; top:12px; left:-9px;
                   width:20px; text-align:center; color:var(--acc); font-size:14px; }
  #splitCtl { position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
              width:min(420px,60%); z-index:5; }
  .tag { position:absolute; top:10px; padding:3px 10px; border-radius:20px;
         background:rgba(10,14,20,.75); font-size:11.5px; z-index:4;
         color:var(--txt); }
  .tag.l { left:10px; } .tag.r { right:10px; }
  pre { background:#0b0f15; border:1px solid var(--line); border-radius:8px;
        padding:12px; font:11.5px/1.45 ui-monospace,Menlo,monospace; overflow:auto;
        max-height:260px; color:#a9b4c6; margin:0; }
  .status { font-size:12.5px; color:var(--dim); margin-top:8px; min-height:18px; }
  .err { color:#ff7b72; }
  a { color:var(--acc); text-decoration:none; }
</style>
</head>
<body>
<header>
  <h1>RCRE-SR <span style="color:var(--acc)">·</span> Document Super-Resolution</h1>
  <span class="sub">structure-preserving pipeline on the RCRE C engine —
    no GPU, no model weights, runs on your CPU</span>
</header>
<main>
  <aside>
    <div class="card">
      <h2>Input</h2>
      <div class="row"><button id="pick" style="margin-top:0">Choose image…</button></div>
      <input type="file" id="file" accept="image/png" hidden>
      <div class="status" id="finfo"></div>
    </div>
    <div class="card">
      <h2>Pipeline</h2>
      <label>Scale</label>
      <select id="scale">
        <option value="1">1× (no resize)</option>
        <option value="2" selected>2× (Lanczos-3)</option>
        <option value="3">3×</option>
        <option value="4">4×</option>
      </select>
      <label>Style</label>
      <select id="style">
        <option value="auto" selected>auto (adaptive)</option>
        <option value="sharp">sharp</option>
        <option value="upscale">upscale</option>
        <option value="clean">clean</option>
        <option value="binarize">binarize (OCR)</option>
      </select>
      <label>Sharpen</label>
      <div class="row">
        <input type="range" id="sharpen" min="-1" max="2" step="0.05" value="-1">
        <span id="sharpenV">auto</span>
      </div>
      <div class="check">
        <input type="checkbox" id="deblur" checked>
        <span>allow deblur (Wiener/FFT)</span>
      </div>
      <button id="go" disabled>Enhance</button>
      <div class="status" id="status"></div>
    </div>
    <div class="card">
      <h2>Report</h2>
      <pre id="report">—</pre>
    </div>
  </aside>
  <section class="stage">
    <div id="drop" class="drop">
      <div style="font-size:34px">📄</div>
      <b>Drop a document photo or scan</b><br>
      8-bit PNG · up to 8 MP<br>
      <span style="font-size:12px">e.g. a slightly blurred or skewed text page</span>
    </div>
    <div id="viewer" hidden>
      <span class="tag l">original</span>
      <span class="tag r" id="afterTag">enhanced</span>
      <img id="before" alt="original">
      <img id="after" alt="enhanced">
      <div id="handle"></div>
      <input type="range" id="splitCtl" min="0" max="100" value="50">
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
let haveImage = false;

const sharpen = $("sharpen");
sharpen.oninput = () => {
  $("sharpenV").textContent = parseFloat(sharpen.value) < 0
    ? "auto" : parseFloat(sharpen.value).toFixed(2);
};
$("splitCtl").oninput = e =>
  document.getElementById("viewer").style.setProperty("--split", e.target.value + "%");

function loadFile(f) {
  const finfo = $("finfo");
  finfo.classList.remove("err");
  const r = new FileReader();
  r.onload = async () => {
    try {
      const buf = new Uint8Array(r.result);
      const bytes = buf.buffer;
      const resp = await fetch("/api/enhance", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream",
                   "x-scale": $("scale").value,
                   "x-style": $("style").value,
                   "x-sharpen": sharpen.value,
                   "x-deblur": $("deblur").checked ? "1" : "0" },
        body: bytes
      });
      const j = await resp.json();
      if (!resp.ok) throw new Error(j.error || "enhance failed");
      $("before").src = "data:image/png;base64," + j.input_b64;
      $("after").src = "data:image/png;base64," + j.output_b64;
      $("afterTag").textContent =
        `enhanced ${j.out_w}×${j.out_h} · ${j.ms.toFixed(0)} ms`;
      $("report").textContent = j.report;
      $("drop").hidden = true;
      $("viewer").hidden = false;
      $("viewer").style.setProperty("--split", "50%");
      $("splitCtl").value = 50;
      $("status").textContent = `done in ${j.ms.toFixed(0)} ms`;
      $("status").classList.remove("err");
      haveImage = true;
      finfo.textContent = `${f.name} — ${j.w}×${j.h}`;
    } catch (e) {
      finfo.textContent = String(e.message || e);
      finfo.classList.add("err");
      $("status").textContent = String(e.message || e);
      $("status").classList.add("err");
    }
  };
  r.readAsArrayBuffer(f);
}
$("pick").onclick = () => $("file").click();
$("file").onchange = e => e.target.files[0] && loadFile(e.target.files[0]);
$("go").onclick = () => $("file").files[0] && loadFile($("file").files[0]);

const drop = $("drop");
["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.add("hot");
}));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => {
  e.preventDefault(); drop.classList.remove("hot");
}));
drop.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) { $("file").files = e.dataTransfer.files; loadFile(f); }
});
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "RCRESr/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + (fmt % args) + "\n")

    def _send(self, code, body: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/enhance":
            self._send(404, b"not found", "text/plain")
            return
        try:
            if not os.path.exists(CLI):
                raise RuntimeError("bin/sr_cli not found — run `make` first")
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            if not data:
                raise ValueError("empty body")
            scale = max(1, min(4, int(self.headers.get("x-scale", "2"))))
            style = self.headers.get("x-style", "auto")
            if style not in ("auto", "sharp", "clean", "binarize", "upscale"):
                style = "auto"
            sharpen = float(self.headers.get("x-sharpen", "-1"))
            deblur = self.headers.get("x-deblur", "1") == "1"

            w, h, ch, pixels = png_decode(data)
            if w * h > MAX_INPUT_PX:
                raise ValueError(f"image too large ({w * h} px > {MAX_INPUT_PX})")
            gray = gray_from_pixels(w, h, ch, pixels)
            in_png = png_encode(w, h, 1, gray)

            tmp = tempfile.mkdtemp(prefix="rcre_sr_")
            try:
                pgm_in = os.path.join(tmp, "in.pgm")
                pgm_out = os.path.join(tmp, "out.pgm")
                write_pgm(pgm_in, w, h, gray)
                report, ms = run_sr_cli(pgm_in, pgm_out, scale, style,
                                        sharpen, deblur)
                ow, oh, obytes = read_pgm(pgm_out)
                out_png = png_encode(ow, oh, 1, obytes)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

            body = json.dumps({
                "w": w, "h": h, "out_w": ow, "out_h": oh,
                "ms": ms,
                "input_b64": base64.b64encode(in_png).decode(),
                "output_b64": base64.b64encode(out_png).decode(),
                "report": report,
            }).encode()
            self._send(200, body, "application/json")
        except Exception as e:  # noqa: BLE001
            body = json.dumps({"error": str(e)}).encode()
            self._send(500, body, "application/json")


def main():
    if not os.path.exists(CLI):
        print(f"warning: {CLI} not found — run `make` in {ROOT} first",
              file=sys.stderr)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RCRE-SR web demo  →  http://127.0.0.1:{port}")
    print("(Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
