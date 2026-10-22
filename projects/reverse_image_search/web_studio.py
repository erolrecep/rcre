#!/usr/bin/env python3
"""
web_studio.py: Standalone Web Studio & Visual Clustering Server for RCRE Lens
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PORT = 8085
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
GALLERY_DIR = BASE_DIR / "gallery"
UPLOADS_DIR = BASE_DIR / "uploads"
INDEX_FILE = BASE_DIR / "gallery.rcre_idx"
BINARY = BASE_DIR / "bin" / "rcre-lens"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

class LensHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.send_json(self.get_system_status())
            return
        elif path == "/api/gallery":
            self.send_json(self.get_gallery_list())
            return
        elif path.startswith("/gallery/"):
            # Serve gallery image
            rel_file = path.replace("/gallery/", "")
            file_path = GALLERY_DIR / rel_file
            if file_path.exists() and file_path.is_file():
                self.serve_file(file_path)
            else:
                self.send_error(404, "Image not found")
            return
        elif path.startswith("/uploads/"):
            rel_file = path.replace("/uploads/", "")
            file_path = UPLOADS_DIR / rel_file
            if file_path.exists() and file_path.is_file():
                self.serve_file(file_path)
            else:
                self.send_error(404, "Upload not found")
            return

        # Serve static web frontend
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/upload":
            self.handle_upload()
            return
        elif path == "/api/index":
            self.handle_index()
            return
        elif path == "/api/cluster":
            k = int(query.get("k", [4])[0])
            self.handle_cluster(k)
            return
        elif path == "/api/search":
            top_k = int(query.get("k", [5])[0])
            self.handle_search(top_k)
            return

        self.send_error(404, "API endpoint not found")

    def serve_file(self, file_path: Path):
        content_type = "image/jpeg"
        if file_path.suffix.lower() == ".png":
            content_type = "image/png"
        elif file_path.suffix.lower() == ".bmp":
            content_type = "image/bmp"

        with open(file_path, "rb") as f:
            data = f.read()

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def get_system_status(self):
        indexed = 0
        file_size_kb = 0.0
        if INDEX_FILE.exists():
            file_size_kb = INDEX_FILE.stat().st_size / 1024.0

        # Run info via binary
        try:
            res = subprocess.run([str(BINARY), "list", "-i", str(INDEX_FILE), "--json"],
                                 capture_output=True, text=True, check=True)
            info = json.loads(res.stdout)
            indexed = info.get("total_images", 0)
        except Exception:
            pass

        return {
            "version": "1.1.0",
            "cpu_cores": os.cpu_count() or 4,
            "indexed_images": indexed,
            "index_size_kb": round(file_size_kb, 2),
            "gallery_path": str(GALLERY_DIR),
            "binary_exists": BINARY.exists()
        }

    def get_gallery_list(self):
        images = []
        for f in GALLERY_DIR.iterdir():
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                images.append({
                    "filename": f.name,
                    "url": f"/gallery/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024.0, 1)
                })
        images.sort(key=lambda x: x["filename"])
        return {"images": images, "total": len(images)}

    def handle_upload(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "Content-Type must be multipart/form-data"}, status=400)
            return

        import cgi
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]}
        )

        uploaded = []
        for key in form.keys():
            item = form[key]
            if item.filename:
                out_path = GALLERY_DIR / Path(item.filename).name
                with open(out_path, "wb") as f:
                    f.write(item.file.read())
                uploaded.append(item.filename)

        # Automatically re-index if requested
        self.run_reindex()
        self.send_json({"status": "success", "uploaded_files": uploaded, "count": len(uploaded)})

    def run_reindex(self):
        if not BINARY.exists():
            return False
        cmd = [str(BINARY), "index", str(GALLERY_DIR), "-o", str(INDEX_FILE), "--threads", "0"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return res.returncode == 0

    def handle_index(self):
        success = self.run_reindex()
        status_info = self.get_system_status()
        self.send_json({"status": "indexed" if success else "failed", **status_info})

    def handle_cluster(self, k: int):
        if not INDEX_FILE.exists():
            self.run_reindex()

        cmd = [str(BINARY), "cluster", "-i", str(INDEX_FILE), "-k", str(k), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            self.send_json({"error": "Clustering execution failed", "stderr": res.stderr}, status=500)
            return

        try:
            data = json.loads(res.stdout)
            # Reformat image paths to relative web urls
            for c in data.get("clusters", []):
                rel_imgs = []
                for p in c.get("images", []):
                    rel_imgs.append({
                        "filename": Path(p).name,
                        "url": f"/gallery/{Path(p).name}"
                    })
                c["images"] = rel_imgs
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": f"JSON parse error: {e}", "raw": res.stdout}, status=500)

    def handle_search(self, top_k: int):
        import cgi
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers["Content-Type"]}
        )

        query_file = None
        for key in form.keys():
            item = form[key]
            if item.filename:
                out_path = UPLOADS_DIR / "query_probe.bmp"
                with open(out_path, "wb") as f:
                    f.write(item.file.read())
                query_file = out_path
                break

        if not query_file or not query_file.exists():
            self.send_json({"error": "No query file provided"}, status=400)
            return

        if not INDEX_FILE.exists():
            self.run_reindex()

        cmd = [str(BINARY), "search", str(query_file), "-i", str(INDEX_FILE), "-k", str(top_k), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            self.send_json({"error": "Search failed", "stderr": res.stderr}, status=500)
            return

        try:
            data = json.loads(res.stdout)
            for item in data.get("results", []):
                fname = Path(item["filepath"]).name
                item["filename"] = fname
                item["url"] = f"/gallery/{fname}"
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": f"Search parse error: {e}", "raw": res.stdout}, status=500)


def main():
    print("===================================================================")
    print(" 🚀 RCRE Lens Visual Studio — Standalone Web Engine (v1.1.0)       ")
    print(f"    Serving live at: http://localhost:{PORT}                       ")
    print("===================================================================")

    if not INDEX_FILE.exists() and any(GALLERY_DIR.iterdir()):
        print("⚡ Pre-indexing initial gallery...")
        subprocess.run([str(BINARY), "index", str(GALLERY_DIR), "-o", str(INDEX_FILE), "--threads", "0"])

    httpd = HTTPServer(("0.0.0.0", PORT), LensHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RCRE Lens Studio server.")
        httpd.server_close()

if __name__ == "__main__":
    main()
