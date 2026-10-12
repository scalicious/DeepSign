"""Local DeepSign UI and underwater-image enhancement service."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2

from enhance import enhance_underwater_image


ROOT = Path(__file__).resolve().parent
UI_PATH = ROOT / "data" / "UI-refresh.html"
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class DeepSignHandler(BaseHTTPRequestHandler):
    server_version = "DeepSignEnhance/1.0"

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route == "/api/health":
            self._send_json(200, {"status": "ok", "enhancement": "ready"})
            return
        if route not in ("/", "/UI-refresh.html"):
            self._send_json(404, {"error": "Not found"})
            return

        try:
            page = UI_PATH.read_bytes()
        except OSError:
            self._send_json(500, {"error": "UI-refresh.html could not be read"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/enhance":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if length <= 0:
            self._send_json(400, {"error": "An image file is required"})
            return
        if length > MAX_IMAGE_BYTES:
            self._send_json(413, {"error": "Image exceeds the 20 MB limit"})
            return

        image_bytes = self.rfile.read(length)
        if len(image_bytes) != length:
            self._send_json(400, {"error": "Incomplete image upload"})
            return
        try:
            enhanced = enhance_underwater_image(image_bytes)
            ok, encoded = cv2.imencode(
                ".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            if not ok:
                raise ValueError("Could not encode enhanced image")
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        except cv2.error:
            self._send_json(422, {"error": "Image enhancement failed"})
            return

        payload = encoded.tobytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, body: dict[str, str]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"{self.log_date_time_string()} {format_string % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), DeepSignHandler)
    print(f"DeepSign UI: http://{host}:{args.port}/")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DeepSign server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
