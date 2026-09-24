"""Browser-based click tool that writes configs/scene.json (RUNBOOK P1.1, schema: docs/SPEC.md §4).

The runtime ships headless OpenCV (no windows, SPEC §12.12), so the tool is a local web page instead of
cv2.imshow: this script serves configs/reference.jpg and scripts/calibrate_scene.html on 127.0.0.1 and
opens the browser. In the page: pick a layer (keys 1-9, 0, q, w, e or the list), left-click points,
Enter to finish a shape, U to undo, Esc to cancel, S to save; wheel zooms, right-drag pans, F fits.
Attributes (ids, lane direction group, allowed exits, signal ids, world coordinates of homography
points) are asked in a form after each shape. Saving writes scene.json (the previous one is kept as
scene.json.bak) and lists any consistency problems (roadwatch.scene.calibration.validate_scene).

Usage:
  python scripts/calibrate_scene.py --video samples/C3902.MP4   # first time: builds the reference frame
  python scripts/calibrate_scene.py                             # edit the existing scene
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roadwatch.config import CONFIG_DIR, REPO_ROOT, SCENE_PATH  # noqa: E402
from roadwatch.scene.calibration import (  # noqa: E402
    LAYERS,
    make_reference,
    scene_to_shapes,
    shapes_to_scene,
    validate_scene,
)
from roadwatch.scene.overlay import LAYER_COLOURS  # noqa: E402

PAGE = Path(__file__).with_suffix(".html")
REFERENCE_PATH = CONFIG_DIR / "reference.jpg"


class CalibrationApp:
    """State behind the page: the reference image, the scene file and the layer definitions."""

    def __init__(self, reference: Path, scene_path: Path) -> None:
        img = cv2.imread(str(reference))
        if img is None:
            raise FileNotFoundError(f"cannot read the reference frame {reference}")
        self.reference = reference
        self.image_size = (img.shape[1], img.shape[0])
        self.scene_path = scene_path

    def state(self) -> dict[str, Any]:
        scene = json.loads(self.scene_path.read_text(encoding="utf-8")) if self.scene_path.exists() else {}
        colours = {name: "#{2:02x}{1:02x}{0:02x}".format(*bgr) for name, bgr in LAYER_COLOURS.items()}
        colours["lane_directions"] = colours["lanes"]
        return {
            "layers": LAYERS,
            "colours": colours,
            "image_size": self.image_size,
            "shapes": scene_to_shapes(scene),
            "u_turn_prohibited_everywhere": bool(scene.get("u_turn_prohibited_everywhere", False)),
            "problems": validate_scene(scene) if scene else [],
        }

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            reference = self.reference.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            reference = str(self.reference)
        scene = shapes_to_scene(
            body["shapes"], self.image_size, reference, bool(body.get("u_turn_prohibited_everywhere"))
        )
        if self.scene_path.exists():
            shutil.copyfile(self.scene_path, self.scene_path.with_suffix(".json.bak"))
        tmp = self.scene_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(scene, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self.scene_path)
        return {"path": str(self.scene_path), "problems": validate_scene(scene)}


def make_handler(app: CalibrationApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/reference.jpg":
                self._send(200, app.reference.read_bytes(), "image/jpeg")
            elif self.path == "/api/state":
                self._json(200, app.state())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/save":
                self._json(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._json(200, app.save(body))
            except (ValueError, KeyError, TypeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, help="sample video to build the reference frame from")
    ap.add_argument("--frames", type=int, default=49, help="frames in the reference median (default 49)")
    ap.add_argument("--remake-reference", action="store_true", help="rebuild the reference even if it exists")
    ap.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    ap.add_argument("--scene", type=Path, default=SCENE_PATH)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.video and (args.remake_reference or not args.reference.exists()):
        print(f"building the reference frame: median of {args.frames} frames of {args.video} ...", flush=True)
        cv2.imwrite(
            str(args.reference), make_reference(args.video, args.frames), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        print(f"wrote {args.reference}")
    if not args.reference.exists():
        ap.error(f"{args.reference} does not exist: pass --video <sample.mp4> once to build it")

    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(CalibrationApp(args.reference, args.scene))
    )
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"calibration tool on {url} (Ctrl+C to stop); saving to {args.scene}", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
