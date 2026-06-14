import cv2
import threading
import time
import os
from flask import Flask, Response, jsonify, render_template

import main as pipeline

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── Pipeline state ───────────────────────────────────────────────────────────
_pipeline_lock    = threading.Lock()
_pipeline_started = False


# ─── MJPEG helpers ────────────────────────────────────────────────────────────
PLACEHOLDER = None

def _placeholder_frame():
    global PLACEHOLDER
    if PLACEHOLDER is None:
        import numpy as np
        img = cv2.imencode(
            '.jpg',
            __import__('numpy').zeros((360, 640, 3), dtype='uint8'),
            [cv2.IMWRITE_JPEG_QUALITY, 60]
        )[1]
        PLACEHOLDER = img.tobytes()
    return PLACEHOLDER


def _gen_stream(cam: str):
    while True:
        frame = (pipeline.latest_frame_cam1
                 if cam == "cam1"
                 else pipeline.latest_frame_cam2)
        if frame is not None:
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg = buf.tobytes() if ok else _placeholder_frame()
        else:
            jpeg = _placeholder_frame()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        time.sleep(0.07)   # ~14 fps


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/stream/cam1')
def stream_cam1():
    return Response(_gen_stream('cam1'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream/cam2')
def stream_cam2():
    return Response(_gen_stream('cam2'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    """Thread-safe snapshot of live_counts from main.py."""
    with pipeline.counts_lock:
        c = dict(pipeline.live_counts)

    return jsonify({
        "cam1_active":      c.get("cam1_active",    0),
        "cam1_total":       c.get("cam1_total",     0),
        "cam2_active":      c.get("cam2_active",    0),
        "cam2_total":       c.get("cam2_total",     0),
        "matches":          c.get("matches",         0),
        "unique":           c.get("unique",          0),
        "cam1_done":        c.get("cam1_done",       False),
        "cam2_done":        c.get("cam2_done",       False),
        "pipeline_done":    pipeline.pipeline_done,
        "pipeline_running": pipeline.pipeline_running,
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Starts (or restarts) the pipeline in a background thread."""
    global _pipeline_started

    with _pipeline_lock:
        if pipeline.pipeline_running:
            return jsonify({"ok": False, "error": "Pipeline already running"}), 409

        pipeline.reset_pipeline()
        _pipeline_started = True

    def _run():
        pipeline.run_pipeline()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"ok": True, "message": "Pipeline started"})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n>>> Dashboard: http://127.0.0.1:5000  <<<\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)