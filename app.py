from flask import Flask, render_template
import sqlite3
import subprocess
import time
import sys
import os
import threading
import time
import cv2
import numpy as np
from flask import Response
import main

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start")
def start():
    if not hasattr(app, "pipeline_thread") or not app.pipeline_thread.is_alive():
        main.pipeline_done = False
        app.pipeline_thread = threading.Thread(target=main.run_pipeline)
        app.pipeline_thread.start()
    
    return render_template("live.html")

def gen_frames(cam_id):
    while True:
        if cam_id == "cam1":
            frame = main.latest_frame_cam1
        else:
            frame = main.latest_frame_cam2
            
        if frame is None:
            frame = np.zeros((512, 640, 3), dtype=np.uint8)
            text = "Waiting for Camera 1..." if cam_id == "cam2" else "Starting..."
            cv2.putText(frame, text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route("/video_feed/1")
def video_feed_1():
    return Response(gen_frames("cam1"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed/2")
def video_feed_2():
    return Response(gen_frames("cam2"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    if not main.pipeline_done:
        return {"status": "running"}

    db_path = os.path.join(BASE, "database.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT * FROM stats")
    data = cur.fetchone()
    conn.close()

    if data is None:
        return {"status": "running"}

    return {
        "status": "done",
        "cam1": data[0],
        "cam2": data[1],
        "unique": data[2],
        "matches": data[3]
    }

@app.route("/results")
def results():
    if not main.pipeline_done:
        return "Pipeline is still running, analyzing surveillance footage... Please wait and refresh.", 202

    time_stamp = int(time.time())

    db_path = os.path.join(BASE, "database.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT * FROM stats")
    data = cur.fetchone()

    conn.close()

    if data is None:
        data = (0,0,0,0)

    return render_template(
        "results.html",
        cam1=data[0],
        cam2=data[1],
        unique=data[2],
        matches=data[3],
        time=time_stamp
    )

if __name__ == "__main__":
    app.run(debug=True)