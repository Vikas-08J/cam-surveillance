import cv2
import os
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import torch
import torchvision.transforms as transforms
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import torchreid

# ==============================
# GLOBALS
# ==============================
latest_frame_cam1 = None
latest_frame_cam2 = None
pipeline_done     = False
pipeline_running  = False   # NEW: True only while actively processing

BASE = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE, "static/outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE, "static/extracted/cam1"), exist_ok=True)
os.makedirs(os.path.join(BASE, "static/extracted/cam2"), exist_ok=True)

# ==============================
# REAL-TIME LIVE COUNTS
# ==============================
counts_lock = threading.Lock()
live_counts = {
    "cam1_active": 0,
    "cam1_total":  0,
    "cam2_active": 0,
    "cam2_total":  0,
    "matches":     0,
    "unique":      0,
    "cam1_done":   False,
    "cam2_done":   False,
}

# ==============================
# SHARED CROSS-CAM FEATURE STORES
# ==============================
cross_cam_lock     = threading.Lock()
cam1_feature_store = {}   # { display_id: feature_vector }
cam2_feature_store = {}
matched_pairs      = set()

CROSS_CAM_THRESHOLD = 0.4


# ==============================
# RESET — called before each new pipeline run
# ==============================
def reset_pipeline():
    """Clears all state so the pipeline can be started fresh."""
    global latest_frame_cam1, latest_frame_cam2, pipeline_done, pipeline_running

    latest_frame_cam1 = None
    latest_frame_cam2 = None
    pipeline_done     = False
    pipeline_running  = False

    with counts_lock:
        live_counts.update({
            "cam1_active": 0,
            "cam1_total":  0,
            "cam2_active": 0,
            "cam2_total":  0,
            "matches":     0,
            "unique":      0,
            "cam1_done":   False,
            "cam2_done":   False,
        })

    with cross_cam_lock:
        cam1_feature_store.clear()
        cam2_feature_store.clear()
        matched_pairs.clear()


# ==============================
# LIVE CROSS-CAM MATCHING
# ==============================
def _recompute_live_unique():
    """
    Must be called inside cross_cam_lock + counts_lock.
    Uses Hungarian algorithm over current feature stores and updates live_counts.
    """
    f1_items = list(cam1_feature_store.items())
    f2_items = list(cam2_feature_store.items())

    if not f1_items or not f2_items:
        live_counts["matches"] = 0
        live_counts["unique"]  = (live_counts["cam1_total"] +
                                  live_counts["cam2_total"])
        return

    feats1 = [x[1] for x in f1_items]
    feats2 = [x[1] for x in f2_items]

    dist_matrix = np.zeros((len(feats1), len(feats2)))
    for i, f1 in enumerate(feats1):
        for j, f2 in enumerate(feats2):
            dist_matrix[i, j] = cosine(f1, f2)

    rows, cols = linear_sum_assignment(dist_matrix)

    matches = sum(
        1 for r, c in zip(rows, cols)
        if dist_matrix[r][c] < CROSS_CAM_THRESHOLD
    )

    live_counts["matches"] = matches
    live_counts["unique"]  = max(
        0,
        live_counts["cam1_total"] + live_counts["cam2_total"] - matches
    )

    print(f"[CROSS-CAM LIVE] Cam1: {live_counts['cam1_total']} | "
          f"Cam2: {live_counts['cam2_total']} | "
          f"Matches: {matches} | "
          f"Unique: {live_counts['unique']}")


# ==============================
# REID MODEL (OSNet)
# ==============================
print("Loading ReID model...")
reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
reid_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

feature_lock = threading.Lock()


def extract_feature(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        feat = reid_model(img)
    feat = feat.cpu().numpy().flatten()
    return feat / np.linalg.norm(feat)


# ==============================
# CLUSTERING
# ==============================
def cluster_features(data, threshold=0.25):
    clusters = []
    for item in data:
        matched = False
        for c in clusters:
            if cosine(item['feature'], c['feature']) < threshold:
                matched = True
                break
        if not matched:
            clusters.append(item)
    return clusters


# ==============================
# CAMERA PROCESSING
# ==============================
REID_THRESHOLD = 0.25


def process_camera(video_path, cam, tracker, detector):
    global latest_frame_cam1, latest_frame_cam2

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(3))
    height = int(cap.get(4))
    fps    = int(cap.get(5))

    out = cv2.VideoWriter(
        os.path.join(BASE, f"static/outputs/{cam}.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        max(1, fps // 3),
        (width, height)
    )

    id_mapping     = {}   # DeepSort track_id → display_id
    next_id        = 1
    track_features = {}   # display_id → feature (within-cam ReID)
    seen_ids       = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame, classes=[0], conf=0.5, iou=0.5, verbose=False)

        detections = []
        for r in results:
            boxes  = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                if w < 30 or h < 60:
                    continue
                detections.append(([x1, y1, w, h], float(score), 'person'))

        tracks     = tracker.update_tracks(detections, frame=frame)
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            ds_id      = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            crop       = frame[t:b, l:r]
            if crop.size == 0:
                continue

            # ── Within-camera ReID ─────────────────────────────────────────
            if ds_id not in id_mapping:
                with feature_lock:
                    feat = extract_feature(crop)

                best_match = None
                best_dist  = REID_THRESHOLD

                for did, f in track_features.items():
                    dist = cosine(feat, f)
                    if dist < best_dist:
                        best_dist  = dist
                        best_match = did

                if best_match is not None:
                    id_mapping[ds_id] = best_match
                    # Running average keeps features fresh
                    updated = track_features[best_match] * 0.7 + feat * 0.3
                    track_features[best_match] = updated / np.linalg.norm(updated)
                else:
                    id_mapping[ds_id] = next_id
                    next_id += 1

                track_features[id_mapping[ds_id]] = feat

            display_id = id_mapping[ds_id]
            active_ids.add(display_id)

            # ── New person → update counts + trigger live cross-cam match ──
            if display_id not in seen_ids:
                seen_ids.add(display_id)

                # 1. Update this camera's total instantly
                with counts_lock:
                    live_counts[f"{cam}_total"] = len(seen_ids)

                # 2. Write feature to shared cross-cam store
                with cross_cam_lock:
                    if cam == "cam1":
                        cam1_feature_store[display_id] = track_features[display_id]
                    else:
                        cam2_feature_store[display_id] = track_features[display_id]

                    # 3. Re-run live Hungarian matching immediately
                    with counts_lock:
                        _recompute_live_unique()

            # ── Update active count every frame ────────────────────────────
            with counts_lock:
                live_counts[f"{cam}_active"] = len(active_ids)

            # Draw bounding box + ID
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {display_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── HUD overlay ───────────────────────────────────────────────────
        with counts_lock:
            c1a = live_counts["cam1_active"]
            c2a = live_counts["cam2_active"]
            c1t = live_counts["cam1_total"]
            c2t = live_counts["cam2_total"]
            mtc = live_counts["matches"]
            unq = live_counts["unique"]

        hud = (
            [
                f"[CAM1] Active: {c1a}  Total: {c1t}",
                f"[CAM2] Active: {c2a}  Total: {c2t}",
                f"Matches: {mtc}  |  Unique: {unq}",
            ]
            if cam == "cam1" else
            [
                f"[CAM2] Active: {c2a}  Total: {c2t}",
                f"[CAM1] Active: {c1a}  Total: {c1t}",
                f"Matches: {mtc}  |  Unique: {unq}",
            ]
        )

        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        if cam == "cam1":
            latest_frame_cam1 = frame.copy()
        else:
            latest_frame_cam2 = frame.copy()

        out.write(frame)

    cap.release()
    out.release()

    with counts_lock:
        live_counts[f"{cam}_done"] = True

    return list(track_features.values()), len(seen_ids)


# ==============================
# MAIN PIPELINE
# ==============================
def run_pipeline():
    global pipeline_done, pipeline_running

    pipeline_running = True

    detector1 = YOLO("yolov8s.pt")
    detector2 = YOLO("yolov8s.pt")

    tracker_cfg = dict(
        max_age=120,
        n_init=5,
        max_cosine_distance=0.2,
        nn_budget=500
    )

    tracker1 = DeepSort(**tracker_cfg)
    tracker2 = DeepSort(**tracker_cfg)

    cam1_path = os.path.join(BASE, "inputs/cam1.mp4")
    cam2_path = os.path.join(BASE, "inputs/cam2.mp4")

    result = {}

    def cam1():
        result['cam1'] = process_camera(cam1_path, "cam1", tracker1, detector1)

    def cam2():
        result['cam2'] = process_camera(cam2_path, "cam2", tracker2, detector2)

    t1 = threading.Thread(target=cam1)
    t2 = threading.Thread(target=cam2)
    t1.start(); t2.start()
    t1.join();  t2.join()

    # Final definitive cross-cam match after both cameras finish
    with cross_cam_lock:
        with counts_lock:
            _recompute_live_unique()
            final_matches = live_counts["matches"]
            final_unique  = live_counts["unique"]

    print("\n===== FINAL STATS =====")
    print(f"Cam1 Total : {live_counts['cam1_total']}")
    print(f"Cam2 Total : {live_counts['cam2_total']}")
    print(f"Matches    : {final_matches}")
    print(f"Unique     : {final_unique}")

    pipeline_running = False
    pipeline_done    = True