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
MAX_GALLERY_SIZE = 10
FEATURE_UPDATE_INTERVAL = 30
TEMP_PHASE_FRAMES = 10
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

    gal1_list = [x[1] for x in f1_items]
    gal2_list = [x[1] for x in f2_items]

    dist_matrix = np.zeros((len(gal1_list), len(gal2_list)))
    for i, gal1 in enumerate(gal1_list):
        for j, gal2 in enumerate(gal2_list):
            if not gal1 or not gal2:
                dist_matrix[i, j] = 1.0
            else:
                dist_matrix[i, j] = min([cosine(f1, f2) for f1 in gal1 for f2 in gal2])

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading ReID model on {device}...")

reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=9,
    loss='softmax',
    pretrained=False
)

weight_path = os.path.join(BASE, 'log', 'osnet', 'model', 'model.pth.tar-30')
try:
    torchreid.utils.load_pretrained_weights(reid_model, weight_path)
    print(f"✅ Successfully loaded custom trained weights from {weight_path}")
except Exception as e:
    print(f"⚠️ Failed to load custom weights: {e}")

reid_model.to(device)
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
    img = transform(img).unsqueeze(0).to(device)
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

    id_mapping          = {}   # DeepSort track_id → display_id
    next_id             = 1
    track_features      = {}   # display_id → list of features (within-cam ReID gallery)
    track_frame_count   = {}   # DeepSort track_id → frame count since last feature extraction
    track_age           = {}   # DeepSort track_id → frame count since track started
    temp_track_features = {}   # DeepSort track_id → list of temporary features
    seen_ids            = set()

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

        current_active_ds_ids = [t.track_id for t in tracks if t.is_confirmed() and t.time_since_update <= 1]
        currently_active_display_ids = set([id_mapping[dsid] for dsid in current_active_ds_ids if dsid in id_mapping and not str(id_mapping[dsid]).startswith("Temp")])

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            ds_id      = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            
            if ds_id not in track_age:
                track_age[ds_id] = 0
            track_age[ds_id] += 1
            crop       = frame[t:b, l:r]
            if crop.size == 0:
                continue

            # ── Within-camera ReID ─────────────────────────────────────────
            if ds_id not in id_mapping:
                id_mapping[ds_id] = f"Temp-{ds_id}"
                temp_track_features[ds_id] = []

            if str(id_mapping[ds_id]).startswith("Temp"):
                new_feat_added = False
                # Extract sparingly to preserve FPS
                if track_age[ds_id] in [1, max(2, TEMP_PHASE_FRAMES // 2), TEMP_PHASE_FRAMES]:
                    with feature_lock:
                        feat = extract_feature(crop)
                    temp_track_features[ds_id].append(feat)
                    new_feat_added = True

                if new_feat_added:
                    best_match = None
                    best_dist  = REID_THRESHOLD

                    for did, gallery in track_features.items():
                        if did in currently_active_display_ids:
                            continue # SPATIAL CONSTRAINT
                        min_dist = min([cosine(f1, f2) for f1 in temp_track_features[ds_id] for f2 in gallery]) if gallery else 1.0
                        if min_dist < best_dist:
                            best_dist  = min_dist
                            best_match = did

                    if best_match is not None:
                        # Match Found! Graduate to Permanent
                        
                        # --- FIX GHOST TRACK STEALING ---------------------
                        # If a sleeping 'ghost' track still owns this ID, it means DeepSORT lost them but hasn't deleted them.
                        # We must forcefully unmap the ghost track so if it wakes up, it doesn't draw a duplicate ID on screen!
                        ghosts = [old_ds for old_ds, old_did in id_mapping.items() if old_did == best_match]
                        for g in ghosts:
                            del id_mapping[g]
                        # --------------------------------------------------

                        id_mapping[ds_id] = best_match
                        currently_active_display_ids.add(best_match)
                        track_features[best_match].extend(temp_track_features[ds_id])
                        track_features[best_match] = track_features[best_match][-MAX_GALLERY_SIZE:]
                        del temp_track_features[ds_id]
                        track_frame_count[ds_id] = 0

                # Check Phase Expiration
                if str(id_mapping[ds_id]).startswith("Temp") and track_age[ds_id] >= TEMP_PHASE_FRAMES:
                    # New Person!
                    id_mapping[ds_id] = next_id
                    next_id += 1
                    track_features[id_mapping[ds_id]] = temp_track_features[ds_id]
                    del temp_track_features[ds_id]
                    track_frame_count[ds_id] = 0
                    currently_active_display_ids.add(id_mapping[ds_id])

            else:
                # Permanent Track, update gallery periodically
                track_frame_count[ds_id] += 1
                if track_frame_count[ds_id] >= FEATURE_UPDATE_INTERVAL:
                    track_frame_count[ds_id] = 0
                    with feature_lock:
                        feat = extract_feature(crop)
                    display_id = id_mapping[ds_id]
                    track_features[display_id].append(feat)
                    if len(track_features[display_id]) > MAX_GALLERY_SIZE:
                        track_features[display_id].pop(0)
                    
                    with cross_cam_lock:
                        if cam == "cam1":
                            cam1_feature_store[display_id] = track_features[display_id]
                        else:
                            cam2_feature_store[display_id] = track_features[display_id]
                        with counts_lock:
                            _recompute_live_unique()

            display_id = id_mapping[ds_id]
            
            # ── New person → update counts + trigger live cross-cam match ──
            if not str(display_id).startswith("Temp"):
                active_ids.add(display_id)
                if display_id not in seen_ids:
                    seen_ids.add(display_id)

                    with counts_lock:
                        live_counts[f"{cam}_total"] = len(seen_ids)

                    with cross_cam_lock:
                        if cam == "cam1":
                            cam1_feature_store[display_id] = track_features[display_id]
                        else:
                            cam2_feature_store[display_id] = track_features[display_id]

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Preparing to load YOLO on {device}...")
    
    detector1 = YOLO("yolov8s.pt")
    detector1.to(device)
    detector2 = YOLO("yolov8s.pt")
    detector2.to(device)
    
    print(f"✅ YOLO Detector 1 is running on: {detector1.device}")
    print(f"✅ YOLO Detector 2 is running on: {detector2.device}")

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