import cv2
import os
import sqlite3
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading

yolo_lock = threading.Lock()
latest_frame_cam1 = None
latest_frame_cam2 = None
pipeline_done = False

print("\nStarting Surveillance Pipeline\n")

# --------------------------------------------------
# Base path (IMPORTANT for Flask)
# --------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# Create required folders
# --------------------------------------------------

os.makedirs(os.path.join(BASE,"static/outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE,"static/extracted/cam1"), exist_ok=True)
os.makedirs(os.path.join(BASE,"static/extracted/cam2"), exist_ok=True)

# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

detector = YOLO("yolov8n.pt")

# --------------------------------------------------
# Feature extractor
# --------------------------------------------------

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor()
])

def extract_feature(img):

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        f = model(img)

    return f.numpy().flatten()

def cluster_features(raw_data_list, threshold=0.15):
    if not raw_data_list: return []
    unique_clusters = []
    
    import numpy as np
    from scipy.spatial.distance import cosine
    
    for item in raw_data_list:
        f = item['feature']
        matched = False
        for i, cluster in enumerate(unique_clusters):
            uf = cluster['feature']
            dist = cosine(f, uf)
            if dist < threshold:
                # Keep the largest crop image as the representative
                if item['area'] > cluster['area']:
                    unique_clusters[i]['crop'] = item['crop']
                    unique_clusters[i]['area'] = item['area']
                
                # Merge feature representations mathematically
                merged = (uf + f) / 2.0
                norm = np.linalg.norm(merged)
                unique_clusters[i]['feature'] = merged / norm if norm > 0 else merged
                matched = True
                break
        if not matched:
            norm = np.linalg.norm(f)
            item['feature'] = f / norm if norm > 0 else f
            unique_clusters.append(item)
            
    return unique_clusters

# --------------------------------------------------
# Camera processing
# --------------------------------------------------

def process_camera(video_path, cam, tracker):

    global latest_frame_cam1, latest_frame_cam2

    print(f"\nProcessing {cam} video:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video:", video_path)
        return [], 0

    persons = []
    seen_ids = set()
    best_crops = {}

    id_mapping = {}
    next_id = 1

    frame_no = 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    process_fps = max(1, int(fps / 3))

    # video output
    out_path = os.path.join(BASE,f"static/outputs/{cam}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path,fourcc,process_fps,(width,height))

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1

        if frame_no % 3 != 0:
            continue
            
        with yolo_lock:
            # Lower confidence to catch difficult-to-detect people
            results = detector(frame, classes=[0], conf=0.35, verbose=False)

        detections = []

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()

            for box, cls, score in zip(boxes, classes, scores):

                if int(cls) == 0:

                    x1,y1,x2,y2 = map(int,box)

                    w = x2-x1
                    h = y2-y1

                    detections.append(([x1,y1,w,h],score,'person'))

        tracks = tracker.update_tracks(detections,frame=frame)

        for track in tracks:

            if not track.is_confirmed() or track.time_since_update > 2:
                continue

            track_id = track.track_id
            
            if track_id not in id_mapping:
                id_mapping[track_id] = next_id
                next_id += 1
            
            display_id = id_mapping[track_id]

            l,t,r,b = map(int,track.to_ltrb())
            
            l = max(0, l)
            t = max(0, t)
            r = min(width, r)
            b = min(height, b)

            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)

            cv2.putText(frame,f"ID {display_id}",(l,t-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            seen_ids.add(display_id)

            crop = frame[t:b,l:r]

            if crop.size == 0 or track.time_since_update > 0:
                continue
                
            area = (r - l) * (b - t)
            if display_id not in best_crops or area > best_crops[display_id]['area']:
                best_crops[display_id] = {'area': area, 'crop': crop.copy()}

        if cam == "cam1":
            latest_frame_cam1 = frame.copy()
        else:
            latest_frame_cam2 = frame.copy()

        out.write(frame)

    cap.release()
    out.release()

    raw_data = []

    for display_id, data in best_crops.items():
        crop = data['crop']
        feature = extract_feature(crop)
        raw_data.append({'feature': feature, 'crop': crop, 'id': display_id, 'area': data['area']})

    return raw_data, len(seen_ids)

def run_pipeline():
    global pipeline_done
    
    # --------------------------------------------------
    # Reset tracking and DB
    # --------------------------------------------------
    db_path = os.path.join(BASE,"database.db")
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS stats(
    cam1 INTEGER,
    cam2 INTEGER,
    unique_people INTEGER,
    matches INTEGER
    )
    """)
    cur.execute("DELETE FROM stats")
    
    for cam_dir in ["cam1", "cam2"]:
        dir_path = os.path.join(BASE, f"static/extracted/{cam_dir}")
        if os.path.exists(dir_path):    
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))
        else:
            os.makedirs(dir_path, exist_ok=True)
            
    tracker1 = DeepSort(max_age=100, n_init=3, max_cosine_distance=0.5, max_iou_distance=0.9, nn_budget=200)
    tracker2 = DeepSort(max_age=100, n_init=3, max_cosine_distance=0.5, max_iou_distance=0.9, nn_budget=200)

    # --------------------------------------------------
    # Input videos
    # --------------------------------------------------
    cam1_video = os.path.join(BASE,"inputs","cam1.mp4")
    cam2_video = os.path.join(BASE,"inputs","cam2.mp4")

    # --------------------------------------------------
    # Run cameras
    # --------------------------------------------------
    cam1_raw_features, cam1_raw_count = process_camera(cam1_video, "cam1", tracker1)
    cam2_raw_features, cam2_raw_count = process_camera(cam2_video, "cam2", tracker2)

    # --------------------------------------------------
    # Intra-camera Clustering for Re-ID Compression
    # --------------------------------------------------
    cam1_clusters = cluster_features(cam1_raw_features, threshold=0.20)
    cam2_clusters = cluster_features(cam2_raw_features, threshold=0.20)

    cam1_count = len(cam1_clusters)
    cam2_count = len(cam2_clusters)
    
    # Save the strictly clustered, deduplicated images
    cam1_features = []
    for final_id, item in enumerate(cam1_clusters, 1):
        path = os.path.join(BASE, f"static/extracted/cam1/{final_id}.jpg")
        cv2.imwrite(path, item['crop'])
        cam1_features.append(item['feature'])

    cam2_features = []
    for final_id, item in enumerate(cam2_clusters, 1):
        path = os.path.join(BASE, f"static/extracted/cam2/{final_id}.jpg")
        cv2.imwrite(path, item['crop'])
        cam2_features.append(item['feature'])

    print(f"\nExtraction & Intra-Camera Clustering completed.")
    print(f"Cam1 Unique True Persons: {cam1_count} (raw tracker IDs: {cam1_raw_count})")
    print(f"Cam2 Unique True Persons: {cam2_count} (raw tracker IDs: {cam2_raw_count})")

    # --------------------------------------------------
    # Matching
    # --------------------------------------------------
    matches = 0

    if len(cam1_features) > 0 and len(cam2_features) > 0:
        distance_matrix = np.ones((len(cam1_features), len(cam2_features)))
        
        for i, f1 in enumerate(cam1_features):
            for j, f2 in enumerate(cam2_features):
                distance_matrix[i, j] = cosine(f1, f2)
                
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] < 0.4:
                matches += 1

    unique_people = cam1_count + cam2_count - matches

    print("Matches:",matches)
    print("Unique people:",unique_people)

    # --------------------------------------------------
    # Save stats
    # --------------------------------------------------
    cur.execute("INSERT INTO stats VALUES (?,?,?,?)",
                (cam1_count,cam2_count,unique_people,matches))

    conn.commit()
    conn.close()

    print("\nPipeline Completed\n")
    pipeline_done = True