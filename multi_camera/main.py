import cv2
import json
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
import argparse
from collections import defaultdict


# --------------------------- Config ---------------------------
YOLO_MODEL_PATH = "yolov8x.pt"
IOU_THRESHOLD = 0.55
VIDEO_SOURCE = 0  
debug=True
# --------------------------- Argument Parsing ---------------------------
def parse_args():
    
    parser = argparse.ArgumentParser(description="Smart Parking System")
    parser.add_argument("--live", action="store_true", help="Enable live video feed")
    parser.add_argument("--camera", type=str, default="camera1", help="Camera name (e.g., camera1)")
    parser.add_argument("--format", type=str, default="jpg", help="static_image_detection_format")

    return parser.parse_args()

# --------------------------- JSON Utilities ---------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# --------------------------- IOU Utility ---------------------------
def box_polygon_iou(car_box, slot_polygon, shape):
    car_polygon = np.array([
        [car_box[0], car_box[1]],
        [car_box[2], car_box[1]],
        [car_box[2], car_box[3]],
        [car_box[0], car_box[3]]
    ], dtype=np.int32)

    mask_slot = np.zeros(shape, dtype=np.uint8)
    mask_car = np.zeros(shape, dtype=np.uint8)

    cv2.fillPoly(mask_slot, [np.array(slot_polygon, dtype=np.int32)], 255)
    cv2.fillPoly(mask_car, [car_polygon], 255)

    intersection = cv2.bitwise_and(mask_slot, mask_car)
    inter_area = np.count_nonzero(intersection)
    slot_area = np.count_nonzero(mask_slot)

    return inter_area / slot_area if slot_area != 0 else 0.0

def centroid_of_polygon(polygon):
    polygon = np.array(polygon)
    M = cv2.moments(polygon)
    if M["m00"] != 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return tuple(polygon[0])

def build_segment_lookup(mapping_path="cameras/segment_mapping.json"):
    mapping = load_json(mapping_path)
    lookup = {}
    for segment_id, camera_list in mapping.items():
        for cam in camera_list:
            lookup[cam] = segment_id
    return lookup


# --------------------------- Slot Status Update ---------------------------
def update_slot_log(logs, states, slot_name, current_status):
    now = datetime.now().isoformat()
    previous = states.get(slot_name, {})

    if previous.get("status") != current_status:
        duration = (datetime.now() - datetime.fromisoformat(previous.get("timestamp", now))).total_seconds()
        logs.append({
            "slot": slot_name,
            "new_status": current_status,
            "previous_status": previous.get("status", "Unknown"),
            "timestamp": now,
            "duration_since_last_change": round(duration, 2)
        })
        states[slot_name] = {"status": current_status, "timestamp": now}

    return logs, states

# --------------------------- Drawing & Ranking ---------------------------
def draw_annotations(frame, slots, entries, occupied_status):
    centroids = {}
    empty_slots = []
    shape = frame.shape[:2]

    for slot_data in slots:
        coords = slot_data["coords"]
        name = slot_data["name"]
        occupied = occupied_status.get(name, False)

        color = (0, 0, 255) if occupied else (0, 255, 0)
        pts = np.array(coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, name, tuple(coords[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if not occupied:
            empty_slots.append(name)
            centroids[name] = centroid_of_polygon(coords)

    for i, (x, y) in enumerate(entries):
        cv2.circle(frame, (x, y), 6, (255, 0, 0), -1)
        cv2.putText(frame, f"Entry {i+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame, centroids, empty_slots

def rank_slots(empty_centroids, entries):
    ranking = []
    for slot_name, (sx, sy) in empty_centroids.items():
        closest_dist = float("inf")
        closest_entry = None
        for i, (ex, ey) in enumerate(entries):
            dist = np.linalg.norm([sx - ex, sy - ey])
            if dist < closest_dist:
                closest_dist = dist
                closest_entry = f"Entry {i+1}"
        ranking.append({
            "slot": slot_name,
            "closest_entry": closest_entry,
            "distance": str(round(closest_dist, 1)),
            "timestamp": datetime.now().isoformat()
        })
    return sorted(ranking, key=lambda x: x["distance"])

#--------------------------- Debugging Function --------------------------
def debugger(var_name,param):
    if(debug):
        print(f"\nvalue of {var_name} is : {param}\n") 

# --------------------------- Processing ---------------------------
def process_frame(frame, model, slots, entries, logs, nearest_logs, slot_states, segment_lookup, camera_name):
    results = model(frame)
    car_boxes = results[0].boxes.xyxy.cpu().numpy()
    car_scores = results[0].boxes.conf.cpu().numpy()
    occupied_status = {}

    for slot in slots:
        name = slot["name"]
        coords = slot["coords"]
        global_id = slot.get("global_id")
        max_iou = 0
        best_confidence = 0

        for car_box, conf in zip(car_boxes, car_scores):
            iou = box_polygon_iou(car_box, coords, frame.shape[:2])
            if iou > max_iou:
                max_iou = iou
                best_confidence = conf

        occupied = max_iou > IOU_THRESHOLD
        occupied_status[name] = occupied
        status = "Occupied" if occupied else "Empty"
        logs, slot_states = update_slot_log(logs, slot_states, name, status)

        if global_id:
            segment_id = segment_lookup.get(camera_name, "unknown_segment")
            write_confidence_vote(segment_id, global_id, camera_name, status, float(round(best_confidence, 2)))

    frame, centroids, _ = draw_annotations(frame, slots, entries, occupied_status)
    ranking = rank_slots(centroids, entries)
    return frame, ranking, logs, slot_states, nearest_logs + ranking

# --------------------------------------- Raw Voting -----------------------
def write_confidence_vote(segment_id, slot_id, camera_name, status, confidence):
    raw_vote_path = os.path.join("cameras", "SEGMENTATION_LOGS", segment_id, "raw_votes.json")
    os.makedirs(os.path.dirname(raw_vote_path), exist_ok=True)

    vote = {
        "slot_id": slot_id,
        "camera": camera_name,
        "status": status,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

    # Append to raw_votes.json
    existing_votes = load_json(raw_vote_path)
    if not isinstance(existing_votes, list):
        existing_votes = []
    existing_votes.append(vote)
    save_json(raw_vote_path, existing_votes)


# --------------------------- Main ---------------------------
def main(LIVE_FEED=False, camera_name="camera1",format="jpg"):
    base_path = os.path.join("cameras", camera_name)
    region_file = os.path.join(base_path, "slot_regions.json")
    image_path = f'static_detection_images/{camera_name}.{format}'
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timelog_path = os.path.join(log_dir, "timelogs.json")
    nearestlog_path = os.path.join(log_dir, "nearestlogs.json")
    slotstate_path = os.path.join(log_dir, "slot_states.json")
    loaded_slots_data = load_json(region_file)
    print(f"loaded_slots_data: {loaded_slots_data}")

    if 'slots' not in loaded_slots_data or 'entries' not in loaded_slots_data:
        print(f"❌ ERROR: 'slots' or 'entries' not found in {region_file}")
        return  # Or handle it more gracefully

    slots = loaded_slots_data['slots']
    entries = loaded_slots_data['entries']

    logs = load_json(timelog_path)
    nearest_logs = load_json(nearestlog_path)
    slot_states = load_json(slotstate_path)

    model = YOLO(YOLO_MODEL_PATH)

    if LIVE_FEED:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            segment_mapping = load_json(os.path.join("cameras", "segment_mapping.json"))
            segment_lookup = build_segment_lookup('cameras/segment_mapping.json')
            for seg, cams in segment_mapping.items():
                for cam in cams:
                    segment_lookup[cam] = seg

            frame, ranking, logs, slot_states, nearest_logs = process_frame(frame, model, slots, entries, logs, nearest_logs, slot_states, segment_lookup, camera_name)
            save_json(timelog_path, logs)
            save_json(slotstate_path, slot_states)
            save_json(nearestlog_path, nearest_logs)
            cv2.imshow("Parking Slot Status", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(image_path)        
        segment_mapping = load_json(os.path.join("cameras", "segment_mapping.json"))
        segment_lookup = build_segment_lookup('cameras/segment_mapping.json')
        for seg, cams in segment_mapping.items():
            for cam in cams:
                segment_lookup[cam] = seg
        frame, ranking, logs, slot_states, nearest_logs = process_frame(frame, model, slots, entries, logs, nearest_logs, slot_states, segment_lookup, camera_name)
        save_json(timelog_path, logs)
        save_json(slotstate_path, slot_states)
        save_json(nearestlog_path, nearest_logs)

        print("\n🟩 Empty Slots Ranked by Closest Entry:")
        for r in ranking:
            print(f"  - {r['slot']} is closest to {r['closest_entry']} ({r['distance']} px)")

        cv2.imshow("Parking Slot Status", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# --------------------------- Entry Point ---------------------------
if __name__ == "__main__":

    

    
    args = parse_args()
    debugger("camera activated" ,args.camera)
    main(LIVE_FEED=args.live, camera_name=args.camera,format=args.format)
