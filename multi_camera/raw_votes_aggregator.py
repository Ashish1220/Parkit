import os
import json
from collections import defaultdict
from datetime import datetime
import threading
import time

DEBUG = True

def debug(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")
def sync_single_camera_segments(camera_dir="cameras", segment_base_dir="cameras/SEGMENTATION_LOGS", segment_map_file="segment_mapping.json"):
    segment_map = load_json(os.path.join(camera_dir, segment_map_file), {})

    for segment_id, cameras in segment_map.items():
        if len(cameras) == 1:
            cam_name = cameras[0]
            source_log_dir = os.path.join(camera_dir, cam_name, "logs")
            target_segment_dir = os.path.join(segment_base_dir, segment_id)
            os.makedirs(target_segment_dir, exist_ok=True)

            # Process slot_states.json
            slot_states_path = os.path.join(source_log_dir, "slot_states.json")
            if os.path.exists(slot_states_path):
                slot_states = load_json(slot_states_path, {})
                updated_states = {
                    f"{cam_name}.{slot_name}": state
                    for slot_name, state in slot_states.items()
                }
                save_json(os.path.join(target_segment_dir, "slot_states.json"), updated_states)

            # Process timelogs.json
            timelogs_path = os.path.join(source_log_dir, "timelogs.json")
            if os.path.exists(timelogs_path):
                timelogs = load_json(timelogs_path, [])
                updated_logs = []
                for entry in timelogs:
                    entry = entry.copy()
                    if "slot" in entry:
                        entry["slot"] = f"{cam_name}.{entry['slot']}"
                    updated_logs.append(entry)
                save_json(os.path.join(target_segment_dir, "timelogs.json"), updated_logs)

            # Optional: nearestlogs.json (not camera-prefixed, but we copy as-is)
            nearest_path = os.path.join(source_log_dir, "nearestlogs.json")
            if os.path.exists(nearest_path):
                nearest_logs = load_json(nearest_path, [])
                save_json(os.path.join(target_segment_dir, "nearestlogs.json"), nearest_logs)


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return default
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def get_cameras_in_segment(segment_id, camera_dir="cameras", segment_file="segment_mapping.json"):
    segment_path = os.path.join(camera_dir, segment_file)
    mapping = load_json(segment_path, {})
    return mapping.get(segment_id, [])

def get_slot_global_map(camera_name):
    slot_file = os.path.join("cameras", camera_name, "slot_regions.json")
    slot_data = load_json(slot_file, {})
    slot_map = {}
    for slot in slot_data.get("slots", []):
        name = slot["name"]
        global_id = slot.get("global_id")
        slot_map[name] = global_id 
    return slot_map

def aggregate_segment_votes(segment_id, base_dir="cameras/SEGMENTATION_LOGS"):
    segment_path = os.path.join(base_dir, segment_id)
    os.makedirs(segment_path, exist_ok=True)

    raw_vote_path = os.path.join(segment_path, "raw_votes.json")
    slot_states_path = os.path.join(segment_path, "slot_states.json")
    timelog_path = os.path.join(segment_path, "timelogs.json")

    votes = load_json(raw_vote_path, [])
    slot_states = load_json(slot_states_path, {})
    timelogs = load_json(timelog_path, [])

    grouped_votes = defaultdict(list)
    for vote in votes:
        grouped_votes[vote["slot_id"]].append(vote)

    now = datetime.now().isoformat()

    # 1. Process global slots from raw votes
    for slot_id, vote_list in grouped_votes.items():
        status_score = defaultdict(float)
        for vote in vote_list:
            status_score[vote["status"]] += vote.get("confidence", 0)

        final_status = max(status_score.items(), key=lambda x: x[1])[0]
        prev_data = slot_states.get(slot_id, {})
        prev_status = prev_data.get("status")
        prev_time = prev_data.get("timestamp", now)

        try:
            duration = (datetime.now() - datetime.fromisoformat(prev_time)).total_seconds()
        except Exception:
            duration = 0.0

        if final_status != prev_status:
            timelogs.append({
                "slot": slot_id,
                "previous_status": prev_status or "Unknown",
                "new_status": final_status,
                "timestamp": now,
                "duration_since_last_change": round(duration, 2)
            })

        slot_states[slot_id] = {"status": final_status, "timestamp": now}

    # 2. Always process local slots from camera logs
    cameras = get_cameras_in_segment(segment_id)
    for camera in cameras:
        local_slot_path = os.path.join("cameras", camera, "logs", "slot_states.json")
        cam_states = load_json(local_slot_path, {})
        slot_map = get_slot_global_map(camera)

        for slot_name, state in cam_states.items():
            global_id = slot_map.get(slot_name)
            if not global_id:  # local-only slot
                formatted_name = f"{camera}.{slot_name}"
                prev_data = slot_states.get(formatted_name, {})
                prev_status = prev_data.get("status")
                prev_time = prev_data.get("timestamp", now)

                try:
                    duration = (datetime.now() - datetime.fromisoformat(prev_time)).total_seconds()
                except Exception:
                    duration = 0.0

                if prev_status != state["status"]:
                    timelogs.append({
                        "slot": formatted_name,
                        "previous_status": prev_status or "Unknown",
                        "new_status": state["status"],
                        "timestamp": now,
                        "duration_since_last_change": round(duration, 2)
                    })

                slot_states[formatted_name] = {
                    "status": state["status"],
                    "timestamp": now
                }

    # 3. Save everything
    save_json(slot_states_path, slot_states)
    save_json(timelog_path, timelogs)
    save_json(raw_vote_path, [])  # Only clears global votes
    debug(f"✅ Aggregated and saved results for {segment_id}")

    sync_single_camera_segments()

def aggregate_all_segments(base_dir="cameras/SEGMENTATION_LOGS"):
    if not os.path.exists(base_dir):
        return
    for segment_id in os.listdir(base_dir):
        segment_path = os.path.join(base_dir, segment_id)
        if os.path.isdir(segment_path):
            aggregate_segment_votes(segment_id, base_dir)

def aggregator_loop(interval_sec=1):
    print("🔁 Aggregating raw votes every", interval_sec, "seconds...")
    while True:
        try:
            aggregate_all_segments()
        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
        time.sleep(interval_sec)

if __name__ == "__main__":
    t = threading.Thread(target=aggregator_loop, daemon=True)
    t.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Aggregator stopped.")
