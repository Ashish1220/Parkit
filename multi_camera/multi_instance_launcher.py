import subprocess
from multiprocessing import Process
import os 
import json

from dag_creator import dag_creator
# ---------------- Global Configuration ----------------
CAMERA_LIST = ["camera1", "camera2","camera3","camera4","camera7"]
LIVE_MODE = False 
camera_Segmentation=True
debug=True
dag_creation=False

# ------------------------------------------------------
def run_camera(camera_name, live_mode):
    args = ["python", "main.py", "--camera", camera_name]
    if live_mode:
        args.append("--live")
    subprocess.run(args)

#--------------------------- Debugging Function --------------------------
def debugger(var_name,param):
    if(debug):
        print(f"\nvalue of {var_name} is : {param}\n") 


# -------------------------------------------PREPROCESS THE SEGMENTS -----------------

def generate_intersecting_segments(camera_dir="cameras"):
    from collections import defaultdict

    camera_graph = defaultdict(set)
    camera_to_slots = defaultdict(list)

    for camera_name in os.listdir(camera_dir):
        region_path = os.path.join(camera_dir, camera_name, "slot_regions.json")
        if not os.path.isfile(region_path):
            continue

        with open(region_path, "r") as f:
            data = json.load(f)

        for slot in data.get("slots", []):
            global_id = slot.get("global_id")
            if global_id:
                camera_to_slots[global_id].append(camera_name)

    for cams in camera_to_slots.values():
        for cam1 in cams:
            for cam2 in cams:
                if cam1 != cam2:
                    camera_graph[cam1].add(cam2)
                    camera_graph[cam2].add(cam1)

    all_cameras = set([
    cam for cam in os.listdir(camera_dir)
    if os.path.isfile(os.path.join(camera_dir, cam, "slot_regions.json"))])  

# Ensure all cameras are keys in camera_graph (even if disconnected)
    for cam in all_cameras:
        if cam not in camera_graph:
            camera_graph[cam] = set()
  

    visited = set()
    segments = []

    def dfs(cam, group):
        visited.add(cam)
        group.add(cam)
        for neighbor in camera_graph[cam]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for cam in all_cameras:
        if cam not in visited:
            group = set()
            dfs(cam, group)
            segments.append(sorted(list(group)))

    result = {f"segment {i+1}": group for i, group in enumerate(segments)}

    output_path = os.path.join(camera_dir, "segment_mapping.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Segment mapping saved to {output_path}")
    return result

if __name__ == "__main__":
    if(camera_Segmentation):
        debugger("camera_segment_mapping",generate_intersecting_segments())
    
    if(dag_creation):
        debugger("DAG CREATOR RUNNING",dag_creator())
    

    processes = []
    for cam in CAMERA_LIST:
        p = Process(target=run_camera, args=(cam, LIVE_MODE))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All camera detection processes completed.")
