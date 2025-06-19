import os
import json

def create_camera_structure(camera_number):
    camera_name = f"camera{camera_number}"
    base_path = os.path.join(os.getcwd(), camera_name)
    log_path = os.path.join(base_path, "logs")

    # Create directories
    os.makedirs(log_path, exist_ok=True)

    # File paths with default content
    files = {
        os.path.join(base_path, "config.json"): {},
        os.path.join(base_path, "slot_regions.json"): {"slots": [], "entries": []},
        os.path.join(log_path, "timelogs.json"): [],
        os.path.join(log_path, "nearestlogs.json"): [],
        os.path.join(log_path, "slot_states.json"): {}
    }

    # Create each file
    for path, content in files.items():
        if not os.path.exists(path):
            with open(path, "w") as f:
                json.dump(content, f, indent=4)
            print(f"✅ Created: {path}")
        else:
            print(f"ℹ️ Already exists: {path}")

if __name__ == "__main__":
    input_str = input("Enter camera numbers separated by space (e.g. 1 2 3): ").strip()
    camera_numbers = input_str.split()

    for cam_no in camera_numbers:
        print(f"\n📦 Setting up structure for camera{cam_no}")
        create_camera_structure(cam_no)
