import cv2
import json
import numpy as np
import os
import argparse

# -------------------- Parse Arguments --------------------
parser = argparse.ArgumentParser(description="Mark Parking Slots for a Camera")
parser.add_argument("--camera", type=str, default="camera1", help="Camera name (e.g., camera1)")
parser.add_argument("--format", type=str, default="jpg", help="Static picture Format")
args = parser.parse_args()
camera_name = args.camera
static_picture_format=args.format
# -------------------- Directory Setup --------------------
base_dir = os.path.join("cameras", camera_name)
os.makedirs(base_dir, exist_ok=True)

region_file_path = os.path.join(base_dir, "slot_regions.json")
image_path = f"static_marking_images/{camera_name}.{static_picture_format}"

# -------------------- Image Loading --------------------
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ {image_path} not found. Please add a static.{static_picture_format} for {camera_name}.")

image = cv2.imread(image_path)
image_copy = image.copy()

# -------------------- State Variables --------------------
slots = []
entries = []
current_slot = []
slot_count = 0
mode = "slot"  # Default mode

# -------------------- Load Existing Data --------------------
if os.path.exists(region_file_path):
    with open(region_file_path, "r") as f:
        data = json.load(f)
        slots = data.get("slots", [])
        entries = data.get("entries", [])
        slot_count = len(slots)
        print(f"📂 Loaded {slot_count} slots and {len(entries)} entries for {camera_name}.")

# -------------------- Redraw --------------------
def redraw_image():
    global image_copy
    image_copy = image.copy()

    for slot in slots:
        pts = np.array(slot["coords"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_copy, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = slot["coords"][0]
        cv2.putText(image_copy, slot["name"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (x, y) in enumerate(entries):
        cv2.circle(image_copy, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(image_copy, f"Entry {i+1}", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for pt in current_slot:
        cv2.circle(image_copy, tuple(pt), 4, (0, 255, 255), -1)

    cv2.imshow("Mark Parking Slots & Entries", image_copy)

# -------------------- Mouse Callback --------------------
def draw(event, x, y, flags, param):
    global current_slot, slots, entries, slot_count, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == "slot":
            current_slot.append([x, y])
            if len(current_slot) == 4:
                slot_count += 1
                slots.append({
                    "name": f"Slot {slot_count}",
                    "coords": current_slot.copy()
                })
                current_slot = []
                redraw_image()
        elif mode == "entry":
            entries.append([x, y])
            print(f"📍 Entry marked at ({x}, {y})")
            redraw_image()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode == "slot" and slots:
            slots.pop()
            slot_count -= 1
            redraw_image()
        elif mode == "entry" and entries:
            entries.pop()
            redraw_image()

# -------------------- Setup --------------------
cv2.namedWindow("Mark Parking Slots & Entries")
cv2.setMouseCallback("Mark Parking Slots & Entries", draw)

print(f"🎥 Editing slot regions for: {camera_name}")
print("🖱️ Left-click to mark.")
print("🧼 Right-click to undo.")
print("⌨️ Press 's' for Slot Mode, 'e' for Entry Mode, 'q' to quit and save.")

# -------------------- Main Loop --------------------
while True:
    redraw_image()
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        mode = "slot"
        print("✏️ Switched to SLOT mode.")
    elif key == ord("e"):
        mode = "entry"
        print("🚪 Switched to ENTRY mode.")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

# -------------------- Save --------------------
data_to_save = {
    "slots": slots,
    "entries": entries
}

with open(region_file_path, "w") as f:
    json.dump(data_to_save, f, indent=4)

print(f"✅ Saved slot & entry data to '{region_file_path}'")
