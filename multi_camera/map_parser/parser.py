import matplotlib.pyplot as plt

def plot_parking_lot(data):
    def normalize_coordinates(lat, lon, base_lat, base_lon):
        # Simple scaling for small areas
        x = (lon - base_lon) * 100000
        y = (lat - base_lat) * 100000
        return x, y

    base_lat, base_lon = data['entry']['latitude'], data['entry']['longitude']

    plt.figure(figsize=(12, 10))

    # Plot Entry & Exit
    for point_name in ['entry', 'exit']:
        lat, lon = data[point_name]['latitude'], data[point_name]['longitude']
        x, y = normalize_coordinates(lat, lon, base_lat, base_lon)
        plt.scatter(x, y, c='green' if point_name == 'entry' else 'red', s=200, label=point_name.capitalize())
        plt.text(x, y, point_name.capitalize(), fontsize=12)

    # Plot Slots
    for slot in data['slots']:
        x, y = normalize_coordinates(slot['latitude'], slot['longitude'], base_lat, base_lon)
        plt.scatter(x, y, c='blue', s=100)
        plt.text(x, y, slot['id'], fontsize=10)

    # Plot Cameras
    for cam in data['cameras']:
        x, y = normalize_coordinates(cam['latitude'], cam['longitude'], base_lat, base_lon)
        plt.scatter(x, y, c='orange', s=150)
        plt.text(x, y, cam['id'], fontsize=10)

    # Plot Paths (avoid duplicate labels)
    for idx, path in enumerate(data['paths']):
        path_points = path['longitude_latitude']
        xs, ys = [], []
        for pt in path_points:
            x, y = normalize_coordinates(pt['latitude'], pt['longitude'], base_lat, base_lon)
            xs.append(x)
            ys.append(y)
        label = f"Path {path['path_name']}" if idx == 0 else None  # Label only once
        plt.plot(xs, ys, linestyle='--', color='purple', label=label)

    plt.title("Local Parking Lot Map Visualization")
    plt.xlabel("Relative X (scaled)")
    plt.ylabel("Relative Y (scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()
import json
import math

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Compute minimum distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)
    
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y), t, (nearest_x, nearest_y)

def compute_total_distance(data, slot_id):
    """Compute total distance from entry to parking slot (path + perpendicular)."""
    def normalize(lat, lon):
        base_lat, base_lon = data['entry']['latitude'], data['entry']['longitude']
        return (lon - base_lon) * 100000, (lat - base_lat) * 100000

    # Get slot coordinates
    for slot in data['slots']:
        if slot['id'] == slot_id:
            slot_coord = normalize(slot['latitude'], slot['longitude'])
            break
    else:
        raise ValueError(f"Slot ID '{slot_id}' not found")

    total_path_distance = 0
    min_total_distance = float('inf')

    for path in data['paths']:
        path_points = path['longitude_latitude']
        path_distance = 0
        for i in range(len(path_points) - 1):
            p1 = normalize(path_points[i]['latitude'], path_points[i]['longitude'])
            p2 = normalize(path_points[i + 1]['latitude'], path_points[i + 1]['longitude'])
            segment_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

            # Check perpendicular distance from slot to current segment
            perp_distance, t, (nearest_x, nearest_y) = point_to_segment_distance(
                slot_coord[0], slot_coord[1], p1[0], p1[1], p2[0], p2[1]
            )

            # Total distance traveled along the path + perpendicular distance
            traveled_distance = total_path_distance + t * segment_length
            total_distance = traveled_distance + perp_distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance

            path_distance += segment_length
            total_path_distance += segment_length  # Continue accumulating for next segments

    return round(min_total_distance, 2)



def compare_slots_by_total_distance(data, slot_id_1, slot_id_2):
    """Compare total distance from entry between two slots and print which one is nearer."""
    distance1 = compute_total_distance(data, slot_id_1)
    distance2 = compute_total_distance(data, slot_id_2)

    print(f"Total distance from Entry to {slot_id_1}: {distance1}")
    print(f"Total distance from Entry to {slot_id_2}: {distance2}")

    if distance1 < distance2:
        print(f"{slot_id_1} is nearer to Entry than {slot_id_2}")
    elif distance2 < distance1:
        print(f"{slot_id_2} is nearer to Entry than {slot_id_1}")
    else:
        print(f"Both {slot_id_1} and {slot_id_2} are equally distant from Entry.")
import json

# Load your parking data
with open('input.json') as f:
    parking_data = json.load(f)

# Compare two slots
compare_slots_by_total_distance(parking_data, 'slot_3', 'slot_5')
plot_parking_lot(parking_data)