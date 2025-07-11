import matplotlib.pyplot as plt
import json
import math
import networkx as nx

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1), 0, (x1, y1)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y), t, (nearest_x, nearest_y)


def normalize(lat, lon, base_lat, base_lon):
    return (lon - base_lon) * 100000, (lat - base_lat) * 100000

def auto_connect_nearby_nodes(G, node_coords, tolerance=1e4):
    """
    Auto-connect nearby nodes within the given tolerance (in normalized units).
    Adds edges with small weight (default: 0.1).
    """
    node_ids = list(node_coords.keys())
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            u, v = node_ids[i], node_ids[j]
            if G.has_edge(u, v):
                continue  # Skip already connected nodes
            x1, y1 = node_coords[u]
            x2, y2 = node_coords[v]
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < tolerance * 100000:  # Scale tolerance to match your normalization
                G.add_edge(u, v, weight=0.1)  # Small weight for cross-connection

def build_graph(data):
    G = nx.Graph()
    base_lat, base_lon = data['entry']['latitude'], data['entry']['longitude']

    point_id = 0
    node_coords = {}

    for path in data['paths']:
        path_points = path['longitude_latitude']
        for i in range(len(path_points)):
            lat, lon = path_points[i]['latitude'], path_points[i]['longitude']
            coord = normalize(lat, lon, base_lat, base_lon)
            node_coords[point_id] = coord
            point_id += 1

    # Re-add edges after points are collected
    point_id = 0
    for path in data['paths']:
        path_points = path['longitude_latitude']
        for i in range(len(path_points) - 1):
            coord1 = node_coords[point_id]
            coord2 = node_coords[point_id + 1]
            dist = math.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1])
            G.add_edge(point_id, point_id + 1, weight=dist)
            point_id += 1
        point_id += 1  # Move to next path (non-overlapping nodes)
    
    auto_connect_nearby_nodes(G, node_coords, tolerance=0.0002)

    return G, node_coords


def find_entry_node(node_coords, data):
    base_lat, base_lon = data['entry']['latitude'], data['entry']['longitude']
    entry_x, entry_y = normalize(data['entry']['latitude'], data['entry']['longitude'], base_lat, base_lon)

    # Find nearest node to entry point
    nearest_node = None
    min_dist = float('inf')
    for node, (x, y) in node_coords.items():
        dist = math.hypot(x - entry_x, y - entry_y)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    return nearest_node


def find_nearest_slot(data, slot_ids):
    base_lat, base_lon = data['entry']['latitude'], data['entry']['longitude']
    G, node_coords = build_graph(data)
    entry_node = find_entry_node(node_coords, data)

    # Precompute shortest path lengths from Entry
    shortest_paths = nx.single_source_dijkstra_path_length(G, entry_node, weight='weight')

    slot_results = []
    slot_paths = {}

    for slot_id in slot_ids:
        for slot in data['slots']:
            if slot['id'] == slot_id:
                slot_coord = normalize(slot['latitude'], slot['longitude'], base_lat, base_lon)
                break
        else:
            raise ValueError(f"Slot {slot_id} not found.")

        min_total_distance = float('inf')
        best_info = None

        for (u, v, d) in G.edges(data='weight'):
            x1, y1 = node_coords[u]
            x2, y2 = node_coords[v]
            segment_length = d

            perp_distance, t, nearest_pt = point_to_segment_distance(
                slot_coord[0], slot_coord[1], x1, y1, x2, y2
            )

            driving_distance = shortest_paths[u] + t * segment_length
            total_distance = driving_distance + perp_distance

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                best_info = {
                    'total_distance': total_distance,
                    'path_node': u,
                    'segment': (x1, y1, x2, y2),
                    'nearest_point': nearest_pt,
                    'slot_coord': slot_coord,
                    'slot_id': slot_id,
                    'path_to_node': nx.shortest_path(G, entry_node, u, weight='weight')
                }

        slot_results.append((slot_id, min_total_distance))
        slot_paths[slot_id] = best_info

    slot_results.sort(key=lambda x: x[1])
    nearest_slot_id = slot_results[0][0]
    nearest_info = slot_paths[nearest_slot_id]

    # Print Rankings
    print("\n--- Slot Distance Ranking from Entry ---")
    for idx, (slot_id, dist) in enumerate(slot_results, 1):
        print(f"{idx}. {slot_id}: {round(dist, 2)}")

    print(f"\nNearest Slot: {nearest_slot_id} (Distance: {round(slot_results[0][1], 2)})")

    # Plotting Map
    plt.figure(figsize=(12, 10))

    # Plot Entry & Exit
    for point_name in ['entry', 'exit']:
        lat, lon = data[point_name]['latitude'], data[point_name]['longitude']
        x, y = normalize(lat, lon, base_lat, base_lon)
        plt.scatter(x, y, c='green' if point_name == 'entry' else 'red', s=200, label=point_name.capitalize())
        plt.text(x, y, point_name.capitalize(), fontsize=12)

    # Plot Slots
    for slot in data['slots']:
        x, y = normalize(slot['latitude'], slot['longitude'], base_lat, base_lon)
        color = 'cyan' if slot['id'] == nearest_slot_id else 'blue'
        plt.scatter(x, y, c=color, s=100)
        plt.text(x, y, slot['id'], fontsize=10)

    # Plot Paths
    for u, v in G.edges:
        x1, y1 = node_coords[u]
        x2, y2 = node_coords[v]
        plt.plot([x1, x2], [y1, y2], color='purple', linestyle='--')

    # Plot Driving Path to Nearest Slot
    path_nodes = nearest_info['path_to_node']
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        x1, y1 = node_coords[u]
        x2, y2 = node_coords[v]
        plt.plot([x1, x2], [y1, y2], color='gold', linewidth=3)

    # Highlight Segment & Connector to Slot
    x1, y1, x2, y2 = nearest_info['segment']
    nearest_x, nearest_y = nearest_info['nearest_point']
    slot_x, slot_y = nearest_info['slot_coord']
    plt.plot([x1, x2], [y1, y2], color='gold', linewidth=3)
    plt.plot([nearest_x, slot_x], [nearest_y, slot_y], color='black', linestyle=':', linewidth=2)

    plt.title("Parking Lot Map with Driving Route to Nearest Slot")
    plt.xlabel("Relative X (scaled)")
    plt.ylabel("Relative Y (scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return nearest_slot_id, round(slot_results[0][1], 2)


# Load Data & Run
with open('sample_input.json') as f:
    parking_data = json.load(f)
