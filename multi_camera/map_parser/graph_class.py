import json
import matplotlib.pyplot as plt
import networkx as nx
import math
import heapq
import os

DEBUG = True


class ParkingGraph:
    def __init__(self, data=None):
        self.data = data
        self.graph = {}
        self.entry = None
        self.exit = None
        self.build_graph()
    
    def node_key(self, lat, lon):
        """Generate a unique key for each node (coordinate pair)."""
        return f"{lat:.6f}_{lon:.6f}"

    def add_node(self, lat, lon):
        key = self.node_key(lat, lon)
        if key not in self.graph:
            self.graph[key] = {
                "latitude": lat,
                "longitude": lon,
                "neighbors": set()
            }
        return key
    
    def build_graph(self):
        graph_cache_file = "graph_cache.json"
        
        if self.data is None:
            # No data provided → Load from cache
            if os.path.exists(graph_cache_file):
                with open(graph_cache_file, 'r') as f:
                    cache = json.load(f)
                    self.graph = cache["graph"]
                    self.entry = cache["entry"]
                    self.exit = cache["exit"]
                
                # Restore neighbors as sets
                for node in self.graph.values():
                    node["neighbors"] = set(node["neighbors"])
                print("[INFO] Graph loaded from cache.")
            else:
                raise ValueError("No input data provided and graph_cache.json not found.")
        else:
            # Build from provided data
            print("[INFO] Building graph from provided data...")
            entry = self.data["entry"]
            exit_ = self.data["exit"]
            self.entry = entry
            self.exit = exit_
            entry_key = self.add_node(entry["latitude"], entry["longitude"])
            exit_key = self.add_node(exit_["latitude"], exit_["longitude"])

            for path in self.data["paths"]:
                points = path["longitude_latitude"]
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i + 1]
                    key1 = self.add_node(p1["latitude"], p1["longitude"])
                    key2 = self.add_node(p2["latitude"], p2["longitude"])
                    self.graph[key1]["neighbors"].add(key2)
                    self.graph[key2]["neighbors"].add(key1)

            # Ensure entry/exit connected
            for key in list(self.graph.keys()):
                node = self.graph[key]
                if node["latitude"] == entry["latitude"] and node["longitude"] == entry["longitude"]:
                    self.graph[entry_key]["neighbors"].update(node["neighbors"])
                if node["latitude"] == exit_["latitude"] and node["longitude"] == exit_["longitude"]:
                    self.graph[exit_key]["neighbors"].update(node["neighbors"])

            # Convert neighbors to list for JSON serialization
            for node in self.graph.values():
                node["neighbors"] = list(node["neighbors"])

            graph_cache = {
                "graph": self.graph,
                "entry": self.entry,
                "exit": self.exit
            }

            # Save graph to cache
            with open(graph_cache_file, 'w') as f:
                json.dump(graph_cache, f, indent=2)
            print(f"[INFO] Graph saved to {graph_cache_file}.")

    def to_json(self):
        """Return the graph in JSON format."""
        return json.dumps(self.graph, indent=2)
    
    def get_graph(self):
        """Return graph as Python dict."""
        return self.graph

    def bfs(self, start_lat, start_lon):
        """Simple BFS traversal from a given coordinate."""
        start_key = self.node_key(start_lat, start_lon)
        visited = set()
        queue = [start_key]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.graph[node]["neighbors"])
        return visited
    
    def visualize_graph(self, highlight_lat=None, highlight_lon=None):
        """Visualize the graph using networkx and matplotlib."""
        G = nx.Graph()
        pos = {}

        for node_key, node_data in self.graph.items():
            lon = node_data["longitude"]
            lat = node_data["latitude"]
            pos[node_key] = (lon, lat)
            for neighbor_key in node_data["neighbors"]:
                G.add_edge(node_key, neighbor_key)

        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_weight='bold')
        plt.title("Parking Lot Path Graph (Longitude vs Latitude)", fontsize=16)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)

        if highlight_lat is not None and highlight_lon is not None:
            print(f"--Marking the lat and long as {highlight_lat} || {highlight_lon}--")
            plt.scatter(highlight_lon, highlight_lat, color='red', s=150, marker='X', label="Target Point",zorder=5)
            plt.legend()

        plt.show()

    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        """Calculate straight-line distance between two points."""
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    def find_closest_node(self, target_lat, target_lon):
        """Find the closest graph node to a target point."""
        min_dist = float('inf')
        closest_node = None
        for key, node in self.graph.items():
            dist = self.euclidean_distance(target_lat, target_lon, node['latitude'], node['longitude'])
            if dist < min_dist:
                min_dist = dist
                closest_node = key
        return closest_node, min_dist

    def dijkstra_shortest_path(self, start_node, target_node):
        """Dijkstra's algorithm to find shortest path between start_node and target_node."""
        distances = {node: float('inf') for node in self.graph}
        prev = {node: None for node in self.graph}
        distances[start_node] = 0

        heap = [(0, start_node)]
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            if current_node == target_node:
                break
            for neighbor in self.graph[current_node]['neighbors']:
                alt = current_dist + self.euclidean_distance(
                    self.graph[current_node]['latitude'], self.graph[current_node]['longitude'],
                    self.graph[neighbor]['latitude'], self.graph[neighbor]['longitude']
                )
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    prev[neighbor] = current_node
                    heapq.heappush(heap, (alt, neighbor))

        # Reconstruct path
        path = []
        node = target_node
        while node:
            path.insert(0, node)
            node = prev[node]
        return path, distances[target_node]

    def find_closest_path(self, target_lat, target_lon):
        """Find closest graph node to target point and shortest path from entry."""
        closest_node, distance = self.find_closest_node(target_lat, target_lon)
        print(f"Closest Node to ({target_lat}, {target_lon}): {closest_node} [Distance: {distance}]")

        entry_lat = self.entry["latitude"]
        entry_lon = self.entry["longitude"]
        entry_node = self.node_key(entry_lat, entry_lon)

        path, total_distance = self.dijkstra_shortest_path(entry_node, closest_node)

        directions=self.get_directions_from_path(path)
        
        if DEBUG:
            self.visualize_graph(target_lat, target_lon)
        
        return {
            "closest_node": closest_node,
            "distance_to_closest_node": distance,
            "shortest_path_from_entry": path,
            "path_total_distance": total_distance,
            "directions": directions
        }

    def get_directions_from_path(self, path, angle_threshold=0.3):
        directions = []
        i = 1  # Start after entry node
        prev_node = self.graph[path[0]]

        while i < len(path) - 1:
            curr_node = self.graph[path[i]]
            next_node = self.graph[path[i + 1]]

            neighbors = self.graph[path[i]]['neighbors']

            if len(neighbors) >= 3 or i == 1:  # Always check at start too
                # Calculate direction
                v1_x = curr_node['longitude'] - prev_node['longitude']
                v1_y = curr_node['latitude'] - prev_node['latitude']
                v2_x = next_node['longitude'] - curr_node['longitude']
                v2_y = next_node['latitude'] - curr_node['latitude']  # ✅ Fixed here

                angle1 = math.atan2(v1_y, v1_x)
                angle2 = math.atan2(v2_y, v2_x)
                delta_angle = (angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi  # Normalize [-π, π]

                if abs(delta_angle) < angle_threshold:
                    directions.append(f"At junction {path[i]}: Go Straight")
                elif delta_angle > 0:
                    directions.append(f"At junction {path[i]}: Turn Left")
                else:
                    directions.append(f"At junction {path[i]}: Turn Right")

            prev_node = curr_node
            i += 1  

        return directions
            




class ParkingLotMap:
    def __init__(self, data=None,create_new_graph=True):
        """
        Initialize ParkingLotMap with the parking layout data.
        """
        self.entry = data["entry"]
        self.exit = data["exit"]
        self.slots = data["slots"]
        self.cameras = data.get("cameras", [])
        self.paths = data["paths"]
        if(create_new_graph):
            self.graph = ParkingGraph(data)  # This will be set later after graph construction.
        else:
            self.graph = ParkingGraph()
        if(self.graph):
            print("[INFO] ParkingGraph linked to ParkingLotMap.")
        self.camera_slot_distance=self.compute_camera_slot_distances()

    def summarize(self):
        """
        Print a summary of the parking lot map.
        """
        print(f"Entry Point: {self.entry}")
        print(f"Exit Point: {self.exit}")
        print(f"Total Parking Slots: {len(self.slots)}")
        print(f"Total Cameras: {len(self.cameras)}")
        print(f"Total Driving Paths: {len(self.paths)}")

    def get_slot_coordinates(self):
        """
        Return list of parking slot coordinates.
        """
        return [(slot["id"], slot["latitude"], slot["longitude"]) for slot in self.slots]

    def set_graph(self, graph_instance):
        """
        Link a ParkingGraph object to this map.
        """
        self.graph = graph_instance

    def save_to_file(self, filename):
        """
        Save the current map data to a JSON file.
        """
        with open(filename, 'w') as f:
            json.dump({
                "entry": self.entry,
                "exit": self.exit,
                "slots": self.slots,
                "cameras": self.cameras,
                "paths": self.paths
            }, f, indent=2)
        print(f"[INFO] ParkingLotMap saved to {filename}.")

    def get_data(self):
        """
        Return raw map data.
        """
        return {
            "entry": self.entry,
            "exit": self.exit,
            "slots": self.slots,
            "cameras": self.cameras,
            "paths": self.paths
        }
    def compute_camera_slot_distances(self, output_file='camera_slot_distances.json'):
        """Precompute normalized distances between cameras and slots, store to JSON."""
        camera_distances = {}
        for camera in self.cameras:
            cam_lat = camera['latitude']
            cam_lon = camera['longitude']
            distances = {}
            for slot in self.slots:
                slot_id = slot['id']
                dist = self.graph.euclidean_distance(cam_lat, cam_lon, slot['latitude'], slot['longitude'])
                distances[slot_id] = dist
            
            # Normalize distances
            min_dist = min(distances.values())
            max_dist = max(distances.values())
            normalized = {}
            for slot_id, dist in distances.items():
                if max_dist > min_dist:
                    normalized[slot_id] = (dist - min_dist) / (max_dist - min_dist)
                else:
                    normalized[slot_id] = 0.0  # Avoid division by zero
            camera_distances[camera['id']] = normalized
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(camera_distances, f, indent=2)
        
        print(f"[INFO] Camera-slot normalized distance matrix created and saved at '{output_file}'.")

        return camera_distances
 
if __name__ == "__main__":
    with open('sample_input.json') as f:
        data = json.load(f)

    lot_map = ParkingLotMap(data)

    print(lot_map.camera_slot_distance)
    
