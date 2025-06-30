import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import argparse

SEGMENT_PATH = "cameras/segment_mapping.json"
GRAPH_OUTPUT = "cameras/segment_graph.json"
ENTRY_NODE = "ENTRY"
EXIT_NODE = "EXIT"
DISCONNECT_WEIGHT = 9  # Do not draw/add these

def load_segments():
    with open(SEGMENT_PATH, "r") as f:
        return list(json.load(f).keys())

def draw_graph(G, pos=None):
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 6))

    # Filter out edges with weight == DISCONNECT_WEIGHT
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight") != DISCONNECT_WEIGHT]

    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, edge_color="gray", arrows=True)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True) if d["weight"] != DISCONNECT_WEIGHT},
        font_size=10
    )
    plt.title("📍 Segment Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def safe_add_edge(G, u, v, weight):
    if weight != DISCONNECT_WEIGHT:
        G.add_edge(u, v, weight=weight)

def build_graph_with_input(segments):
    G = nx.DiGraph()
    segments = sorted(set(segments))
    G.add_nodes_from(segments)
    G.add_node(ENTRY_NODE)
    G.add_node(EXIT_NODE)

    print("🧠 Segments found:", ", ".join(segments))
    print("👉 Enter distances (use '9' to mark as disconnected)")

    asked = set()
    for i, src in enumerate(segments):
        for j in range(i + 1, len(segments)):
            dst = segments[j]
            if (src, dst) not in asked and (dst, src) not in asked:
                dist = input(f"Distance between '{src}' ⬌ '{dst}': ").strip()
                try:
                    d = float(dist)
                    safe_add_edge(G, src, dst, d)
                    safe_add_edge(G, dst, src, d)
                except ValueError:
                    print("⚠️ Invalid number. Skipping.")
                asked.add((src, dst))

    print("🔁 Add ENTRY → segment distances:")
    for seg in segments:
        dist = input(f"Distance from ENTRY → '{seg}': ").strip()
        try:
            d = float(dist)
            safe_add_edge(G, ENTRY_NODE, seg, d)
        except ValueError:
            print("⚠️ Invalid number. Skipping.")

    print("🔁 Add segment → EXIT distances:")
    for seg in segments:
        dist = input(f"Distance from '{seg}' → EXIT: ").strip()
        try:
            d = float(dist)
            safe_add_edge(G, seg, EXIT_NODE, d)
        except ValueError:
            print("⚠️ Invalid number. Skipping.")

    return G

def save_graph(G):
    data = {u: {v: d["weight"] for v, d in G[u].items()} for u in G.nodes}
    os.makedirs(os.path.dirname(GRAPH_OUTPUT), exist_ok=True)
    with open(GRAPH_OUTPUT, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Graph saved to {GRAPH_OUTPUT}")

def load_existing_graph():
    with open(GRAPH_OUTPUT, "r") as f:
        raw = json.load(f)
    G = nx.DiGraph()
    for src, targets in raw.items():
        for dst, weight in targets.items():
            G.add_edge(src, dst, weight=weight)
    return G

def dag_creator():
    parser = argparse.ArgumentParser(description="Segment Graph Tool")
    parser.add_argument("--draw-only", action="store_true", help="Draw existing graph only")
    parser.add_argument("--interactive", action="store_true", help="Create graph interactively")
    args = parser.parse_args()

    if not os.path.exists(SEGMENT_PATH):
        print(f"❌ {SEGMENT_PATH} not found.")
        return

    if args.draw_only:
        if not os.path.exists(GRAPH_OUTPUT):
            print("❌ No graph saved yet.")
            return
        G = load_existing_graph()
        draw_graph(G)
        return

    segments = load_segments()
    if args.interactive:
        G = build_graph_with_input(segments)
        draw_graph(G)
        save_graph(G)
    else:
        print("👉 Use --interactive to build or --draw-only to view.")

if __name__ == "__main__":
    dag_creator()
