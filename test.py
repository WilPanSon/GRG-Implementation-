import json

import dot_product
import grg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_grg(grg, figsize=(12, 8)):
    # Build graph
    G = nx.DiGraph()
    for node in grg.nodes:
        muts = [f"m{m}" for m in node.mut if m is not None]
        label = f"{node.n}\n" + ",".join(muts) if muts else f"{node.n}"
        G.add_node(node, label=label)

    for node in grg.nodes:
        for child in node.adj:
            G.add_edge(node, child)

    endpoint_nodes = set(grg.haplotype_endpoints.values())

    # Topological order
    topo = list(nx.topological_sort(G))

    # Assign levels (distance from endpoints upward)
    level = {n: None for n in G.nodes}
    for n in reversed(topo):  # start from endpoints
        if n in endpoint_nodes:
            level[n] = 0
        else:
            child_levels = [level[c] for c in G.successors(n) if level[c] is not None]
            level[n] = (max(child_levels) + 1) if child_levels else 1

    # Group by levels
    layers = {}
    for n, l in level.items():
        layers.setdefault(l, []).append(n)

    # Assign positions (spread nodes in each layer horizontally)
    pos = {}
    max_level = max(layers.keys())
    for l, nodes in layers.items():
        spacing = 3  # horizontal spacing
        x_offset = -((len(nodes) - 1) * spacing) / 2
        for i, n in enumerate(nodes):
            pos[n] = (x_offset + i * spacing, max_level - l)

    # Style nodes
    node_colors = []
    node_edges = []
    for node in G.nodes:
        if node in endpoint_nodes:
            node_colors.append("lightcoral")
            node_edges.append("red")
        else:
            node_colors.append("skyblue")
            node_edges.append("black")

    plt.figure(figsize=figsize)

    nx.draw(
        G, pos,
        with_labels=False,
        node_size=2500,
        node_color=node_colors,
        edgecolors=node_edges,
        edge_color="gray",
        arrowsize=15,
        width=1.5,
        alpha=0.9,
    )

    # Labels
    labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")

    plt.title("Genome Representation Graph (GRG)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()


# mutations which arise in a node are stored in a tuple
n0 = grg.Node(0, (3, 6, 7))
n1 = grg.Node(1, (4))
n2 = grg.Node(2, (11))
n3 = grg.Node(3, (1))
n4 = grg.Node(4, (2, 9))
n5 = grg.Node(5, (0))
n6 = grg.Node(6, (8))
n7 = grg.Node(7, (5, 10))

# edges are tuples of parent and child nodes
n6.connect(n0)
n6.connect(n5)
n7.connect(n0)
n7.connect(n4)
n5.connect(n1)
n5.connect(n4)
n4.connect(n2)
n4.connect(n3)

all_nodes = {n0, n1, n2, n3, n4, n5, n6, n7}

endpoints_map = {
    "hapA": n0,  # Path: 6 -> 0 or 7 -> 0
    "hapB": n1,  # Path: 6 -> 5 -> 1
    "hapC": n2,  # Path: 6 -> 5 -> 4 -> 2
    "hapD": n3,  # Path: 7 -> 4 -> 3
}
individual_map = {
    "hapA": "Individual_1",
    "hapB": "Individual_1",
    "hapC": "Individual_2",
    "hapD": "Individual_2",
}

grg_ = grg.GRG(all_nodes, endpoints_map, individual_map)

mutation_map = grg_.get_mutation_to_individuals()

print(json.dumps(mutation_map, indent=2))

individual_to_mutation_map = grg_.get_individuals_to_mutations()
print(json.dumps(individual_to_mutation_map, indent=2))

matrix = grg_.to_matrix()
print(matrix)

xtv = dot_product.xtv_mutation_vector(grg_)
print(xtv)

print(dot_product.haplotype_dot_vector(grg_, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))

plot_grg(grg_)