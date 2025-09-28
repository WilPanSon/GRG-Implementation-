import json
import os

import dot_product
import grg
import numpy as np


# Load toy data from JSON file
def load_toy_data(json_file_path):
    """Load tree sequence data from JSON file."""
    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


# Load the toy data
toy_data = load_toy_data("toy_data.json")

# Use the data directly from JSON
nodes = toy_data["nodes"]
edges = toy_data["edges"]
sites = toy_data["sites"]
mutations = toy_data["mutations"]

# Create a mapping from node ID to mutations
node_mutations = {}
for mut in mutations:
    node_id = mut["node"]
    if node_id not in node_mutations:
        node_mutations[node_id] = []
    node_mutations[node_id].append(mut["site"])

# Create GRG Node objects
# Sample nodes (0-7) are leaf nodes, internal nodes (8-14) are ancestral nodes
grg_nodes = []
for i, node_data in enumerate(nodes):
    # Get mutations for this node, or empty tuple if none
    muts = tuple(node_mutations.get(i, []))
    grg_node = grg.Node(i, muts)
    grg_nodes.append(grg_node)

# Create edges (parent-child relationships)
# Note: In GRG, we connect parent to child (parent.connect(child))
for edge in edges:
    parent_id = edge["parent"]
    child_id = edge["child"]
    grg_nodes[parent_id].connect(grg_nodes[child_id])

# Create all_nodes set
all_nodes = set(grg_nodes)

# Create haplotype endpoints (sample nodes 0-7)
endpoints_map = {}
for i in range(8):  # Sample nodes 0-7
    endpoints_map[f"hap{i}"] = grg_nodes[i]

# Create individual mapping (group haplotypes into individuals)
# Let's create 4 individuals with 2 haplotypes each
individual_map = {}
individuals = ["Individual_1", "Individual_2", "Individual_3", "Individual_4"]
for i in range(8):
    hap_name = f"hap{i}"
    individual_map[hap_name] = individuals[
        i // 2
    ]  # Pair haplotypes 0,1 -> Individual_1, etc.

# Create the GRG
grg_ = grg.GRG(all_nodes, endpoints_map, individual_map)

print("=== Mutation to Individuals Map ===")
mutation_map = grg_.get_mutation_to_individuals()
print(json.dumps(mutation_map, indent=2))

print("\n=== Individual to Mutations Map ===")
individual_to_mutation_map = grg_.get_individuals_to_mutations()
print(json.dumps(individual_to_mutation_map, indent=2))

print("\n=== Genotype Matrix ===")
matrix = grg_.to_matrix()
print(matrix)

print("\n=== X^T v Mutation Vector ===")
xtv = dot_product.xtv_mutation_vector(grg_)
print(xtv)

print("\n=== Dot Product Test ===")
# Create a test vector for dot product computation
# The vector should have entries for all possible mutations (0-3 based on sites)
test_vector = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dot_result = dot_product.dot(grg_, test_vector)
print(f"Dot product result: {dot_result}")

print("\n=== Summary ===")
print(f"Number of nodes: {len(all_nodes)}")
print(f"Number of mutations: {len(grg_.mutations)}")
print(f"Number of haplotypes: {len(endpoints_map)}")
print(f"Number of individuals: {len(set(individual_map.values()))}")
print(f"Matrix shape: {matrix.shape}")
print(f"X^T v vector length: {len(xtv)}")
