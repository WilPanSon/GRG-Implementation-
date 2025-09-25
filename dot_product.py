import numpy as np
from collections import defaultdict

def xtv_mutation_vector(grg):
    """
    Compute X^T v for the GRG without making X^T.

    For each mutation m, returns sum of v[node.n] over all sample nodes (haplotypes)
    that carry mutation m. The output is a vector where index == mutation id.
    """

    # Collect all mutations and determine the vector length
    all_mutations = set()
    max_mutation_id = -1
    for node in grg.nodes:
        for m in node.mut:
            if m is None:
                continue
            all_mutations.add(m)
            if m > max_mutation_id:
                max_mutation_id = m

    if max_mutation_id < 0:
        # No mutations present
        return np.zeros(0, dtype=int)

    vec = np.zeros(max_mutation_id + 1, dtype=int)

    # Identify sample nodes as leaf nodes (no children)
    sample_nodes = {node for node in grg.nodes if len(node.adj) == 0}
    if not sample_nodes:
        return vec  # No samples to contribute

    # For each node that carries mutations, traverse forward to all reachable sample nodes.
    # We will add the reachable sample node's n to each mutation carried by the source node.
    # To avoid repeated traversal work, we can cache reachability from each starting node.
    reachability_cache = {}

    def reachable_samples_from(start_node):
        """Return the set of sample nodes reachable from start_node (including start_node if leaf)."""
        if start_node in reachability_cache:
            return reachability_cache[start_node]
        visited = set()
        stack = [start_node]
        reachable_samples = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if len(current.adj) == 0:
                # Leaf (sample) node
                reachable_samples.add(current)
            else:
                for child in current.adj:
                    if child not in visited:
                        stack.append(child)
        reachability_cache[start_node] = reachable_samples
        return reachable_samples

    # For each mutation-bearing node, add contributions from reachable samples
    for node in grg.nodes:
        # Skip nodes without valid mutations
        node_mutations = [m for m in node.mut if m is not None]
        if not node_mutations:
            continue

        reachable_samples = reachable_samples_from(node)
        if not reachable_samples:
            continue

        # Sum contributions per mutation: add each reachable sample's n to the index of the mutation
        for sample in reachable_samples:
            n_val = sample.n
            for m in node_mutations:
                # m is guaranteed to be >= 0 integer by construction
                vec[m] += n_val

    return vec


