from collections import defaultdict, deque

import numpy as np


def haplotype_dot_vector(grg, y):
    """
    Returns a vector: one entry per haplotype = dot product of its path with y.
    """
    # Stable node list for indexing
    nodes_list = list(grg.nodes)
    node_to_id = {node: i for i, node in enumerate(nodes_list)}
    n = len(nodes_list)

    # Precompute mutation sums
    node_mut_sum = np.zeros(n, dtype=np.float64)
    for i, node in enumerate(nodes_list):
        muts = [m for m in node.mut if m is not None]
        if muts:
            node_mut_sum[i] = np.sum(y[muts])

    parents = [[] for _ in range(n)]
    for i, node in enumerate(nodes_list):
        for child in node.adj:
            j = node_to_id[child]
            parents[j].append(i)

    # DP cache: total path weight for each node
    cache = {}

    def dfs(u):
        if u in cache:
            return cache[u]
        total = node_mut_sum[u]
        for p in parents[u]:
            total += dfs(p)
        cache[u] = total
        return total

    # Compute vector for each haplotype
    hap_ids = sorted(grg.haplotype_endpoints.keys())
    hap_values = np.zeros(len(hap_ids), dtype=np.float64)
    for i, hap_id in enumerate(hap_ids):
        node = grg.haplotype_endpoints[hap_id]
        hap_values[i] = dfs(node_to_id[node])

    return hap_values


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
