from collections import defaultdict, deque

import numpy as np


def dot(grg, y):
    nodes_list = list(grg.nodes)
    node_to_id = {node: i for i, node in enumerate(nodes_list)}
    n = len(nodes_list)

    node_mut_sum = np.zeros(n, dtype=np.float64)
    for i, node in enumerate(nodes_list):
        muts = [m for m in node.mut if m is not None]
        if muts:
            node_mut_sum[i] = np.sum(y[muts])

    # Build adjacency and indegree
    children = [[] for _ in range(n)]
    indegree = np.zeros(n, dtype=np.int64)
    for i, node in enumerate(nodes_list):
        for child in node.adj:
            j = node_to_id[child]
            children[i].append(j)
            indegree[j] += 1

    q = deque([i for i in range(n) if indegree[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in children[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    endpoint_ids = {node_to_id[node] for node in grg.haplotype_endpoints.values()}
    endpoint_count = np.zeros(n, dtype=np.int64)

    # Process nodes in reverse topo order
    for u in reversed(topo):
        # Count 1 if this node is a haplotype endpoint
        count = 1 if u in endpoint_ids else 0
        # Add counts from children
        for v in children[u]:
            count += endpoint_count[v]
        endpoint_count[u] = count

    total = np.sum(endpoint_count * node_mut_sum)
    return total


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
