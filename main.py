from collections import defaultdict
import numpy as np
import json

class Node:
    def __init__(self, n: int, mut):
        self.n = n
        self.adj = set()
        self.mut = (
            mut if isinstance(mut, tuple) else (mut,)
        )  # checks if mut is a tuple. If it is, it unpacks it. If it is not, it wraps it in a tuple.

    def connect(self, node: "Node"):  # adjacency list contains child nodes
        self.adj.add(node)

    def __repr__(self):
        return f'Node("{self.n}, {self.mut}")'
    def verify(self):
        """Checks that the Node's attributes are the correct types."""
        if not isinstance(self.n, int):
            raise TypeError("Node ID 'n' must be an integer.")
        if not isinstance(self.mut, tuple):
            raise TypeError("Node 'mut' attribute must be a tuple.")
        if not isinstance(self.adj, set):
            raise TypeError("Node 'adj' attribute must be a set.")
        return True

class GRG:
    def __init__(
        self,
        nodes: set[Node],
        samples: list[str] = None,):
        
        self.nodes = nodes
        self.mutations = [
            m for node in nodes for m in node.mut if m is not None
        ]  # iterate through nodes and mutations and add them to the mutations list, including m0
        self.samples = samples if samples is not None else []
        self.verify()
        
    def verify(self):
        # 1. Check top-level attributes
        if not isinstance(self.nodes, set):
            raise TypeError("GRG 'nodes' attribute must be a set.")
        
        # 2. Verify each node and its connections
        for node in self.nodes:
            node.verify() # Verify the node's internal state
            for child in node.adj:
                if not isinstance(child, Node):
                    raise TypeError(f"Node {node} has an invalid child of type {type(child)}.")
                if child not in self.nodes:
                    raise ValueError(f"Node {node} connects to a child node that is not in the main GRG node set.")
        
        # 3. Check for cycles to ensure it's a DAG
        if self._has_cycles():
            raise ValueError("The graph contains a cycle and is not a valid DAG.")
            
        return True
        
    def _has_cycles(self):
        """DFS-based utility to detect cycles in the graph."""
        visiting = set()  # Nodes currently in the recursion stack for the current DFS path
        visited = set()   # All nodes that have been visited at least once

        for node in self.nodes:
            if node not in visited:
                if self._cycle_util(node, visiting, visited):
                    return True
        return False

    def _cycle_util(self, node, visiting, visited):
        visiting.add(node)
        visited.add(node)

        for child in node.adj:
            if child in visiting:
                return True  # Cycle detected
            if child not in visited:
                if self._cycle_util(child, visiting, visited):
                    return True
        
        visiting.remove(node)
        return False
        
    def _get_all_parents(self):
        # Build a map of all parent-child relationships (allowing multiple parents)
        all_parents = {}

        for node in self.nodes:
            for child in node.adj:
                if child not in all_parents:
                    all_parents[child] = []
                all_parents[child].append(node)

        # Add root nodes (nodes with no parents)
        all_children = set(all_parents.keys())
        roots = [node for node in self.nodes if node not in all_children]

        return all_parents, roots

    def get_individuals_to_mutations(
        self, haplotype_endpoints, haplotype_to_individual_map
    ):
        """
        Returns a dictionary mapping each individual to all mutations
        present on either of their haplotypes.
        """
        all_parents, roots = self._get_all_parents()

        haplotype_mutations = {}
        for hap_id, terminal_node in haplotype_endpoints.items():
            muts = set()
            visited = set()
            stack = [terminal_node]

            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)

                muts.update(current.mut)

                # Add all parents to the stack
                if current in all_parents:
                    for parent in all_parents[current]:
                        if parent not in visited:
                            stack.append(parent)

            haplotype_mutations[hap_id] = muts

        individual_to_mut_sets = defaultdict(set)
        for hap_id, muts in haplotype_mutations.items():
            ind_id = haplotype_to_individual_map.get(hap_id)
            if ind_id:
                individual_to_mut_sets[ind_id].update(muts)

        return {ind: sorted(muts) for ind, muts in individual_to_mut_sets.items()}

    def get_mutation_to_individuals(
        self, haplotype_endpoints, haplotype_to_individual_map):
        # Step 1: Call the primary method to get the data
        individual_to_mutations = self.get_individuals_to_mutations(
            haplotype_endpoints,
            haplotype_to_individual_map
        )

        mutation_to_ind_sets = defaultdict(set)
        for individual, mutations in individual_to_mutations.items():
            for mut in mutations:
                mutation_to_ind_sets[mut].add(individual)

        return {
            mut: sorted(list(inds))
            for mut, inds in mutation_to_ind_sets.items()
        }
    
    def verify_matrix(self, matrix, haplotype_endpoints, mutations_map):
        dimensions = matrix.shape
        if dimensions[0] != len(haplotype_endpoints):
            raise ValueError("Matrix doesn't have as many rows as haplotypes.")
        if dimensions[1] != len(mutations_map):
            raise ValueError("Matrix doesn't have as many columns as mutations.")
        if not np.all((matrix == 0) | (matrix == 1)):
            raise ValueError("Matrix breaches 0/1 rule.")
        return
    
    # Akshay
    def to_matrix(self, haplotype_endpoints, haplotype_to_individual_map):
        """
        Returns a matrix where each row is a haplotype and each column is a mutation.
        Entry is 1 if the haplotype has the mutation, 0 otherwise.
        """
        # Get all mutations in sorted order
        all_mutations = sorted(set(m for node in self.nodes for m in node.mut if m is not None))
        mutation_to_col = {mut: idx for idx, mut in enumerate(all_mutations)}

        # Get haplotype IDs in sorted order
        haplotype_ids = sorted(haplotype_endpoints.keys())
        num_haplotypes = len(haplotype_ids)
        num_mutations = len(all_mutations)

        # For each haplotype, collect all mutations on its path
        matrix = np.zeros((num_haplotypes, num_mutations), dtype=int)
        all_parents, _ = self._get_all_parents()

        for row_idx, hap_id in enumerate(haplotype_ids):
            terminal_node = haplotype_endpoints[hap_id]
            muts = set()
            visited = set()
            stack = [terminal_node]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                muts.update(current.mut)
                if current in all_parents:
                    stack.extend(all_parents[current])
            # Set matrix entries for this haplotype
            for mut in muts:
                col_idx = mutation_to_col[mut]
                matrix[row_idx, col_idx] = 1
        
        self.verify_matrix(matrix, haplotype_endpoints, all_mutations)
        return matrix
