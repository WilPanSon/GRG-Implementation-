import json

import dot_product
import grg
import numpy as np

# mutations which arise in a node are stored in a tuple
# Creating nodes based on the haplotype mappings
n0 = grg.Node(0, (2, 8))  # H0: m2, m8
n1 = grg.Node(1, ())  # H1: m2, m8 (inherited from parent)
n2 = grg.Node(2, (1, 2))  # H2: m1, m2
n3 = grg.Node(3, ())  # H3: m1, m2 (inherited from parent)
n4 = grg.Node(4, ())  # H4: m1, m2 (inherited from parent)
n5 = grg.Node(5, ())  # H5: m1, m2 (inherited from parent)
n6 = grg.Node(6, ())  # H6: m1, m2 (inherited from parent)
n7 = grg.Node(7, (8))  # H7: m1, m2, m8 (inherits m1,m2, gets m8)

# edges are tuples of parent and child nodes
# Setting up the tree structure where mutations are inherited
n0.connect(n1)  # H0 -> H1 (H1 inherits m2, m8)
n2.connect(n3)  # H2 -> H3 (H3 inherits m1, m2)
n2.connect(n4)  # H2 -> H4 (H4 inherits m1, m2)
n2.connect(n5)  # H2 -> H5 (H5 inherits m1, m2)
n2.connect(n6)  # H2 -> H6 (H6 inherits m1, m2)
n2.connect(n7)  # H2 -> H7 (H7 inherits m1, m2, then gets m8)

all_nodes = {n0, n1, n2, n3, n4, n5, n6, n7}

endpoints_map = {
    "hapA": n0,  # H0: m2, m8
    "hapB": n1,  # H1: m2, m8
    "hapC": n2,  # H2: m1, m2
    "hapD": n3,  # H3: m1, m2
    "hapE": n4,  # H4: m1, m2
    "hapF": n5,  # H5: m1, m2
    "hapG": n6,  # H6: m1, m2
    "hapH": n7,  # H7: m1, m2, m8
}

individual_map = {
    "hapA": "Individual_1",
    "hapB": "Individual_1",
    "hapC": "Individual_2",
    "hapD": "Individual_2",
    "hapE": "Individual_3",
    "hapF": "Individual_3",
    "hapG": "Individual_4",
    "hapH": "Individual_4",
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

print(dot_product.dot(grg_, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
