import json
import grg
import dot_product
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

mutation_map = grg_.get_mutation_to_individuals(endpoints_map, individual_map)

print(json.dumps(mutation_map, indent=2))

individual_to_mutation_map = grg_.get_individuals_to_mutations(
    endpoints_map, individual_map
)
print(json.dumps(individual_to_mutation_map, indent=2))

matrix = grg_.to_matrix(endpoints_map, individual_map)
print(matrix)


