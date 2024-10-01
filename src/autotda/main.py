from autotda.functions.relationship_functions import DatasetDiscovery, get_adjacent_nodes
from autotda.functions.tree_functions import HCIAutoFeat


if __name__ == "__main__":
    ddisc = DatasetDiscovery(matcher="Jaccard", data_repositories=["credit"])
    ddisc.find_relationships()

    autofeat = HCIAutoFeat(
        base_table="credit/table_0_0.csv",
        target_variable="class",
        relations=ddisc.relations,
    )

    # neighbours_relations = get_adjacent_nodes(relations=ddisc.relations, node="credit/table_0_0.csv")
    
    # for n in neighbours_relations.keys():
    #     print(n)
    #     print(list(map(lambda x: vars(x), neighbours_relations[n])))

    autofeat.streaming_feature_selection(queue={autofeat.base_table})

    trees = dict(sorted(autofeat.join_tree_maping.items(), key=lambda item: item[1][0].rank, reverse=True))
    for tr in trees.keys():
        print(tr)
        tree_node, filename = autofeat.join_tree_maping[tr]
        print(f"\t\t{tree_node.rank}")
        print(filename)
        # print(list(map(lambda x: vars(x), tree_node.relations)))
        
