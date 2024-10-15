from typing import Dict, List, Tuple
from polars import DataFrame

from app.graph.graph_model import from_relations_to_graph
from autotda.functions.tree_functions import JoinTree


class DisplayJoinTree:
    rank: float 
    elements: Tuple[List, List]
    table_data: DataFrame
    selected_features: DataFrame
    discarded_features: DataFrame
    join_tree_filename: str
    join_tree_id: str 

    def __init__(self, nodes: List[dict], edges: List[dict], rank: int, dataframe: DataFrame,
                 join_tree_id: str, 
                 selected_features: DataFrame, discarded_features: DataFrame, 
                 join_tree_filename: str) -> None:
        self.elements = {
            "nodes": nodes,
            "edges": edges
        }
        self.rank = rank
        self.table_data = dataframe
        self.join_tree_id = join_tree_id
        self.selected_features = selected_features
        self.discarded_features = discarded_features
        self.join_tree_filename = join_tree_filename


def from_join_keys_to_dataframe(join_keys: List[str]) -> DataFrame:
    tables = []
    columns = []
    for jk in join_keys:
        tables.append(".".join(jk.split(".")[:-1]))
        columns.append(jk.split(".")[-1])

    return DataFrame({"table_name": tables, "join_column": columns})



def print_join_trees(join_trees: Dict[str, Tuple[JoinTree, str]], top_k_trees: int) -> List[DisplayJoinTree]:
    trees = []

    for i, (k, v) in enumerate(join_trees.items()):
        if i == top_k_trees:
            break

        tree_node: JoinTree = v[0]

        nodes, edges = from_relations_to_graph(tree_node.relations)
        dataframe = from_join_keys_to_dataframe(tree_node.join_keys)
        display_tree = DisplayJoinTree(nodes=nodes, edges=edges, rank=tree_node.rank, dataframe=dataframe,
                                       join_tree_id=k,
                                       selected_features=tree_node.selected_features,
                                       discarded_features=tree_node.discarded_features, 
                                       join_tree_filename=v[1])
        trees.append(display_tree)

    return trees




