from typing import Dict, List, Tuple
from polars import DataFrame

from app.graph.graph_model import from_relations_to_graph
from autotda.functions.tree_functions import JoinTree


class DisplayJoinTree:
    rank: float 
    elements: Tuple[List, List]
    table_data: DataFrame

    def __init__(self, nodes: List[dict], edges: List[dict], rank: int, dataframe: DataFrame) -> None:
        self.elements = {
            "nodes": nodes,
            "edges": edges
        }
        self.rank = rank
        self.table_data = dataframe


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

        print(tree_node.rank)
        nodes, edges = from_relations_to_graph(tree_node.relations)
        dataframe = from_join_keys_to_dataframe(tree_node.join_keys)
        display_tree = DisplayJoinTree(nodes=nodes, edges=edges, rank=tree_node.rank, dataframe=dataframe)
        trees.append(display_tree)

    return trees



