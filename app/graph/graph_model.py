from typing import List, Tuple
import uuid
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

from src.autotda.functions.relationship_functions import Relation

# Style node & edge groups
node_styles = [
    NodeStyle("TABLE", "#FF7F3E", "name"),
]

edge_styles = [
    EdgeStyle("RELATED", caption='weight', directed=False),
]

class NodeTable:
    id: str
    name: str
    label: str

    def __init__(self, table_name: str) -> None:
         self.id = str(uuid.uuid4())
         self.name = table_name
         self.label = "TABLE"

    def to_dict(self): 
         data = {
              "data": vars(self)
        }
         return data
    
class EdgeRelation:
    id: str
    source: str
    target: str
    source_name: str
    target_name: str
    source_column: str
    target_column: str
    weight: float 
    label: str

    def __init__(self, 
                 source_id: str,
                 target_id: str,
                 source_name: str, 
                 target_name: str, 
                 source_column: str, 
                 target_column: str, 
                 weight=1) -> None:
         
         self.id = str(uuid.uuid4())
         self.source = source_id
         self.target = target_id
         self.source_name = source_name
         self.target_name = target_name
         self.source_column = source_column
         self.target_column = target_column
         self.weight = weight
         self.label = "RELATED"

    def to_dict(self):
         data = {
              "data": vars(self)
         }
         return data
    

def from_relations_to_graph(relations: List[Relation]) -> Tuple[List, List]:
     nodes = []
     edges = []

     visited = {}
     for relation in relations:
        if not relation.from_table in visited.keys():     
            node_source = NodeTable(relation.from_table) 
            visited[relation.from_table] = node_source
            nodes.append(node_source.to_dict())

        if not relation.to_table in visited.keys():     
            node_target = NodeTable(relation.to_table) 
            visited[relation.to_table] = node_target
            nodes.append(node_target.to_dict())

        edge = EdgeRelation(
            source_id=visited[relation.from_table].id, 
            target_id=visited[relation.to_table].id,
            source_name=visited[relation.from_table].name,
            target_name=visited[relation.to_table].name,
            source_column=relation.from_col,
            target_column=relation.to_col,
            weight=relation.similarity
        )
        edges.append(edge.to_dict())
    
     return nodes, edges

