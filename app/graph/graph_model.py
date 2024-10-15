from typing import Dict, List, Tuple
import uuid
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

from autotda.functions.tree_functions import JoinTree
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
    alias: int

    def __init__(self, table_name: str, alias: int = None) -> None:
         self.id = str(uuid.uuid4())
         self.name = table_name
         self.label = "TABLE"
         self.alias = alias

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

    def __eq__(self, value: object) -> bool:
        if self.source_name == value.source_name and self.source_column == value.source_column and self.weight == value.weight and self.target_name == value.target_name and self.target_column == value.target_column:
            return True
        
        if self.source_name == value.target_name and self.source_column == value.target_column and self.weight == value.weight and self.target_name == value.source_name and self.target_column == value.source_column:
            return True 
        
        return False 
            
        
    def to_dict(self):
         data = {
              "data": vars(self)
         }
         return data
    

def from_relations_to_graph(relations: List[Relation]) -> Tuple[List, List]:
     nodes = []
     edges = []

     visited_edges = []

     visited = {}
     alias = {}
     index = 0 
     for relation in relations:
        if relation.from_table not in alias:
            alias[relation.from_table] = index
            index += 1
        if relation.to_table not in alias:
            alias[relation.to_table] = index
            index += 1
        if not relation.from_table in visited.keys():     
            node_source = NodeTable(relation.from_table, alias=alias[relation.from_table]) 
            visited[relation.from_table] = node_source
            nodes.append(node_source.to_dict())

        if not relation.to_table in visited.keys():     
            node_target = NodeTable(relation.to_table, alias=alias[relation.to_table]) 
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

        if not edge in visited_edges:
            visited_edges.append(edge)
            edges.append(edge.to_dict())
    
     return nodes, edges


