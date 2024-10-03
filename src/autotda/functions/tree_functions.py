import tempfile
from pydantic import BaseModel
from autotda.data_models.dataset_model import Dataset
# from autotda.functions.classes import Tree, Join, Weight
import networkx
from matplotlib import pyplot as plt
from pathlib import Path as pt
import pandas as pd
import polars as pl
from copy import deepcopy
# from autotda.functions.helper_functions import get_df_with_prefix
import uuid
import datetime
from typing import Dict, List, Set, Tuple, Optional
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
# import autotda.functions.evaluation_functions as evaluation_functions
# import autotda.functions.feature_functions as feature_functions
# import autotda.functions.evaluation_functions as evaluation_functions
# from networkx.drawing.nx_pydot import graphviz_layout

from autotda.functions.relationship_functions import Relation, get_adjacent_nodes
from autotda.functions.relevance_redundancy import RelevanceRedundancy
from autotda.config import DATA_FOLDER


TEMP_DIR = tempfile.TemporaryDirectory()
TEMP_ROOT_PATH = pt(TEMP_DIR.name)

class JoinTree():
    rank: float
    relations: List[Relation]
    features: List[str]
    join_keys: List[str] 

    def __init__(self, features: List[str], join_keys: List[str]=None, 
                 rank: float=None, relations: List[Relation]=None) -> None:
        self.features = features
        self.join_keys = join_keys if join_keys else []
        self.rank = rank if rank else 0.0 
        self.relations = relations if relations else []

    def __str__(self) -> str:
        print(str(vars(self)))


class HCIAutoFeat:
    base_table: str 
    target_variable: str 
    top_k_features: int  
    top_k_join_trees: int 
    non_null_ratio: float 
    join_tree_maping: Dict[str, Tuple[str, JoinTree]]
    discovered: Set[str]
    sample_size: int = 3000

    def __init__(self,
                base_table: str,
                target_variable: str, 
                relations: List[Relation],
                top_k_features: int = 15,  
                top_k_join_trees: int = 5, 
                non_null_ratio: float = 0.65) -> None:
        
        self.base_table = base_table
        self.target_variable = target_variable
        self.top_k_features = top_k_features
        self.top_k_join_trees = top_k_join_trees
        self.non_null_ratio = non_null_ratio
        self.discovered = set()
        self.join_tree_maping = {}

        self.relations = relations
        self.rel_red = RelevanceRedundancy(self.target_variable)

        self.init_tree_root()

    def init_tree_root(self):
        from sklearn.model_selection import train_test_split

        # Read dataframe
        base_table_df = get_df_with_prefix(
            self.base_table, self.target_variable
        )
        partial_join_filename = TEMP_ROOT_PATH / f"{self.base_table.replace('/', '_')}.parquet"

        # Stratified sampling
        if self.sample_size < base_table_df.shape[0]:
            X_train, _ = train_test_split(
                base_table_df,
                train_size=self.sample_size,
                stratify=base_table_df[self.target_variable],
                random_state=42,
            )
        else:
            X_train = base_table_df

        X_train.write_parquet(partial_join_filename)

        # Base table features are the selected features
        features = list(X_train.columns)
        if self.target_variable in features:
            features.remove(self.target_variable)

        root_node = JoinTree(features=features)

        self.join_tree_maping[self.base_table] = (root_node, partial_join_filename) 

    # def compute_join_trees(self, autofeat, top_k_features, non_null_ratio_threshold: float = 0.5, explain=False, verbose=True):
    #     autofeat.trees = []
    #     autofeat.set_base_table(autofeat.base_table, autofeat.targetColumn)
    #     emptyTree = Tree(begin=autofeat.base_table, joins=[], rank=0)
    #     emptyTree.features = autofeat.partial_join_selected_features[str(autofeat.base_table)]
    #     emptyTree.id = 0
    #     autofeat.trees.append(emptyTree)
    #     if verbose:
    #         print("Calculating join trees...")
    #     self.stream_feature_selection(autofeat=autofeat, queue={str(self.base_table)})


    def streaming_feature_selection(self, queue: set, previous_queue: set = None):
        if len(queue) == 0:
            return
        if previous_queue is None:
            previous_queue = queue.copy()

        all_neighbours = set()

        while len(queue) > 0:
            base_node_id = queue.pop()
            self.discovered.add(base_node_id)

            # near_weights = get_adjacent_nodes(autofeat, {base_node_id}, 0.5)
            neighbours_relations = get_adjacent_nodes(relations=self.relations, node=base_node_id)
            # neighbours = [n.to_table for n in near_weights]
            neighbours = set(neighbours_relations.keys())
            neighbours = sorted((set(neighbours).difference(self.discovered)))
            all_neighbours.update(neighbours)
            for node in neighbours:
                self.discovered.add(node)
                # join_keys = autofeat.get_weights_from_and_to_table(base_node_id, n)
                join_keys = neighbours_relations[node]
                max_similarity = join_keys[0].similarity
                join_keys = [rel for rel in join_keys if rel.similarity == max_similarity]

                right_df = get_df_with_prefix(node)

                current_queue = set()
                while len(previous_queue) > 0:
                    previous_table_join = previous_queue.pop()
                  
                    # key_path = TEMP_ROOT_PATH / previous_table_join
                    # previous_join = pd.read_parquet(key_path)

                    prop: Relation
                    for prop in join_keys:
                        join_list = f"{previous_table_join}--{prop.to_table}-{prop.to_col}"

                        joined_df, join_filename, keys = self.step_join(relation=prop, left_filename=previous_table_join, right_df=right_df)

                        # sampled_right_df = right_df.groupby(prop.get_to_prefix()).sample(n=1, random_state=42)
                        # if previous_join[prop.get_from_prefix()].dtype != sampled_right_df[prop.get_to_prefix()].dtype:
                        #     current_queue.append(previous_table_join)
                        #     continue
                        # joined_df = pd.merge(left=previous_join, right=sampled_right_df, left_on=(prop.get_from_prefix()),
                        #                     right_on=(prop.get_to_prefix()), how="left")
                        
                        # joined_df.to_parquet(pt(autofeat.temp_dir.name) / filename)
                        
                        non_null_ratio = self.non_null_ratio_calculation(joined_df, keys[1])
                        if not non_null_ratio:
                            current_queue.add(previous_table_join)
                            continue
                        
                        join_tree_node, _ = self.join_tree_maping[previous_table_join]
                        result = self.streaming_relevance_redundancy(
                            dataframe=joined_df.clone(),
                            new_features=list(right_df.columns),
                            selected_features=join_tree_node.features
                        )
                        
                        if result is not None:
                            score, rel_score, red_score, final_features, rel_discarded, red_discarded = result
                            # remaining_rel_score = [i for i in rel_score if i[0] in final_features]
                            # remaining_red_score = [i for i in red_score if i[0] in final_features]
                            # red_discarded = red_discarded + [i for i in red_score if i[0] not in final_features]
                            # rel_discarded = rel_discarded + [i for i in rel_score if i[0] not in final_features]
                            # join = Join(prop.from_table, prop.to_table, 
                            #             prop.from_col, prop.to_col, non_null_ratio, {"rel": remaining_rel_score, 
                            #                                                     "red": remaining_red_score}, 
                            #             {"rel": rel_discarded, "red": red_discarded})
                            all_features = join_tree_node.features.copy()
                            all_features.extend(final_features)

                            relations = join_tree_node.relations.copy()
                            relations.append(prop)
                            final_keys = join_tree_node.join_keys.copy()
                            final_keys.extend(keys)
                            tree_node = JoinTree(rank=score, features=all_features, join_keys=final_keys, relations=relations)
                            self.join_tree_maping[join_list] = (tree_node, join_filename)
                            # autofeat.partial_join_selected_features[str(join_list)] = all_features

                            # tree = deepcopy(autofeat.tree_hash[str(previous_table_join)])
                            # tree.features = all_features

                            # join_keys = autofeat.join_keys[str(previous_table_join)]
                            # join_keys.extend([prop.get_from_prefix(), prop.get_to_prefix()])
                            # autofeat.join_keys[str(join_list)] = join_keys
                            # tree.join_keys = join_keys
                            # tree.rank = score
                            # tree.add_join(join)
                            # tree.id = len(autofeat.trees)
                            # autofeat.tree_hash[str(join_list)] = tree
                            # autofeat.trees.append(deepcopy(tree))
                        else:
                            self.join_tree_maping[join_list] = (join_tree_node, join_filename)
                            # autofeat.partial_join_selected_features[str(join_list)] = \
                            #     autofeat.partial_join_selected_features[str(previous_table_join)]
                            # autofeat.tree_hash[str(join_list)] = autofeat.tree_hash[str(previous_table_join)]
                            # autofeat.join_keys[str(join_list)] = autofeat.join_keys[str(previous_table_join)]
                        
                        # autofeat.join_name_mapping[str(join_list)] = filename
                        current_queue.add(join_list)
                previous_queue.update(current_queue)
                
        self.streaming_feature_selection(all_neighbours, previous_queue)


    def step_join(
        self,
        relation: Relation,
        left_filename: str,
        right_df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, str, list]:
        
        # key_path = TEMP_ROOT_PATH / left_filename
        left_df = pl.read_parquet(self.join_tree_maping[left_filename][1])

        # Step - Sample neighbour data - Transform to 1:1 or M:1
        sampled_right_df = right_df.filter(
            pl.int_range(0, pl.count()).shuffle(seed=42).over(f"{relation.to_table}.{relation.to_col}") < 1
        )

        # File naming convention as the filename can be gigantic
        join_filename = TEMP_ROOT_PATH / f"{str(uuid.uuid4())}.parquet"

        # Join
        left_on = f"{relation.from_table}.{relation.from_col}"
        right_on = f"{relation.to_table}.{relation.to_col}"

        if left_df[left_on].dtype != sampled_right_df[right_on].dtype:
            return None

        new_names = [f"{x}_tmp" for x in [left_on, right_on]]

        # replicate join columns with new names
        df1 = left_df.with_columns(pl.col(left_on).alias(new_names[0]))
        df2 = sampled_right_df.with_columns(pl.col(right_on).alias(new_names[1]))

        # perform join and drop columns
        joined_df = df1.join(df2, left_on=new_names[0], right_on=new_names[1], how="left").drop(new_names[0])
        joined_df.write_parquet(join_filename)

        return joined_df, join_filename, [left_on, right_on]
    
    def non_null_ratio_calculation(self, joined_df: pl.DataFrame, right_key: str) -> float:
        # total_length = joined_df.shape[0]
        # non_nulls = joined_df[prop.get_to_prefix()].count()
        # return non_nulls / total_length
        if joined_df[right_key].count() / joined_df.shape[0] < self.non_null_ratio:
            # logging.debug(f"\t\tRight column value ration below {self.value_ratio}.\nSKIPPED Join")
            return False
        return True
    
    def streaming_relevance_redundancy(
        self, dataframe: pl.DataFrame, new_features: list[str], selected_features: list[str]
    ) -> Optional[Tuple]:
    
        # t = datetime.datetime.now()
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False
        ).fit_transform(X=dataframe.to_pandas(), random_state=42, random_seed=42)

        X = df.drop(columns=[self.target_variable])
        y = df[self.target_variable]

        features = list(set(X.columns).intersection(set(new_features)))
        top_feat = len(features) if len(features) < self.top_k_features else self.top_k_features

        m = 1
        feature_score_relevance, rel_discarded_features = self.rel_red.measure_relevance(
            dataframe=X, new_features=features, target_column=y
        )
        feature_score_relevance = feature_score_relevance[:top_feat]
        rel_discarded_features += feature_score_relevance[top_feat:]

        if len(feature_score_relevance) == 0:
            return None
        
        relevant_features = list(dict(feature_score_relevance).keys())
        m = len(feature_score_relevance) if len(feature_score_relevance) > 0 else m
        sum_m = sum(list(map(lambda x: x[1], feature_score_relevance)))
        
        final_features = relevant_features
        sum_o = 0
        o = 1
        feature_score_redundancy, red_discarded_features = self.rel_red.measure_redundancy(
            dataframe=X, selected_features=selected_features, relevant_features=relevant_features, target_column=y
        )

        if len(feature_score_redundancy) == 0:
            return None

        o = len(feature_score_redundancy) if feature_score_redundancy else o
        sum_o = sum(list(map(lambda x: x[1], feature_score_redundancy)))
        final_features = list(dict(feature_score_redundancy).keys())

        score = (o * sum_m + m * sum_o) / (m * o)

        return (score, feature_score_relevance, feature_score_redundancy, final_features, 
                rel_discarded_features, red_discarded_features)
    

def get_df_with_prefix(node_id: str, target_column=None) -> pl.DataFrame:
    """
    Get the node from the database, read the file identified by node_id and prefix the column names with the node label.

    :param node_id: ID of the node - used to retrieve the corresponding node from the database
    :param target_column: Optional parameter. The name of the label/target column containing the classes,
            only needed when the dataset to read contains the class.
    :return: 0: A pandas dataframe whose columns are prefixed with the node label, 1: the node label
    """
    dataframe = pl.read_csv(str(DATA_FOLDER / node_id), encoding="utf8", quote_char='"')
    if target_column:
        dataframe = dataframe.rename(
            lambda col_name: f"{node_id}.{col_name}" if col_name != target_column else col_name
        )
    else:
        dataframe = dataframe.rename(lambda col_name: f"{node_id}.{col_name}")

    # dataframe = dataframe.to_pandas()
    # if target_column:
    #     dataframe = dataframe.set_index([target_column]).reset_index()

    return dataframe




# def display_join_trees(self, top_k: None):
#     if top_k is None:
#         top_k = len(self.trees)
#     sorted_trees = sorted(self.trees, key=lambda x: x.rank, reverse=True)[:top_k]
#     for index, tree in enumerate(sorted_trees):
#         if len(tree.joins) > 0:
#             plt.figure(figsize=(5, 2))
#             graph = networkx.DiGraph()
#             plt.gca()
#             labels = {}
#             mapping = {tree.joins[0].from_table: 0}
#             count = 0
#             for i in tree.joins:
#                 count += 1
#                 mapping[i.to_table] = count
#                 graph.add_edge(mapping[i.from_table], mapping[i.to_table])
#                 from_col = i.from_col if len(i.from_col) < 20 else i.from_col[:10] + "..." + i.from_col[-10:]
#                 to_col = i.to_col if len(i.to_col) < 20 else i.to_col[:10] + "..." + i.to_col[-10:]
#                 labels[(mapping[i.from_table], mapping[i.to_table])] = (from_col + " -> " + to_col)
#             ids = list(mapping.values())
#             names = list(mapping.keys())
#             df = pd.DataFrame({"Node ID": ids, "Table Name": names})
#             pos = graphviz_layout(graph, prog="dot")
#             networkx.draw(graph, pos=pos, with_labels=True)
#             # networkx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels, font_size=10)
#             plt.title(f"Join Tree ID: {tree.id}. Rank: {('%.2f' % tree.rank)}")
#             plt.table(cellText=df.values, cellLoc="center", colLabels=df.columns, loc="right").auto_set_column_width([0,1])
#             plt.show()


# def display_join_tree(self, tree_id):
#     tree = get_tree_by_id(self, tree_id)
#     if tree is None or len(tree.joins) == 0:
#         return
#     graph = networkx.DiGraph()
#     plt.gca()
#     labels = {}
#     mapping = {tree.joins[0].from_table: 0}
#     count = 0
#     for i in tree.joins:
#         count += 1
#         mapping[i.to_table] = count
#         graph.add_edge(mapping[i.from_table], mapping[i.to_table])
#         from_col = i.from_col if len(i.from_col) < 20 else i.from_col[:10] + "..." + i.from_col[-10:]
#         to_col = i.to_col if len(i.to_col) < 20 else i.to_col[:10] + "..." + i.to_col[-10:]
#         labels[(mapping[i.from_table], mapping[i.to_table])] = (from_col + " -> " + to_col)
#     ids = list(mapping.values())
#     names = list(mapping.keys())
#     df = pd.DataFrame({"Node ID": ids, "Feature Name": names})
#     pos = graphviz_layout(graph, prog="dot")
#     networkx.draw(graph, pos=pos, with_labels=True)
#     # networkx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels, font_size=10)
#     plt.title(f"Join Tree ID: {tree.id}. Rank: {('%.2f' % tree.rank)}")
#     plt.table(cellText=df.values, cellLoc="center", colLabels=df.columns, loc="right").auto_set_column_width([0, 1])
#     plt.show()





# def inspect_join_tree(self, tree_id):
#     tree = get_tree_by_id(self, tree_id)
#     if tree is None:
#         return
#     print(tree)
#     feature_functions.show_features(self, tree_id)


# def materialise_join_tree(self, tree_id):
#     tree = get_tree_by_id(self, tree_id)
#     base_df = get_df_with_prefix(self.base_table, self.targetColumn)
#     i: Join
#     for i in tree.joins:
#         df = get_df_with_prefix(i.to_table)
#         base_df = pd.merge(base_df, df, left_on=i.get_from_prefix(), right_on=i.get_to_prefix(), how="left")
#     base_df = base_df[tree.features]
#     # Filter on selected features in rel_red
#     return base_df


# def get_tree_by_id(self, tree_id) -> Tree:
#     list = [i for i in self.trees if i.id == tree_id]
#     if len(list) == 0:
#         return None
#     return list[0]


# def get_adjacent_nodes(self, nodes: list, threshold: float) -> [Weight]:
#     return_list = list()
#     for n in nodes:
#         for x in self.get_weights_from_table(n):
#             return_list.append(x)
#     return return_list



# def remove_join_from_tree(self, tree_id: int, table: str):
#     tree = get_tree_by_id(self, tree_id)
#     if tree is None:
#         print("Can not find path")
#         return
#     for index, join in enumerate(tree.joins):
#         if join.to_table == table:
#             features = [i[0] for i in join.rel_red["rel"]] + [i[0] for i in join.rel_red["red"]]
#             tree.features = list(set(tree.features).difference(set(features)))
#             tree.joins.pop(index)
#             print("Path removed")
#     # evaluation_functions.rerun(self)


# def rerun(autofeat):
#     if len(autofeat.trees) > 0:
#         print("Recalculating trees")
#         compute_join_trees(autofeat, top_k_features=autofeat.top_k_features)
#         evaluation_functions.rerun(autofeat)


# def explain_tree(self, tree_id: int):
#     tree = get_tree_by_id(self, tree_id)
#     if tree is None:
#         return
#     print(tree.explain())