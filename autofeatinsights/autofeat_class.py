import logging
import glob
import tempfile
import functions.relationship_functions as relationship_functions
import functions.tree_functions as tree_functions
import functions.feature_functions as feature_functions
import functions.evaluation_functions as evaluation_functions
from functions.helper_functions import RelevanceRedundancy, get_df_with_prefix
from typing import List, Set
from functions.classes import Weight, Path, Result
import pandas as pd
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO)


class FeatureDiscovery:
    targetColumn: str
    threshold: float
    paths: [Path]
    weights: [Weight]
    results: [Result]
    base_dataset: str
    partial_join: pd.DataFrame
    extra_tables: [(str, str)]
    exlude_tables: [(str, str)]
    partial_join_selected_features: dict = {}
    explore: bool
    null_ratio_threshold: float

    def __init__(self):

        # self.base_dataset = base_dataset
        # self.weight_treshold = weight_treshold
        self.datasets = []
        self.weights = []
        self.results = []
        self.discovered: Set[str] = set()
        self.extra_tables = []
        self.exlude_tables = []
        self.definite_features = []
        self.exclude_features = []
        self.temp_dir = tempfile.TemporaryDirectory()
        # self.explore = explore
        # self.top_k = top_k
        # self.rel_red = RelevanceRedundancy(targetColumn, jmi=jmi, pearson=pearson)
        self.paths = []
        self.join_name_mapping = {}

    def set_base_table(self, base_table: str, target_column: str):
        self.base_table = base_table
        self.targetColumn = target_column
        X_train, X_test = train_test_split(
            get_df_with_prefix(self.base_table, self.targetColumn), random_state=42)
        self.partial_join = X_train.copy()
        features = list(X_train.columns)
        features.remove(target_column)
        self.partial_join_selected_features[str([base_table])] = features
        self.rel_red = RelevanceRedundancy(target_column)

    def set_dataset_repository(self, dataset_repository: List[str] = [], all_tables: bool = False):
        if len(dataset_repository) > 0 and all_tables:
            raise Exception("You can't set all_tables to True and specify a dataset repository.")
        if len(dataset_repository) == 0 and not all_tables:
            raise Exception("You need to specify a dataset repository or set all_tables to True.")
        if all_tables:
            datasets = [i.split("/")[-1] for i in glob.glob("data/benchmark/*")]
            self.datasets = datasets
        else:
            self.datasets = dataset_repository

    def get_tables_repository(self):
        tables = []
        for dataset in self.datasets:
            for table in glob.glob("data/benchmark/" + dataset + "/*.csv"):
                tables.append((table.split("/")[-2]) + "/" + table.split("/")[-1])
        return tables

    # def display_join_paths_dsplot(self, top_k: None):
    #     if top_k is None:
    #         top_k = len(self.paths)
    #     sorted_paths = sorted(self.paths, key=lambda x: x.rank, reverse=True)[:top_k]
    #     for index, path in enumerate(sorted_paths):
    #         graph_dic = {}
    #         graph_dic[path.begin] = []
    #         i: Join
    #         for i in path.joins:
    #             if i.from_table not in graph_dic:
    #                 graph_dic[i.from_table] = []
    #             if i.to_table not in graph_dic:
    #                 graph_dic[i.to_table] = []
    #         for i in path.joins:
    #             graph_dic[i.from_table].append(i.to_table)
    #         dsGraph(graph_dic, directed=True).plot(output_path=(f"graph-{index}.png"))

    # def add_table(self, dataset: str, table: str):
    #     logging.warning("This means that the algorithm has to recalculate it's weights and paths.")
    #     self.extra_tables.append((dataset, table))
    #     self.data_discover(silent=True)

    # def remove_table(self, dataset: str, table: str):
    #     logging.warning("This means that the algorithm has to recalculate it's weights and paths.")
    #     if (dataset, table) in self.extra_tables:
    #         self.extra_tables.remove((dataset, table))
    #     self.exclude_tables.append((dataset, table))
    #     self.data_discover(silent=True)

    # def add_feature(self, dataset: str, table: str, feature: str):
    #     # logging.warnin("This means that the algorithm has to recalculate it's weights and paths.")
    #     self.definite_features.append(dataset + "/" + table + "." + feature)

    # def remove_feature(self, dataset: str, table: str, feature: str):
    #     if (dataset + "/" + table + "." + feature) in self.definite_features:
    #         self.definite_features.remove(dataset + "/" + table + "." + feature)
    #     self.exclude_features.append(dataset + "/" + table + "." + feature)  
    
    # def get_path_length(self, path: str) -> int:
    #     path_tokens = path.split("--")
    #     return len(path_tokens) - 1

    def get_weights_from_table(self, table: str):
        return [i for i in self.weights if i.from_table == table]
    
    def get_weights_from_and_to_table(self, from_table, to_table):
        return [i for i in self.weights if i.from_table == from_table and i.to_table == to_table]

    def find_relationships(self, threshold: float = 0.5):
        logging.info("Step 1: Finding relationships")
        relationship_functions.find_relationships(self, threshold)

    def display_best_relationships(self):
        relationship_functions.display_best_relationships(self)
    
    def display_table_relationship(self, table1: str, table2: str):
        relationship_functions.display_table_relationship(self, table1, table2)

    def compute_join_paths(self, top_k_features: int = 10):
        tree_functions.compute_join_paths(self, top_k_features)

    def show_features(self, path_id: int, show_discarded_features: bool = False):
        feature_functions.show_features(self, path_id, show_discarded_features)

    def display_join_paths(self, top_k: None):
        tree_functions.display_join_paths(self, top_k)

    def explain_relationship(self, table1: str, table2: str):
        relationship_functions.explain_relationship(self, table1, table2)

    def inspect_join_path(self, path_id: int):
        tree_functions.inspect_join_path(self, path_id)

    def evalute_paths(self, top_k_paths: int = 2):
        evaluation_functions.evaluate_paths(self, top_k_paths)

if __name__ == "__main__":

    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="credit/table_0_0.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["credit"])
    autofeat.find_relationships(threshold=0.8)
    # autofeat.read_relationships()
    # # autofeat.display_best_relationships()
    # autofeat.display_table_relationship("credit/table_0_0.csv", "credit/table_1_1.csv")
    # autofeat.explain_relationship("credit/table_0_0.csv", "credit/table_1_1.csv")
    autofeat.compute_join_paths(top_k_features=2)
    # autofeat.inspect_join_path(2)
    autofeat.show_features(path_id=3, show_discarded_features=True)
    # autofeat.display_join_paths(top_k=2)
    # df = autofeat.materialise_join_path(path_id=1)
    # print(list(df.columns))
    # autofeat.evaluate_paths(top_k_paths=2)
