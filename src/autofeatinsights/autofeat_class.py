import logging
import glob
import tempfile
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import src.autofeatinsights.functions.relationship_functions as relationship_functions
import src.autofeatinsights.functions.tree_functions as tree_functions
import src.autofeatinsights.functions.feature_functions as feature_functions
import src.autofeatinsights.functions.evaluation_functions as evaluation_functions
from src.autofeatinsights.functions.helper_functions import RelevanceRedundancy, get_df_with_prefix
from typing import List, Set
from src.autofeatinsights.functions.classes import Weight, Tree, Result
import pandas as pd
from sklearn.model_selection import train_test_split
logging.getLogger().setLevel(logging.WARNING)


class FeatureDiscovery:
    targetColumn: str
    threshold: float
    paths: [Tree]
    weights: [Weight]
    results: [Result]
    base_dataset: str
    partial_join: pd.DataFrame
    extra_tables: [(str, str)]
    exlude_tables: [(str, str)]
    partial_join_selected_features: dict = {}
    join_keys: dict = {}
    explore: bool
    non_null_ratio_threshold: float

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
        self.trees = []
        self.join_name_mapping = {}

    def set_base_table(self, base_table: str, target_column: str):
        """
        Sets the base table and target column for feature generation.

        Args:
            base_table (str): The name of the base table.
            target_column (str): The name of the target column.

        Returns:
            None
        """
        self.base_table = base_table
        self.targetColumn = target_column
        X_train = get_df_with_prefix(self.base_table, self.targetColumn)
        self.partial_join = X_train.copy()
        features = list(X_train.columns)
        features.remove(target_column)
        self.partial_join_selected_features[str(base_table)] = features
        self.join_keys[str(base_table)] = []
        self.tree_hash = {}
        self.rel_red = RelevanceRedundancy(target_column)

    def set_dataset_repository(self, dataset_repository: List[str] = [], all_tables: bool = False):
        """
        Sets the dataset repository for the AutofeatClass object.

        Args:
            dataset_repository (List[str]): A list of dataset paths.
            all_tables (bool): Flag indicating whether to use all tables in the repository.

        Raises:
            Exception: If both dataset_repository and all_tables are specified.
            Exception: If neither dataset_repository nor all_tables are specified.

        Returns:
            None

        """
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
        """
        Retrieves the tables from the repository.

        Returns:
            tables (list): A list of table paths.
        """
        tables = []
        for dataset in self.datasets:
            for table in glob.glob("data/benchmark/" + dataset + "/*.csv"):
                tables.append((table.split("/")[-2]) + "/" + table.split("/")[-1])
        return tables

    def add_table(self, table: str):
        """
        Adds an extra table to the list of tables used for feature generation.

        Args:
            table (str): The name of the table to be added.

        Returns:
            None
        """
        self.extra_tables.append(table)
        if self.relationship_threshold is not None and self.matcher is not None:
            relationship_functions.rerun(self, self.relationship_threshold, self.matcher)

    def remove_table(self, table: str):
        """
        Removes a table from the list of extra tables and adds it to the list of excluded tables.

        Args:
            table (str): The name of the table to be removed.
        
        Returns:
            None
        """
        if table in self.extra_tables:
            self.extra_tables.remove(table)
        self.exclude_tables.append(table)
        if self.relationship_threshold is not None and self.matcher is not None:
            relationship_functions.rerun(self, self.relationship_threshold, self.matcher)

    def get_weights_from_table(self, table: str):
        """
        Returns a list of weights from the specified table.

        Args:
            table (str): The name of the table.

        Returns:
            list: A list of weights from the specified table.
        """
        return [i for i in self.weights if i.from_table == table]
    
    def get_weights_from_and_to_table(self, from_table, to_table):
        """
        Returns a list of weights that have the specified 'from_table' and 'to_table' values.

        Args:
            from_table (str): The source table name.
            to_table (str): The destination table name.

        Returns:
            list: A list of weights that match the specified 'from_table' and 'to_table' values.
        """
        return [i for i in self.weights if i.from_table == from_table and i.to_table == to_table]

    def find_relationships(self, matcher="coma", relationship_threshold: float = 0.5, explain=False, 
                           use_cache=True, verbose=True):
        """
        Finds relationships between features in the dataset.

        Args:
            matcher (str, optional): The name of the matcher to use for finding relationships. Defaults to "coma".
            relationship_threshold (float, optional): The threshold value for determining the strength of a relationship. Defaults to 0.5.
            explain (bool, optional): Whether to provide an explanation for the relationships found. Defaults to False.
            use_cache (bool, optional): Whether to use a cache for storing previously computed relationships. Defaults to True.
            verbose (bool, optional): Whether to print verbose output during the process. Defaults to True.

        Returns:
            None
        """
        self.matcher = matcher
        self.relation_threshold = relationship_threshold
        relationship_functions.find_relationships(self, relationship_threshold, matcher, explain, 
                                                  use_cache=use_cache, verbose=verbose)

    def read_relationships(self, file_path):
        """
        Reads the relationships from a file and updates the object's internal state.

        Args:
            file_path (str): The path to the file containing the relationships.

        Returns:
            None
        """
        relationship_functions.read_relationships(self, file_path)

    def display_best_relationships(self):
        """
        Displays the best relationships found by  FeatureDiscovery.

        Returns:
            None
        """
        relationship_functions.display_best_relationships(self)

    def add_relationship(self, table1: str, col1: str, table2: str, col2: str, weight: float):
        """
        Adds a relationship between two columns in different tables.

        Args:
            table1 (str): The name of the first table.
            col1 (str): The name of the column in the first table.
            table2 (str): The name of the second table.
            col2 (str): The name of the column in the second table.
            weight (float): The weight of the relationship.

        Returns:
            None
        """
        relationship_functions.add_relationship(self, table1, col1, table2, col2, weight)

    def remove_relationship(self, table1: str, col1: str, table2: str, col2: str):
        """
        Removes a relationship between two columns in different tables.

        Args:
            table1 (str): The name of the first table.
            col1 (str): The name of the column in the first table.
            table2 (str): The name of the second table.
            col2 (str): The name of the column in the second table.

        Returns:
            None
        """
        relationship_functions.remove_relationship(self, table1, col1, table2, col2)

    def update_relationship(self, table1: str, col1: str, table2: str, col2: str, weight: float):
        """
        Update the relationship between two tables and their respective columns with a given weight.

        Args:
            table1 (str): The name of the first table.
            col1 (str): The name of the column in the first table.
            table2 (str): The name of the second table.
            col2 (str): The name of the column in the second table.
            weight (float): The weight of the relationship.

        Returns:
            None
        """
        relationship_functions.update_relationship(self, table1, col1, table2, col2, weight)
    
    def display_table_relationship(self, table1: str, table2: str):
        """
        Display the relationship between two tables.

        Args:
            table1 (str): The name of the first table.
            table2 (str): The name of the second table.
        
        Returns:
            None
        """
        relationship_functions.display_table_relationship(self, table1, table2)

    def compute_join_trees(self, top_k_features: int = 10, non_null_threshold=0.5, explain=False, verbose=True):
        """
        Compute join trees for feature selection.

        Args:
            top_k_features (int): Number of top features to select. Defaults to 10.
            non_null_threshold (float): The threshold value for determining the non-null ratio of a feature. Defaults to 0.5.
            explain (bool): Whether to provide an explanation for the join trees. Defaults to False.
            verbose (bool): Whether to print verbose output during the process. Defaults to True.

        Returns:
            None
        """
        tree_functions.compute_join_trees(self, top_k_features, non_null_ratio_threshold=non_null_threshold, 
                                          explain=explain, verbose=verbose)

    def show_features(self, tree_id: int, show_discarded_features: bool = False):
        """
        Display the features for a given tree ID.

        Args:
            tree_id (int): The ID of the tree.
            show_discarded_features (bool): Whether to show discarded features or not. Default is False.

        Returns:
            None
        """
        feature_functions.show_features(self, tree_id, show_discarded_features)

    def display_join_trees(self, top_k: int = None):
        """
        Display the join trees for the AutoFeatClass instance.

        Args:
            top_k (int): The number of join trees to display. If None, display all join trees.

        Returns:
            None
        """
        tree_functions.display_join_trees(self, top_k)
    
    def display_join_tree(self, tree_id):
        """
        Display the join path with the given tree_id.

        Args:
            tree_id: The ID of the join path to display.
        
        Returns:
            None
        """
        tree_functions.display_join_tree(self, tree_id)

    def explain_relationship(self, table1: str, table2: str):
        """
        Explains the relationship between two tables.

        Args:
            table1 (str): The name of the first table.
            table2 (str): The name of the second table.

        Returns:
            None
        """
        relationship_functions.explain_relationship(self, table1, table2)
    
    def explain_tree(self, tree_id: int):
        """
        Explain the tree identified by the given tree_id.

        Args:
            tree_id (int): The ID of the tree to explain.
        
        Returns:
            None
        """
        tree_functions.explain_tree(self, tree_id)

    def remove_join_path_from_tree(self, tree_id: int, table: str):
        """
        Removes a join path from the tree.

        Args:
            tree_id (int): The ID of the tree.
            table (str): The name of the table to remove the join path from.

        Returns:
            None
        """
        tree_functions.remove_join_from_tree(self, tree_id, table)

    def explain_result(self, tree_id: int, model: str = 'GBM'):
        """
        Explain the result of a specific tree in the AutoFeat pipeline.

        Args:
            tree_id (int): The ID of the tree to explain.
            model (str, optional): The model to use for explanation. Defaults to 'GBM'.

        Returns:
            None
        """
        evaluation_functions.explain_result(self, tree_id, model)

    def inspect_join_tree(self, tree_id: int):
        """
        Inspects the join tree with the given tree_id.

        Args:
            tree_id (int): The ID of the join tree to inspect.

        Returns:
            None
        """
        tree_functions.inspect_join_tree(self, tree_id)

    def evaluate_trees(self, algorithm='GBM', top_k_trees: int = 3, verbose=True, explain=False):
        """
        Evaluate the performance of the generated trees.

        Args:
            algorithm (str): The algorithm to use for evaluation. Default is 'GBM'.
            top_k_paths (int): The number of top paths to consider. Default is 3.
            verbose (bool): Whether to print verbose output. Default is True.
            explain (bool): Whether to explain the evaluation results. Default is False.
        """
        evaluation_functions.evalute_trees(self, algorithm, top_k_trees, verbose=verbose, explain=explain)

    def get_best_result(self):
        """
        Returns:
            Result: The best result by accuracy.
        """
        return evaluation_functions.get_best_result(self)

    def evaluate_augmented_table(self, tree_id: int, algorithm='GBM', verbose=False):
        """
        Evaluate the augmented table using the specified algorithm and tree ID.

        Args:
            tree_id (int): The ID of the tree to use for evaluation.
            algorithm (str): The algorithm to use for evaluation. Default is 'GBM'.
            verbose (bool): Whether to print verbose output. Default is False.

        Returns:
            None
        """
        evaluation_functions.evaluate_table(self, algorithm, tree_id, verbose)
        evaluation_functions.explain_result(self, tree_id, algorithm)

    def adjust_relevance_value(self, tree_id: int, feature: str, value: float):
        """
        Adjusts the relevance value of a feature for a specific tree.

        Args:
            tree_id (int): The ID of the tree.
            feature (str): The name of the feature.
            value (float): The new relevance value.

        Returns:
            None
        """
        feature_functions.adjust_relevance_value(self, tree_id, feature, value)

    def adjust_redundancy_value(self, tree_id: int, feature: str, value: float):
        """
        Adjusts the redundancy value for a specific feature in a given tree.

        Args:
            tree_id (int): The ID of the tree.
            feature (str): The name of the feature.
            value (float): The new redundancy value.
        
        Returns:
            None
        """
        feature_functions.adjust_redundancy_value(self, tree_id, feature, value)

    def adjust_non_null_ratio(self, tree_id: int, table: str, value: float):
        """
        Adjusts the non-null ratio for a specific tree and table.

        Args:
            tree_id (int): The ID of the tree.
            table (str): The name of the table.
            value (float): The new non-null ratio value.
        
        Returns:
            None
        """
        feature_functions.adjust_non_null_ratio(self, tree_id, table, value)

    def move_features_to_discarded(self, tree_id: int, features: [str]):
        """
        Moves the specified features to the discarded list for the given tree.

        Args:
            tree_id (int): The ID of the tree.
            features (list[str]): The list of features to be moved to the discarded list.

        Returns:
            None
        """
        feature_functions.move_features_to_discarded(self, tree_id, features)

    def move_features_to_selected(self, tree_id: int, features: [str]):
        """
        Moves the specified features from discarded to the selected features list for the given tree.

        Args:
            tree_id (int): The ID of the tree.
            features (list[str]): The list of features to be moved.

        Returns:
            None
        """
        feature_functions.move_features_to_selected(self, tree_id, features)

    def materialise_join_tree(self, tree_id: int):
        """
        Materializes the join tree with the given tree_id.

        Args:
            tree_id (int): The ID of the join tree to materialize.

        Returns:
            The materialized join tree.
        """
        return tree_functions.materialise_join_tree(self, tree_id)

    def augment_dataset(self, algorithm="GBM", relation_threshold: float = 0.5, non_null_threshold=0.5, matcher="coma", 
                        top_k_features: int = 10, 
                        top_k_trees: int = 3, explain=True, verbose=True, use_cache=True):
        """
        Augments the dataset by finding relationships between features, computing join trees, and evaluating the trees.
        
        Args:
            algorithm (str): The algorithm to use for tree evaluation. Default is "GBM".
            relation_threshold (float): The threshold for considering a relationship between features. Default is 0.5.
            non_null_threshold: The threshold for considering a feature as non-null. Default is 0.5.
            matcher (str): The matcher to use for finding relationships. Default is "coma".
            top_k_features (int): The number of top features to select. Default is 10.
            top_k_paths (int): The number of top paths to select. Default is 3.
            explain (bool): Whether to explain the process. Default is True.
            verbose (bool): Whether to print verbose output. Default is True.
            use_cache (bool): Whether to use cached relationship weights. Default is True.

        Returns:
            None
        """
        if use_cache:
            if os.path.isfile(f"saved_weights/{self.base_table}_{relation_threshold}_{matcher}_weights.txt"):
                if verbose:
                    print("Reading from cache file: " + f"saved_weights/{self.base_table}_{relation_threshold}_{matcher}_weights.txt")
                    self.read_relationships(f"saved_weights/{self.base_table}_{relation_threshold}_{matcher}_weights.txt")
            else:
                self.find_relationships(relationship_threshold=relation_threshold, matcher=matcher, 
                                        explain=explain, verbose=verbose)
        else:
            self.find_relationships(relationship_threshold=relation_threshold, matcher=matcher, 
                                    explain=explain, verbose=verbose)
        self.compute_join_trees(top_k_features=top_k_features, explain=explain, non_null_threshold=non_null_threshold, 
                                verbose=verbose)
        self.evaluate_trees(algorithm=algorithm, top_k_trees=top_k_trees, explain=explain)


if __name__ == "__main__":

    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])
    autofeat.find_relationships()
    autofeat.display_best_relationships()

    # autofeat.augment_dataset(explain=True)
    # autofeat.read_relationships("saved_weights/school/base.csv_0.5_coma_weights.txt")
    # autofeat.compute_join_trees(top_k_features=5)
    # autofeat.display_join_path(2)
    # autofeat.augment_dataset(explain=True)
    # autofeat.read_relationships()
    # autofeat.compute_join_paths(top_k_features=5)
    # autofeat.display_join_path(1)
    # # autofeat.show_features(1, show_discarded_features=True)
    # df = autofeat.materialise_join_path(1)
    # print(df)
# autofeat.update_relationship(table1="school_best/base.csv", col1="DBN", table2="school_best/qr.csv", col2="DBN", 
    # weight=0.2)
    # autofeat.find_relationships(relationship_threshold=0.8, matcher="jaccard")
    # autofeat.add_table("school_best/")
    # autofeat.read_relationships()
    # autofeat.display_best_relationships()
    # autofeat.display_table_relationship("credit/table_0_0.csv", "credit/table_1_1.csv")
    # autofeat.explain_relationship("credit/table_0_0.csv", "credit/table_1_1.csv")
    # autofeat.compute_join_paths()
    # autofeat.show_features(1, show_discarded_features=True)
    # autofeat.move_feature_to_discarded(1, "credit/table_1_1.csv.other_parties")
    # # autofeat.adjust_relevance_value(1, "credit/table_1_1.csv.other_parties", 0.5)
    # # autofeat.adjust_null_ratio(1, "credit/table_1_1.csv", 0.5)
    # autofeat.show_features(1, show_discarded_features=True)
    # autofeat.move_feature_to_selected(1, "credit/table_1_1.csv.other_parties")
    # autofeat.show_features(1, show_discarded_features=True)
    # autofeat.inspect_join_path(2)
    # autofeat.show_features(path_id=3, show_discarded_features=True)
    # autofeat.display_join_paths(top_k=2)
    # df = autofeat.materialise_join_path(path_id=1)
    # print(list(df.columns))
    # autofeat.evaluate_paths(top_k_paths=2)
    # autofeat.add_relationship("credit/table_0_0.csv", "residence_since", "credit/table_1_1.csv", "housing", 0.8)
