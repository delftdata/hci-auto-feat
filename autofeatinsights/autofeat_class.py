import uuid
import logging
import glob
import itertools
import warnings
import tempfile
import networkx
from valentine.algorithms import Coma
from joblib import Parallel, delayed
from valentine import valentine_match
from functions.helper_functions import RelevanceRedundancy, get_df_with_prefix
from multiprocessing import Manager
from typing import List, Set, Tuple, Optional
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from functions.classes import Weight, Path, Result, Join
from copy import deepcopy
from pathlib import Path as pt
from dsplot.graph import Graph as dsGraph
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
import tqdm
from autogluon.tabular import TabularPredictor
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
    data_quality_threshhold: float

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
        # self.data_quality_threshhold = data_quality_threshhold
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

    def read_relationships(self):
        f = open("weights.txt", "r")
        stringlist = f.read().split(",")
        for i in stringlist:
            if i != "":
                table1, table2, col1, col2, weight = i.split("--")
                self.weights.append(Weight(table1, table2, col1, col2, float(weight)))
        f.close()
        tables = self.get_tables_repository()
        self.weight_string_mapping = {}
        for t in tables:
            if len(t) > 20:
                new_string = t.split("/")[0] + "/" + t.split("/")[1][:3] + "..." + t.split("/")[1][-7:]
                self.weight_string_mapping[t] = new_string
            else:
                self.weight_string_mapping[t] = t

    def find_relationships(self, threshold: float = 0.5):
        manager = Manager()
        temp = manager.list()

        def profile(combination):
            (table1, table2) = combination
            df1 = pd.read_csv("data/benchmark/" + table1)
            df2 = pd.read_csv("data/benchmark/" + table2)
            matches = self.calculate_coma(df1, df2)
            for m in matches.items():
                ((_, col_from), (_, col_to)), similarity = m
                if similarity > threshold:
                    temp.append(Weight(table1, table2, col_from, col_to, similarity))
                    temp.append(Weight(table2, table1, col_to, col_from, similarity))
        tables = self.get_tables_repository()
        self.weight_string_mapping = {}
        for t in tables:
            if len(t) > 20:
                new_string = t.split("/")[0] + "/" + t.split("/")[1][:3] + "..." + t.split("/")[1][-7:]
                self.weight_string_mapping[t] = new_string
            else:
                self.weight_string_mapping[t] = t
        Parallel(n_jobs=-1)(delayed(profile)(combination) 
                            for combination in tqdm.tqdm(itertools.combinations(tables, 2), 
                                                         total=len(tables) * (len(tables) - 1) / 2))
        print(temp)
        self.weights = temp

        # Uncomment for saving weights to file.
        # f = open("weights.txt", "w")
        # stringlist = []
        # for i in self.weights:
        #     stringlist.append(f"{i.from_table}--{i.to_table}--{i.from_col}--{i.to_col}--{i.weight},")
        # f.writelines(stringlist)
        # f.close()

    def get_tables_repository(self):
        tables = []
        for dataset in self.datasets:
            for table in glob.glob("data/benchmark/" + dataset + "/*.csv"):
                tables.append((table.split("/")[-2]) + "/" + table.split("/")[-1])
        return tables
    
    def display_relationships(self):
        tables = self.get_tables_repository()
        highest_weights = []
        for table1 in tables:
            for table2 in tables:
                if table1 == table2:
                    highest_weights.append([self.weight_string_mapping[table1], self.weight_string_mapping[table2], 1])
                else:
                    weight = self.get_best_weight(table1, table2)
                    if weight is not None:
                        highest_weights.append([self.weight_string_mapping[weight.from_table], 
                                                self.weight_string_mapping[weight.to_table], 
                                                weight.weight])
        df = pd.DataFrame(highest_weights, columns=["from_table", "to_table", "weight"])
        seaborn.heatmap(df.pivot(index="from_table", columns="to_table", values="weight"), square=True)
        plt.xticks(fontsize="small", rotation=30) 
        plt.show()

    def get_best_weight(self, table1: str, table2: str) -> Weight:
        weights = [i for i in self.weights if i.from_table == table1 and i.to_table == table2]
        if len(weights) == 0:
            return None
        return max(weights, key=lambda x: x.weight)
    
    # This function calculates the COMA weights between 2 tables in the datasets.
    def calculate_coma(self, table1: pd.DataFrame, table2: pd.DataFrame) -> dict:
        matches = valentine_match(table1, table2, Coma())
        for m in matches.items():
            logging.debug(m)
        return matches
    
    def add_relationship(self, table1: str, col1: str, table2: str, col2: str, weight: float):
        self.weights.append(Weight(table1, table2, col1, col2, weight))
        self.weights.append(Weight(table2, table1, col2, col1, weight))

    def remove_relationship(self, table1: str, col1: str, table2: str, col2):
        weights = [i for i in self.weights if i.from_table == table1 
                   and i.to_table == table2 and i.from_col == col1 
                   and i.to_col == col2]
        weights = weights + [i for i in self.weights if i.from_table == table2 and i.to_table == table1 
                             and i.from_col == col2 and i.to_col == col1]
        if len(weights) == 0:
            return
        for i in weights:
            self.weights.remove(i)
    
    def update_relationship(self, table1: str, col1: str, table2: str, col2: str, weight: float):
        self.remove_relationship(table1, col1, table2, col2)
        self.add_relationship(table1, col1, table2, col2, weight)

    def compute_join_paths(self, data_qualtiy_threshold: float = 0.5):
        logging.info("Step 2: Calculating paths")
        emptyPath = Path(begin=self.base_table, joins=[], rank=0)
        emptyPath.id = 0
        self.paths.append(emptyPath)
        self.__stream_feature_selection(queue={self.base_table}, 
                                        path=Path(begin=(self.base_table), joins=[], rank=0, ), 
                                        data_quality_threshhold=data_qualtiy_threshold)

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

    def display_join_paths(self, top_k: None):
        if top_k is None:
            top_k = len(self.paths)
        sorted_paths = sorted(self.paths, key=lambda x: x.rank, reverse=True)[:top_k]
        for index, path in enumerate(sorted_paths):
            graph = networkx.DiGraph()
            labels = {}
            for i in path.joins:
                graph.add_edge(i.from_table, i.to_table)
                labels[(i.from_table, i.to_table)] = i.from_col + " -> " + i.to_col
            pos = networkx.spring_layout(graph)
            networkx.draw(graph, pos=pos, with_labels=True)
            networkx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels, font_size=10)
            plt.title(f"Rank: {path.rank}")
            plt.show()

    def __stream_feature_selection(self, path: Path, queue: set, data_quality_threshhold: float,
                                   previous_queue: List = None):
        if len(queue) == 0:
            return
        if previous_queue is None:
            previous_queue = [[queue.copy().pop()]]
        all_neighbours = set()
        while len(queue) > 0:
            base_node_id = queue.pop()
            self.discovered.add(base_node_id)
            near_weights = self.__get_adjacent_nodes({base_node_id}, 0.5)
            neighbours = [n.to_table for n in near_weights]
            neighbours = list(set(neighbours).difference(self.discovered))
            all_neighbours.update(neighbours)
            for n in neighbours:
                self.discovered.add(n)
                join_keys = self.get_weights_from_and_to_table(base_node_id, n)
                max_val = 0
                highest_join_keys = []
                for jk in join_keys:
                    if jk.weight > max_val:
                        max_val = jk.weight
                        highest_join_keys = [jk]
                    elif jk.weight == max_val:
                        highest_join_keys.append(jk)
                right_df = get_df_with_prefix(highest_join_keys[0].to_table)
                current_queue = list()
                while len(previous_queue) > 0:
                    previous_table_join: [Join] = previous_queue.pop()
                    previous_join = None
                    if previous_table_join == [self.base_table]:
                        previous_join = self.partial_join.copy()
                    else:
                        read_file = self.join_name_mapping[str(previous_table_join)]
                        key_path = pt(self.temp_dir.name) / read_file
                        previous_join = pd.read_parquet(key_path)
                    prop: Weight
                    for prop in highest_join_keys:
                        join_list: [str] = previous_table_join + [prop.to_table]
                        filename = f"{self.base_table.replace('/', '-')}_{str(uuid.uuid4())}.parquet"
                        # TODO: Check if column types are similar.
                        joined_df = pd.merge(left=previous_join, right=right_df, left_on=(prop.get_from_prefix()),
                                             right_on=(prop.get_to_prefix()), how="left")
                        joined_df.to_parquet(pt(self.temp_dir.name) / filename)
                        data_quality = self.data_quality_calculation(joined_df, prop)
                        if data_quality < data_quality_threshhold:
                            continue
                        result = self.streaming_relevance_redundancy(
                            dataframe=joined_df.copy(),
                            new_features=list(right_df.columns),
                            selected_features=self.partial_join_selected_features[str(previous_table_join)],
                        )
                        if result is not None:
                            score, rel_score, red_score, features, rel_discarded, red_discarded = result
                            join = Join(prop.from_table, prop.to_table, 
                                        prop.from_col, prop.to_col, data_quality, {"rel": rel_score, "red": red_score}, 
                                        {"rel": rel_discarded, "red": red_discarded})
                            all_features = self.partial_join_selected_features[str(previous_table_join)]
                            all_features.extend(features)
                            self.partial_join_selected_features[str(join_list)] = all_features
                            path.features = all_features
                            path.rank = score
                        path.add_join(join)
                        path.id = len(self.paths)
                        self.paths.append(deepcopy(path))
                        self.join_name_mapping[str(join_list)] = filename
                        current_queue.append(join_list)
                previous_queue += current_queue
        self.__stream_feature_selection(path, all_neighbours, previous_queue)

    def inspect_join_path(self, path_id):
        path = self.get_path_by_id(self, path_id)
        print(path)

    def materialise_join_path(self, path_id):
        path = self.get_path_by_id(path_id)
        base_df = get_df_with_prefix(self.base_table, self.targetColumn)
        i: Join
        for i in path.joins:
            df = get_df_with_prefix(i.to_table)
            base_df = pd.merge(base_df, df, left_on=i.get_from_prefix(), right_on=i.get_to_prefix(), how="left")
        return base_df
    
    def get_path_by_id(self, path_id) -> Path:
        return [i for i in self.paths if i.id == path_id][0]
    
    def showWeights(self, table1=None, table2=None):
        if table1 is None and table2 is None:
            return self.weights
        elif table1 is not None and table2 is None:
            return self.weights[table1]
        elif table1 is None and table2 is not None:
            return self.weights[table2]
        else:
            return self.weights[table1][table2]

    def add_table(self, dataset: str, table: str):
        logging.warning("This means that the algorithm has to recalculate it's weights and paths.")
        self.extra_tables.append((dataset, table))
        self.data_discover(silent=True)

    def remove_table(self, dataset: str, table: str):
        logging.warning("This means that the algorithm has to recalculate it's weights and paths.")
        if (dataset, table) in self.extra_tables:
            self.extra_tables.remove((dataset, table))
        self.exclude_tables.append((dataset, table))
        self.data_discover(silent=True)

    def add_feature(self, dataset: str, table: str, feature: str):
        # logging.warnin("This means that the algorithm has to recalculate it's weights and paths.")
        self.definite_features.append(dataset + "/" + table + "." + feature)

    def remove_feature(self, dataset: str, table: str, feature: str):
        if (dataset + "/" + table + "." + feature) in self.definite_features:
            self.definite_features.remove(dataset + "/" + table + "." + feature)
        self.exclude_features.append(dataset + "/" + table + "." + feature)

    def streaming_relevance_redundancy(
        self, dataframe: pd.DataFrame, new_features: List[str], selected_features: List[str]
    ) -> Optional[Tuple[float, List[dict]]]:
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False, verbosity=0
        ).fit_transform(X=dataframe, random_state=42, random_seed=42)

        X = df.drop(columns=[self.targetColumn])
        y = df[self.targetColumn]

        features = list(set(X.columns).intersection(set(new_features)))
        # top_feat = len(features) if len(features) < self.top_k else self.top_k

        relevant_features = new_features
        sum_m = 0
        m = 1
        feature_score_relevance, rel_discarded_features = self.rel_red.measure_relevance(
            dataframe=X, new_features=features, target_column=y
        )
        # feature_score_relevance = feature_score_relevance[:top_feat]
        feature_score_relevance = feature_score_relevance
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

        return score, feature_score_relevance, feature_score_redundancy, final_features, rel_discarded_features, red_discarded_features

    def data_quality_calculation(self, joined_df: pd.DataFrame, prop: Weight) -> float:
        total_length = joined_df.shape[0]
        non_nulls = joined_df[prop.get_to_prefix()].count()
        return non_nulls / total_length
    
    def __get_adjacent_nodes(self, nodes: list, threshold: float) -> [Weight]:
        return_list = list()
        for n in nodes:
            for x in self.get_weights_from_table(n):
                return_list.append(x)
        return return_list
    
    def get_path_length(self, path: str) -> int:
        path_tokens = path.split("--")
        return len(path_tokens) - 1
    
    def evaluate_paths(self, top_k_paths: int = 2):
        logging.info("Step 3: Evaluating paths")
        sorted_paths = sorted(self.paths, key=lambda x: x.rank, reverse=True)[:top_k_paths]
        print(sorted_paths)
        for path in tqdm.tqdm(sorted_paths, total=len(sorted_paths)):
            self.evaluate_table(path.id)
        
    def evaluate_table(self, path_id: int):
        path = self.get_path_by_id(path_id)
        base_df = get_df_with_prefix(self.base_table, self.targetColumn)
        i: Join
        for i in path.joins:
            df = get_df_with_prefix(i.to_table)
            base_df = pd.merge(base_df, df, left_on=i.get_from_prefix(), right_on=i.get_to_prefix(), how="left")
        df = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False, enable_text_ngram_features=False, 
            verbosity=0).fit_transform(X=base_df)
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[self.targetColumn]), 
                                                            df[self.targetColumn], test_size=0.2, random_state=10)
        X_train[self.targetColumn] = y_train
        X_test[self.targetColumn] = y_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = TabularPredictor(label=self.targetColumn,
                                         problem_type="binary",
                                         verbosity=0,
                                         path="AutogluonModels/" + "models").fit(
                                             train_data=X_train, hyperparameters={'LR': {'penalty': 'L1'}})
        model_names = predictor.model_names()
        for model in model_names[:-1]:
            result = Result()
            res = predictor.evaluate(X_test, model=model)
            result.accuracy = (res['accuracy'])
            ft_imp = predictor.feature_importance(data=X_test, model=model, feature_stage="original")
            result.feature_importance = dict(zip(list(ft_imp.index), ft_imp["importance"])),
            result.model = model
            result.rank = path.rank
            result.path = path
            self.add_result(result)

    def show_features(self, path_id: int, show_discarded_features: bool = False):
        path = self.get_path_by_id(path_id)
        path.show_table(show_discarded_features)

    def get_weights_from_table(self, table: str):
        return [i for i in self.weights if i.from_table == table]
    
    def get_weights_from_and_to_table(self, from_table, to_table):
        return [i for i in self.weights if i.from_table == from_table and i.to_table == to_table]
    
    def add_result(self, result):
        self.results.append(result)

    def show_result(self, id: str):
        return self.results[id].show_graph


if __name__ == "__main__":
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="credit/table_0_0.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["credit"])
    autofeat.find_relationships(threshold=0.5)
    # autofeat.read_relationships()
    autofeat.display_relationships()
    autofeat.compute_join_paths()
    autofeat.show_features(path_id=1, show_discarded_features=True)
    autofeat.display_join_paths(top_k=2)
    # df = autofeat.materialise_join_path(path_id=1)
    # print(list(df.columns))
    # autofeat.evaluate_paths(top_k_paths=2)
