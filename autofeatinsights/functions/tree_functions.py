import logging
from functions.classes import Path, Join, Weight
import networkx
from matplotlib import pyplot as plt
from pathlib import Path as pt
import pandas as pd
from copy import deepcopy
from functions.helper_functions import get_df_with_prefix
import uuid
from typing import Tuple, Optional
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


def compute_join_paths(autofeat, null_ratio_threshold: float = 0.5):
    logging.info("Step 2: Calculating paths")
    emptyPath = Path(begin=autofeat.base_table, joins=[], rank=0)
    emptyPath.id = 0
    autofeat.paths.append(emptyPath)
    stream_feature_selection(autofeat=autofeat, queue={autofeat.base_table}, 
                             path=Path(begin=(autofeat.base_table), joins=[], rank=0, ), 
                             null_ratio_threshold=null_ratio_threshold)


def stream_feature_selection(autofeat, path: Path, queue: set, null_ratio_threshold: float, 
                             previous_queue: list = None):
    if len(queue) == 0:
        return
    if previous_queue is None:
        previous_queue = [[queue.copy().pop()]]
    all_neighbours = set()
    while len(queue) > 0:
        base_node_id = queue.pop()
        autofeat.discovered.add(base_node_id)
        near_weights = get_adjacent_nodes(autofeat, {base_node_id}, 0.5)
        neighbours = [n.to_table for n in near_weights]
        neighbours = list(set(neighbours).difference(autofeat.discovered))
        all_neighbours.update(neighbours)
        for n in neighbours:
            autofeat.discovered.add(n)
            join_keys = autofeat.get_weights_from_and_to_table(base_node_id, n)
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
                if previous_table_join == [autofeat.base_table]:
                    previous_join = autofeat.partial_join.copy()
                else:
                    read_file = autofeat.join_name_mapping[str(previous_table_join)]
                    key_path = pt(autofeat.temp_dir.name) / read_file
                    previous_join = pd.read_parquet(key_path)
                prop: Weight
                for prop in highest_join_keys:
                    join_list: [str] = previous_table_join + [prop.to_table]
                    filename = f"{autofeat.base_table.replace('/', '-')}_{str(uuid.uuid4())}.parquet"
                    if previous_join[prop.get_from_prefix()].dtype != right_df[prop.get_to_prefix()].dtype:
                        continue
                    joined_df = pd.merge(left=previous_join, right=right_df, left_on=(prop.get_from_prefix()),
                                         right_on=(prop.get_to_prefix()), how="left")
                    joined_df.to_parquet(pt(autofeat.temp_dir.name) / filename)
                    null_ratio = null_ratio_calculation(autofeat, joined_df, prop)
                    if null_ratio < null_ratio_threshold:
                        continue
                    result = streaming_relevance_redundancy(
                        autofeat,
                        dataframe=joined_df.copy(),
                        new_features=list(right_df.columns),
                        selected_features=autofeat.partial_join_selected_features[str(previous_table_join)],
                    )
                    if result is not None:
                        score, rel_score, red_score, features, rel_discarded, red_discarded = result
                        join = Join(prop.from_table, prop.to_table, 
                                    prop.from_col, prop.to_col, null_ratio, {"rel": rel_score, "red": red_score}, 
                                    {"rel": rel_discarded, "red": red_discarded})
                        all_features = autofeat.partial_join_selected_features[str(previous_table_join)]
                        all_features.extend(features)
                        autofeat.partial_join_selected_features[str(join_list)] = all_features
                        path.features = all_features
                        path.rank = score
                    path.add_join(join)
                    path.id = len(autofeat.paths)
                    autofeat.paths.append(deepcopy(path))
                    autofeat.join_name_mapping[str(join_list)] = filename
                    current_queue.append(join_list)
            previous_queue += current_queue
    stream_feature_selection(autofeat, path, all_neighbours, null_ratio_threshold, previous_queue)


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


def streaming_relevance_redundancy(
    self, dataframe: pd.DataFrame, new_features: list[str], selected_features: list[str]
) -> Optional[Tuple[float, list[dict]]]:
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

    return (score, feature_score_relevance, feature_score_redundancy, final_features, 
            rel_discarded_features, red_discarded_features)


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


def get_adjacent_nodes(self, nodes: list, threshold: float) -> [Weight]:
    return_list = list()
    for n in nodes:
        for x in self.get_weights_from_table(n):
            return_list.append(x)
    return return_list


def null_ratio_calculation(self, joined_df: pd.DataFrame, prop: Weight) -> float:
    total_length = joined_df.shape[0]
    non_nulls = joined_df[prop.get_to_prefix()].count()
    return non_nulls / total_length