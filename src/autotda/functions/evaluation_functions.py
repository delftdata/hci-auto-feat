import tqdm
from src.autotda.functions.tree_functions import JoinTree
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import polars as pl 

hyper_parameters = [
    {"RF": {}},
    {"GBM": {}},
    {"XT": {}},
    {"XGB": {}},
    {'KNN': {}},
    {'LR': {'penalty': 'L1'}},
]

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple


@dataclass
class Result:
    TFD_PATH = "TFD_PATH"
    TFD = "AutoFeat"
    TFD_REL = "AutoFeat_Rel"
    TFD_RED = "AutoFeat_Red"
    TFD_Pearson = "AutoFeat-Pearson-MRMR"
    TFD_Pearson_JMI = "AutoFeat-Pearson-JMI"
    TFD_JMI = "AutoFeat-Spearman-JMI"
    ARDA = "ARDA"
    JOIN_ALL_BFS = "Join_All_BFS"
    JOIN_ALL_BFS_F = "Join_All_BFS_Filter"
    JOIN_ALL_BFS_W = "Join_All_BFS_Wrapper"
    JOIN_ALL_DFS = "Join_All_DFS"
    JOIN_ALL_DFS_F = "Join_All_DFS_Filter"
    JOIN_ALL_DFS_W = "Join_All_DFS_Wrapper"
    BASE = "BASE"

    model: str
    data_path: str = None
    approach: str = None
    data_label: str = None
    join_time: Optional[float] = None
    total_time: float = 0.0
    feature_selection_time: Optional[float] = None
    depth: Optional[int] = None
    accuracy: Optional[float] = None
    train_time: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    join_path_features: List[str] = None
    cutoff_threshold: Optional[float] = None
    redundancy_threshold: Optional[float] = None
    rank: Optional[int] = None
    top_k: int = None

    def __post_init__(self):
        if self.join_time is not None:
            self.total_time += self.join_time

        if self.train_time is not None:
            self.total_time += self.train_time

        if self.feature_selection_time is not None:
            self.total_time += self.feature_selection_time


def get_hyperparameters(algorithm: str = None) -> list[dict]:
    if algorithm is None:
        return hyper_parameters

    if algorithm == 'LR':
        return [{'LR': {'penalty': 'L1'}}]
    
    model = {algorithm: {}}
    if model in hyper_parameters:
        return [model]
    else:
        raise Exception(
            "Unsupported algorithm. Choose one from the list: [RF, GBM, XT, XGB, KNN, LR]."
        )
    

# def evalute_trees(join_trees: Dict[str, Tuple[JoinTree, str]], target_variable: str, ml_model: str, top_k_results: int):
    # autofeat.results = []
    # autofeat.algorithm = algorithm
    # autofeat.top_k_results = top_k_results
    # sorted_trees = sorted(autofeat.trees, key=lambda x: x.rank, reverse=True)[:top_k_results]
    
    # if verbose:
    #     print("Evaluating join trees...")
    # total_number = len(join_trees.keys())
    
    # for k, v in tqdm.tqdm(join_trees.items(), total=total_number):
    #     join_tree, _ = v
    #     evaluate_join_tree(autofeat, algorithm)
    
    # if explain:
    #     best_result = sorted(autofeat.results, key=lambda x: x.accuracy, reverse=True)[0]
    #     print(f"AutoFeat creates {len(autofeat.trees)} join trees: the best performing join tree is tree: {best_result.tree.id}")
    #     tree = tree_functions.get_tree_by_id(autofeat, best_result.tree.id)
    #     print(tree.explain())
    #     autofeat.show_features(best_result.tree.id)
    #     print(best_result.explain())


def evaluate_join_tree(join_trees: Dict[str, Tuple[JoinTree, str]], join_tree_id: str, target_variable: str, ml_model: str):
    results = []

    join_tree: JoinTree
    join_tree, join_filename = join_trees[join_tree_id]
    
    columns_to_select = join_tree.features.copy()
    columns_to_select.append(target_variable)
    columns_to_drop = set(join_tree.features).intersection(set(join_tree.join_keys))
    materialised_join = pl.read_parquet(join_filename).select(columns_to_select).drop(columns_to_drop)

    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False, 
        verbosity=0).fit_transform(X=materialised_join.to_pandas())
    
    hyper_parameters = get_hyperparameters(ml_model)
    for hyperparam in hyper_parameters:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_variable]),
                                                            df[target_variable], test_size=0.2, random_state=10)        
        X_train[target_variable] = y_train
        X_test[target_variable] = y_test
        predictor = TabularPredictor(label=target_variable,
                                     problem_type="binary",
                                     verbosity=0,
                                     path="AutogluonModels/" + "models").fit(
                                         train_data=X_train, hyperparameters=hyperparam)
        model_names = predictor.get_model_names()
        for model in model_names[:-1]:
            evaluation_result = predictor.evaluate(X_test, model=model)
            ft_imp = predictor.feature_importance(data=X_train, model=model, feature_stage="original")            
            
            result = Result(model=model)
            result.accuracy = evaluation_result['accuracy']
            result.feature_importance = dict(zip(list(ft_imp.index), ft_imp["importance"]))
            result.rank = join_tree.rank
            result.join_path_features = join_tree.features
            results.append(result)

    return results


# def add_result(self, result):
#     self.results.append(result)


# def show_result(self, id: str):
#     return self.results[id].show_graph


# def get_best_result(autofeat):
#     return max(autofeat.results, key=lambda x: x.accuracy)


# def rerun(autofeat):
#     if len(autofeat.results) > 0:
#         print("Recalculating results...")
#         evalute_trees(autofeat, autofeat.algorithm, top_k_results=autofeat.top_k_results)


# def explain_result(self, tree_id: int, model: str):
#     for i in self.results:
#         if i.tree.id == tree_id and i.model == model:
#             print(i.explain())
#             return
#     print("no result found")