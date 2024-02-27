from src.autofeatinsights.functions.helper_functions import get_df_with_prefix
import tqdm
from src.autofeatinsights.functions.classes import Result, Join
import src.autofeatinsights.functions.tree_functions as tree_functions
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd

hyper_parameters = [
    {"RF": {}},
    {"GBM": {}},
    {"XT": {}},
    {"XGB": {}},
    {'KNN': {}},
    {'LR': {'penalty': 'L1'}},
]


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
    

def evalute_trees(autofeat, algorithm, top_k_results: int = 5,
                  explain=False, verbose=True):
    autofeat.results = []
    autofeat.top_k_results = top_k_results
    sorted_trees = sorted(autofeat.trees, key=lambda x: x.rank, reverse=True)[:top_k_results]
    if verbose:
        print("Evaluating join trees...")
    for tree in tqdm.tqdm(sorted_trees, total=len(sorted_trees)):
        evaluate_table(autofeat, algorithm, tree.id, verbose=verbose, multiple=True)
    if explain:
        best_result = sorted(autofeat.results, key=lambda x: x.accuracy, reverse=True)[0]
        print(f"AutoFeat creates {len(autofeat.trees)} join trees: the best performing join tree is tree: {best_result.tree.id}")
        tree = tree_functions.get_tree_by_id(autofeat, best_result.tree.id)
        print(tree.explain())
        autofeat.show_features(best_result.tree.id)
        print(best_result.explain())


def evaluate_table(autofeat, algorithm, tree_id: int, verbose=False, multiple=False):
    if not multiple:
        autofeat.results = []
    tree = tree_functions.get_tree_by_id(autofeat, tree_id)
    base_df = get_df_with_prefix(autofeat.base_table, autofeat.targetColumn)
    i: Join
    for i in tree.joins:
        df = get_df_with_prefix(i.to_table).groupby(i.get_to_prefix()).sample(n=1, random_state=42)
        base_df = pd.merge(base_df, df, left_on=i.get_from_prefix(), right_on=i.get_to_prefix(), how="left")
    base_df = base_df[tree.features + [autofeat.targetColumn]]
    columns_to_drop = set(base_df.columns).intersection(set(tree.join_keys))
    base_df.drop(columns=list(columns_to_drop), inplace=True)
    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False, 
        verbosity=0).fit_transform(X=base_df)
    hyper_parameters = get_hyperparameters(algorithm)
    for hyperparam in hyper_parameters:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[autofeat.targetColumn]),
                                                            df[autofeat.targetColumn], test_size=0.2, random_state=10)
        
        X_train[autofeat.targetColumn] = y_train
        X_test[autofeat.targetColumn] = y_test
        predictor = TabularPredictor(label=autofeat.targetColumn,
                                     problem_type="binary",
                                     verbosity=0,
                                     path="AutogluonModels/" + "models").fit(
                                         train_data=X_train, hyperparameters=hyperparam)
        model_names = predictor.get_model_names()
        for model in model_names[:-1]:
            result = Result()
            res = predictor.evaluate(X_test, model=model)
            result.accuracy = res['accuracy']
            ft_imp = predictor.feature_importance(data=X_test, model=model, feature_stage="original")
            result.feature_importance = dict(zip(list(ft_imp.index), ft_imp["importance"]))
            result.model = list(hyperparam.keys())[0]
            result.model_full_name = model
            result.rank = tree.rank
            result.tree = tree
            autofeat.results.append(result)


def add_result(self, result):
    self.results.append(result)


def show_result(self, id: str):
    return self.results[id].show_graph


def get_best_result(autofeat):
    return max(autofeat.results, key=lambda x: x.accuracy)


def rerun(autofeat):
    if len(autofeat.results) > 0:
        print("Recalculating results...")
        evalute_trees(autofeat, autofeat.top_k_results)


def explain_result(self, tree_id: int, model: str):
    for i in self.results:
        if i.tree.id == tree_id and i.model == model:
            print(i.explain())
            return
    print("no result found")