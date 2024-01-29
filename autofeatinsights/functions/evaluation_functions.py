import logging
from autofeatinsights.functions.helper_functions import get_df_with_prefix
import tqdm
from autofeatinsights.functions.classes import Result, Join
import autofeatinsights.functions.tree_functions as tree_functions
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


def get_hyperparameters(algorithm: [str] = None) -> list[dict]:
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
    

def evaluate_paths(autofeat, algorithms, top_k_results: int = 5, verbose=False):
    logging.info("Step 3: Evaluating paths")
    autofeat.top_k_results = top_k_results
    sorted_paths = sorted(autofeat.paths, key=lambda x: x.rank, reverse=True)[:top_k_results]
    for path in tqdm.tqdm(sorted_paths, total=len(sorted_paths)):
        evaluate_table(autofeat, algorithms, path.id, verbose=verbose)
    

def evaluate_table(autofeat, algorithms, path_id: int, verbose=False):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    base_df = get_df_with_prefix(autofeat.base_table, autofeat.targetColumn)
    i: Join
    for i in path.joins:
        df = get_df_with_prefix(i.to_table)
        base_df = pd.merge(base_df, df, left_on=i.get_from_prefix(), right_on=i.get_to_prefix(), how="left")
        # Filter selected featurs
    df = AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False, enable_text_ngram_features=False, 
        verbosity=0).fit_transform(X=base_df)
    hyper_parameters = get_hyperparameters(algorithms)
    for model in hyper_parameters:
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[autofeat.targetColumn]), 
                                                            df[autofeat.targetColumn], test_size=0.2, random_state=10)
        X_train[autofeat.targetColumn] = y_train
        X_test[autofeat.targetColumn] = y_test
        predictor = TabularPredictor(label=autofeat.targetColumn,
                                     problem_type="binary",
                                     verbosity=(2 if verbose else 0),
                                     path="AutogluonModels/" + "models").fit(
                                         train_data=X_train, hyperparameters={model})
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
            add_result(autofeat, result)
            if verbose:
                print(result.explain())


def add_result(self, result):
    self.results.append(result)


def show_result(self, id: str):
    return self.results[id].show_graph


def rerun(autofeat):
    if len(autofeat.results) > 0:
        print("Recalculating results...")
        evaluate_paths(autofeat, autofeat.top_k_results)


def explain_result(self, path_id: int, model: str):
    for i in self.results:
        if i.path.id == path_id and i.model == model:
            print(i.explain())
            return
    print("no result found")