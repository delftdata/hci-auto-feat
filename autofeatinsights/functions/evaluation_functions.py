import logging
from functions.helper_functions import get_df_with_prefix
import tqdm
from functions.classes import Result, Join
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
import pandas as pd


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


def add_result(self, result):
    self.results.append(result)


def show_result(self, id: str):
    return self.results[id].show_graph