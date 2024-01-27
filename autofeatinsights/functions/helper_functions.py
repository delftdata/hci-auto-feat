
import xxhash
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from ITMO_FS.filters.multivariate import CMIM, MRMR
from ITMO_FS.utils.information_theory import entropy, conditional_mutual_information
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import rankdata


xxh = xxhash.xxh64(seed=42)


class RelevanceRedundancy:

    def __init__(self, target_column: str, jmi: bool = False, pearson: bool = False):
        self.target_column = target_column
        self.target_entropy = None
        self.dataframe_entropy = {}
        self.dataframe_conditional_entropy = {}
        self.jmi = jmi
        self.pearson = pearson

    def measure_relevance(
        self, dataframe: pd.DataFrame, new_features: List[str], target_column: pd.Series
    ) -> List[tuple]:

        if self.target_column in new_features:
            new_features.remove(self.target_column)

        new_common_features = list(set(dataframe.columns).intersection(set(new_features)))
        if len(new_common_features) == 0:
            return []

        if self.pearson:
            correlation_score = abs(
                pearson_correlation(np.array(dataframe[new_common_features]), np.array(target_column))
            )
        else:
            correlation_score = abs(
                spearman_correlation(np.array(dataframe[new_common_features]), np.array(target_column))
            )

        final_feature_scores_rel = []
        discarded_features = []
        for value, name in list(zip(correlation_score, new_common_features)):
            # value 0 means that the features are completely redundant
            # value 1 means that the features are perfectly correlated, which is still undesirable
            if 0 < value < 1:
                final_feature_scores_rel.append((name, value))
            else:
                discarded_features.append((name, value))

        if len(final_feature_scores_rel) == 0:
            return []

        return sorted(final_feature_scores_rel, key=lambda s: s[1], reverse=True), discarded_features

    def measure_redundancy(
        self,
        dataframe: pd.DataFrame,
        selected_features: List[str],
        relevant_features: List[str],
        target_column: pd.Series,
    ) -> List[tuple]:
        """
        Measure the redundancy of selected features with respect to the relevant features.

        Args:
            dataframe (pd.DataFrame): The input dataframe.
            selected_features (List[str]): The list of selected features.
            relevant_features (List[str]): The list of relevant features.
            target_column (pd.Series): The target column.

        Returns:
            List[tuple]: A list of tuples containing the name of the feature and its redundancy score.
        """
        selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
        new_features_int = [i for i, value in enumerate(dataframe.columns) if value in relevant_features]

        est = KBinsDiscretizer(strategy='uniform', encode='ordinal', subsample=1000)
        try:
            discr_dataframe = est.fit_transform(dataframe)
        except ValueError:
            discr_dataframe = dataframe

        relevance = np.apply_along_axis(
            self.cached_mutual_information, 0, np.array(discr_dataframe)[:, np.array(new_features_int)], target_column
        )
        redundancy = np.vectorize(
            lambda free_feature: np.sum(
                np.apply_along_axis(
                    self.cached_mutual_information,
                    0,
                    np.array(discr_dataframe)[:, np.array(selected_features_int)],
                    np.array(discr_dataframe)[:, free_feature],
                )
            )
        )(new_features_int)

        if self.jmi:
            cond_dependency = np.vectorize(
                lambda free_feature: np.sum(
                    np.apply_along_axis(
                        self.cached_conditional_mutual_information,
                        0,
                        np.array(dataframe)[:, np.array(selected_features_int)],
                        np.array(dataframe)[:, free_feature],
                        target_column,
                    )
                )
            )(np.array(new_features_int))
            mrmr_scores = (
                relevance
                - (1 / np.array(selected_features_int).size) * redundancy
                + (1 / np.array(selected_features_int).size) * cond_dependency
            )
        else:
            mrmr_scores = relevance - (1 / np.array(selected_features_int).size) * redundancy

        if np.all(np.array(mrmr_scores) == 0):
            return []

        max_mrmr = np.max(mrmr_scores)
        min_mrmr = np.min(mrmr_scores)
        if max_mrmr == min_mrmr:
            normalised_scores = (mrmr_scores - min_mrmr) / min_mrmr
        else:
            normalised_scores = (mrmr_scores - min_mrmr) / (max_mrmr - min_mrmr)

        feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
        final_feature_scores_mrmr = [(name, value) for name, value in feature_scores if value > 0]
        discarded_red_scores = list(set(feature_scores) - set(final_feature_scores_mrmr))
        
        return sorted(final_feature_scores_mrmr, key=lambda s: s[1], reverse=True), discarded_red_scores

    def measure_relevance_and_redundancy(
        self, dataframe: pd.DataFrame, selected_features: List[str], new_features: List[str], target_column: pd.Series
    ) -> Tuple[list, list]:
        final_feature_scores_rel = self.measure_relevance(dataframe, new_features, target_column)
        final_feature_scores_mrmr = self.measure_redundancy(
            dataframe, selected_features, list(dict(final_feature_scores_rel).keys()), target_column
        )
        return final_feature_scores_rel, final_feature_scores_mrmr

    def cached_mutual_information(self, x, y):
        xxh.update(x)
        xxh.update(np.array(y))
        h = xxh.intdigest()
        xxh.reset()
        if h in self.dataframe_conditional_entropy:
            cond_entropy = self.dataframe_conditional_entropy[h]
        else:
            cond_entropy = conditional_entropy(x, np.array(y))
            self.dataframe_conditional_entropy[h] = cond_entropy

        xxh.update(np.array(y))
        # h_entropy = hash(str(y))
        h_entropy = xxh.intdigest()
        xxh.reset()

        if h_entropy in self.dataframe_entropy:
            entr = self.dataframe_entropy[h_entropy]
        else:
            entr = entropy(y)
            self.dataframe_entropy[h_entropy] = entr
        return entr - cond_entropy

    def cached_conditional_mutual_information(self, x, y, z):
        h1 = hash(str(list(zip(x, z))))
        if h1 in self.dataframe_entropy:
            entropy_xz = self.dataframe_entropy[h1]
        else:
            entropy_xz = entropy(list(zip(x, z)))
            self.dataframe_entropy[h1] = entropy_xz

        h2 = hash(str(list(zip(y, z))))
        if h2 in self.dataframe_entropy:
            entropy_yz = self.dataframe_entropy[h2]
        else:
            entropy_yz = entropy(list(zip(y, z)))
            self.dataframe_entropy[h2] = entropy_yz

        h3 = hash(str(list(zip(x, y, z))))
        if h3 in self.dataframe_entropy:
            entropy_xyz = self.dataframe_entropy[h3]
        else:
            entropy_xyz = entropy(list(zip(x, y, z)))
            self.dataframe_entropy[h3] = entropy_xyz

        h4 = hash(str(z))
        if h4 in self.dataframe_entropy:
            entropy_z = self.dataframe_entropy[h4]
        else:
            entropy_z = entropy(z)
            self.dataframe_entropy[h4] = entropy_z

        return entropy_xz + entropy_yz - entropy_xyz - entropy_z


def measure_relevance(
    dataframe: pd.DataFrame, feature_names: List[str], target_column: pd.Series
) -> Tuple[Optional[list], list]:
    common_features = list(set(dataframe.columns).intersection(set(feature_names)))

    if len(common_features) == 0:
        return None, []

    features = dataframe[common_features]
    # scores = information_gain(np.array(features), np.array(target_column)) / max(entropy(features),
    #                                                                              entropy(target_column))
    scores = abs(spearman_correlation(np.array(features), np.array(target_column)))

    final_feature_scores = []
    final_features = []
    for value, name in list(zip(scores, common_features)):
        # value 0 means that the features are completely redundant
        # value 1 means that the features are perfectly correlated, which is still undesirable
        if 0 < value < 1:
            final_feature_scores.append((name, value))
            final_features.append(name)

    return final_feature_scores, final_features


def measure_conditional_redundancy(
    dataframe: pd.DataFrame,
    selected_features: List[str],
    new_features: List[str],
    target_column: pd.Series,
    conditional_redundancy_threshold: float = 0.5,
) -> Tuple[Optional[list], list]:
    selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
    new_features_int = [i for i, value in enumerate(dataframe.columns) if value in new_features]

    scores = CMIM(
        np.array(selected_features_int), np.array(new_features_int), np.array(dataframe), np.array(target_column)
    )

    if np.all(np.array(scores) == 0):
        return None, []
    # normalise
    normalised_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
    final_feature_scores = [(name, value) for name, value in feature_scores if value > 0]
    final_feature_names = [feat for feat, _ in final_feature_scores]

    return final_feature_scores, final_feature_names


def measure_redundancy(dataframe, feature_group: List[str], target_column) -> Tuple[Optional[list], List[str]]:
    if len(feature_group) == 1 or len(feature_group) == 0:
        return None, feature_group

    scores = np.vectorize(
        lambda feature: np.mean(
            np.apply_along_axis(
                lambda x, y, z: conditional_mutual_information(y, z, x),
                0,
                np.array(dataframe[np.setdiff1d(feature_group, feature)]),
                np.array(dataframe[feature]),
                np.array(target_column),
            )
        )
    )(feature_group)
    feature_scores = list(zip(np.array(feature_group), scores))
    final_feature_scores = [(name, value) for name, value in feature_scores if value > 0]
    final_feature_names = [feat for feat, _ in final_feature_scores]

    return final_feature_scores, final_feature_names


def measure_joint_mutual_information(
    dataframe: pd.DataFrame, selected_features: List[str], new_features: List[str], target_column: pd.Series
) -> Tuple[Optional[list], list]:
    selected_features_int = [i for i, value in enumerate(dataframe.columns) if value in selected_features]
    new_features_int = [i for i, value in enumerate(dataframe.columns) if value in new_features]

    scores = MRMR(
        np.array(selected_features_int), np.array(new_features_int), np.array(dataframe), np.array(target_column)
    )

    if np.all(np.array(scores) == 0):
        return None, []

    if np.max(scores) == np.min(scores):
        normalised_scores = (scores - np.min(scores)) / np.max(scores)
    else:
        normalised_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    feature_scores = list(zip(np.array(dataframe.columns)[np.array(new_features_int)], normalised_scores))
    final_feature_scores = [(name, value) for name, value in feature_scores if value > 0]
    final_feature_names = [feat for feat, _ in final_feature_scores]

    return final_feature_scores, final_feature_names


def get_df_with_prefix(node_id: str, targetColumn=None):
    prefix = node_id + "."
    if targetColumn:
        dataframe = pd.read_csv('data/benchmark/' + node_id).set_index(targetColumn) \
            .add_prefix(prefix).reset_index()
    else:
        dataframe = pd.read_csv('data/benchmark/' + node_id).add_prefix(prefix)
    return dataframe


def conditional_entropy(x_j: np.ndarray, y: np.ndarray) -> np.float64:
    # Find unique values and their corresponding indices
    unique_x, inverse_x = np.unique(x_j, return_inverse=True)
    unique_y, inverse_y = np.unique(y, return_inverse=True)

    # Create a contingency table
    contingency_table = np.zeros((len(unique_x), len(unique_y)), dtype=int)
    np.add.at(contingency_table, (inverse_x, inverse_y), 1)

    # Compute counts for each unique value in x_j
    counts_x = np.sum(contingency_table, axis=1)

    # Calculate the probabilities P(Y|X)
    prob_y_given_x = contingency_table / counts_x[:, None]

    # Compute entropy for each unique value in x_j
    part_entropies = -np.sum(elog(prob_y_given_x), axis=1)

    # Compute the weighted average of the part_entropies
    entropy = np.sum(counts_x / len(x_j) * part_entropies)

    return entropy


def elog(value: np.ndarray) -> np.ndarray:
    log_values = np.zeros_like(value)
    mask = value > 0
    log_values[mask] = np.log(value[mask])
    return value * log_values


def pearson_correlation(x, y):
    x_dev = x - np.mean(x, axis=0)
    y_dev = y - np.mean(y)
    sq_dev_x = x_dev * x_dev
    sq_dev_y = y_dev * y_dev
    sum_dev = y_dev.T.dot(x_dev).reshape((x.shape[1],))
    denominators = np.sqrt(np.sum(sq_dev_y) * np.sum(sq_dev_x, axis=0))

    results = np.array(
        [(sum_dev[i] / denominators[i]) if denominators[i] > 0.0 else 0 for i
         in range(len(denominators))])
    return results


def spearman_correlation(x, y):
    n = x.shape[0]
    if n < 2:
        raise ValueError("The input should contain more than 1 sample")

    x_ranks = np.apply_along_axis(rankdata, 0, x)
    y_ranks = rankdata(y)

    return pearson_correlation(x_ranks, y_ranks)
