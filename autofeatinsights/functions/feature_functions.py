import autofeatinsights.functions.tree_functions as tree_functions
import autofeatinsights.functions.evaluation_functions as evaluation_functions


def show_features(autofeat, path_id: int, show_discarded_features: bool = False):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    path.show_table(show_discarded_features)


def adjust_relevance_value(autofeat, path_id, feature_name, value):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red["rel"]):
            if item[0] == feature_name:
                new_item = (item[0], value)
                join.rel_red["rel"][index] = new_item
    evaluation_functions.rerun(autofeat)


def adjust_redundancy_value(autofeat, path_id, feature_name, value):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red["red"]):
            if item[0] == feature_name:
                new_item = (item[0], value)
                join.rel_red["red"][index] = new_item
    evaluation_functions.rerun(autofeat)


def adjust_non_null_ratio(autofeat, path_id, table_name, value):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    if path is None:
        return
    for join in path.joins:
        if join.to_table == table_name:
            join.non_null_ratio = value
    evaluation_functions.rerun(autofeat)


def move_features_to_discarded(autofeat, path_id, features):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red["rel"]):
            if item[0] in features:
                join.rel_red["rel"].pop(index)
                join.rel_red_discarded["rel"].append(item)
        for index, item in enumerate(join.rel_red["red"]):
            if item[0] in features:
                join.rel_red["red"].pop(index)
                join.rel_red_discarded["red"].append(item)
    new_features = set(path.features).difference(set(features))
    path.features = list(new_features)
    # evaluation_functions.rerun(autofeat)


def move_features_to_selected(autofeat, path_id, features):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red_discarded["rel"]):
            if item[0] in features:
                join.rel_red_discarded["rel"].pop(index)
                join.rel_red["rel"].append(item)
        for index, item in enumerate(join.rel_red_discarded["red"]):
            if item[0] in features:
                join.rel_red_discarded["red"].pop(index)
                join.rel_red["red"].append(item)
    path.features.extend(features)
    # evaluation_functions.rerun(autofeat)
