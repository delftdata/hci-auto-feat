import autofeatinsights.functions.tree_functions as tree_functions
import autofeatinsights.functions.evaluation_functions as evaluation_functions


def show_features(autofeat, tree_id: int, show_discarded_features: bool = False):
    tree = tree_functions.get_tree_by_id(autofeat, tree_id)
    tree.show_table(show_discarded_features)


def adjust_relevance_value(autofeat, tree_id, feature_name, value):
    tree = tree_functions.get_tree_by_id(autofeat, tree_id)
    if tree is None:
        return
    for join in tree.joins:
        for index, item in enumerate(join.rel_red["rel"]):
            if item[0] == feature_name:
                new_item = (item[0], value)
                join.rel_red["rel"][index] = new_item
    evaluation_functions.rerun(autofeat)


def adjust_redundancy_value(autofeat, tree_id, feature_name, value):
    tree = tree_functions.get_tree_by_id(autofeat, tree_id=tree_id)
    if tree is None:
        return
    for join in tree.joins:
        for index, item in enumerate(join.rel_red["red"]):
            if item[0] == feature_name:
                new_item = (item[0], value)
                join.rel_red["red"][index] = new_item
    evaluation_functions.rerun(autofeat)


def adjust_non_null_ratio(autofeat, tree_id, table_name, value):
    tree = tree_functions.get_tree_by_id(autofeat, tree_id)
    if tree is None:
        return
    for join in tree.joins:
        if join.to_table == table_name:
            join.non_null_ratio = value
    evaluation_functions.rerun(autofeat)


def move_features_to_discarded(autofeat, tree_id, features):
    tree = tree_functions.get_tree_by_id(autofeat, tree)
    if tree is None:
        return
    for join in tree.joins:
        for index, item in enumerate(join.rel_red["rel"]):
            if item[0] in features:
                join.rel_red["rel"].pop(index)
                join.rel_red_discarded["rel"].append(item)
        for index, item in enumerate(join.rel_red["red"]):
            if item[0] in features:
                join.rel_red["red"].pop(index)
                join.rel_red_discarded["red"].append(item)
    new_features = set(tree.features).difference(set(features))
    tree.features = list(new_features)
    # evaluation_functions.rerun(autofeat)


def move_features_to_selected(autofeat, tree_id, features):
    tree = tree_functions.get_tree_by_id(autofeat, tree_id)
    if tree is None:
        return
    for join in tree.joins:
        for index, item in enumerate(join.rel_red_discarded["rel"]):
            if item[0] in features:
                join.rel_red_discarded["rel"].pop(index)
                join.rel_red["rel"].append(item)
        for index, item in enumerate(join.rel_red_discarded["red"]):
            if item[0] in features:
                join.rel_red_discarded["red"].pop(index)
                join.rel_red["red"].append(item)
    tree.features.extend(features)
    # evaluation_functions.rerun(autofeat)
