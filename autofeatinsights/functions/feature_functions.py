import functions.tree_functions as tree_functions


def show_features(autofeat, path_id: int, show_discarded_features: bool = False):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    path.show_table(show_discarded_features)


def adjust_relevance_value(self, path_id, feature_name, value):
    path = tree_functions.get_path_by_id(self, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red["rel"]):
            if item[0] == feature_name:
                new_item = (item[0], value)
                join.rel_red["rel"][index] = new_item


def adjust_redundancy_value(self, path_id, feature_name, value):
    path = tree_functions.get_path_by_id(self, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red["red"]):
            if item[0] == feature_name:
                new_item = (item[0], value)
                join.rel_red["red"][index] = new_item


def adjust_null_ratio(self, path_id, table_name, value):
    path = tree_functions.get_path_by_id(self, path_id)
    if path is None:
        return
    for join in path.joins:
        if join.to_table == table_name:
            join.null_ratio = value


def move_feature_to_discarded(self, path_id, feature_name):
    path = tree_functions.get_path_by_id(self, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red["rel"]):
            if item[0] == feature_name:
                join.rel_red["rel"].pop(index)
                join.rel_red_discarded["rel"].append(item)
        for index, item in enumerate(join.rel_red["red"]):
            if item[0] == feature_name:
                join.rel_red["red"].pop(index)
                join.rel_red_discarded["red"].append(item)


def move_feature_to_selected(self, path_id, feature_name):
    path = tree_functions.get_path_by_id(self, path_id)
    if path is None:
        return
    for join in path.joins:
        for index, item in enumerate(join.rel_red_discarded["rel"]):
            if item[0] == feature_name:
                join.rel_red_discarded["rel"].pop(index)
                join.rel_red["rel"].append(item)
        for index, item in enumerate(join.rel_red_discarded["red"]):
            if item[0] == feature_name:
                join.rel_red_discarded["red"].pop(index)
                join.rel_red["red"].append(item)