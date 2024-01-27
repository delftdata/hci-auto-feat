import functions.tree_functions as tree_functions


def show_features(autofeat, path_id: int, show_discarded_features: bool = False):
    path = tree_functions.get_path_by_id(autofeat, path_id)
    path.show_table(show_discarded_features)