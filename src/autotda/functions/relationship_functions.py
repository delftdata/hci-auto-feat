from enum import Enum
from glob import glob
import os
from typing import Dict, List

from autotda.data_models.dataset_model import ALL_DATASETS, Dataset, filter_datasets, init_datasets
import pandas as pd
from joblib import Parallel, delayed
import itertools
import tqdm
import seaborn
from autotda.config import CONNECTIONS, DATA_FOLDER, RELATIONS_FOLDER
from valentine.algorithms import Coma, JaccardDistanceMatcher
from valentine import valentine_match
import matplotlib.pyplot as plt
from tabulate import tabulate
# import autotda.functions.tree_functions as tree_functions


MATCHER = {
    "Jaccard": JaccardDistanceMatcher,
    "Coma": Coma
}

class Relation:
    def __init__(self, from_table, to_table, from_col, to_col, similarity):
        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.similarity = similarity

    def __eq__(self, other):
        if self.from_table != other.from_table:
            return False 
        if self.to_table != other.to_table:
            return False
        if self.from_col != other.from_col: 
            return False 
        if self.to_col != other.to_col:
            return False 
        if self.similarity != other.similarity:
            return False
        return True 
    
    def __str__(self) -> str:
        return str(vars(self))


class DatasetDiscovery:
    matcher: str
    relations_filename: str = None
    similarity_threshold: float = 0.65
    relations: List[Relation] = None
    table_repository: List[str]
    data_repository: List[Dataset]

    def __init__(self, matcher: str, data_repositories: List[str]=None):
        self.matcher = matcher
        self.data_repository = ALL_DATASETS
        if data_repositories:
            self.set_dataset_repository(dataset_repository=data_repositories)
        self.set_table_repository()

    def set_relations_filename(self):
        filename = '_'.join(list(map(lambda f: f.base_table_label, self.data_repository)))
        self.relations_filename = f"{filename}_{self.similarity_threshold}_{self.matcher}_weights.csv"
        

    def set_dataset_repository(self, dataset_repository: List[str]):
        """
        Sets the dataset repository for the AutofeatClass object.

        Args:
            dataset_repository (List[str]): A list of dataset paths.

        Returns:
            None

        """
        self.data_repository = filter_datasets(dataset_labels=dataset_repository)

        self.set_relations_filename()


    def set_table_repository(self):
        """
        Retrieves the tables from the repository.

        Returns:
            tables (list): A list of table paths.
        """
        tables = []

        for dataset in self.data_repository:
            files = glob(f"{DATA_FOLDER}/{dataset.base_table_path}/*.csv", recursive=True)
            files = [f for f in files if CONNECTIONS not in f and f.endswith("csv")]

        for f in files:
            table_path = f.partition(f"{DATA_FOLDER}/")[2]
            table_name = table_path.split("/")[-1]
            tables.append(table_path)

        self.table_repository = tables    

    def set_similarity_threshold(self, threshold):
        self.similarity_threshold = threshold

        self.set_relations_filename()


    # def find_relationships(autofeat, relationship_threshold: float = 0.5, matcher: str = "coma", explain=False, verbose=True, use_cache=True):
    def find_relationships(self, use_cache=True):
        # tables = autofeat.get_tables_repository()
        # tables.extend(autofeat.extra_tables)
        # tables = [i for i in tables if i not in autofeat.exclude_tables]
        # if explain:
        #     print(f" 1. AutoFeat computes the relationships between {len(tables)} tables from the datasets: {autofeat.datasets}."
        #         + f" extra tables: {autofeat.extra_tables} and excludes: {autofeat.exclude_tables}" 
        #         + f" repository, using {matcher} similarity score with a threshold of {relationship_threshold} "
        #         + f"(i.e., all the relationships with a similarity < {relationship_threshold} will be discarded).")
        # if verbose:
        #     print("Calculating relationships...")

        if os.path.isfile(RELATIONS_FOLDER / self.relations_filename):
            df = pd.read_csv(RELATIONS_FOLDER / self.relations_filename)
            self.relations = [Relation(row.from_table, row.to_table, row.from_col, row.to_col, row.similarity) for index, row in df.iterrows()]  
            return

        # manager = Manager()
        # temp = manager.list()

        temp = [] 
        
        def profile(table_pair):
            (table1, table2) = table_pair
            df1 = pd.read_csv(DATA_FOLDER / table1)
            df2 = pd.read_csv(DATA_FOLDER / table2)

            valentine_matcher = None
            if self.matcher in MATCHER:
                valentine_matcher = MATCHER[self.matcher]
            else:
                KeyError("The matcher does not exist.")
                return
            matches = valentine_match(df1, df2, valentine_matcher())

            for item in matches.items():
                ((_, col_from), (_, col_to)), similarity = item
                if similarity > self.similarity_threshold:
                    temp.append(Relation(table1, table2, col_from, col_to, similarity))
                    temp.append(Relation(table2, table1, col_to, col_from, similarity))

        # If the name is too long 
        # autofeat.weight_string_mapping = {}
        # for t in tables:
        #     if len(t) > 20:
        #         new_string = t.split("/")[0] + "/" + t.split("/")[1][:3] + "..." + t.split("/")[1][-7:]
        #         autofeat.weight_string_mapping[t] = new_string
        #     else:
        #         autofeat.weight_string_mapping[t] = t

        # Parallel(n_jobs=-1)(delayed(profile)(combination)
        #                     for combination in tqdm.tqdm(itertools.combinations(self.table_repository, r=2)))

        for combination in tqdm.tqdm(itertools.combinations(self.table_repository, r=2)):
            profile(combination)

        self.relations = temp

        if use_cache:
            pd.DataFrame.from_records([vars(s) for s in temp]).to_csv(RELATIONS_FOLDER / self.relations_filename, index=False)

def get_adjacent_nodes(relations: List[Relation], node: str) -> Dict[str, List[Relation]]:
    neighbours = {}
    for relation in relations:
        if relation.from_table == node:
            if relation.to_table in neighbours.keys():
                if relation not in neighbours[relation.to_table]:
                    neighbours[relation.to_table].append(relation)
            else: 
                neighbours[relation.to_table] = [relation]

        # if relation.to_table == node:
        #     if relation.from_table in neighbours.keys():
        #         if relation not in neighbours[relation.from_table]:
        #             neighbours[relation.from_table].append(relation)
        #     else:
        #         neighbours[relation.from_table] = [relation]

    for n in neighbours.keys():
        sorted(neighbours[n], key = lambda x : x.similarity)
    return neighbours
    
    # def read_relationships(self):
    #     self.weights = []
    #     f = open(file_path, "r")
    #     stringlist = f.read().split(",")
    #     for i in stringlist:
    #         if i != "":
    #             table1, table2, col1, col2, weight = i.split("--")
    #             self.weights.append(Weight(table1, table2, col1, col2, float(weight)))
    #     f.close()
        # tables = self.get_tables_repository()
        # self.weight_string_mapping = {}
        # for t in tables:
        #     if len(t) > 20:
        #         new_string = t.split("/")[0] + "/" + t.split("/")[1][:3] + "..." + t.split("/")[1][-7:]
        #         self.weight_string_mapping[t] = new_string
        #     else:
        #         self.weight_string_mapping[t] = t    


# def add_relationship(autofeat, table1: str, col1: str, table2: str, col2: str, weight: float, update: bool = True):
#     autofeat.weights.append(Weight(table1, table2, col1, col2, weight))
#     autofeat.weights.append(Weight(table2, table1, col2, col1, weight))
#     if update:
#         tree_functions.rerun(autofeat)


# def remove_relationship(autofeat, table1: str, col1: str, table2: str, col2, update: bool = True):
#     weights = [i for i in autofeat.weights if i.from_table == table1 
#                and i.to_table == table2 and i.from_col == col1 
#                and i.to_col == col2]
#     weights = weights + [i for i in autofeat.weights if i.from_table == table2 and i.to_table == table1 
#                          and i.from_col == col2 and i.to_col == col1]
#     if len(weights) == 0:
#         return
#     for i in weights:
#         if i in autofeat.weights:
#             autofeat.weights.remove(i)
#     if update:
#         tree_functions.rerun(autofeat)


# def update_relationship(autofeat, table1: str, col1: str, table2: str, col2: str, weight: float):
#     remove_relationship(autofeat, table1, col1, table2, col2, update=False)
#     add_relationship(autofeat, table1, col1, table2, col2, weight, update=False)
#     tree_functions.rerun(autofeat)


# def display_best_relationships(autofeat):
#     tables = autofeat.get_tables_repository()
#     highest_weights = []
#     for table1 in tables:
#         for table2 in tables:
#             if table1 == table2:
#                 highest_weights.append([autofeat.weight_string_mapping[table1], 
#                                         autofeat.weight_string_mapping[table2], 1])
#             else:
#                 weight = get_best_weight(autofeat, table1, table2)
#                 if weight is not None:
#                     highest_weights.append([autofeat.weight_string_mapping[weight.from_table], 
#                                             autofeat.weight_string_mapping[weight.to_table], 
#                                             weight.weight])
#     if len(autofeat.datasets) == 1:
#         highest_weights = [[i[0].split("/")[-1], i[1].split("/")[-1], i[2]] for i in highest_weights]
#     df = pd.DataFrame(highest_weights, columns=["from_table", "to_table", "weight"])
#     seaborn.heatmap(df.pivot(index="from_table", columns="to_table", values="weight"), square=True, cmap="PiYG",
#                     vmin=0, vmax=1)
#     plt.xlabel("")
#     plt.ylabel("")
#     plt.xticks(fontsize="small", rotation=60)
#     plt.yticks(fontsize="small", rotation=0)
#     plt.savefig("heatmap.pdf", dpi=300, bbox_inches='tight')


# def display_table_relationship(autofeat, table1: str, table2: str):
#     weights = [i for i in autofeat.weights if i.from_table == table1 and i.to_table == table2]
#     if len(weights) == 0:
#         print("No Weights found")
#         return
#     df = pd.DataFrame([[i.from_col, i.to_col, i.weight] for i in weights], 
#                       columns=["from_column", "to_column", "weight"])
#     plt.figure(figsize=(15, 3))
#     seaborn.heatmap(df.pivot(index="from_column", columns="to_column", values="weight"), square=True, cmap="PiYG", 
#                     vmin=0, vmax=1)
#     plt.xlabel(table2)
#     plt.ylabel(table1)
#     tab1_name = table1.split('/')[-1].split('.')[0]
#     tab2_name = table2.split('/')[-1].split('.')[0]
#     plt.xticks(fontsize="small", rotation=60)
#     plt.yticks(fontsize="small", rotation=0)
#     plt.savefig(f"heatmap_{tab1_name}_{tab2_name}.pdf", dpi=300, bbox_inches='tight')


# def explain_relationship(autofeat, table1: str, table2: str):
#     weights = [i for i in autofeat.weights if (i.from_table == table1 or i.from_table == table2) 
#                and (i.to_table == table1 or i.to_table == table2)]
#     rows = []
#     if len(weights) == 0:
#         print(f"There are no relationships between {table1} and {table2}.")
#         return
#     for i in weights:
#         rows.append([autofeat.weight_string_mapping[i.from_table], 
#                      autofeat.weight_string_mapping[i.to_table], i.weight])
#     print(f"Relationships between {table1} and {table2}:")
#     table = tabulate(rows, headers=["from_table", "to_table", "weight"])
#     print(table)
    

# def get_best_weight(autofeat, table1: str, table2: str) -> Weight:
#     weights = [i for i in autofeat.weights if i.from_table == table1 and i.to_table == table2]
#     if len(weights) == 0:
#         return None
#     return max(weights, key=lambda x: x.weight)


# def rerun(autofeat, threshold, matcher):
#     if len(autofeat.weights) > 0:
#         print("Relationships are recaclculated.")
#         find_relationships(autofeat, threshold, matcher)
#         tree_functions.rerun(autofeat)


if __name__ == "__main__":
    ddisc = DatasetDiscovery(matcher="Jaccard", data_repositories=["school"])
    ddisc.find_relationships()
