from functions.classes import Weight
import pandas as pd
from multiprocessing import Manager
from joblib import Parallel, delayed
import itertools
import tqdm
import seaborn
from valentine.algorithms import Coma, JaccardDistanceMatcher
from valentine import valentine_match
import matplotlib.pyplot as plt
from tabulate import tabulate
import functions.tree_functions as tree_functions


def read_relationships(self):
    self.weights = []
    f = open("weights.txt", "r")
    stringlist = f.read().split(",")
    for i in stringlist:
        if i != "":
            table1, table2, col1, col2, weight = i.split("--")
            self.weights.append(Weight(table1, table2, col1, col2, float(weight)))
    f.close()
    tables = self.get_tables_repository()
    self.weight_string_mapping = {}
    for t in tables:
        if len(t) > 20:
            new_string = t.split("/")[0] + "/" + t.split("/")[1][:3] + "..." + t.split("/")[1][-7:]
            self.weight_string_mapping[t] = new_string
        else:
            self.weight_string_mapping[t] = t


def find_relationships(autofeat, relationship_threshold: float = 0.5, matcher: str = "coma", explain=False):
    manager = Manager()
    temp = manager.list()
    autofeat.relationship_threshold = relationship_threshold
    autofeat.matcher = matcher
    
    # This function calculates the COMA weights between 2 tables in the datasets.
    def calculate_matches(table1: pd.DataFrame, table2: pd.DataFrame, matcher: str) -> dict:
        if matcher == "jaccard":
            matches = valentine_match(table1, table2, JaccardDistanceMatcher())
        else:
            matches = valentine_match(table1, table2, Coma())
        return matches
    
    def profile(combination, matcher="coma"):
        (table1, table2) = combination
        df1 = pd.read_csv("data/benchmark/" + table1)
        df2 = pd.read_csv("data/benchmark/" + table2)
        matches = calculate_matches(df1, df2, matcher)
        for m in matches.items():
            ((_, col_from), (_, col_to)), similarity = m
            if similarity > relationship_threshold:
                temp.append(Weight(table1, table2, col_from, col_to, similarity))
                temp.append(Weight(table2, table1, col_to, col_from, similarity))
    tables = autofeat.get_tables_repository()
    autofeat.weight_string_mapping = {}
    for t in tables:
        if len(t) > 20:
            new_string = t.split("/")[0] + "/" + t.split("/")[1][:3] + "..." + t.split("/")[1][-7:]
            autofeat.weight_string_mapping[t] = new_string
        else:
            autofeat.weight_string_mapping[t] = t
    if explain:
        print(f"AutoFeat computes the relationships between N tables from the {autofeat.datasets}" 
              + f" repository, using {matcher} similarity score with a threshold of {relationship_threshold}" 
              + f"(i.e., all the relationships with a similarity < {relationship_threshold} will be discarded).")
    Parallel(n_jobs=-1)(delayed(profile)(combination, matcher)
                        for combination in tqdm.tqdm(itertools.combinations(tables, 2), 
                                                     total=len(tables) * (len(tables) - 1) / 2))
    autofeat.weights = temp
    # Uncomment for saving weights to file.
    f = open("weights.txt", "w")
    stringlist = []
    for i in autofeat.weights:
        stringlist.append(f"{i.from_table}--{i.to_table}--{i.from_col}--{i.to_col}--{i.weight},")
    f.writelines(stringlist)
    f.close()


def add_relationship(autofeat, table1: str, col1: str, table2: str, col2: str, weight: float, update: bool = True):
    autofeat.weights.append(Weight(table1, table2, col1, col2, weight))
    autofeat.weights.append(Weight(table2, table1, col2, col1, weight))
    if update:
        tree_functions.rerun(autofeat)


def remove_relationship(autofeat, table1: str, col1: str, table2: str, col2, update: bool = True):
    weights = [i for i in autofeat.weights if i.from_table == table1 
               and i.to_table == table2 and i.from_col == col1 
               and i.to_col == col2]
    weights = weights + [i for i in autofeat.weights if i.from_table == table2 and i.to_table == table1 
                         and i.from_col == col2 and i.to_col == col1]
    if len(weights) == 0:
        return
    for i in weights:
        if i in autofeat.weights:
            autofeat.weights.remove(i)
    if update:
        tree_functions.rerun(autofeat)


def update_relationship(autofeat, table1: str, col1: str, table2: str, col2: str, weight: float):
    remove_relationship(autofeat, table1, col1, table2, col2, update=False)
    add_relationship(autofeat, table1, col1, table2, col2, weight, update=False)
    tree_functions.rerun(autofeat)


def display_best_relationships(autofeat):
    tables = autofeat.get_tables_repository()
    highest_weights = []
    for table1 in tables:
        for table2 in tables:
            if table1 == table2:
                highest_weights.append([autofeat.weight_string_mapping[table1], 
                                        autofeat.weight_string_mapping[table2], 1])
            else:
                weight = get_best_weight(autofeat, table1, table2)
                if weight is not None:
                    highest_weights.append([autofeat.weight_string_mapping[weight.from_table], 
                                            autofeat.weight_string_mapping[weight.to_table], 
                                            weight.weight])
    if len(autofeat.datasets) == 1:
        highest_weights = [[i[0].split("/")[-1], i[1].split("/")[-1], i[2]] for i in highest_weights]
    df = pd.DataFrame(highest_weights, columns=["from_table", "to_table", "weight"])
    seaborn.heatmap(df.pivot(index="from_table", columns="to_table", values="weight"), square=True, cmap="PiYG",
                    vmin=autofeat.relation_threshold, vmax=1)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontsize="small", rotation=30) 
    plt.savefig("heatmap.pdf", dpi=300, bbox_inches='tight')


def display_table_relationship(autofeat, table1: str, table2: str):
    weights = [i for i in autofeat.weights if i.from_table == table1 and i.to_table == table2]
    if len(weights) == 0:
        return
    df = pd.DataFrame([[i.from_col, i.to_col, i.weight] for i in weights], 
                      columns=["from_column", "to_column", "weight"])
    seaborn.heatmap(df.pivot(index="from_column", columns="to_column", values="weight"), square=True, cmap="PiYG", 
                    vmin=autofeat.relation_threshold, vmax=1)
    plt.xlabel(table2)
    plt.ylabel(table1)
    plt.xticks(fontsize="small", rotation=30) 
    plt.yticks(fontsize="small", rotation=0)
    plt.savefig("heatmap_base_crime.pdf", dpi=300, bbox_inches='tight')


def explain_relationship(autofeat, table1: str, table2: str):
    weights = [i for i in autofeat.weights if (i.from_table == table1 or i.from_table == table2) 
               and (i.to_table == table1 or i.to_table == table2)]
    rows = []
    if len(weights) == 0:
        print(f"There are no relationships between {table1} and {table2}.")
        return
    for i in weights:
        rows.append([autofeat.weight_string_mapping[i.from_table], 
                     autofeat.weight_string_mapping[i.to_table], i.weight])
    print(f"Relationships between {table1} and {table2}:")
    table = tabulate(rows, headers=["from_table", "to_table", "weight"])
    print(table)
    

def get_best_weight(autofeat, table1: str, table2: str) -> Weight:
    weights = [i for i in autofeat.weights if i.from_table == table1 and i.to_table == table2]
    if len(weights) == 0:
        return None
    return max(weights, key=lambda x: x.weight)


def rerun(autofeat, threshold, matcher):
    if len(autofeat.weights) > 0:
        print("Relationships are recaclculated.")
        find_relationships(autofeat, threshold, matcher)
        tree_functions.rerun(autofeat)
