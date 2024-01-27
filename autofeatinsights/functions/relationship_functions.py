from functions.classes import Weight
import pandas as pd
from multiprocessing import Manager
from joblib import Parallel, delayed
import itertools
import tqdm
import seaborn
from valentine.algorithms import Coma
from valentine import valentine_match
import matplotlib.pyplot as plt


def read_relationships(self):
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


def find_relationships(autofeat, threshold: float = 0.5):
    manager = Manager()
    temp = manager.list()

    # This function calculates the COMA weights between 2 tables in the datasets.
    def calculate_coma(table1: pd.DataFrame, table2: pd.DataFrame) -> dict:
        matches = valentine_match(table1, table2, Coma())
        return matches
    
    def profile(combination):
        (table1, table2) = combination
        df1 = pd.read_csv("data/benchmark/" + table1)
        df2 = pd.read_csv("data/benchmark/" + table2)
        matches = calculate_coma(df1, df2)
        for m in matches.items():
            ((_, col_from), (_, col_to)), similarity = m
            if similarity > threshold:
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
    Parallel(n_jobs=-1)(delayed(profile)(combination) 
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


def add_relationship(autofeat, table1: str, col1: str, table2: str, col2: str, weight: float):
    autofeat.weights.append(Weight(table1, table2, col1, col2, weight))
    autofeat.weights.append(Weight(table2, table1, col2, col1, weight))


def remove_relationship(autofeat, table1: str, col1: str, table2: str, col2):
    weights = [i for i in autofeat.weights if i.from_table == table1 
               and i.to_table == table2 and i.from_col == col1 
               and i.to_col == col2]
    weights = weights + [i for i in autofeat.weights if i.from_table == table2 and i.to_table == table1 
                         and i.from_col == col2 and i.to_col == col1]
    if len(weights) == 0:
        return
    for i in weights:
        autofeat.weights.remove(i)


def update_relationship(autofeat, table1: str, col1: str, table2: str, col2: str, weight: float):
    autofeat.remove_relationship(table1, col1, table2, col2)
    autofeat.add_relationship(table1, col1, table2, col2, weight)


def display_best_relationships(autofeat):
    tables = autofeat.get_tables_repository()
    highest_weights = []
    for table1 in tables:
        for table2 in tables:
            if table1 == table2:
                highest_weights.append([autofeat.weight_string_mapping[table1], autofeat.weight_string_mapping[table2], 1])
            else:
                weight = get_best_weight(autofeat, table1, table2)
                if weight is not None:
                    highest_weights.append([autofeat.weight_string_mapping[weight.from_table], 
                                            autofeat.weight_string_mapping[weight.to_table], 
                                            weight.weight])
    df = pd.DataFrame(highest_weights, columns=["from_table", "to_table", "weight"])
    seaborn.heatmap(df.pivot(index="from_table", columns="to_table", values="weight"), square=True)
    plt.xticks(fontsize="small", rotation=30) 
    plt.show()

   
    
def get_best_weight(autofeat, table1: str, table2: str) -> Weight:
    weights = [i for i in autofeat.weights if i.from_table == table1 and i.to_table == table2]
    if len(weights) == 0:
        return None
    return max(weights, key=lambda x: x.weight)