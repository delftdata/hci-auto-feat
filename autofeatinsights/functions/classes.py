"""
This file contains the classes for the AutoFeatInsights package.
"""
import pandas as pd
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt


class Join():

    def __init__(self, from_table: str,
                 to_table: str, from_col: str, to_col: str,
                 non_null_ratio: float, rel_red: dict, rel_red_discarded: dict):

        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.non_null_ratio = non_null_ratio
        self.rel_red = rel_red
        self.rel_red_discarded = rel_red_discarded
   
    def get_from_prefix(self):
        return self.from_table + "." + self.from_col
  
    def get_to_prefix(self):
        return self.to_table + "." + self.to_col
   
    def __str__(self) -> str:
        return "Join from " + self.from_table + "." + self.from_col + " to " \
            + self.to_table + "." + self.to_col + " with Non-Null ratio " \
            + str(self.non_null_ratio) + "."

    def __repr__(self) -> str:
        return f"{self.from_table}.{self.from_col} -> {self.to_table}.{self.to_col}"
 
    def explain(self) -> str:
        return "This is a join from: " + self.get_from_prefix() \
            + " to " + self.get_to_prefix() \
            + " with Non-Null Ratio: " + str(self.nnon_ull_ratio) + "." 


class Path:

    def __init__(self, begin: str, joins: [Join] = None, rank: float = None):
        self.begin = begin
        if joins is None:
            self.joins = []
        else:
            self.joins = joins
        self.rank = rank
    
    def add_join(self, join):
        self.joins.append(join)

    def get_rel_red(self):
        rel_rel_dict = {}
        i: Join
        for i in self.joins:
            # dict[i.to_table + "." + i.to_col] = {}
            for rel_red in i.rel_red:
                for j in i.rel_red[rel_red]:
                    name, val = j
                    if name not in rel_rel_dict:
                        rel_rel_dict[name] = {"non_null_ratio": 0, "rel": None, "red": None}
                    rel_rel_dict[name][rel_red] = val
                    if rel_rel_dict[name]["non_null_ratio"] == 0:
                        rel_rel_dict[name]["non_null_ratio"] = i.non_null_ratio
        return rel_rel_dict
    
    def get_discarded_rel_red(self):
        rel_rel_dict = {}
        i: Join
        for i in self.joins:
            # dict[i.to_table + "." + i.to_col] = {}
            for rel_red in i.rel_red_discarded:
                for j in i.rel_red_discarded[rel_red]:
                    name, val = j
                    if name not in rel_rel_dict:
                        rel_rel_dict[name] = {"non_null_ratio": 0, "rel": None, "red": None}
                    rel_rel_dict[name][rel_red] = val
                    if rel_rel_dict[name]["non_null_ratio"] == 0:
                        rel_rel_dict[name]["non_null_ratio"] = i.non_null_ratio
        return rel_rel_dict
    
    def show_table(self, discarded_features: bool = False):
        scores = self.get_rel_red()
        table_data = []
        for key, values in scores.items():
            if len(key) > 20:
                key = key[:10] + "..." + key[-10:]
            row = [key, values["non_null_ratio"], values['rel'], values['red']]
            table_data.append(row)
        # Displaying the table 
        table = tabulate(table_data, headers=["Key", "Non-Null Ratio", "Relevance", "Redundancy"], tablefmt="grid")
        print(table)
        if discarded_features:
            print("Discarded Features")
            discarded_scores = self.get_discarded_rel_red()
            table_data = []
            for key, values in discarded_scores.items():
                if len(key) > 20:
                    key = key[:10] + "..." + key[-10:]
                row = [key, values["non_null_ratio"], values['rel'], values['red']]
                table_data.append(row)
            # Displaying the table
            table = tabulate(table_data, headers=["Key", "Non-Null Ratio", "Relevance", "Redundancy"], tablefmt="grid")
            print(table)

    def __str__(self) -> str:
        ret_string = "Rank: " + ("%.2f" % self.rank)
        ret_string += "\nFrom base table: " + self.begin
        table = tabulate([[i.from_table + "." + i.from_col, i.to_table + "." + i.to_col, i.non_null_ratio] 
                          for i in self.joins], headers=["From Table.Column", "To Table.Column", "Non-Null Ratio"], tablefmt="grid")
        ret_string += "\nJoin paths: \n" + table
        return ret_string
    
    def __repr__(self) -> str:
        return self.begin + " -> " + str(self.joins)
   
    def explain(self) -> str:
        ret_str = "The join tree starts at the table " + self.begin \
            + ". \nThe join tree has the following joins: "
        for i in self.joins:
            ret_str += "\n \t from (table.column) " + i.get_from_prefix() \
                + " to (table.column)" + i.get_to_prefix() + " with Non-Null ratio " \
                + ("%.2f" % i.non_null_ratio) + "."
        ret_str += "\nThe rank of the path is " + ("%.2f" % self.rank) + "."
        return ret_str
    

class Result():

    # rank: float
    path: Path
    accuracy: float
    feature_importance: dict
    model: str
    data: pd.DataFrame

    def __init__(self):
        self.feature_importance = {}

    def getFeatureImportance(self, tableName=None, featureName=None):
        if tableName is None and featureName is None:
            return self.result.feature_importance
        elif tableName is not None and featureName is None:
            return self.getFeatureImportanceByTable(self, tableName)
        elif tableName is not None and featureName is not None:
            return self.getFeatureImportanceByTableAndFeature(self, tableName,
                                                              featureName)
    
    def getFeatureImportanceByTable(self, tableName: str):
        list = []
        for i in self.feature_importance:
            if tableName in i:
                list.append(i)
        return list
    
    def getFeatureImportanceByTableAndFeature(self, tableName: str,
                                              featureName: str):
        return self.feature_importance[tableName + "." + featureName]
    
    def show_graph(self, ax=None, plot=True):
        G = nx.Graph()
        if len(self.path.joins) == 0:
            G.add_node(self.path.begin)
        else:
            for i in self.path.joins:
                G.add_edge(i.from_table, i.to_table)
        if plot:
            plt.figure()
            plt.title(self.model + " : " + str(self.accuracy) + " accuracy")
        if ax is None:
            nx.draw(G, with_labels=True, font_weight='bold')
        else:
            nx.draw(G, ax=ax, with_labels=True, font_weight='bold')
            ax.set_title(self.model + ": " + str(self.accuracy) + " accuracy")
        if plot:
            plt.show()

    def __str__(self) -> str:
        ret_string = "Result with model" + self.model \
            + "\n \t rank:" + ("%.2f" % self.rank) \
            + " \n \t with path:" + str(self.path) \
            + " \n \t Accuracy:" + ("%.2f" % self.accuracy) \
            + " \n \t Feature importance:"
        for i in self.feature_importance[0]:
            ret_string += "\n \t \t " + str(i) + " : " \
                + str(self.feature_importance[0][i])
        return ret_string
    
    def explain(self) -> str:
        feature_list = []
        for i in self.feature_importance[0]:
            if len(i) > 20:
                i_string = i[:10] + "..." + i[-10:]
            else:
                i_string = i
            feature_list.append([i_string, self.feature_importance[0][i]])
        return "The result is calculated by evaluating the path with the AutoML algorithm AutoGluon." \
            + ". \n The accuracy of the model is " + str(self.accuracy) \
            + ". \n The feature importance of the model is \n" \
            + tabulate(feature_list, headers=["Feature", "Importance"], tablefmt="grid")


class Weight():
    from_table: str
    to_table: str
    from_col: str
    to_col: str
    weight: float

    def __init__(self, from_table, to_table, from_col, to_col, weight):
        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.weight = weight

    def get_from_prefix(self):
        return self.from_table + "." + self.from_col
    
    def get_to_prefix(self):
        return self.to_table + "." + self.to_col
    
    def __str__(self) -> str:
        return "Weight from " + self.get_from_prefix() \
            + " to " + self.get_to_prefix() \
            + " with weight " + str(self.weight)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def explain(self) -> str:
        return "This weight is calculated by the COMA algorithm. \
            This calculates the similarity between the columns of the tables. \
            The higher the similarity, the higher the weight. \n \
            The weight from " + self.getFromPrefix() + " to " + self.getToPrefix() + " is " + str(self.weight) + "."
