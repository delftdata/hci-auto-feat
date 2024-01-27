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
                 null_ratio: float, rel_red: dict, rel_red_discarded: dict):

        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.null_ratio = null_ratio
        self.rel_red = rel_red
        self.rel_red_discarded = rel_red_discarded
   
    def get_from_prefix(self):
        return self.from_table + "." + self.from_col
  
    def get_to_prefix(self):
        return self.to_table + "." + self.to_col
   
    def __str__(self) -> str:
        return "Join from " + self.from_table + "." + self.from_col + " to " \
            + self.to_table + "." + self.to_col + " with null ratio " \
            + str(self.null_ratio) + "."

    def __repr__(self) -> str:
        return f"{self.from_table}.{self.from_col} -> {self.to_table}.{self.to_col}"
 
    def explain(self) -> str:
        return "This is a join from: " + self.get_from_prefix() \
            + " to " + self.get_to_prefix() \
            + " with Null Ratio: " + str(self.null_ratio) + "." 

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
                        rel_rel_dict[name] = {"null_ratio": 0, "rel": None, "red": None}
                    rel_rel_dict[name][rel_red] = val
                    if rel_rel_dict[name]["null_ratio"] == 0:
                        rel_rel_dict[name]["null_ratio"] = i.null_ratio
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
                        rel_rel_dict[name] = {"null_ratio": 0, "rel": None, "red": None}
                    rel_rel_dict[name][rel_red] = val
                    if rel_rel_dict[name]["null_ratio"] == 0:
                        rel_rel_dict[name]["null_ratio"] = i.null_ratio
        return rel_rel_dict
    
    def show_table(self, discarded_features: bool = False):
        scores = self.get_rel_red()
        table_data = []
        for key, values in scores.items():
            row = [key, values["null_ratio"], values['rel'], values['red']]
            table_data.append(row)
        # Displaying the table 
        table = tabulate(table_data, headers=["Key", "Null Ratio", "Relevance", "Redundancy"], tablefmt="grid")
        print(table)
        if discarded_features:
            print("Discarded Features")
            discarded_scores = self.get_discarded_rel_red()
            table_data = []
            for key, values in discarded_scores.items():
                row = [key, values["null_ratio"], values['rel'], values['red']]
                table_data.append(row)
            # Displaying the table
            table = tabulate(table_data, headers=["Key", "Null Ratio", "Relevance", "Redundancy"], tablefmt="grid")
            print(table)

    def __str__(self) -> str:
        ret_string = "Begin: " + self.begin
        table = tabulate([[i.from_table + "." + i.from_col, i.to_table + "." + i.to_col, i.null_ratio] 
                          for i in self.joins], headers=["From", "To", "Null Ratio"], tablefmt="grid")
        ret_string += "\nJoins: \n" + table
        ret_string += "\nRank: " + ("%.2f" % self.rank)
        return ret_string
    
    def __repr__(self) -> str:
        return self.begin + " -> " + str(self.joins)
   
    def explain(self) -> str:
        ret_str = "The path starts at the table " + self.begin \
            + ". \n The path has the following joins: "
        for i in self.path:
            ret_str += "\n \t from (table.column) " + i.get_from_prefix() \
                + " to (table.column)" + i.get_to_prefix()
        ret_str += "\n The rank of the path is " + ("%.2f" % self.rank) + "."
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
        return "The result is calculated by evaluating the path with the AutoML algorithm AutoGluon. \
            The AutoML algorithm is run on the path " + str(self.path) \
            + ". \n The accuracy of the model is " + str(self.accuracy) \
            + ". \n The feature importance of the model is " \
            + str(self.feature_importance) + "."


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
