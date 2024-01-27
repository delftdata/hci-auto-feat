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
                 data_quality: float, rel_red: dict):

        self.from_table = from_table
        self.to_table = to_table
        self.from_col = from_col
        self.to_col = to_col
        self.data_quality = data_quality
        self.rel_red = rel_red
   
    def get_from_prefix(self):
        return self.from_table + "." + self.from_col
  
    def get_to_prefix(self):
        return self.to_table + "." + self.to_col
   
    def __str__(self) -> str:
        return "Join from " + self.from_table + "." + self.from_col + " to " \
            + self.to_table + "." + self.to_col + " with data quality " \
            + str(self.data_quality) + " and rel_red " + str(self.rel_red)

    def __repr__(self) -> str:
        return f"{self.from_table}.{self.from_col} -> {self.to_table}.{self.to_col}"
 
    def explain(self) -> str:
        return "This is a join from: " + self.get_from_prefix() \
            + " to " + self.get_to_prefix() \
            + " with Data Quality: " + str(self.data_quality) \
            + " and Relevance/Redundancy " + str(self.rel_red)
 

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
                        rel_rel_dict[name] = {"rel": 0, "red": 0,
                                              "data_quality": 0}
                    rel_rel_dict[name][rel_red] = val
                    if rel_rel_dict[name]["data_quality"] == 0:
                        rel_rel_dict[name]["data_quality"] = i.data_quality
        return rel_rel_dict
    
    def showTable(self):
        scores = self.get_rel_red()
        table_data = []
        for key, values in scores.items():
            row = [key, values['rel'], values['red'], values["data_quality"]]
            table_data.append(row)
        # Displaying the table
        table = tabulate(table_data, headers=["Key", "Relevance", "Redundancy", "Data Quality"], tablefmt="grid")
        print(table)

    def __str__(self) -> str:
        ret_string = "Begin: " + self.begin
        for i in self.joins:
            ret_string += "\n \t" + str(i)
        ret_string += "\n \t rank: " + str(self.rank)
        return ret_string
    
    def __repr__(self) -> str:
        return self.begin + " -> " + str(self.joins)
   
    def explain(self) -> str:
        ret_str = "The path starts at the table " + self.begin \
            + ". \n The path has the following joins: "
        for i in self.path:
            ret_str += "\n \t from (table.column) " + i.get_from_prefix() \
                + " to (table.column)" + i.get_to_prefix()
        ret_str += "\n The rank of the path is " + str(self.rank) + "."
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
            + "\n \t rank:" + str(self.rank) \
            + " \n \t with path:" + str(self.path) \
            + " \n \t Accuracy:" + str(self.accuracy) \
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
