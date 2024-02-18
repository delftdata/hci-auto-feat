.. _use_cases:

Use Cases
=========
These use cases gives you examples to show how to use the package. Not all methods are covered. 
For that, please have a look at the :ref:`API`.

Run entire algorithm
--------------------
This code runs the entire algorithm on the repository "school" and the base table "school/base.csv" with the target column "class". 

.. code-block:: python

    from autofeatinsights.autofeat_class import FeatureDiscovery
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])
    autofeat.augment_dataset(explain=True)


Run algorithm with initermediate steps
--------------------------------------

This code runs the algorithm with the intermediate steps.

.. code-block:: python

    from autofeatinsights.autofeat_class import FeatureDiscovery
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])

    autofeat.find_relationships()
    autofeat.calculate_join_trees()
    autofeat.evaluate_trees()

Add and Remove tables from the repository
-----------------------------------------

.. code-block:: python

    from autofeatinsights.autofeat_class import FeatureDiscovery
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])
    autofeat.add_table(table="dataset/table.csv")
    autofeat.remove_table(table="dataset/table2.csv")

Calculate Relationships with Insights and Control
-------------------------------------------------

This code calculates the relationships between the columns in the base table and shows results.
Furthermore, the relationships are adjusted.

.. code-block:: python

    from autofeatinsights.autofeat_class import FeatureDiscovery
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])
    autofeat.find_relationships()
    # Shows the best relationships
    autofeat.display_best_relationships()
    # Shows the best relationship between 2 tables.
    autofeat.display_table_relationship(table1="school/base.csv", table2="school/qe.csv")
    # Removes the relationship between 2 columns in different tables.
    autofeat.remove_relationship(table1="school/ap.csv", col1="SchoolName" table2="school/qe.csv", col2="School Name")
    # Adjust the relationship between 2 tables in different tables.
    autofeat.update_relationship(table1="school/ap.csv", col1="SchoolName" table2="school/qe.csv", col2="School Name", weight=0.2)
    # Add relationship between 2 columns in different tables.
    autofeat.add_relationship(table1="school/ap.csv", col1="SchoolName" table2="school/qe.csv", col2="School Name", weight=0.2)
    # Explains the relationships betweeen 2 tables
    autofeat.explain_relationship(table1="school/ap.csv", table2="school/qe.csv") 



Calculate Join Trees with Insights and Control
----------------------------------------------

This code calculates the join trees between the columns in the base table and shows results.
Furthermore, the join trees are adjusted.

.. code-block:: python

    from autofeatinsights.autofeat_class import FeatureDiscovery
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])
    autofeat.find_relationships()
    autofeat.calculate_join_trees()
    # Shows all the trees
    autofeat.display_join_trees()
    # Shows the best tree
    autofeat.display_join_trees(top_k=1)
    # Shows a single tree by id
    autofeat.display_join_tree(tree_id=1)
    # Shows all the details of a tree by id
    autofeat.inspect_join_tree(tree_id=3)
    # Remove a table from a tree
    autofeat.remove_join_path_from_tree(tree_id=1, table="school/ap.csv")
    # Show selected features from tree 1 and with discarded features
    autofeat.show_features(tree_id=1, show_discarded=True)
    # Explains the join tree
    autofeat.explain_tree(tree_id=1)


Evaluate Join Trees with Insights and Control
---------------------------------------------
This code evaluates the join trees and shows results.

.. code-block:: python
    
    from autofeatinsights.autofeat_class import FeatureDiscovery
    autofeat = FeatureDiscovery()
    autofeat.set_base_table(base_table="school/base.csv", target_column="class")
    autofeat.set_dataset_repository(dataset_repository=["school"])
    autofeat.find_relationships()
    autofeat.calculate_join_trees()
    # Evaluate all trees
    autofeat.evaluate_trees(algorithm='GBM', top_k_paths: int = 3, verbose=True, explain=False)
    # Explains results
    autofeat.explain_result(tree_id=1, model="GBM")
    # Retuns the best result
    best_result = get_best_result()
    # Evaluate a single tree
    autofeat.evaluate_augmented_table(tree_id=1, algorithm='GBM', verbose=False)
    





   

