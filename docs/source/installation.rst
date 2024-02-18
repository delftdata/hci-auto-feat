Installation
============

Prerequisites
-------------
This package uses **python 3.10**.

Set-up
------

Install the autofeat package using pip:

.. code-block:: python

    pip install autofeat

Then you can use the package in your code by importing the FeatureDiscovery class in it:

.. code-block:: python

    from autofeat import FeatureDiscovery


Usage
-----

To use the framework create an Autofeat Class

.. code-block:: python

    autofeat = FeatureDiscovery()

Then you have to set up the base table and the target column:

.. code-block:: python

   autofeat.set_base_table(base_table="school/base.csv", target_column="class")

Furthermore, you have to select what repositories you want to use in the feature discovery process.
The repositories should be located in ./data/*repository_name*

.. code-block:: python

   autofeat.set_dataset_repository(dataset_repository=["school"])

Alternatively, you can select all repsitories in the ./data directory by using the following command:

.. code-block:: python

   autofeat.set_dataset_repository(all_tables=True)

Running
-------

Finally you can run the entire feature discovery process:

.. code-block:: python

    autofeat.augment_dataset()

The function has multiple parameters to tune your feature discovery process.


.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1
    
    * - Parameter
      - Description
      - Type
      - Default
    * - algorithm
      - | The algorithm to use for the evaluation of each found tree. Possible options are:
        | - RF (Random Forrest)
        | - GBM (Gradient Boosting Machine)
        | - XT (Extra Trees)
        | - XGB (XGBoost)
        | - KNN (K-Nearest Neighbors)
        | - LR (Logistic Regression)
      - String
      - GBM
    * - relation_threshold
      - The threshold to select relations between columns.
      - float
      - 0.5
    * - non_null_threshold
      - The threshold of non-null values in the resulting table after a possible join.
      - float
      - 0.5
    * - matcher
      - The matcher to use for the join.
      - str
      - COMA
    * - top_k_features
      - The number of top features to select from the feature discovery process.
      - int
      - 10
    * - top_k_paths
      - The number of top paths to select from the feature discovery process.
      - int
      - 3
    * - explain
      - If True, the function will print the explanation of the feature discovery process.
      - bool
      - False
    * - verbose
      - If True, the function will print the progress of the feature discovery process.
      - bool
      - False
    * - use_cache
      - If True, the function will use saved relationships to load the results of earlier relation discovery processes.
      - bool
      - True
    * - save_cache
      - If True, the function will save the relationships found in the relation discovery process.
      - bool
      - True