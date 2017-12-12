# H2O AutoML Tutorial

AutoML is a function in H2O that automates the process of building a large number of models, with the goal of finding the "best" model without any prior knowledge or effort by the Data Scientist.  

The current version of AutoML (in H2O 3.16.*) trains and cross-validates a default Random Forest, an Extremely-Randomized Forest, a random grid of Gradient Boosting Machines (GBMs), a random grid of Deep Neural Nets, a fixed grid of GLMs, and then trains two Stacked Ensemble models at the end. One ensemble contains all the models (optimized for model performance), and the second ensemble contains just the best performing model from each algorithm class/family (optimized for production use).

- More information and code examples are available in the [AutoML User Guide](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html).

- New features and improvements planned for AutoML are listed [here](https://0xdata.atlassian.net/issues/?filter=21603).

## Part 1: Binary Classification

For the AutoML binary classification demo, we use a subset of the [Product Backorders](https://www.kaggle.com/tiredgeek/predict-bo-trial/data) dataset.  The goal here is to predict whether or not a product will be put on backorder status, given a number of product metrics such as current inventory, transit time, demand forecasts and prior sales.

In this tutorial, you will:

- Specify a training frame.
- Specify the response variable and predictor variables.
- Run AutoML where stopping is based on max number of models.
- View the leaderboard (based on cross-validation metrics).
- Explore the ensemble composition.
- Save the leader model (binary format & MOJO format).

Demo Notebooks:

 - [R/automl\_binary\_classification\_product\_backorders.Rmd](./R/automl_binary_classification_product_backorders.Rmd)  [<img src="https://www.r-project.org//favicon-16x16.png" width=18>](https://www.r-project.org//favicon-16x16.png)
 - [Python/automl\_binary\_classification\_product\_backorders.ipynb](./Python/automl_binary_classification_product_backorders.ipynb) [<img src="https://www.python.org/static/favicon.ico"  width=16>](https://www.python.org/static/favicon.ico)


## Part 2: Regression

For the AutoML regression demo, we use the [Combined Cycle Power Plant](http://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) dataset.  The goal here is to predict the energy output (in megawatts), given the temperature, ambient pressure, relative humidity and exhaust vacuum values.  In this demo, you will use H2O's AutoML to outperform the [state-of-the-art results](https://www.sciencedirect.com/science/article/pii/S0142061514000908) on this task.

In this tutorial, you will:

- Split the data into train/test sets.
- Specify a training frame and leaderboard (test) frame.
- Specify the response variable.
- Run AutoML where stopping is based on max runtime, using training frame (80%).
- Run AutoML where stopping is based on max runtime, using original frame (100%).
- View leaderboard (based on test set metrics).
- Compare the leaderboards of the two AutoML runs.
- Predict using the AutoML leader model.
- Compute performance of the AutoML leader model on a test set.

Demo Notebooks:

 - [R/automl\_regression\_powerplant\_output.Rmd](./R/automl_regression_powerplant_output.Rmd) [<img src="https://www.r-project.org//favicon-16x16.png" width=18>](https://www.r-project.org//favicon-16x16.png)
 - [Python/automl\_regression\_powerplant\_output.ipynb](./Python/automl_regression_powerplant_output.ipynb) [<img src="https://www.python.org/static/favicon.ico"  width=16>](https://www.python.org/static/favicon.ico)
